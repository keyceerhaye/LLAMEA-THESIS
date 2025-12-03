## MADA 3.0 Implementation Plan

This plan translates the roadmap in `mada3.0.md` into concrete code changes for `LLAMEA-THESIS/LLAMEA-MADA`. Each phase lists the relevant modules, the refactors to perform, fallbacks to preserve benchmarking integrity, and the validation steps required to ensure AOCC/fitness logging stays correct.

### Baseline Architecture Check

- **Population loop** – `llamea-experimentation/experiments/benchmarks/main-thesis.py` (and sibling scripts under `experiments/benchmarks/`) orchestrate initialization, evaluation, and AOCC logging via `evaluate_algorithm`. Any MADA changes must keep the interface `code → class name → fitness` untouched so log files (`log.jsonl`, `try-*-aucs.txt`) remain consistent.
- **MADA stack** – `src/llamea/mada/parser.py`, `src/llamea/mada/operator.py`, `src/llamea/mada/ds_ts.py` and `src/llamea/solution.py` already contain the extraction logic, bandit controller, sampler math, and metadata helpers that downstream benchmarking depends on.
- **LLM contracts** – `src/llamea/llm.py` and `managers.py` (in the root thesis folder) supply `generate_block_snippet`, `refine_algorithm`, etc. When prompts are updated we must preserve the format `# Name/# Code` and code fences, otherwise `AlgorithmManager.extract_algorithm_code` will throw and benchmarking halts.

### Phase 1 – Semantic Extraction & Instrumentation

1. **Introduce a classifier helper**
   - File: `src/llamea/mada/parser.py`
   - Add `_classify_method(node: ast.FunctionDef) -> tuple[str | None, float]` that scores each function against the four canonical roles using heuristics:
     - Name similarity (`"select"`, `"mutation"`, `"recombine"`, etc.).
     - Signature shape (argument names like `population`, `offspring`, `candidate`).
     - Keyword frequency (search for `self.archive`, `parents`, `fitness`).
   - Keep a fallback `_llm_labeler(code_snippet)` gated behind a config flag so we can optionally call a tiny LLM for ambiguous cases without blocking offline runs.
2. **Store expanded metadata**
   - Extend `ParsedAlgorithm` to include:
     - `method_roles: Dict[str, str]` mapping original names to canonical roles.
     - `dependencies: Dict[str, Dict[str, set]]` capturing helper calls (`self.update_archive()`), class attributes touched, and imported symbols.
   - Update `BlockParser.extract` to:
     - Iterate through all methods, call `_classify_method`, and populate `blocks` even when the method name differs.
     - Record placeholder insertions along with reasons (e.g., “no candidate block found” vs “ambiguous classification”).
3. **Coverage metrics**
   - After extraction, compute counts: number of real blocks, placeholder blocks, helper functions detected.
   - Persist them via `ParsedAlgorithm.to_metadata()` and the existing `Solution.set_mada_blocks` call inside `MADAOperator.ensure_blocks`.
   - Add light-weight logging (e.g., `self.algorithm_manager.logger.log_conversation`) annotating placeholder ratios whenever they exceed a threshold; this gives immediate feedback during benchmarks.
4. **Fallback safety**
   - Ensure `_synthetic_algorithm` still returns valid placeholders so `evaluate_algorithm` never receives incomplete code.
   - Gate all new heuristics behind try/except to fall back on literal matching if AST analysis fails; this prevents crashes mid-benchmark.

### Phase 2 – Shared State Preservation & Prompt Guardrails

1. **Helper bundle representation**
   - Augment `ParsedAlgorithm` with a `helper_groups` structure: for each block, record the helper methods and class attributes it references (from Phase 1 dependencies).
   - Update `BlockParser.assemble` to accept an additional `helper_overrides` map. When a block override requires helpers not present in the base class, automatically splice in the donor’s helper definitions and ensure `__init__` contains required attribute initializations (inject missing assignments).
2. **Compatibility checks**
   - In `MADAOperator._recombination_offspring` and `_innovative_offspring`:
     - After deciding the new block, call a `_verify_dependencies(block_name, parsed_template, overrides)` helper that ensures every referenced helper/attribute exists; otherwise clone it from the snippet’s source.
     - If cloning fails (e.g., helper defined outside the class), skip that block change and mark the decision with `reward_override = 0` so DS-TS isn’t misled.
3. **Prompt updates (mandatory)**
   - Yes, we must update the LLM prompts so generated code respects shared-state rules and improved survivor selection. Without this, the parser and helper checks introduced above will keep rejecting snippets, stalling evolution.
   - **Where to edit**
     - `src/llamea/mada/operator.py::_request_innovation_block`: rewrite the prompt body to include the bullet points below and to embed a serialized description of required helpers/attributes.
     - `LLAMEA-THESIS/managers.py::refine_algorithm` (and any mutation prompts reused elsewhere): append a short paragraph after the performance feedback reminding the LLM to keep global state changes inside helper methods/`__init__`, reuse helper calls, and avoid roulette-style survivor selection.
     - If `llamea/llm.py` contains reusable prompt templates (e.g., `self.output_format_prompt`, mutation prompts), add the same guardrails there so legacy modes remain compatible.
   - **Required wording to include**
     ```
     - Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.
     - If the block depends on helpers such as update_archive(), call them instead of duplicating logic.
     - Avoid fitness-proportional (roulette) survivor selection; prefer deterministic truncation, rank-based, or inverse-fitness probabilities.
     ```
   - Update `AlgorithmManager.generate_block_snippet` call sites to pass descriptions of required helpers (grabbed from `helper_groups`) so the LLM knows which functions already exist and can reference them directly.
4. **Regression safety**
   - Add unit-style smoke tests (even simple ones) under `llamea-experimentation/tests` covering:
     - Recombination between two mock algorithms where mutation references `self.archive`.
     - Verified assembly retains helper code.
   - Preserve existing AOCC evaluation by ensuring assembled modules still define a single optimizer class consumed by `evaluate_algorithm`.

### Phase 3 – Hybrid Warm-Start & Bandit Telemetry

1. **Warm-start pipeline**
   - In the benchmark driver (e.g., `experiments/benchmarks/main-thesis.py` and CLI utilities inside `llamea-experimentation/docs/…`):
     - Add CLI flags (`--mada-warmstart-gens`, `--mada-prior-weight`, `--mada-enable-warmstart`).
     - Modify the initialization phase so, before instantiating `MADAOperator`, we can:
       - Run standard LLaMEA for N generations (reuse existing evolutionary loop).
       - Serialize the resulting parents’ code/fitness to a cache (JSON file inside the experiment directory).
2. **Seeding DS-TS**
   - Extend `DiscountedThompsonSampler` with a `seed(block_id, arm_name, reward, weight)` method that directly manipulates `discounted_count` and `discounted_sum`.
   - After warm-start extraction, call `MADAOperator.seed_bandits(parent_solutions)` which:
     - Ensures `ensure_blocks` is run on the archived parents to get block hashes and their AOCC scores.
     - Calls `seed` with `reward = parent.fitness` (or normalized delta) scaled by `--mada-prior-weight`.
   - Persist the seeded state in the logger for reproducibility (write to `experiments/.../log.jsonl` or a sidecar file).
3. **Telemetry hooks**
   - Add a `diagnostics` entry to `OffspringProposal` with:
     - `placeholder_ratio`
     - `helper_bundles_attached`
     - `blocks_changed`
     - `bandit_snapshot` (pull from `DiscountedThompsonSampler.get_state_snapshot`)
   - Flush these diagnostics into the run log (`experiment_logger.log_conversation` or JSON) so we can trace DS-TS behavior post-warm-start.
4. **Ensuring AOCC continuity**
   - Warm-start must not consume the primary benchmarking budget; document the sequence in README/CLI help so that official runs still evaluate exactly `budget` algorithms used for AOCC comparison.
   - Guard the warm-start code behind CLI flags; default behavior remains the current one to avoid breaking old scripts.

### Phase 4 – Survivor Selection Upgrades & Partial Recombination

1. **Per-block DS-TS decisions**
   - Refactor `_innovative_offspring`/`_recombination_offspring` loops to:
     - Sample arms per block as today.
     - Introduce a “no-op/retain” arm that simply copies the template block without change.
     - Update the lineage entry to record `changed: bool`.
2. **Roulette rejection**
   - Enhance `_valid_block` in `operator.py` to parse the snippet AST and detect fitness-proportional randomness (e.g., look for `np.random.choice` with weights derived directly from `fitness`). Reject such blocks and fallback to donor implementations with `reward_override = 0`.
   - Update prompts (from Phase 2) to encourage deterministic or inverse-fitness schemes.
3. **Helper modules as blocks**
   - Extend `BlockParser.target_blocks` to include optional helper categories (e.g., `"archive_helpers"`, `"local_search_helpers"`). These can be treated as DS-TS-controlled “modules” so that dependent survivor-selection/mutation blocks keep their siblings.
   - Update `MADAOperator.bandits` to include the new block ids and ensure `assemble` injects helper modules at the right location (before the canonical methods).
4. **Selection safety**
   - Ensure `Solution.metadata["mada_lineage"]` captures the number of changed blocks; when the benchmarking script computes rewards (`child.fitness - best_parent.fitness`), include this metric in the logged lineage to correlate performance with scope of changes.
   - Maintain legacy strategy weights (40/40/20) but allow CLI overrides so experiments can disable partial recombination if needed.

### Validation & Testing Checklist

1. **Unit / component tests**
   - Add parser tests covering semantic classification, helper bundling, and placeholder metrics (e.g., `tests/test_mada_parser.py` – create if absent).
   - Add DS-TS seeding tests verifying `seed()` updates the posterior as expected.
   - Add operator tests to ensure `_verify_dependencies` brings helpers along and that “retain block” arms leave code unchanged.
2. **End-to-end smoke**
   - Run `python experiments/benchmarks/main-thesis.py ... --budget 8 --mada-enable-warmstart` using the Dummy LLM to ensure no crashes.
   - Run a short real benchmark (≥ 50 evaluations) and confirm:
     - `log.jsonl` contains diagnostics entries.
     - AOCC values are still computed and stored in `try-*-aucs.txt`.
3. **Performance regression guard**
   - Track placeholder ratios and AOCC progress before/after each phase; a sudden drop indicates a parsing or assembly issue.
   - Keep the warm-start optional so historical comparisons (0.54 baseline) remain achievable.

### Execution Order Summary

1. Implement Phase 1 parser/metrics changes → add tests → run smoke benchmark.
2. Layer Phase 2 helper preservation and prompt updates → re-run tests + smoke.
3. Add Phase 3 warm-start seeding and telemetry (behind flags) → validate DS-TS state dumps.
4. Finish with Phase 4 per-block recombination and survivor-selection safeguards → final long benchmark for thesis numbers.

Following this plan ensures each enhancement is isolated, testable, and backwards-compatible with the existing benchmarking tooling, so AOCC calculations and log formats remain intact.
