## MADA 3.0 Roadmap

### 1. Context

- Latest benchmark (`exp-12-03_001917-…`) shows high-performing parents like `AdaptiveDE` still meet the four-block signature contract yet expose richer behavior through auxiliary helpers (`archive`, stochastic ranking, adaptive parameters).
- Current parser only captures blocks whose method names literally match `parent_selection`, `recombination`, `mutation`, `survivor_selection`, so sophisticated logic hidden behind other names collapses to placeholders. The log excerpt confirms we lose archive-aware mutation logic even though it exists in practice.
- DS-TS bandits cold-start every run, which makes early MADA generations behave like random recombination of placeholders. Roulette-style survivor selection also keeps resurfacing because prompts do not discourage it, harming minimization tasks.

### 2. Extraction & Instrumentation (Phase 1)

1. **Semantic Block Matching**
   - Extend `BlockParser` to classify _every_ `ast.FunctionDef` using heuristics (name similarity, signature/parameter pattern, keywords like `parents`, `archive`, `fitness`).
   - When heuristics are inconclusive, fall back to a lightweight LLM tagger to map methods (e.g., `tournament_select`) to canonical roles.
   - Preserve the original method name plus detected role in `ParsedAlgorithm` so recomposition can restore helper references faithfully.
2. **Dependency Capture**
   - During parsing, record helper usage: which methods reference `self.archive`, which helper functions are called, and what class attributes are read/written.
   - Include this metadata alongside the block snippets to guide later phases that preserve shared state.
3. **Coverage Logging**
   - Store per-individual metrics (`real_blocks`, `placeholder_blocks`, helper counts) inside `Solution.metadata["mada_blocks"]["metrics"]`.
   - Aggregate stats in `MADAOperator.ensure_blocks` (e.g., rolling average of placeholder ratios) so experiments can verify Phase 1 impact.

### 3. Shared State & Prompt Guardrails (Phase 2)

1. **Helper Bundles**
   - Promote frequently referenced helpers (`update_archive`, `rank_survivors`) and class attributes into explicit bundles tied to each block.
   - Update `BlockParser.assemble` so when a block override depends on a helper bundle, the bundle is carried over automatically or cloned from the donor.
2. **Prompt Updates**
   - Modify `_request_innovation_block` prompts to insist that global state changes live in helper methods or `__init__`, and that blocks merely _invoke_ those helpers.
   - Reject innovations that introduce hidden globals or omit required helpers; fall back to donor code with a neutral reward if validation fails.
3. **State Compatibility Checks**
   - Before final assembly, verify that every referenced helper/function/attribute exists; if not, bring it along from the source block or abort the recombination prior to evaluation.

### 4. Hybrid Warm-Start & Bandit Telemetry (Phase 3)

1. **Warm LLAMEA Run**
   - Let vanilla LLaMEA run 10–15 generations (or until a configurable AOCC threshold) to build a diverse parent archive.
   - Parse these parents with the improved semantic parser and treat their block performances as pseudo-observations.
2. **Seeding DS-TS**
   - Initialize each block’s bandit statistics (`discounted_count`, `discounted_sum`) with the observed AOCCs so priors reflect actual block quality (e.g., CMA-style survivor selection starts with a higher mean than placeholders).
   - Provide CLI knobs (`--mada-warmstart-gens`, `--mada-prior-weight`) to control how much influence the warm-start data has.
3. **Reward Diagnostics**
   - Log per-block reward baselines, best-parent fitness, and child deltas in the lineage metadata.
   - Emit experiment-level telemetry (histograms of rewards per arm, placeholder ratios, helper coverage) to confirm warm-start shortens exploration time.

### 5. Survivor Selection & Partial Recombination (Phase 4)

1. **Prompt Guardrails**
   - Bias survivor-selection prompts toward deterministic truncation, rank-based elitism, or inverse-fitness probabilities; automatically reject roulette/fitness-proportional schemes during `_valid_block`.
2. **Per-Block Decisions**
   - Allow DS-TS to decide _per block_ whether to replace it; untouched blocks should be copied verbatim from the base template to avoid breaking cohesive modules when only one component needs tweaking.
   - Record how many blocks changed per offspring and correlate with resulting rewards to detect when “full rewrites” hurt performance.
3. **Helper Blocks as First-Class Citizens**
   - Treat complex helpers (`archive_update`, `local_search_step`) as optional block types so mutation/survivor-selection can carry their dependencies together.

### 6. Validation Loop

- After completing each phase, rerun the ≥50-generation benchmark and compare AOCC curves to the 0.54 baseline.
- Use the new coverage and bandit telemetry to verify improvements (e.g., increased fraction of real blocks, reduced placeholder usage, faster convergence due to warm-start).
- Document results per phase inside the thesis repo for clean ablation studies.

This file consolidates the “highest-impact steps” plus actionable phases so the next implementation pass can follow a deterministic checklist toward MADA 3.0.
