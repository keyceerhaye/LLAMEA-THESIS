# MADA Long-Term Fix Plan

This plan translates the issues captured in `MADA-SUGGESTIONS.md` into actionable implementation workstreams so MADA-LLAMEA consistently produces modular algorithms, unlocks recombination/innovation, and stays observable.

## 1. Objectives

- Guarantee every generated optimizer exposes concrete implementations of the four core blocks (`parent_selection`, `recombination`, `mutation`, `survivor_selection`) plus any helpers they rely on.
- Keep the MADA operator adaptive instead of collapsing into legacy-only fallbacks.
- Surface structure/strategy metrics each generation to guide future tuning.

## 2. Workstreams

### 2.1 Prompt & Template Redesign

1. **Scaffold injection**

   - Update `AlgorithmManager._base_init_prompt` and `refine_algorithm` (both in `LLAMEA-THESIS/LLAMEA-MADA/managers.py`) so the LLM must return a class template containing the four target blocks, stubs for helpers, and clear instructions about reusing helper calls instead of inlining logic.
   - Provide a starter template (e.g., `templates/mada_optimizer_base.py`) with placeholder implementations that reference helper hooks; feed it into `last_algorithm` before refinements so the LLM edits concrete code rather than starting from scratch.

2. **Block-aware innovation prompts**

   - Extend `_request_innovation_block` in `llamea-experimentation/src/llamea/mada/operator.py` to always include the helper/attribute context plus explicit “required helpers” bullet lists. Propagate parser coverage stats into the guardrail block so the LLM sees when a block is missing entirely.

3. **Guardrail feedback loop**
   - After each parent is parsed, if `placeholder_ratio >= 0.3`, push a `high_placeholder_ratio` note into `AlgorithmManager.update_guardrail_feedback`. The next legacy refinement then carries a system reminder to decompose the logic.

### 2.2 Operator Tuning

1. **Placeholder thresholds**
   - Once the scaffold is enforced, lower `placeholder_block_threshold` back to ~0.3; until then, gate it on `parsed.coverage["real_blocks"]` (e.g., allow recombination whenever both parents expose ≥2 non-placeholder blocks).
2. **Partial innovation rewards**
   - Replace the current “abort on ≥50 % fallbacks” rule with per-block rewards: accept the child, set `reward_override = 0` only for reverted blocks, and let DS-TS update arms for blocks that truly changed.
3. **Strategy availability**
   - Decouple API transport failures from `record_strategy_outcome` by retrying LLM calls upstream; only increment failure counts when the generated code cannot be parsed or evaluated.

### 2.4 Monitoring & Tooling

Add a lightweight diagnostics module writing to `exp-*/monitoring.json` per generation:

- Block coverage (`real_blocks`, `placeholder_ratio`) for each parent.
- Strategy counts per generation (legacy / innovation / recombination) and DS-TS arm probabilities per block.
- Innovation failure taxonomy (dependency issues, parser errors, evaluation crashes).

Wire this into the experiment logger (e.g., `ExperimentLogger.log_monitoring_snapshot`) so we can trend regressions over multiple runs.

## 3. Implementation Sequence

1. **Template + prompt update** (Workstream 2.1) – unblock real block coverage.
2. **Operator adjustments** (2.2) – let recombination/innovation run on the new structure.
3. **Monitoring hooks** (2.4) – emit metrics once the pipeline produces meaningful signals.

Each stage should be verified by rerunning a short-budget experiment and inspecting:

- Parser coverage report (expect placeholder ratio < 0.25 by generation 3).
- `reward-log.csv` showing non-zero recombination rows and improving innovation rewards.
- Monitoring snapshot confirming DS-TS exploration and surfacing any blocks still stuck on placeholders.

Executing this plan will transition MADA-LLAMEA from a legacy-only search into a modular, observable evolutionary system that can keep improving with future experimentation.
