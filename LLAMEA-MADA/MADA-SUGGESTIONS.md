# MADA Suggestions – 2025-12-04

## 1. Observed Operator Usage
- `reward-log.csv` in `exp-12-04_180002-gemini-2.0-flash-thesis-experiment-evolutionary-elitism` records **71** legacy offspring, **12** innovation offspring, and **0** recombination offspring (no rows contain `,recombination,`).
- Every reported “🎉 NEW BEST” in the run log was produced via legacy refinement; recombination never executed and innovation attempts rarely survived evaluation.

## 2. Why Legacy Dominates
1. **Missing block structure**  
   - The prompts in `managers.py` only demand `__init__` and `__call__`, so generated optimizers (see `BEST_ALGORITHM.py`) are monolithic classes without `parent_selection`, `recombination`, `mutation`, or `survivor_selection`.  
   - The block parser therefore fills each target block with placeholders, pushing the placeholder ratio above the thresholds used by MADA.

2. **Recombination is pre-emptively aborted**  
   - `_recombination_offspring` raises `PlaceholderAbort` whenever either parent exceeds `self.placeholder_block_threshold = 0.3`. Because almost every method is a placeholder, recombination instantly falls back to `_legacy_offspring`, so DS-TS never sees a recombination reward.

3. **Innovation collapses to α**  
   - `_innovative_offspring` throws `InnovationAbort` once half the blocks revert to the α parent. With no helper methods or stored attributes to borrow, dependency checks in `_verify_dependencies` fail, so most blocks fall back and the abort path is triggered.
   - Even when innovation produces code, the offspring often crashes during evaluation (rows 3, 49, 58, 71, 72, 80 in `reward-log.csv`), clipping the reward to −0.3 and teaching DS-TS to avoid the innovation arm.

4. **External API instability disables exploration**  
   - Between API calls 47–77 the transcript shows repeated HTTP 500/`ECONNREFUSED` errors. Each exception increments MADA’s `strategy_failure_counts`; after two failures the affected strategy is cooled down for five offspring (`record_strategy_outcome`). This shrinks the menu to whatever arm (usually legacy) still functions.
   - Stagnation handling (`force_strategy = "legacy"`) also injects guaranteed legacy offspring whenever the boost window is active, further biasing the counters toward legacy.

## 3. Consequences
- Because the only stable path is legacy refinement, all improvements (e.g., `AdaptiveDifferentialEvolutionWithArchiveAndRestartV13`) come from monolithic rewrites rather than MADA’s block-level innovation. The evolutionary loop effectively degenerates into “legacy-only,” wasting the recombination/innovation infrastructure and keeping DS-TS priors stagnant.

## 4. Recommendations

### 4.1 Prompt and Template Changes (highest impact)
- **Enforce the four-block scaffold at generation time.** Update both initialization and refinement prompts (`AlgorithmManager._base_init_prompt` and `refine_algorithm`) to *require* concrete implementations of `parent_selection`, `recombination`, `mutation`, and `survivor_selection`, or inject a template class that already defines these stubs before the LLM mutates it.
- **Provide helper descriptions in innovation prompts.** Expand `_request_innovation_block` to pass the template’s helper list and attribute requirements so generated snippets can satisfy `_verify_dependencies`.
- **Reject monolithic submissions early.** After parsing a parent, warn the LLM (via guardrail feedback) when placeholder ratios exceed 0.3, nudging it to supply decomposed logic in subsequent refinements.

### 4.2 Operator Guardrail Adjustments (short term)
- **Temporarily relax placeholder thresholds** (e.g., raise `placeholder_block_threshold` to 0.8) so recombination can at least run while prompts are being improved. Expect low-quality offspring until modular code arrives, but DS-TS will begin collecting data.
- **Allow partial innovation rewards.** Instead of aborting when ≥50 % of blocks fall back, accept the partial child, set `reward_override` to 0 for the reverted blocks, and let DS-TS update the few blocks that did change.
- **Detect structural issues once per generation** and prefer recombination/innovation only when the selected parents expose ≥2 real blocks, preventing wasted API calls.

### 4.3 Robustness to API Failures
- Wrap `generate_block_snippet`/`refine_algorithm` requests with exponential backoff outside of `record_strategy_outcome` so transient HTTP 500s do not immediately disable exploration strategies.
- Consider caching the last valid snippet per block so innovation can retry locally when the API endpoint is down.

### 4.4 Monitoring
- Add a summary table (per generation) of block coverage and strategy usage to the experiment log to flag when MADA regresses to legacy-only behavior.
- Track DS-TS arm probabilities per block; alert when any block’s innovation arm goes stale for >N generations.

Implementing the prompt/template changes is the critical path: once parents consistently expose real block implementations, the existing MADA operator (bandits, dependency checks, lineage logging) will have meaningful material to recombine and innovate on, reducing reliance on legacy rewrites.

