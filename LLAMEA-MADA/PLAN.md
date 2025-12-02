# MADA Integration Plan

## Objective

- Embed the MADA pipeline (Discounted Thompson Sampling + AST-driven recomposition) into the evolutionary stack inside `LLAMEA-THESIS/LLAMEA-MADA`, ensuring the new operator coexists with the current LLaMEA refinement loop without deleting legacy behaviors.
- Conform to the Discounted Thompson Sampling (DS-TS) math described in _Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery_ ([PDF](file:///C:\Users\Kukoy\Documents\MADA-LLAMEA-IMPLEMENTATION\LLAMEA-THESIS\LLAMEA-MADA\D-TS.pdf)).

## Mandatory Integration Points

### `llamea/llamea.py`

- Keep `LLaMEA.run` unchanged for the thesis driver but expose MADA hooks by:
  - Using `Solution.metadata` to stash `mada_lineage` (arm choices + block provenance) produced by the new operator so downstream evaluation and logging can access it.
  - Relying on `Solution.copy()/empty_copy()` to propagate parent IDs so MADA can attribute rewards correctly.
- No direct modifications are required for core sampling logic, but the file constrains how far we can diverge from Solution semantics.

### `llamea/solution.py`

- `Solution.metadata` already exists and will store:
  - `mada_blocks`: extracted AST segments per block.
  - `mada_lineage`: for each block, the arm (`alpha`, `beta`, `innovation`) and source hash.
  - `mada_reward_ready`: boolean for whether AOCC deltas were computed.
- We will add thin helpers (e.g., `set_lineage`, `get_lineage`) so other modules do not touch `metadata` directly.

### `benchmarks/main-evolutionary.py`

- Used as reference for how `LLaMEA` expects evaluation functions (`evaluate_population_fitness`). Confirms that AOCC is maximized and reinforces the need to keep reward sign conventions consistent.
- No direct edits planned, but it validates how AOCC scores are computed (via `correct_aoc`) which MADA must treat as the reward surface.

### `llamea-experimentation/experiments/benchmarks/main-thesis.py`

- Primary driver that will call the new operator:
  - Extend `run_evolutionary_mode` so each offspring request routes through a `MADAOperator`.
  - Keep existing API with `AlgorithmManager` for legacy mutation (20% of cases) per spec.
  - Store MADA lineage in the `Individual` objects (these map to `Solution` concepts and already hold metadata).
  - Call `MADAOperator.update_bandits(lineage, reward)` right after AOCC is computed.

### `llamea-experimentation/experiments/benchmarks/managers.py`

- `AlgorithmManager` remains the entry point for LLM calls. The MADA operator will reuse `fetch_algorithm/refine_algorithm/extract_algorithm_code` as fallbacks whenever it needs fresh innovation blocks or when stitching fails.

## New `llamea/mada/` Module Structure

```
llamea/mada/
├── __init__.py          # Re-exports MADAOperator, DiscountedThompsonSampler, BlockParser
├── ds_ts.py             # Pure DS-TS math: class DiscountedThompsonSampler
├── parser.py            # AST parser to split/join blocks (parent/recombination/mutation/survivor)
└── operator.py          # Orchestrator implementing the 40/40/20 strategy + bandit updates
```

### `ds_ts.py`

- Implements Discounted Thompson Sampling with Gaussian priors exactly as defined in the PDF:
  - Maintains `discounted_count`, `discounted_sum`, posterior mean, capped variance (`tau_max`), and sampling.
  - Offers `select_arm(block_id)` → returns (`arm_name`, `theta`, internal state snapshot id).
  - Offers `update(block_id, chosen_arm, reward)` to apply discounted updates.
- Handles numerical stability (epsilon on zero counts) and exposes diagnostics for logging.

### `parser.py`

- Uses Python’s `ast` module to:
  - Locate the primary optimizer class and its methods.
  - Extract code segments for `parent_selection`, `recombination`, `mutation`, `survivor_selection`. If absent, synthesize placeholders describing “pass-through” behavior.
  - Provide `assemble(block_map)` to rebuild a valid class definition, preserving imports and helper functions.
  - Return structural metadata (hashes, AST dumps) for lineage tracking.

### `operator.py`

- Holds:
  - Four `DiscountedThompsonSampler` instances (one per block).
  - Strategy scheduler implementing:
    - **40 % Innovative MADA:** Query bandits per block, decide whether to copy from Alpha, Beta, or call LLM (Innovation). Innovation arms use `AlgorithmManager` prompts targeted at the specific block (“Generate a mutation() body ...”).
    - **40 % Pure Recombination:** Deterministic stitching with 50/50 randomized parent block selection, bypassing LLM.
    - **20 % Legacy Mutation:** Delegate to `AlgorithmManager.refine_algorithm` (current behavior).
  - `generate_offspring(parents, context)` returning `(code_str, lineage_metadata)`.
  - `update_bandits(lineage, reward)` that loops over recorded decisions and calls the DS-TS updater.
  - Error handling + fallbacks (see below).

## Data Flow & Reward Propagation

1. **Parent Preparation (main-thesis)**
   - When parents are selected, hand them (including AOCC + code) to `MADAOperator`. The operator caches AST-parsed block dictionaries for reuse to save parse cost.
2. **Strategy Selection**
   - Random draw with weights (0.4 / 0.4 / 0.2) determines which generation path executes.
3. **Block Assembly**
   - For Innovative MADA: per block, call the corresponding DS-TS bandit:
     - If arm = Alpha/Beta → copy block from respective parent’s parsed block.
     - If arm = Innovation → craft a narrow prompt, call LLM, validate/parse the returned snippet, and insert it.
   - For Pure Recombination: skip bandits, but still record lineage as deterministic “recomb”.
   - For Legacy Mutation: call `AlgorithmManager.refine_algorithm` and treat lineage as “legacy”.
4. **Construction & Evaluation**
   - `parser.assemble` emits full Python code. `run_evolutionary_mode` instantiates an `Individual`, attaches `mada_lineage`, and runs `evaluate_algorithm` (unchanged).
5. **Reward Computation**
   - After evaluation, compute AOCC improvement: `reward = child.fitness - best_parent.fitness` (or `0` if evaluation failed). Store reward + parent baseline in lineage.
6. **Bandit Updates**
   - Invoke `MADAOperator.update_bandits(lineage, reward)` to update only the bandits and arms that contributed (Innovation arms share the same reward).
   - Persist DS-TS state on the operator instance so next offspring share the updated priors.

## Fallback & Robustness Strategy

- **Parsing Failures:**
  - If `parser.extract_blocks` cannot parse parent code, mark the parent as “AST-unavailable” and skip it for Alpha/Beta pulls; automatically use Innovation blocks (LLM) or switch to Legacy Mutation.
- **Malformed LLM block:**
  - Validate snippet via `ast.parse` and simple static checks (method signature, `self` usage). On failure, retry once; if still invalid, fall back to copying Alpha block and record a penalty reward (0) for that arm.
- **Assembly Errors:**
  - Wrap `assemble` in try/except. If final class cannot be built, revert to Legacy Mutation for that offspring, and log the incident via `ExperimentLogger`.
- **Evaluation Errors / AOCC zeros:**
  - If `evaluate_algorithm` returns an error, treat reward as `-abs(parent_baseline)` to discourage the chosen arms.
- **Missing Arms:**
  - DS-TS maintains `tau_max` cap to avoid exploding variance even if an arm is rarely used. On initialization all arms share identical priors (mean=0, variance=`tau_max^2`).
- **State Persistence:**
  - Serialize DS-TS internal arrays inside `MADAOperator` (optionally via logger) to resume if experiment restarts mid-run.

## Implementation Phases

1. **Scaffolding:** Create `llamea/mada` package with DS-TS class, parser skeleton, and operator stub; add exports in `__init__.py`.
2. **DS-TS Engine:** Implement `DiscountedThompsonSampler` with full math (discounted counts/sums, capped variance, sampling) and tests.
3. **AST Parser:** Build block extraction + assembly utilities, including heuristics for missing methods.
4. **Operator Logic:** Implement scheduler, MADA generation, lineage tracking, and LLM prompt helpers.
5. **main-thesis Integration:**
   - Instantiate `MADAOperator` inside `run_evolutionary_mode`.
   - Replace direct `refine_algorithm` calls with operator invocations while preserving logging/metrics.
   - Compute AOCC reward and call `update_bandits`.
6. **Telemetry & Tests:** Add lightweight logging to trace decisions, and unit tests for DS-TS + parser using sample algorithms.

With this plan in place we can begin implementing MADA while keeping the current LLaMEA workflow intact.
