## MADA-LLAMEA New Implementation (2025-12)

This document captures the new functionality and wiring introduced in the current MADA-LLAMEA framework. It focuses on what changed relative to the earlier LLaMEA / EoH variants and how the behavioral-diversity pipeline is implemented end-to-end in `main-thesis-mada.py`, `mada_components.py`, and `managers.py`.

### What’s New (high level)
- Behavioral traces are first-class: every algorithm run is wrapped by `TraceCollector`, producing best-so-far trajectories for diversity scoring.
- Diversity-aware reward: composite reward `Δfitness + α·NN-Dist`, with α scheduled over generations and rewards standardized online.
- Discounted Thompson Sampling (D-TS) controls operator choice (mutation vs crossover) using normalized composite rewards; tracks fitness vs diversity contributions separately.
- Bandit and lineage logging: `bandit_snapshots.jsonl` and `mada_offspring.jsonl` record arm posteriors, rewards, and operator lineage per offspring.
- Guardrailed LLM prompting: block/holistic prompts inject recent issues, strict attribute whitelists, and recombination helpers; supports innovation, recombination, and legacy refinement modes.
- Robust evaluation harness: automatic helper injection, budget guarding, safe `np.random.choice`, and BBOB trace aggregation per algorithm.

### Core Components (new/updated)
- `TraceCollector` (in `mada_components.py`)
  - Wraps the IOH problem, records best-so-far per eval, enforces budget, exposes `get_trace()` padded to budget.
  - `aggregate_traces()` provides a median-robust pooled trace for logging.
- Diversity metric
  - `calculate_nn_dist(trace, history, global_min, global_max)`: nearest-neighbor distance on normalized traces; returns 1.0 for the first algorithm.
- Reward shaping
  - `compute_composite_reward(f_child, f_parent, nn_dist, alpha, error, clamp)`: combines exploitation and diversity; penalizes errors; clamps for stability; tiny positive tie-breaker to avoid stagnation.
  - `RewardNormalizer(warmup_period=5)`: Welford online stats; returns raw rewards during warmup, z-scores afterward; exposes `get_stats()`.
- Diversity weight scheduling
  - `AlphaScheduler(alpha_start, alpha_end, t_max, schedule∈{linear, exponential, cosine, constant})`: anneals exploration pressure over generations.
- Bandit (D-TS) extensions in `main-thesis-mada.py`
  - `ArmStatistics` tracks pulls, μ̂, τ, total reward, plus fitness/diversity reward decomposition.
  - `DiscountedThompsonSampler.update_with_decomposition()` stores component rewards alongside normalized reward updates.
  - `get_arm_probabilities()` Monte-Carlo estimates selection probabilities for logging.
- Prompt/guardrail stack in `managers.py`
  - Block innovation, recombination, and holistic refinement prompts include explicit allowed attributes, context blocks, and “recent issues” guardrails.
  - Role prompt discourages DE-style reuse and hyped metaheuristics; elitism and detailed-feedback hooks remain supported.
  - Semantic linter prompt cleans merged helpers when needed.

### Evolutionary Loop Changes (`run_mada_evolutionary_mode`)
- Initialization
  - Generates μ parents via `AlgorithmManager.fetch_algorithm`, deduplicated by SHA-256; evaluates each with trace capture and stores traces in `history_traces`.
  - Tracks global min/max fitness from traces for later NN-Dist normalization.
- Per-generation workflow
  - Alpha schedule computed once per generation; reported in console and logged.
  - D-TS selects operator (`mutation` or `crossover`); snapshot of arm posteriors logged.
  - Offspring evaluation uses `evaluate_algorithm_with_trace()`:
    - Injects safe globals and missing helpers (`lb/ub/bounds/pop/archive` etc.).
    - Wraps the fitness call with `TraceCollector` to capture trajectories; handles `OverBudgetException`.
    - Runs BBOB (24 fids × 3 iids × 3 reps), collects AUCs, detailed AUCs by BBOB family, and aggregated trace.
  - Diversity score: `calculate_nn_dist` against `history_traces`; defaults to 1.0 when no history.
  - Reward: `compute_composite_reward` → `RewardNormalizer.normalize`; decomposition (Δfitness, diversity bonus) fed to D-TS via `update_with_decomposition`.
  - History update: stores trace if present; updates global min/max for normalization drift.
  - Logging: offspring record (operator, rewards, NN-Dist, α, θ, lineage) to `mada_offspring.jsonl`; code and AUCs saved per attempt.
  - Selection: `(μ+λ)` when `--elitism`, otherwise λ-only; `BEST_ALGORITHM.py` persisted at the end with metadata.
- Final summary prints bandit pulls, μ̂, and cumulative fitness/diversity rewards.

### Operator Strategies (current MADA loop)
- Mutation path: LLM mutation prompt with performance breakdown and last error hints; child inherits parent lineage and increments generation.
- Crossover path: implicit crossover prompt combining two best parents’ code and scores.
- If only one parent exists, MADA falls back to mutation.
- Elitism toggle controls whether parents survive into the next generation.

### Logging & Artifacts
- `exp-*/` directory per run with:
  - `ioh/` (IOH experimenter logs), `code/` (all tried algorithms), `conversationlog.txt`.
  - `try-*-aucs.txt` (AUC arrays, optional stats header).
  - `mada_offspring.jsonl` (per-offspring operator, rewards, θ, lineage, errors).
  - `bandit_snapshots.jsonl` (per-generation arm stats, τ, μ̂, selection probabilities, reward stats, α, history size).
  - `BEST_ALGORITHM.py` (best code, fitness, generation, operator).

### CLI Surface (new/updated flags in `main-thesis-mada.py`)
- Bandit: `--discount γ`, `--tau-max`, `--reward-variance`, `--reward-clamp`.
- MADA diversity: `--alpha-start`, `--alpha-end`, `--alpha-schedule` (linear|exponential|cosine|constant).
- Evolution: `--n-parents`, `--n-offspring`, `--generations`, `--elitism`, `--budget` (API calls), `--eval-budget` (per-algo evaluations).
- API/model: `--model`, `--api-key`, `--base-url`, `--max-tokens`, `--experiment-name`.
- Mode: `--evolutionary-mode` required to run the full MADA loop.

### Failure Handling & Guardrails
- Missing code block / class name retries during parent seeding; deduped by code hash.
- Safe `np.random.choice` wrapper avoids errors on small arrays; auto-injects common attributes if absent in candidate algorithms.
- Errors during evaluation set fitness to 0.0, clear traces, penalize reward; D-TS still updated via composite reward.
- `RewardNormalizer` warmup avoids unstable early statistics; reward clamp limits outliers.
- Prompts carry “recent issues” guardrail note to steer LLM away from repeated violations (missing helpers, placeholders, invalid blocks).

### How to Run (examples)
- Balanced exploratory run:
  - `python main-thesis-mada.py --evolutionary-mode --elitism --budget 80 --eval-budget 10000 --n-parents 4 --n-offspring 12 --alpha-start 0.6 --alpha-end 0.0 --alpha-schedule cosine --discount 0.95`
- Faster smoke test:
  - `python main-thesis-mada.py --evolutionary-mode --budget 20 --eval-budget 2000 --n-parents 2 --n-offspring 6 --alpha-start 0.5 --alpha-end 0.1 --alpha-schedule linear`

### Implementation Hotspots (for reference)
- `mada_components.py`: trace, diversity, reward, alpha scheduling helpers.
- `main-thesis-mada.py`: D-TS bandit with reward decomposition; evaluation harness with trace capture; evolutionary loop with diversity-aware reward flow; logging.
- `managers.py`: ExperimentLogger, AlgorithmManager prompts/guardrails, block/holistic prompt builders, semantic linter hook.

### Intended Outcomes
- Encourage exploration early (high α, higher innovation acceptance) while converging later (α→0).
- Stabilize bandit learning via standardized rewards and diversity decomposition.
- Provide reproducible, inspectable runs with full lineage, traces, and bandit dynamics to analyze why an offspring emerged and how operators were credited.

