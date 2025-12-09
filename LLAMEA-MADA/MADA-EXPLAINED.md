# MADA-EXPLAINED

A practical reference for how the Model-Aware Differential Architecture (MADA) layer augments LLaMEA’s evolutionary code generation. It is written for the methodology section of the thesis and aligns with the current implementation in `main-thesis.py` and `llamea-experimentation/src/llamea/mada/`.

---

## 1) Goal and Scope

- Make LLaMEA population-based: evolve μ parents and λ offspring under a fixed API/evaluation budget (`--budget`, `--eval-budget`).
- Treat optimizer source as modular building blocks, not monolithic files.
- Use bandit-guided choices to decide, per block, whether to reuse parent code or ask the LLM to innovate.
- Provide reproducible runs with logged lineage, prompts, outputs, and fitness traces.

## 2) High-Level Loop

1.  **Initialization:** `main-thesis.py --evolutionary-mode` seeds μ parents via the population-generation prompt (Section 6.1). Each parent is evaluated on BBOB (24 noiseless functions; 5-D; bounds [-5, 5]).
2.  **Generation cycle (per λ offspring):**
    - Sample two parents (α is best, β is runner-up).
    - `MADAOperator.generate_offspring` chooses one strategy using normalized CLI weights:
      - `legacy` (AlgorithmManager refine)
      - `recombination` (per-block mix of α/β)
      - `innovation` (bandit-guided block replacement)
    - Offspring code is parsed, executed, and evaluated.
    - Lineage + reward are fed back into the bandits.
3.  **Selection:** `(μ+λ)` if `--elitism` else `(μ,λ)`. Best μ become next-generation parents until the budget or generation limit is hit.

## 3) Block Parsing & Metadata

- **Target blocks:** `parent_selection`, `recombination`, `mutation`, `survivor_selection`.
- `BlockParser` caches per-class metadata: imports/helpers, block hashes (SHA-256), placeholders for missing blocks, helper/attribute references.
- Every `Solution` stores `metadata["mada_lineage"]` capturing which block came from which parent or LLM arm, plus reward and snapshot IDs for debugging.

## 4) Offspring Strategies

- **Legacy mutation (`legacy`):** Call `AlgorithmManager.refine_algorithm` with fitness/error context; closest to original iterative LLaMEA.
- **Recombination (`recombination`):** For each block, pick donor α or β; assemble a hybrid class; keeps helpers/imports coherent via the cached parse.
- **Innovation (`innovation`):** For each tracked block:
  - The per-block bandit samples an arm: `alpha` (reuse α), `beta` (reuse β), or `innovation` (ask LLM for a fresh block).
  - On `innovation`, `_request_innovation_block` prompts the LLM with population summary, helper/attribute context, guardrails, and the current block as reference.
  - Any invalid snippet falls back to α with reward override `0.0` (penalizes the chosen arm).

## 5) Discounted Thompson Sampling (DS-TS)

- **One sampler per block** (`ds_ts.py`) so decisions are independent across `parent_selection`, `recombination`, `mutation`, `survivor_selection`.
- **Arms:** `alpha` (keep best parent), `beta` (import from runner-up), `innovation` (LLM rewrite).
- **Discounting:** Stats are multiplied by `γ = --mada-discount` (default 0.97) before each update to forget stale evidence.
- **Selection:** Draw θ from each arm’s Gaussian posterior (mean/variance clamped by `tau_max`), pick argmax.
- **Reward:** `child_fitness − max(parent fitnesses)`; negative for regressions. Evaluation failures set fitness to `0.0` and strongly penalize innovation.
- **Updates:** After evaluation, lineage decisions with `bandit=True` call `sampler.update(block_id, arm, reward, reward_override)`. Each decision is tagged with a `snapshot_id` (`block:counter`) for traceability.
- **Effect:** Fine-grained credit assignment learns which blocks to keep stable and which to keep probing.

## 6) Prompting System (guardrails injected automatically)

6.1 **Population generation** (`AlgorithmManager.fetch_algorithm`)

- Tasked to write a fresh optimizer for BBOB with `__init__(budget)` and `__call__(func)`.
- Returns `# Name:` + Python code block; includes a simple RandomSearch example as scaffolding.
- Guardrails: keep shared state inside helpers/`__init__`; reuse helpers instead of duplicating; include “recent issues” summary from runtime violations (missing helpers, placeholder overuse, etc.).

  6.2 **Refinement** (`AlgorithmManager.refine_algorithm`)

- Input: last algorithm name, error (if any), AOCC mean/std; optional detailed AOCC table when `--detailed-feedback` or `--elitism` is set.
- Instruction: refine or redesign; same guardrails and optional “recent issues” bullet.

  6.3 **Innovation block** (`MADAOperator._request_innovation_block`)

- Input: population summary, class name, target block name/signature, helper/attribute context, current block body, guardrails + recent issues.
- Output: only the method definition in a Python block with the exact signature; no module-level globals.

  6.4 **Runtime feedback loop**

- `MADAOperator` records violations (`missing_helper`, `missing_attributes`, `invalid_block`, high placeholder ratio).
- A sliding window is summarized and injected as the “Recent issues observed” bullet in all prompts to steer the LLM away from recurring errors.

## 7) Reward Pipeline & Evaluation

- Evaluate each candidate on BBOB; fitness is mean AUC (Area over convergence curve).
- Baseline per child: best parent fitness.
- Reward: improvement over baseline; zero if identical; negative if worse.
- Failure handling: syntax/runtime/budget errors → fitness `0.0`, negative reward, innovation arm penalized.
- Lineage stores `baseline`, `reward`, block-level arm choices, and snapshot IDs for post-hoc analysis.

## 8) Configuration Surface (key flags)

- Evolution: `--evolutionary-mode`, `--n-parents`, `--n-offspring`, `--elitism`, `--generations` (optional fixed).
- Budgets: `--budget` (API calls), `--eval-budget` (function evaluations per algorithm).
- MADA weighting: `--mada-legacy-weight`, `--mada-recombination-weight`, `--mada-innovation-weight` (normalized in `resolve_mada_weights`).
- Bandit dynamics: `--mada-discount` (γ), `--mada-innovation-weight` influences exploration; optional `rng_seed` for reproducibility.
- Logging: `--experiment-name`, `--detailed-feedback`, and standard API settings (`--model`, `--base-url`, `--max-tokens`).

## 9) Logging, Traceability, and Reuse

- `ExperimentLogger` saves raw LLM outputs, parsed code, and per-run AUC traces (`try-*-aucs.txt`).
- Each `Solution` keeps `mada_lineage` with block hashes, arms, rewards, and snapshots to explain “why this child looks this way.”
- Recombination and innovation reuse cached imports/helpers so stitched classes stay executable.

## 10) Why This Works (methodological rationale)

- **Modular credit assignment:** Bandits operate per block, letting the system keep stable components (e.g., survivor selection) while exploring riskier ones (e.g., mutation).
- **Discounting for non-stationarity:** DS-TS forgets stale evidence, matching shifting fitness landscapes across generations.
- **Guardrail feedback:** Runtime violations feed back into prompts, reducing repeated failure modes without hard-coding rules.
- **Balanced search:** Strategy weights trade off exploitation (`legacy`, `alpha`) and exploration (`innovation`, higher γ/innovation weight).
- **Whole-class context:** Innovation prompts include helper/attribute context to avoid orphaned dependencies while still allowing block-level novelty.

## 11) Practical Tips

- Start with balanced weights (e.g., `0.3 legacy / 0.3 recombination / 0.4 innovation`) and adjust based on stagnation vs. instability.
- Use `--elitism` for safer progress; drop it if you need aggressive exploration.
- Set a fixed `rng_seed` when comparing ablations; DS-TS plus LLM sampling otherwise introduces run-to-run variance.
- Inspect `mada_lineage` and guardrail summaries when diagnosing repeated failures or surprising behavior.
