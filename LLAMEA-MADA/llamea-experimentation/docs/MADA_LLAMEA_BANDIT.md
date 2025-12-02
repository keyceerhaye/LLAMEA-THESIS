# MADA–LLaMEA Overview

This note explains how the Model-Aware Differential Architecture (MADA) layer plugs into LLaMEA’s evolutionary loop and how the per-block Discounted Thompson Sampling (DS-TS) bandit guides code generation. The file references the current implementation under `llamea-experimentation/src/llamea/mada/`.

## High-Level Flow

1. **CLI configuration** – `main-thesis.py` exposes `--evolutionary-mode`, population sizes, and the MADA-specific options (`--mada-discount`, `--mada-innovation-weight`, etc.). These are parsed in `resolve_mada_weights` and passed to the evolutionary loop.
2. **Evolutionary loop** – `run_evolutionary_mode` initializes μ parent individuals with fresh LLM calls, evaluates them on BBOB, then iterates λ offspring per generation while enforcing the API/evaluation budget.
3. **MADA operator** – During offspring creation `MADAOperator.generate_offspring` decides whether to run a legacy refinement, recombine two parents, or perform bandit-guided innovation. Each offspring arrives with lineage metadata describing which blocks came from which parents (or the LLM) and which strategy/arm produced them.
4. **Reward propagation** – After fitness is computed, the child’s lineage is fed back into `MADAOperator.update_bandits`, which updates the DS-TS state for every block touched by innovation/recombination. Rewards are defined as the child’s fitness delta against the best parent baseline.
5. **Logging** – `ExperimentLogger` writes the raw LLM output, parsed code, and the per-run AUC traces (`try-*-aucs.txt`) so experiments can be replayed or audited.

## Block Parsing & Metadata

`BlockParser` (`parser.py`) splits an optimizer class into reusable blocks. By default it tracks the four orchestration hooks (`parent_selection`, `recombination`, `mutation`, `survivor_selection`). When a new individual is created, `MADAOperator.ensure_blocks` caches the parsed representation and stores the block metadata inside the `Solution` object:

- **Imports/helpers** – captured and re-emitted so recombined children remain runnable.
- **Block hashes** – SHA-256 digests let MADA detect when two individuals share identical method implementations.
- **Placeholders** – if a class omits a target block, the parser injects a safe default so DS-TS can still ask the LLM to fill it in later.

Because these four hooks always contain meaningful logic, recombination/innovation can swap out targeted behaviors instead of editing unused stubs, keeping MADA’s bandits focused on the components it can actually recombine.

## Offspring Strategies

`MADAOperator` supports three strategies, sampled with the normalized CLI weights:

- **Legacy mutation (`legacy`)** – Calls `AlgorithmManager.refine_algorithm`, priming the system prompt with the focal parent’s code/metrics. This is closest to the original LLaMEA iterative refinement.
- **Recombination (`recombination`)** – Chooses per-block donors from the top-two parents (α, β). The parser assembles a new class that mixes their best blocks and records which parent supplied each piece.
- **Innovation (`innovation`)** – For every tracked block, DS-TS samples an arm:
  - `alpha` – reuse α’s block.
  - `beta` – reuse β’s block.
  - `innovation` – request a fresh snippet from the LLM via `generate_block_snippet`, seeding the prompt with the current population summary and the block’s existing implementation.

Any innovation failure (invalid syntax, missing signature, etc.) falls back to α’s implementation with a reward override of `0.0`, which penalizes the corresponding bandit arm.

## Discounted Thompson Sampling (DS-TS)

Each block maintains its own `DiscountedThompsonSampler` (`ds_ts.py`) so selection decisions remain independent across parts of the optimizer. Key behaviors:

- **Arm state** – For every arm (`alpha`, `beta`, `innovation`) the sampler tracks discounted counts, sums, and squared sums to approximate the posterior mean/variance of rewards.
- **Discounting** – Before updating, all sufficient statistics are multiplied by `γ = --mada-discount` (default `0.97`). This gradually forgets stale evidence so the bandit adapts when the search landscape changes.
- **Sampling** – `select_arm` draws a Gaussian sample θ from each arm’s posterior (`posterior_mean`, `posterior_var`) and picks the argmax. Variance is clamped by `tau_max` to avoid overly aggressive exploration spikes.
- **Updates** – After offspring evaluation, `update_bandits` iterates over the lineage decisions flagged with `bandit=True` and calls `sampler.update(block_id, arm_name, reward, reward_override)`. The reward is `(child fitness − best parent fitness)` unless an override was specified (e.g., invalid snippet → 0).
- **Snapshots** – Every selection returns a `snapshot_id` (`block:counter`) so lineage logs know which posterior state generated the decision. This gets written into the metadata for debugging.

Because DS-TS works on a per-block basis, innovation can learn that (for example) changing the `mutation` method is beneficial while `parent_selection` should currently stick with α. This fine-grained credit assignment is what lets MADA reuse reliable building blocks while still probing new design ideas.

## Reward Shaping & Baselines

Inside `run_evolutionary_mode` the reward pipeline does the following:

1. Evaluate the child code on the BBOB benchmark to produce a mean AUC fitness.
2. Compute `baseline = max(parent fitnesses)` (i.e., best of the current μ parents).
3. Set `reward = child.fitness - baseline`. Improvements are positive; regressions are negative; identical performance yields zero.
4. Attach `baseline` and `reward` to the child’s lineage before calling `update_bandits`.

If evaluation fails (syntax/runtime errors, budget overruns), the child fitness is forced to `0.0`, the reward becomes `-abs(baseline)`, and the innovation arm is strongly penalized. This quickly teaches DS-TS to avoid unreliable block replacements.

## Practical Usage Tips

- **Tuning strategy weights** – Use the CLI flags to bias the search. Example: `--mada-innovation-weight 0.6 --mada-recombination-weight 0.2 --mada-legacy-weight 0.2` favors exploration when the current parents plateau.
- **Adding new blocks** – Extend `PLACEHOLDER_SIGNATURES` plus `BlockParser.target_blocks` with the method name. MADA will automatically start tracking, recombining, and rewarding that block.
- **Resetting bandits** – Instantiate `MADAOperator` with a fixed `rng_seed` for reproducible studies or re-create the operator between runs to drop old posterior state.
- **Debugging lineage** – Every `Solution` stores `metadata["mada_lineage"]` with the sampled arms, block hashes, and rewards; dumping this structure helps explain why an offspring was generated a certain way.

With these pieces combined, MADA enables LLaMEA to treat optimizer source code like a modular genome, applying selection pressure not only at the individual level but also at the level of named methods. The DS-TS bandit keeps the system grounded to empirical feedback, steadily shifting probability mass toward the block sources that consistently improve fitness.

