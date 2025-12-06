## Goal

Produce a safe, incremental plan to fix MADA crossover and prompting while **preserving the existing Discounted Thompson Sampling (DS-TS) machinery** and avoiding regressions to the legacy evaluator. No code edits yet—this is the actionable checklist before touching implementations in `LLAMEA-MADA`.

## What’s in place today (observed)
- `main-thesis.py` drives evolutionary mode and already instantiates `MADAOperator(parser=BlockParser, rng_seed=None)` and calls `generate_offspring(...)` then `update_bandits(...)`.
- Offspring reward sent to bandits is `child.fitness - best_parent_fitness` (AOCC is maximized; reward sign must stay this way for DS-TS).
- Evaluation uses `correct_aoc` + `aoc_logger`; `OverBudgetException` protects budget.
- Prompting guardrails live in `AlgorithmManager` but currently do **not** enforce real recombination semantics.
- `MADA-IMPROVEMENT.md` documents that recombination hooks are placeholders, archive is ignored for parent selection, and crossover success is effectively zero.

## Risks to DS-TS we must avoid
- Changing reward sign or baseline (must stay “child minus parent/baseline”, positive = improvement).
- Resetting or bypassing `update_bandits(lineage, reward)`; any refactor must leave call-site and data shape intact.
- Altering arm semantics (Alpha/Beta/Innovation) without updating lineage encoding.
- Mutating DS-TS state outside its API (no direct field edits; only via sampler methods).
- Removing clipping/bounds checks that keep generated code numerically stable (protects reward quality).

## Change package (ordered; stop after each step to validate)

1) **Strengthen prompts without touching DS-TS math**
   - Replace the current blocky prompt with the checklist from `MADA-IMPROVEMENT.md`: explicit `select_parents → recombine → mutate → survivor_selection`; “no placeholders/no identity recombination”; comments explaining exploration/exploitation.
   - Add self-check list (recombination uses ≥2 parents; archive used; bounds clipped after crossover and mutation; no TODO/placeholder).
   - Keep `AlgorithmManager` interfaces untouched (DS-TS depends on lineage only, not on prompt format).

2) **Parent selection that respects archive diversity**
   - Implement `select_parents(pop_x, pop_f, archive_x, k=2, with_diversity=True)` that samples one elite + one diverse/tail.
   - Wire this into `MADAOperator` generation path (not in DS-TS core); ensure lineage still reports Alpha/Beta/Innovation correctly.

3) **Make recombination real and always invoked**
   - In the generator template, force `recombine()` to be called before `mutate()` in the main loop; recombination must change the genotype (no pass-through).
   - Add crossover operator with success tracking (e.g., SBX/DE blend with bounds clipping). Track per-offspring success rate and downscale step sizes if success < 15% over a window.
   - Ensure mutation is fallback, not the only path. Maintain lineage so DS-TS updates stay consistent.

4) **Bounds and validity guards**
   - Clip/reflect after recombination and after mutation.
   - Reject helpers that return inputs unchanged. Abort generation if placeholders detected; fall back to legacy mutation but still log lineage with a penalty reward of 0 to DS-TS.

5) **Logging & observability (keeps DS-TS trustworthy)**
   - Log crossover success %, operator used, and mean step size per generation.
   - If a full generation yields zero crossover success, emit a warning and penalize the crossover arm with zero reward (without changing reward definition).
   - Surface evaluation exceptions in `ExperimentLogger` and forward a concise error note into the refine prompt.

6) **Safety tests before rollout**
   - Unit sanity: run sphere/rastrigin with small budget; assert offspring differ from parents; DS-TS reward updates are nonzero; success rate > 0 early.
   - Regression: short evolutionary run (budget 20) to verify `update_bandits` is still called with the same reward schema and no DS-TS state resets.
   - Lint/static: `ast.parse` the generated code to ensure no signature mismatches before evaluation.

## Execution checkpoints
- After steps 1–2: prompts updated, parent sampler uses archive; DS-TS untouched. Run smoke test to ensure lineage still updates.
- After step 3: recombination operator live; verify crossover success metric > 0 and AOCC not degraded.
- After step 5: confirm logs show strategy usage, success %, and any penalties applied; DS-TS should continue to sample all three arms.

## Non-goals (to keep scope safe)
- No changes to DS-TS formulas (`gamma`, `tau_max`, posterior mean/variance) or arm definitions.
- No change to AOCC computation or evaluator budget handling.
- No edits outside `LLAMEA-MADA` folder.


