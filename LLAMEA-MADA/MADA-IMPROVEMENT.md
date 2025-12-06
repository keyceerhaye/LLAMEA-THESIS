# Making MADA-LLAMEA Crossover Actually Work

## What went wrong in the 12-05 runs

- All “recombined/innovated” variants (e.g., `AdaptiveGaussianMu_Recomb_*`) are just the original mutation-only loop; the recombination hooks are placeholders and never called.
- Parent selection is fixed to tournament-of-2 on the current population; archive is never used for crossover, so diversity collapses.
- Because crossover is effectively absent, improvement came only from legacy mutation, and recombination trials scored 0.0 AOCC.

## Prompting / scaffolding fixes (no canned code, but enforce intent)

- Drop the category-block prompt style that labels sections as “recombination/mutation/parent selection.” Instead, provide a minimal skeleton with explicit calls to `select_parents()`, `recombine()`, `mutate()`, and `survivor_selection()` inside the main loop, and **state that no helper may be left as a placeholder**.
- Add a checklist the model must satisfy before returning code (e.g., “verify recombination uses ≥2 parents and is called each generation; no helper returns its input unchanged; no TODO/placeholder strings”).
- Constrain the model to keep state local to the class and to reuse `update_archive()` rather than re-implementing it.
- Replace codeblock libraries with **goal-and-signal driven guidance**: ask the model to propose a crossover that (1) leverages differences among elites and diverse samples, (2) perturbs along those directions with adaptive step sizes, and (3) explains in short comments why the operator should help exploration vs exploitation.

## Implementation blueprint for MADA

- Require the main loop to (1) select parents (from population + archive), (2) recombine, (3) mutate, (4) evaluate, (5) survivor select. Reject any returned code that skips step 2.
- Use an **archive-aware parent sampler**: always include at least one elite from archive and one diverse sample from tail or random. This helps crossover have meaningful contrast.
- Track a simple **crossover success rate** (offspring beats parent). If success < 15% over last N offspring, reduce step size or switch operator (e.g., from SBX to DE blend).
- Ensure **bounds-safe** recombination: clip or reflect after crossover, not only after mutation.
- Keep **population statistics** (mean/std) to scale crossover noise; current code uses a fixed `mutation_rate` and ignores population spread.
- Survivor selection: keep `(μ+λ)` with diversity guard (e.g., reject if too close to an existing elite, measured by L2 < ε).

## Extensibility beyond the four blocks

- Allow the LLM to introduce **additional operators** (e.g., local search, restart triggers, adaptive perturbation, coordinate-wise search) as long as they declare (a) when they run, (b) how many evaluations they consume, (c) what they return (new population or candidate).
- Provide a **generic operator registry** interface: each operator exposes `name`, `call(pop_x, pop_f, archive) -> (new_pop_x, new_pop_f, evals_used)` so the framework can schedule it without knowing its internals.
- In the prompt, say: “You may add new helpers if you call them from the main loop and ensure they respect the budget and bounds; avoid placeholders.”
- Treat unknown operators as optional steps between recombination and mutation or as post-processing; log their eval usage to keep accounting correct.
- Keep the core invariants: budget respected, archive updated via `update_archive`, population shape preserved, and no module-level globals.

## Suggested prompt text (replace the current blocky one)

> Implement a population-based optimizer with explicit recombination. You must:
>
> - Implement and call `select_parents`, `recombine`, `mutate`, `survivor_selection`; no placeholders/TODOs.
> - Recombination must use ≥2 parents and change the genotype (no identity pass-through).
> - Design a crossover that intentionally exploits **directions between good and diverse parents**; describe briefly in code comments why it should help.
> - Adapt step sizes or mixing coefficients based on recent offspring success (if success low, shrink/explore differently; if high, exploit more).
> - Always clip to bounds after recombination and mutation.
> - Use archive elites in parent selection for diversity; update archive via `update_archive`.
> - Return `(self.f_opt, self.x_opt)` and respect `budget`.
>   Before finalizing, self-check: recombination called? uses ≥2 parents? no helper returns input unchanged? no placeholders? bounded? budget respected?

## Concrete code changes to make in the generator/templates

- Replace the current placeholder helper methods with real operators (pick two templates above and a tiny success-based switch).
- Force the main loop to call `recombine()` before `mutate()`, and pass a tuple/list of parents.
- Introduce `select_parents(pop_x, pop_f, archive_x, k=2, with_diversity=True)` that draws one from elites and one from random tail.
- Add a small `adapt_operator_stats` dict to track crossover success and scale α/η/F.
- Keep mutation as a fallback, but only when crossover fails to produce improvement after N attempts.

## Quick validation steps

- Unit-check on simple spheres/rastrigin: verify offspring differ from parents and success rate > 0 early in search.
- Log crossover success %, operator used, and mean step size per generation; abort/penalize if success stays 0 for a full generation.
- In LLM evaluation, reject any model output containing “placeholder,” “TODO,” or helper that returns its input.

Implementing the above should make crossover tangible, leverage the archive, and prevent the LLM from emitting no-op recombination blocks.\*\*\*
