# MADA Recovery & Quality Roadmap

This plan replaces the previous phase-by-phase outline. It focuses on eliminating the root causes observed in the latest experiments (invalid snippets, survivor-selection crashes, stagnant AOCC) and on restoring a healthy reward signal. Workstreams are grouped by outcome and listed in recommended execution order.

---

## 1. Code Safety & Parser Guarantees

1. **Canonical scaffold enforcement**
   - Update `AlgorithmManager` prompts and templates so every emitted class starts from the MADA scaffold (helpers already wired). Ensure survivor selection examples unpack `(candidate, fitness)` tuples before calling the objective.
   - Add explicit _good vs. bad_ code snippets to the prompt appendix that show tuple destructuring in loops, `zip` calls, and helper invocations so the LLM has concrete guidance when the parser flags `candidate/fitness` issues.
   - When a legacy/innovation attempt fails validation, feed back a short reminder (“Loop through `population` as `for candidate, fitness in population:`”) alongside the traceback to avoid endless scaffold echoes.
2. **Semantic validator expansion**
   - Extend `llamea/mada/ast_utils.py` to detect tuple-return misuse (e.g., `func((candidate, fitness))`) and missing destructuring in block methods. Reject snippets referencing undefined helper names or calling `mutant()` on tuples.
3. **Typed survivor-selection contracts**
   - Add a post-assembly check that every `survivor_selection` returns a list of tuples and that population updates never feed tuples back into `func`. Normalize offenders before evaluation or mark the block invalid.
4. **Unit coverage**
   - Augment `llamea-experimentation/tests/` with fixtures that verify the new AST guards and scaffold enforcement (legacy and innovation paths).

Deliverables: zero `[TypeError: __call__()]` failures in reward logs, legacy outputs always execute to completion, and parser tests cover tuple misuse scenarios.

---

## 2. Reward Signal & Controller Health

1. **Per-strategy reward baselines (done) audit**
   - Log per-strategy baseline values alongside global reward so we can confirm DS-TS sees positive/negative deltas per arm.
2. **Syntax / runtime reward separation (done) tuning**
   - Increase syntax penalty to a configurable `SYNTAX_ERROR_REWARD` and verify reward logs reflect three tiers: syntax, runtime, success.
3. **Outer DS-TS diagnostics**
   - Persist `outer_controller` posterior and choice counts in `monitoring.jsonl`. Add a summary print each generation so we can see when the outer loop sticks to legacy.
4. **Controller nudges**
   - When legacy accrues N consecutive syntax penalties, temporarily down-weight it in the outer sampler to force innovation/recombination attempts.
   - If all strategies enter syntax cooldown at once, switch to a tutoring mode for the next few offspring: surface the tuple-handling example block directly in the prompt and log the violating block instead of outright rejecting it once, so at least one candidate reaches evaluation for telemetry before re-enabling strict rejection.

Goal: reward-log.csv shows a spread of values rather than monotonic −0.05, and monitoring reveals outer-loop adaptability instead of a legacy lock-in.

---

## 3. Innovation & Legacy Resilience

1. **Legacy-only fallback quality**
   - If legacy is the only available strategy (others cooled down), enforce a checklist (tuple unpacking, helper usage, correct `func` signature) before evaluation. Log guardrail failures with actionable messaging.
2. **Innovation dependency cap (done) refinement**
   - Make `_verify_dependencies` report which helpers/attributes caused the fallback and surface that summary in the prompt so the LLM fixes the right block instead of reintroducing `popsize`.
3. **Refresh mechanism tuning**
   - Record refresh successes/failures separately and gate them on receiving at least one valid algorithm per refresh window.

Success criterion: innovation delivers ≥1 executable child per generation once cooldowns clear, and legacy no longer monopolizes API budget with invalid snippets.

---

## 4. Stagnation & Diversity Controls

1. **Hybrid parent selector**
   - Port `_select_unique` logic into population selection so ties prefer diverse lineages. Record parent family IDs in monitoring snapshots.
2. **Refresh heuristics**
   - Use stagnation stretch + zero improvements in best AOCC to trigger `--allow-refresh` automatically when flag is active; log which parent was replaced and the new fitness delta.
3. **Diverse summary to LLM**
   - Feed population summaries grouped by lineage/operator so the LLM sees the diversity requirement explicitly in each refinement request.

Outcome: parent printouts show distinct lineages, monitoring logs `[REFRESH]` events when stagnation lasts >10 generations, and best AOCC improves beyond 0.09 mid-run.

---

## 5. Observability & Tooling

1. **Structured monitoring schema**
   - Version the `monitoring.jsonl` format and add fields for `syntax_penalties`, `runtime_failures`, and `refresh_events`.
2. **Reward-log enrichment**
   - Append columns for `failure_kind` (syntax/runtime), `strategy_baseline`, and `outer_choice` to align with the new controllers.
3. **Post-run summarizer**
   - Build a lightweight script/notebook that ingests the monitoring + reward logs and surfaces key metrics (best AOCC, cooldown counts, invalid snippet rate) for quick regression analysis.

Verification: each experiment folder contains machine-readable summaries, making it obvious whether an iteration progressed or stalled.

---

## Execution Notes

- Prioritize Workstreams 1 & 2 to stop invalid snippets at the source and stabilize the reward signal.
- After code safety improvements, re-run short (budget 20) experiments to validate outer-loop behavior before tackling deeper diversity/stagnation tweaks.
- Keep tests green (`pytest llamea-experimentation/tests/test_mada_parser.py`) after each milestone. Consider adding CLI smoke tests once legacy output reliability improves.
