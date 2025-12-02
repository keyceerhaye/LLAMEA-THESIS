# MADA Implementation Guide

## 1. Protocol: The "PLAN.MD" First Rule

### CRITICAL INSTRUCTION TO AI:

You are prohibited from writing implementation code (Python) in the first step.

Your ONLY output for the first turn must be a file named `PLAN.md`.

You must scan the provided codebase—specifically `llamea/llamea.py`, `llamea/solution.py`, and `benchmarks/main-evolutionary.py`—and draft a comprehensive architectural plan that includes:

- **Integration Points**: Identification of exactly where to hook the new operator without deleting existing logic.
- **File Structure**: A proposal for the new `llamea/mada/` module.
- **Data Flow**: How reward signals (AOCC) will flow back from the evaluator to the Bandits.
- **Fallback Mechanisms**: How the system handles malformed LLM code generation.

---

## 2. Mathematical Specification: Discounted Thompson Sampling (DS-TS)

You must implement the Discounted Thompson Sampling algorithm with Gaussian Priors exactly as defined in the provided literature.

### Parameters

1. \(\gamma\) **(Discount Factor)**: Controls the memory length (typically \(0.9 - 0.99\))
2. \(\tau\_{max}\) **(Variance Cap)**: Prevents infinite exploration
3. **Arms** (\(K=3\)): Alpha (Best Parent), Beta (Weak Parent), Innovation (LLM)

### State Variables & Update Logic

For every arm \(i\) at generation \(t\):

#### 4. Discounted Count (\(N_t\)):

\[N\_{t+1}(\gamma,i) = \gamma N_t(\gamma,i) + \mathbb{I}\{i_t = i\}\]

#### 5. Discounted Sum of Rewards (\(\tilde{\mu}\_t\)):

\[\tilde{\mu}\_{t+1}(\gamma,i) = \gamma \tilde{\mu}\_t(\gamma,i) + X_t \cdot \mathbb{I}\{i_t = i\}\]

(Where \(X_t\) is the fitness improvement).

#### 6. Posterior Mean (\(\hat{\mu}\_t\)):

\[\hat{\mu}_{t+1}(\gamma,i) = \frac{\tilde{\mu}_{t+1}(\gamma,i)}{N\_{t+1}(\gamma,i)}\]

#### 7. Posterior Variance (\(\tau_t\)):

\[\tau*{t+1}(i) = \min \left( \frac{1}{\sqrt{N*{t+1}(\gamma,i)}}, \tau\_{max} \right)\]

#### 8. Sampling Action

Draw sample \(\theta_t(i)\) from the Normal distribution:

\[\theta*{t}(i) \sim \mathcal{N}(\hat{\mu}*{t}(\gamma,i), \tau\_{t}(i)^{2})\]

#### 9. Select the arm with \(\arg\max \theta_t(i)\)

---

## 3. MADA Architectural Logic

### A. Decomposition

You must implement a parser (AST-based) to decompose a Python Algorithm script into 4 constituent blocks:

10. `parent_selection`
11. `recombination`
12. `mutation`
13. `survivor_selection`

### B. The 4-Bandit System

Instantiate 4 independent DS-TS Bandits, one for each block above.

- **Arm 0 (Alpha)**: Extract block from the High-Fitness Parent
- **Arm 1 (Beta)**: Extract block from the Low-Fitness Parent (Diversity)
- **Arm 2 (Innovation)**: Generate a new block via LLM prompt

### C. Hierarchical Control Strategy

The `generate_offspring` method must follow this probability schedule:

#### 15. 40% Innovative MADA:

- Query the 4 Bandits.
- If Bandits choose Alpha/Beta: Copy code from parents.
- If Bandits choose Innovation: Prompt LLM for that specific function.
- Assemble and return.

#### 40% Pure Recombination (Stitching):

- NO LLM Generation.
- Deterministically stitch blocks from Parent A or B (50/50 chance per block).

#### 20% Legacy Mutation:

- Use the existing LLaMEA mutate function (Random Restart).

---

## 4. Implementation Requirements

### File Structure

Based on the existing folder structure, implement the following:

- `llamea/mada/__init__.py`
- `llamea/mada/ds_ts.py` (The math engine)
- `llamea/mada/parser.py` (AST decomposition)
- `llamea/mada/operator.py` (The Controller/Gatekeeper)

### Credit Assignment

After evaluation, you must calculate the reward \(X_t\) (AOCC improvement) and call a method `update_bandits(lineage, reward)` to update the DS-TS state for the arms used in that generation.

---

## 5. Current Implementation Snapshot

- `llamea/mada/ds_ts.py` implements the Gaussian DS-TS equations with discounted counts, posterior variance capping, and snapshot export for telemetry.
- `llamea/mada/parser.py` provides block extraction/assembly for the four evolutionary operators plus placeholders when methods are missing.
- `llamea/mada/operator.py` orchestrates the 40/40/20 strategy, records lineage metadata, and drives AlgorithmManager fallbacks.
- `llamea-experimentation/experiments/benchmarks/main-thesis.py` now routes offspring generation through `MADAOperator` while preserving evaluation flow.
- CLI knobs `--mada-discount` and `--mada-*-weight` let you tune DS-TS memory and the innovation/recombination/legacy schedule without editing code.
- Regression tests at `LLAMEA-THESIS/tests/test_mada_ds_ts.py` and `LLAMEA-THESIS/tests/test_mada_parser.py` cover the sampler math and AST scaffolding.

## Action:

Using the specifications above, please scan the current codebase and generate `PLAN.md`.
