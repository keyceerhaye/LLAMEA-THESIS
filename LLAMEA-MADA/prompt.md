# MADA–LLaMEA Prompt Catalog

This file documents every prompt template used by the Phase 1/2 MADA integration so you can review or tweak them in one place.

---

## 1. Population Generation Prompt (AlgorithmManager.fetch_algorithm)

**Location:** `managers.py` (`_base_init_prompt` + dynamic guardrail block)  
**Used for:** Initial parent generation in both iterative and evolutionary modes.

```
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
An example of such code (a simple random search), is as follows:
```
class RandomSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <short-name>
# Code:
```python
<code>
```
```

**Guardrails appended automatically:**

```
Please respect the following guardrails:
- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.
- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.
- Recent issues observed: <auto-filled summary from runtime, optional>
```

The “Recent issues” bullet is injected by `AlgorithmManager.update_guardrail_feedback()` using violation telemetry from `MADAOperator` (e.g., “missing helper support: 3; high placeholder ratio: 1”).

---

## 2. Refinement Prompt (AlgorithmManager.refine_algorithm)

**Location:** `managers.py` (`refine_prompt` template)  
**Used for:** Legacy mutation strategy and any fallback where the operator reuses AlgorithmManager.

```
The last proposed algorithm <algorithm_name> got an error: <last_error>,
an average Area over the convergence curve (AOCC, 1.0 is the best) of <auc_mean>,
and a standard deviation of <auc_std>.
Either refine or redesign to improve the algorithm. Give the response in the format:
# Name: <short-name>
# Code:
```python
<code>
```

This refinement must respect the following guardrails:
- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.
- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.
- Recent issues observed: <auto-filled summary, optional>
```

If there was no runtime error, the first sentence omits the “error” clause. When `--detailed-feedback` or `--elitism` is enabled, the prompt also appends the detailed AOCC table and/or best-so-far code as separate messages.

---

## 3. Innovation Block Prompt (MADAOperator._request_innovation_block)

**Location:** `llamea-experimentation/src/llamea/mada/operator.py`  
**Used for:** Bandit innovation arm when requesting a new block implementation from the LLM.

```
The current population summary is:
<population_summary>

Improve the `<block_name>` method of the optimizer class `<class_name>`.
Use the exact signature `<method_signature>` and return only the method definition
inside a Python code block. Keep helper references consistent with the class.

Helper/context for `<block_name>`:
Available helper methods: <list or “none detected; reuse shared helpers only.”>
Shared attributes referenced: <list or “none beyond default state.”>

Please respect the following guardrails:
- Keep shared state updates inside helper methods or __init__; do not introduce module-level globals.
- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.
- Recent issues observed: <summary of latest violations, optional>

Existing implementation for reference:
```python
<current block>
```
```

The helper context is derived from `ParsedAlgorithm.helper_groups` so the LLM knows which helpers/attributes already exist. The guardrail summary is identical to the one used by `AlgorithmManager`.

---

## 4. Runtime Guardrail Feedback Loop

- `MADAOperator` records every dependency failure (`missing_helper`, `missing_attributes`), invalid block (`invalid_block`), and high placeholder ratio.
- A sliding window of recent violations is summarized into human-friendly text (top three counts).
- Before any call to `AlgorithmManager` (legacy fallback or innovation), the operator pushes that summary via `algorithm_manager.update_guardrail_feedback(summary)`.
- `AlgorithmManager` adds the summary as the “Recent issues observed” bullet in both generation/refinement prompts.

This keeps the first two guardrails always on, but the third “bullet” appears only when the system actually observes a recurring issue, which keeps exploration flexible in unknown domains.

