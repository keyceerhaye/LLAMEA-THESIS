# LLaMEA Prompt Documentation

This document contains all prompts used in the LLaMEA implementations.

---

## 1. Baseline LLaMEA (main-thesis.py)

### System Prompt (Role)

```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing.
You may introduce concise helper utilities, restart logic, and archive/step-size management inside the class, provided the public __init__ and __call__ stay intact and total budget is respected.
```

### Initialization Prompt

````
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
An example of such code (a simple random search), is as follows:
```python
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
````

Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:

# Name: <classname>

# Code: <code>

```

### Refine Prompt (with error)
```

The last proposed algorithm {algorithm_name} got an error: {last_error}, an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, and a standard deviation of {auc_std:.02f}. Either refine or redesign to improve the algorithm. Give the response in the format:

# Name: <classname>

# Code: <code>

```

### Refine Prompt (without error)
```

The last proposed algorithm {algorithm_name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {auc_mean:.02f}, and a standard deviation of {auc_std:.02f}. Either refine or redesign to improve the algorithm. Give the response in the format:

# Name: <classname>

# Code: <code>

```

### Detailed Feedback Prompt (optional)
```

The mean AOCC score of the last algorithm on Separable functions was {detailed_aucs[0]:.02f}, on functions with low or moderate conditioning {detailed_aucs[1]:.02f}, on functions with high conditioning and unimodal {detailed_aucs[2]:.02f}, on Multi-modal functions with adequate global structure {detailed_aucs[3]:.02f}, and on Multi-modal functions with weak global structure {detailed_aucs[4]:.02f}

```

### Elitism Prompt (optional)
```

The best so far proposed algorithm got an average AOCC of {current_best_AOCC:.02f} and the code was as follows:
{current_best_algorithm}

```

---

## 2. MADA-LLAMEA v2 (main-thesis-mada-v2.py)

### System Prompt (same as baseline)
```

You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing.
You may introduce concise helper utilities, restart logic, and archive/step-size management inside the class, provided the public **init** and **call** stay intact and total budget is respected.

```

### Initialization Prompt

```

The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
An example of such code (a simple random search), is as follows:

```python
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

# Name: <classname>

# Code: <code>

```

### Mutation Prompt
```

The last proposed algorithm {name} got an average AOCC of {score:.4f}:

```python
{code}
```

{feedback}

Either refine or redesign to improve the algorithm. Give the response in the format:

# Name: <classname>

# Code: <code>

```

**Variables:**
- `{name}` - Parent algorithm's class name
- `{score}` - Parent's AOCC score
- `{code}` - Parent's Python code
- `{feedback}` - Error message if any, otherwise "No specific feedback."

### Crossover Prompt
```

I am designing metaheuristic algorithms for black-box optimization (BBOB benchmark, 5D, bounds [-5, 5]).

Here are two high-scoring solutions:

Solution 1 (Score: {score_a:.4f}):

```python
{code_a}
```

{feedback_a}

Solution 2 (Score: {score_b:.4f}):

```python
{code_b}
```

{feedback_b}
{error_analysis}
Generate a Solution 3 that achieves an even higher score by combining patterns from both.
Follow the same interface: **init**(self, budget) and **call**(self, func)

Format:

# Name: <classname>

# Code: <code>

```

**Variables:**
- `{feedback_a}` / `{feedback_b}` - **OPTIONAL** Performance breakdown per function group (enabled with `--detailed-feedback` flag)
- `{error_analysis}` - Combined error analysis from both parents (if any errors)

### Refine Prompt (with optional detailed feedback - baseline style)
```

The last proposed algorithm {name} got an average AOCC of {score:.4f}:

```python
{code}
```

{performance_breakdown}

{error_feedback}

Either refine or redesign to improve the algorithm. Give the response in the format:

# Name: <classname>

# Code: <code>

```

**Variables:**
- `{name}` - Parent algorithm's class name
- `{performance_breakdown}` - **OPTIONAL** (enabled with `--detailed-feedback` flag). Baseline-style sentence:
  "The mean AOCC score on Separable functions was X.XX, on functions with low or moderate conditioning X.XX, on functions with high conditioning and unimodal X.XX, on Multi-modal functions with adequate global structure X.XX, and on Multi-modal functions with weak global structure X.XX."
- `{error_feedback}` - "The algorithm got an error: ..." (if any)

---

## 3. Key Differences

| Aspect | Baseline LLaMEA | MADA v2 |
|--------|-----------------|---------|
| **System Prompt** | Restricts nature-inspired algorithms | Same as baseline |
| **Operators** | Single refinement operator | 3 operators: mutation, crossover, refine |
| **Operator Selection** | N/A (single operator) | Discounted Thompson Sampling (D-TS) |
| **Feedback Style** | Optional detailed AOCC per function group | Same (optional with `--detailed-feedback`) |
| **Prompt Style** | Simple "refine or redesign" | Same baseline style for mutation/refine |
| **Diversity** | Based on code difference | Trace-based behavioral diversity (NN-Dist) |

---

## 4. Prompt Guardrails (Baseline)

These guardrails are optionally appended to baseline prompts:

```

- You MAY add small helper utilities and restart logic inside the class; keep them class-scoped (no module-level globals).
- If the block depends on helpers such as update_archive(), call them instead of duplicating logic.
- Do not introduce orphaned dependencies; use helpers already present in the Context Block.
- Preserve class structure: only modify the requested block, not **init** or **call** wiring.
- Prefer alternative search moves over classic DE-style mutation/crossover when possible.

```

---

*Last updated: December 11, 2024*
```
