# LLAMEA-MADA Analysis: Why Valid Algorithms Are Not Being Generated

**Date**: December 6, 2025  
**Experiment**: `exp-12-06_134758-gemini-2.0-flash-thesis-experiment-evolutionary-elitism`

---

## Executive Summary

The LLAMEA-MADA framework is currently stuck in a failure loop where **zero valid algorithms** are being produced across all generations. Every algorithm fails either at static validation or preflight checks. The root cause is a **prompt-compliance gap**: the LLM (Gemini 2.0 Flash) consistently ignores explicit guardrails and generates code with forbidden patterns.

---

## 1. Observed Failure Patterns

### 1.1 Primary Error: Forbidden Bounds Usage (90%+ of failures)

```
StaticReject: forbidden bounds usage 'func.bounds.shape'
```

**What the LLM generates (WRONG):**
```python
def __call__(self, func):
    self.dim = func.bounds.shape[1]           # ❌ WRONG
    self.lb, self.ub = func.bounds[0], func.bounds[1]  # ❌ WRONG
```

**What we require (CORRECT):**
```python
def __call__(self, func):
    lb = np.asarray(func.bounds.lb)   # ✓ Correct
    ub = np.asarray(func.bounds.ub)   # ✓ Correct
    dim = len(lb)                      # ✓ Correct
```

### 1.2 Secondary Error: Sampling Without Replacement

```
PreflightError: Cannot take a larger sample than population when 'replace=False'
```

**Root cause**: Tournament selection with `tournament_size = 3` but population might have fewer individuals:
```python
indices = np.random.choice(len(population), tournament_size, replace=False)  # Crashes if pop_size < 3
```

### 1.3 Missing Init Attributes

The MADA block rewrites reference attributes that don't exist yet:
```
Recent issues observed: missing __init__ attributes: 8; invalid block syntax: 2
```

---

## 2. Root Cause Analysis

### 2.1 The LLM Is Ignoring Guardrails

Looking at the conversation log, the prompt clearly states:
```
- Access bounds via lb = np.asarray(func.bounds.lb); ub likewise; do not use func.bounds.shape/len.
```

Yet the very first response uses `func.bounds.shape[1]`. This happens because:

1. **The guardrails are passive suggestions**, not enforced constraints
2. **No working example is provided** that the LLM can copy
3. **The model (Gemini 2.0 Flash) has training data bias** toward BBOB patterns that use `.shape[1]`
4. **Previous context pollution**: Even after rejection, the conversation history contains bad code

### 2.2 Static Validator Catches Errors But Can't Fix Them

The static validator in `main-thesis.py` correctly rejects bad code:
```python
forbidden_bound_tokens = [
    "func.bounds.shape",
    "bounds.shape",
    "self.bounds[0]",
    ...
]
```

But once rejected:
- The error is logged
- `algorithm_manager.record_reject(code)` saves the hash
- The next prompt gets an additional guardrail

This doesn't help because the LLM continues generating similar code.

### 2.3 MADA Block Rewrites Inherit Bad Patterns

When MADA tries to improve individual blocks (parent_selection, recombine, etc.):
- It uses the parent's code as a template
- The parent's code already has forbidden patterns
- The block rewrite inherits these patterns

### 2.4 Population Is Stuck at Fitness 0.0

Since no algorithm passes evaluation:
- All fitness values are 0.0
- Selection has no meaningful gradient
- Evolution cannot progress
- Archive contains only failed algorithms

---

## 3. Code Flow Analysis

```
┌─────────────────────────────────────────────────────────────┐
│                    fetch_algorithm()                        │
│                          │                                  │
│         LLM generates code with func.bounds.shape           │
│                          ▼                                  │
│              evaluate_algorithm()                           │
│                          │                                  │
│         _static_validate_algorithm() → FAIL                 │
│                          │                                  │
│                "StaticReject: forbidden..."                 │
│                          │                                  │
│              record_reject(code)                            │
│                          │                                  │
│         algorithm_manager.last_error = error                │
│                          ▼                                  │
│              Next generation prompt includes:               │
│              "Avoid reusing prior rejected code hashes"     │
│                          │                                  │
│              BUT: LLM generates SAME pattern                │
│                          │                                  │
│                     REPEAT...                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Specific Issues in Current Implementation

### 4.1 `managers.py` - Prompt Construction

**Problem**: The `_base_init_prompt` doesn't include a working example.

```python
self._base_init_prompt = """
Implement a population-based optimizer with explicit recombination. You must:
- Use the exact class interface: __init__(self, budget); __call__(self, func) must infer dim from func.bounds and respect budget.
...
"""
```

**Missing**: An actual working code example that shows correct bounds handling.

### 4.2 `main-thesis.py` - Static Validation

**Current validation (lines 47-83)**:
```python
def _static_validate_algorithm(code: str):
    lower = code.lower()
    
    for tok in forbidden_bound_tokens:
        if tok in lower:
            return False, f"StaticReject: forbidden bounds usage '{tok}'"
```

**Problem**: This is **case-insensitive** (using `lower`), which means `FUNC.BOUNDS.SHAPE` would match, but the actual check for `"func.bounds.lb"` uses lowercase. This could cause false negatives.

### 4.3 MADA Operator - Block Rewrites

In `operator.py`, block rewrites ask for isolated methods:
```python
Improve the `parent_selection` method of the optimizer class `RecombinationOptimizer`.
Use the exact signature `def select_parents(self):` and return only the method definition
```

**Problem**: The rewritten block is spliced into the parent's code, which already has bad `__call__` setup.

### 4.4 Tournament Selection Size

The generated code uses:
```python
tournament_size = 5
indices = np.random.choice(len(self.pop_x), size=(len(self.pop_x), tournament_size), replace=True)
```

But with a small population (e.g., `pop_size = min(budget // 10, 100)` with `budget=10` gives `pop_size=1`), this crashes.

---

## 5. Recommended Fixes

### 5.1 **CRITICAL: Provide a Working Example in the Prompt**

Add a complete, validated example to `_base_init_prompt`:

```python
self._base_init_prompt = """
Implement a population-based optimizer with explicit recombination.

CRITICAL INTERFACE REQUIREMENTS:
- __init__(self, budget): Only budget argument
- __call__(self, func): Infer dimension from bounds

CORRECT BOUNDS HANDLING (MANDATORY):
```python
def __call__(self, func):
    lb = np.asarray(func.bounds.lb)  # Always use .lb
    ub = np.asarray(func.bounds.ub)  # Always use .ub
    dim = len(lb)                     # Get dimension this way
    # NEVER use func.bounds.shape or bounds[0]/bounds[1]
```

...rest of prompt...
"""
```

### 5.2 **Add Code Auto-Correction Before Validation**

Before static validation, attempt to auto-fix common patterns:

```python
def _auto_correct_bounds(code: str) -> str:
    """Attempt to auto-fix common bounds mistakes."""
    # Replace func.bounds.shape[1] with len(np.asarray(func.bounds.lb))
    code = re.sub(
        r'func\.bounds\.shape\[1\]',
        'len(np.asarray(func.bounds.lb))',
        code
    )
    # Replace func.bounds[0], func.bounds[1] with proper lb/ub
    code = re.sub(
        r'self\.lb,\s*self\.ub\s*=\s*func\.bounds\[0\],\s*func\.bounds\[1\]',
        'self.lb = np.asarray(func.bounds.lb)\n        self.ub = np.asarray(func.bounds.ub)',
        code
    )
    return code
```

### 5.3 **Fix Tournament Selection Safety**

Ensure tournament selection handles small populations:

```python
def select_parents(self, population, fitness):
    tournament_size = min(3, len(population))  # Don't exceed population size
    if tournament_size < 2:
        return list(range(len(population)))  # Return all if too small
    indices = np.random.choice(len(population), tournament_size, replace=True)  # Use replace=True
    ...
```

### 5.4 **Seed the Initial Population with a Known-Good Algorithm**

Instead of relying on LLM to generate valid code from scratch, provide one working algorithm:

```python
SEED_ALGORITHM = '''
import numpy as np

class BaseOptimizer:
    def __init__(self, budget):
        self.budget = budget
        self.f_opt = float('inf')
        self.x_opt = None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        dim = len(lb)
        pop_size = max(10, min(self.budget // 10, 50))
        
        population = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.array([func(x) for x in population])
        evals = pop_size
        
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()
        
        while evals < self.budget:
            # Tournament selection
            parents_idx = []
            for _ in range(pop_size):
                candidates = np.random.choice(pop_size, min(3, pop_size), replace=True)
                winner = candidates[np.argmin(fitness[candidates])]
                parents_idx.append(winner)
            parents = population[parents_idx]
            
            # Recombination (uniform crossover)
            offspring = np.empty_like(parents)
            for i in range(0, pop_size, 2):
                if i + 1 < pop_size:
                    mask = np.random.rand(dim) < 0.5
                    offspring[i] = np.where(mask, parents[i], parents[i+1])
                    offspring[i+1] = np.where(mask, parents[i+1], parents[i])
                else:
                    offspring[i] = parents[i].copy()
            
            # Mutation
            mutation = np.random.normal(0, 0.1 * (ub - lb), offspring.shape)
            offspring = offspring + mutation * (np.random.rand(*offspring.shape) < 0.1)
            offspring = np.clip(offspring, lb, ub)
            
            # Evaluate
            offspring_f = np.array([func(x) for x in offspring])
            evals += pop_size
            
            # Survivor selection (elitist)
            combined = np.vstack([population, offspring])
            combined_f = np.concatenate([fitness, offspring_f])
            best_indices = np.argsort(combined_f)[:pop_size]
            population = combined[best_indices]
            fitness = combined_f[best_indices]
            
            if fitness[0] < self.f_opt:
                self.f_opt = fitness[0]
                self.x_opt = population[0].copy()
        
        return self.f_opt, self.x_opt
'''
```

### 5.5 **Consider Using a Different Model**

Gemini 2.0 Flash may not be following complex instructions reliably. Consider:
- **GPT-4 Turbo** or **GPT-4o**: Better instruction following
- **Claude 3 Opus/Sonnet**: Excellent at following precise specifications
- **Gemini 1.5 Pro**: Larger context, better reasoning

### 5.6 **Add Retry Logic with Code Repair**

```python
MAX_RETRIES = 3

def generate_valid_algorithm(manager, eval_budget):
    for attempt in range(MAX_RETRIES):
        message = manager.fetch_algorithm()
        code = manager.extract_algorithm_code(message)
        
        # Try to auto-correct
        code = _auto_correct_bounds(code)
        
        # Validate
        ok, msg = _static_validate_algorithm(code)
        if ok:
            return code, None
        
        # Update guardrails with specific error
        manager.update_guardrail_feedback(f"Attempt {attempt+1} failed: {msg}")
    
    return None, "Failed to generate valid code after retries"
```

---

## 6. Immediate Action Items

| Priority | Action | File | Effort |
|----------|--------|------|--------|
| 🔴 HIGH | Add working example to prompt | `managers.py` | Low |
| 🔴 HIGH | Add auto-correction for bounds | `main-thesis.py` | Medium |
| 🟡 MED | Seed population with valid algorithm | `main-thesis.py` | Low |
| 🟡 MED | Fix tournament selection safety | Template/validation | Low |
| 🟢 LOW | Consider different LLM model | Configuration | Low |
| 🟢 LOW | Add retry with repair logic | `main-thesis.py` | Medium |

---

## 7. Conclusion

The framework's logic is sound, but the LLM is not complying with the prompt instructions. The most effective fix is to:

1. **Show, don't tell**: Provide a complete working example in the prompt
2. **Auto-correct**: Fix common mistakes before validation
3. **Seed the population**: Start with at least one valid algorithm so evolution has something to work with

Without these changes, the system will continue producing zero-fitness algorithms indefinitely.

