# utils.py (root)

## Purpose
BBOB-specific utility functions for evaluation, logging, and error handling in thesis scripts.

## Role in System
Provides specialized utilities for BBOB benchmark evaluation:
- Area Over Convergence Curve (AOCC/AUC) calculation
- Custom IOH loggers
- Exception handling for evaluation
- Budget enforcement

## Key Components

### Exceptions

#### `OverBudgetException`
Raised when algorithm exceeds allowed evaluation budget.

**Usage:**
```python
if log_info.evaluations > self.budget:
    raise OverBudgetException
```

**Caught by:** Main evaluation loops to prevent runaway algorithms.

#### `NoCodeException`
Raised when LLM response doesn't contain extractable code.

**Note:** Duplicate of `llamea.utils.NoCodeException`. Root version for legacy compatibility.

### Functions

#### `correct_aoc(ioh_function, logger, budget) -> float`
Corrects Area Over Convergence (AOC) for early-stopped runs.

**Problem:** 
If algorithm stops before budget exhausted, raw AOC underestimates performance (assumes no progress after stopping).

**Solution:**
Extrapolate final fitness for remaining evaluations.

**Algorithm:**
1. Get final fitness from `ioh_function.state.current_best_internal.y`
2. Clip to logger bounds [lower, upper]
3. Transform (log scale if configured)
4. Calculate fraction of range covered: (f - lower) / (upper - lower)
5. Calculate missing evaluations: budget - actual_evaluations
6. Add: logger.aoc + (missing_evals * fraction)
7. Normalize by budget
8. Return: 1 - normalized_aoc (higher is better)

**Parameters:**
- `ioh_function` - IOH problem in final state (before reset)
- `logger` - aoc_logger in final state (for settings)
- `budget` - Intended maximum evaluations

**Returns:** Normalized AUC in [0, 1], higher is better

**Important:** Call before resetting function or logger.

### Classes

#### `aoc_logger(logger.AbstractLogger)`
Custom IOH logger for tracking Area Over Convergence.

**Purpose:** 
Incrementally computes AOCC during optimization run.

**Key Attributes:**
- `aoc` - Accumulated area (updated each evaluation)
- `lower`, `upper` - Fitness range bounds (default: 1e-8 to 1e8)
- `budget` - Maximum allowed evaluations
- `transform` - Transformation function (log10 or identity)

**Key Methods:**

##### `__init__(budget, lower=1e-8, upper=1e8, scale_log=True)`
Initialize logger with bounds and budget.

**Parameters:**
- `budget` - Evaluation budget
- `lower` - Lower fitness bound (better than this → 0 AOCC contribution)
- `upper` - Upper fitness bound (worse than this → 1 AOCC contribution)
- `scale_log` - Use log10 scale (True) or linear (False)

##### `__call__(log_info: LogInfo)`
Called by IOH after each evaluation.

**Algorithm:**
1. Check if over budget → raise OverBudgetException
2. If at exact budget → skip (final evaluation)
3. Clip fitness to [lower, upper]
4. Transform fitness (log10 or identity)
5. Compute normalized value: (transform(f) - transform(lower)) / (transform(upper) - transform(lower))
6. Add to `self.aoc`

**Purpose:** Accumulates area under curve as algorithm progresses.

##### `reset(func)`
Reset logger for next run.

**Effect:** Sets `self.aoc = 0`

#### `budget_logger(logger.AbstractLogger)`
Simple logger that only enforces budget constraint.

**Purpose:** 
Prevent algorithms from exceeding allowed evaluations.

**Key Methods:**

##### `__init__(budget)`
Initialize with evaluation budget.

##### `__call__(log_info: LogInfo)`
Check budget, raise exception if exceeded.

```python
if log_info.evaluations > self.budget:
    raise OverBudgetException
```

##### `reset()`
Reset logger (no state to clear).

## AOCC Calculation Details

### Area Over Convergence Curve
AOCC measures "how bad" an algorithm is:
- Low AOCC (near 0) = good (reaches low fitness quickly)
- High AOCC (near 1) = bad (stays at high fitness)

### Transformation: 1 - AOCC → AUC
Scripts report AUC (Area Under Convergence):
- AUC = 1 - AOCC
- High AUC (near 1) = good
- Low AUC (near 0) = bad

### Log Scale
Default `scale_log=True` uses log10 transformation:
- Fitness range: [1e-8, 1e8] → [log10(1e-8), log10(1e8)] = [-8, 8]
- Better for optimization where improvements span orders of magnitude
- Matches IOH standard practices

### Bounds
- `lower=1e-8`: Target fitness (optimal)
- `upper=1e2` or `1e8`: Baseline fitness (random)
- Clipping ensures outliers don't break normalization

## Usage Patterns

### Basic AOCC Logging
```python
from utils import aoc_logger, correct_aoc

budget = 10000
logger = aoc_logger(budget, upper=1e2)
problem.attach_logger(logger)

algorithm(problem)

auc = correct_aoc(problem, logger, budget)
print(f"AUC: {auc}")

logger.reset(problem)
problem.reset()
```

### Budget Enforcement Only
```python
from utils import budget_logger, OverBudgetException

budget = 10000
logger = budget_logger(budget)
problem.attach_logger(logger)

try:
    algorithm(problem)
except OverBudgetException:
    print("Algorithm exceeded budget")
```

### Multiple Runs
```python
aucs = []
for seed in range(10):
    np.random.seed(seed)
    problem.reset()
    logger.reset(problem)
    
    try:
        algorithm(problem)
    except OverBudgetException:
        pass
    
    auc = correct_aoc(problem, logger, budget)
    aucs.append(auc)

mean_auc = np.mean(aucs)
```

## Dependencies
- `numpy` - Numerical operations (clipping, etc.)
- `ioh.logger` - AbstractLogger base class
- `ioh.LogInfo` - Evaluation state information

## Data Flow
```
IOH Problem
    ↓
algorithm() calls func()
    ↓
IOH calls logger.__call__(log_info)
    ↓
aoc_logger accumulates AOCC
[optional: raise OverBudgetException]
    ↓
algorithm completes (or stops early)
    ↓
correct_aoc() extrapolates if needed
    ↓
AUC score returned
```

## Error Handling
- **OverBudgetException**: Raised by loggers, caught by evaluation loops
- **Division by zero**: Avoided by clamping to bounds
- **Invalid bounds**: No validation (assumes user provides sensible values)

## Configuration
- **Bounds**: Adjust `lower`, `upper` based on problem difficulty
- **Scale**: Use `scale_log=False` for linear-scale problems
- **Budget**: Must match algorithm budget parameter

## Risks and Quirks
- **State dependency**: `correct_aoc()` must be called before reset
- **Bound selection**: Inappropriate bounds lead to poor AOCC resolution
- **Log scale**: Not suitable for all problems (e.g., problems with negative fitness)
- **Budget precision**: Exact budget match stops logging; over budget raises exception
- **Duplicate exception**: `NoCodeException` also defined in `llamea.utils`

## Performance Considerations
- **Lightweight**: O(1) per evaluation
- **No memory accumulation**: Only stores running sum
- **Log transform**: Negligible overhead

## Integration Points
- **main.py**: Uses for BBOB evaluation
- **main-thesis.py**: Uses for BBOB evaluation
- **main-evolutionary.py**: Uses for BBOB evaluation
- **IOH experimenter**: Implements AbstractLogger interface

## Comparison with Standard IOH Logging
| Feature | aoc_logger | IOH Analyzer |
|---------|------------|--------------|
| AOCC | ✅ Real-time | ✅ Post-hoc |
| Budget enforcement | ✅ | ❌ |
| Early stop correction | Via correct_aoc() | ❌ |
| File output | ❌ | ✅ |
| Full convergence data | ❌ | ✅ |

**Use case:**
- aoc_logger: Quick single-number metric
- IOH Analyzer: Detailed analysis, plotting, comparisons

## Extension Points
To customize AOCC calculation:
1. Subclass `aoc_logger`
2. Override `__call__()` to change accumulation
3. Override `transform` for different scaling
4. Adjust bounds for problem-specific ranges


