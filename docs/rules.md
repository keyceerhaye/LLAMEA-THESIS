# LLaMEA Coding Rules and Conventions

## Overview
This document defines the coding style, naming conventions, architectural patterns, and best practices for the LLaMEA codebase.

## Core Principle

**KISS: Keep It Simple, Stupid** [[memory:2272829]]

All design decisions should favor simplicity over cleverness. Code should be:
- Easy to understand
- Easy to modify
- Easy to debug
- Easy to test

## Python Version

**Minimum:** Python 3.11+

**Rationale:**
- Modern type hinting support
- Performance improvements
- Standard library enhancements

## Code Style

### General Formatting

**Indentation:**
- Always use 4 spaces (never tabs)
- Consistent across all files

**Line Length:**
- Target: 80 characters
- Hard limit: 100 characters
- Exceptions: Long strings, URLs, import statements

**Encoding:**
- UTF-8 for all Python files
- Include `# -*- coding: utf-8 -*-` if non-ASCII characters present

**Imports:**
```python
# Standard library
import os
import re
from typing import List, Optional

# Third-party
import numpy as np
from ConfigSpace import ConfigurationSpace

# Local
from .solution import Solution
from .utils import code_distance
```

**Order:**
1. Standard library imports
2. Third-party library imports
3. Local application imports

Within each group, alphabetically sorted.

### Naming Conventions

#### Classes
```python
# PascalCase, always capitalized
class LLaMEA:
    pass

class ExperimentLogger:
    pass

class OpenAI_LLM:  # Exception: Underscores for acronym separation
    pass
```

#### Functions and Methods
```python
# snake_case, always lowercase
def initialize_population():
    pass

def evaluate_fitness(individual):
    pass

def apply_unified_diff(text, patch):
    pass
```

#### Variables
```python
# snake_case, descriptive names
population_size = 10
best_so_far = None
current_generation = 0

# Single-letter acceptable for:
# - Loop indices: i, j, k
# - Mathematics: x, y, f
# - Dimensions: n, m, d
```

#### Constants
```python
# SCREAMING_SNAKE_CASE
MAX_RETRIES = 5
DEFAULT_TEMPERATURE = 0.8
BBOB_FUNCTION_COUNT = 24
```

#### Private Members
```python
class MyClass:
    def __init__(self):
        self._private_var = 0      # Single underscore: internal use
        self.__very_private = 0    # Double underscore: name mangling
    
    def _internal_method(self):    # Single underscore: internal use
        pass
```

### Documentation

#### Module Docstrings
```python
"""
LLM modules to connect to different LLM providers.

This module provides abstraction layers for interacting with various
Large Language Model APIs including OpenAI, Gemini, and Ollama.
"""
```

#### Class Docstrings
```python
class LLaMEA:
    """
    A class that represents the Language Model powered Evolutionary Algorithm.
    
    This class handles the initialization, evolution, and interaction with
    a language model to generate and refine algorithms.
    """
```

#### Function Docstrings
```python
def apply_unified_diff(text: str, diff: str) -> str:
    """
    Apply a unified diff to the given text using pure Python.
    
    Args:
        text: The original text to patch.
        diff: The unified diff patch.
    
    Returns:
        The patched text as a string.
    
    Raises:
        ValueError: If the diff format is invalid.
    """
```

**Style:** Google-style docstrings preferred.

#### Comments
```python
# Good: Explain WHY, not WHAT
# Normalize score to [0, 1] range because BBOB uses log scale
normalized_score = (score - min_score) / (max_score - min_score)

# Bad: Redundant
# Divide score by max_score
normalized_score = score / max_score
```

### Type Hints

**Required for:**
- Public API functions
- Complex functions
- Functions with non-obvious parameter types

```python
from typing import List, Optional, Callable, Tuple

def evolve_solution(
    self, 
    individual: Solution
) -> Solution:
    pass

def sample_solution(
    self,
    session_messages: list,
    parent_ids: Optional[List[str]] = None,
    HPO: bool = False
) -> Solution:
    pass

# Callable type hints
def __init__(
    self,
    f: Callable[[Solution, Optional[ExperimentLogger]], Solution],
    llm: LLM,
    distance_metric: Optional[Callable[[Solution, Solution], float]] = None
):
    pass
```

## Directory Structure Rules

### Package Organization
```
llamea/                     # Core package
├── __init__.py            # Public API exports
├── llamea.py              # Main algorithm
├── llm.py                 # LLM providers
├── solution.py            # Data structures
├── loggers.py             # Logging
├── utils.py               # Utilities
└── bbobalgs/              # Subpackage for generated algorithms
    ├── __init__.py
    └── *.py
```

### File Naming
- Lowercase with underscores: `experiment_logger.py`
- Test files: `test_*.py`
- Main scripts: `main-*.py` (hyphens for executable scripts)

### Module Boundaries

**Core (`llamea/`):**
- No dependencies on root-level scripts
- No dependencies on examples
- No dependencies on tests
- Minimal external dependencies

**Root scripts:**
- May import from `llamea/`
- May import from `managers.py`, `utils.py`
- Self-contained examples

**Examples (`examples/`):**
- May import from `llamea/`
- May import from root-level utilities
- Each example should be runnable independently

## Design Patterns

### Expected Patterns

#### Dependency Injection
```python
# Good: Inject dependencies
class LLaMEA:
    def __init__(self, f, llm, logger=None):
        self.f = f
        self.llm = llm
        self.logger = logger

# Bad: Hard-coded dependencies
class LLaMEA:
    def __init__(self):
        self.llm = OpenAI_LLM("hardcoded-key")
```

#### Strategy Pattern
```python
# Good: Configurable strategies
niching_strategies = {
    "sharing": apply_fitness_sharing,
    "clearing": apply_clearing,
    None: lambda x: x
}
strategy = niching_strategies[self.niching]
population = strategy(population)
```

#### Composition Over Inheritance
```python
# Good: Compose behaviors
class LLaMEA:
    def __init__(self, llm, logger):
        self.llm = llm           # Has-a LLM
        self.logger = logger     # Has-a Logger

# Avoid: Deep inheritance hierarchies
class AdvancedLLaMEA(LLaMEA):
    class SuperAdvancedLLaMEA(AdvancedLLaMEA):
        # Too deep!
```

### Forbidden Patterns

#### Global State
```python
# Bad: Global mutable state
global_population = []

def evolve():
    global global_population
    global_population.append(...)

# Good: Pass state explicitly
def evolve(population):
    return population + [new_individual]
```

#### God Objects
```python
# Bad: One class does everything
class LLaMEAGod:
    def initialize(self): pass
    def query_llm(self): pass
    def evaluate(self): pass
    def log(self): pass
    def plot(self): pass
    def export(self): pass
    # Too many responsibilities!

# Good: Separate concerns
class LLaMEA: ...         # Evolution logic
class LLM: ...            # LLM interaction
class ExperimentLogger: # Logging
```

## Error Handling

### Exception Hierarchy
```python
# Custom exceptions for domain-specific errors
class NoCodeException(Exception):
    """Could not extract generated code."""
    pass

class OverBudgetException(Exception):
    """Algorithm exceeded evaluation budget."""
    pass
```

### Error Handling Strategy
```python
# Fail fast for programming errors
def evaluate(solution):
    if solution is None:
        raise ValueError("Solution cannot be None")

# Catch and log for runtime errors
try:
    solution = evaluate_fitness(solution)
except Exception as e:
    solution.set_scores(worst_value, f"Error: {e}", repr(e))
    logger.log_individual(solution)
```

### Error Messages
```python
# Good: Specific, actionable
raise ValueError(
    "API key is required. Options:\n"
    "  1. Use --api-key argument\n"
    "  2. Set OPENAI_API_KEY environment variable\n"
    "  3. Create a .env file with OPENAI_API_KEY=your_key"
)

# Bad: Vague
raise ValueError("Missing key")
```

## Testing Conventions

### Test File Organization
```
tests/
├── __init__.py
├── test_llamea.py          # Tests for llamea.py
├── test_llm.py             # Tests for llm.py
├── test_solution.py        # Tests for solution.py
└── test_utils.py           # Tests for utils.py
```

### Test Naming
```python
def test_initialize_creates_population():
    """Test that initialize() creates the correct population size."""
    pass

def test_evaluate_fitness_handles_errors():
    """Test that evaluate_fitness() catches and logs errors."""
    pass
```

### Test Structure (AAA Pattern)
```python
def test_solution_copy():
    # Arrange
    original = Solution(code="x = 1", name="Test")
    
    # Act
    copy = original.copy()
    
    # Assert
    assert copy.id != original.id
    assert copy.parent_ids == [original.id]
    assert copy.generation == original.generation + 1
```

### Mocking
```python
# Use pytest-mock or unittest.mock
def test_llm_query_with_mock(mocker):
    mock_client = mocker.patch('openai.OpenAI')
    mock_client.return_value.chat.completions.create.return_value = ...
    
    llm = OpenAI_LLM(api_key="test")
    response = llm.query([{"role": "user", "content": "test"}])
    
    assert response == expected_response
```

## Configuration Management

### Avoid Magic Numbers
```python
# Bad
for i in range(24):
    problem = get_problem(i + 1, ...)

# Good
BBOB_FUNCTION_COUNT = 24
for fid in range(1, BBOB_FUNCTION_COUNT + 1):
    problem = get_problem(fid, ...)
```

### Configuration Files
```yaml
# benchmark_config.yaml
default:
  llm_provider: "openai"
  budget: 100
  n_parents: 5
  n_offspring: 5
```

### Environment Variables
```python
# Good: Environment variables for secrets
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

# Bad: Hardcoded secrets
api_key = "sk-hardcoded-key"
```

## Logging Conventions

### Log Levels
```python
import logging

logger = logging.getLogger(__name__)

# ERROR: Errors that prevent operation
logger.error(f"Failed to evaluate solution {solution.id}: {error}")

# WARNING: Recoverable issues
logger.warning(f"API rate limit reached, retrying in {delay}s")

# INFO: Progress information
logger.info(f"Generation {gen}, best fitness: {best.fitness}")

# DEBUG: Detailed debugging information
logger.debug(f"Constructed prompt: {prompt[:100]}...")
```

### Structured Logging (JSONL)
```python
# For experiment data
with jsonlines.open("log.jsonl", "a") as f:
    f.write({
        "id": solution.id,
        "fitness": solution.fitness,
        "generation": solution.generation,
        "timestamp": datetime.now().isoformat()
    })
```

## Performance Conventions

### Parallelization
```python
# Use joblib for CPU-bound parallel tasks
from joblib import Parallel, delayed

results = Parallel(n_jobs=max_workers, backend="loky")(
    delayed(evaluate)(solution) for solution in population
)
```

### Avoid Premature Optimization
```python
# Good: Clear and simple first
def compute_distance(a, b):
    return 1 - similarity(a, b)

# Only optimize if profiling shows it's a bottleneck
# Then add:
@lru_cache(maxsize=1000)
def compute_distance(a, b):
    return 1 - similarity(a, b)
```

## Git Conventions

### Commit Messages
```
Format: <type>: <subject>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Adding tests
- chore: Maintenance

Examples:
feat: Add diff mode support to LLM sampling
fix: Correct AUC calculation for early-stopped runs
docs: Update installation instructions
refactor: Simplify niching strategy selection
```

### Branch Naming
```
feature/diff-mode-support
fix/auc-calculation-bug
docs/architecture-document
refactor/llm-abstraction
```

## Framework-Specific Rules

### LLaMEA Configuration
```python
# Good: Explicit configuration
llamea = LLaMEA(
    f=evaluate,
    llm=llm,
    n_parents=5,
    n_offspring=10,
    budget=100,
    niching="sharing",
    niche_radius=0.3,
    experiment_name="my_experiment"
)

# Bad: Relying on defaults without understanding them
llamea = LLaMEA(f=evaluate, llm=llm)
```

### Prompt Engineering
```python
# Good: Clear, structured prompts
task_prompt = """
The optimization algorithm should:
1. Accept a budget and dimension in __init__
2. Implement __call__(self, func)
3. Respect the evaluation budget
4. Work on problems with bounds [-5.0, 5.0]

Example:
```python
class MyAlgorithm:
    ...
```
"""

# Bad: Vague prompts
task_prompt = "Make a good optimization algorithm"
```

### Solution Handling
```python
# Good: Always set scores explicitly
solution.set_scores(
    fitness=0.85,
    feedback="Good performance on test functions",
    error=""
)

# Bad: Modifying attributes directly
solution.fitness = 0.85  # Incomplete
```

## Module Dependencies

### Dependency Rules

**Level 0: Standard Library**
- `os`, `re`, `json`, `datetime`, `uuid`, etc.

**Level 1: Core External**
- `numpy` (numerical operations)
- No optional dependencies here

**Level 2: LLaMEA Core**
- `llamea.solution`
- `llamea.utils`
- `llamea.loggers`

**Level 3: Framework**
- `llamea.llm` (depends on Level 2)
- `llamea.llamea` (depends on all)

**Level 4: Applications**
- Root scripts
- Examples
- Tests

**Rule:** Lower levels cannot import from higher levels.

### Optional Dependencies
```python
# Good: Graceful degradation
try:
    import smac
    HPO_AVAILABLE = True
except ImportError:
    HPO_AVAILABLE = False

if HPO and not HPO_AVAILABLE:
    raise ImportError("HPO requires SMAC. Install with: pip install smac")
```

## Documentation Requirements

### Public API
- All public classes, functions, and methods must have docstrings
- Include type hints for all parameters and return values
- Document exceptions raised
- Provide usage examples for non-trivial APIs

### README Files
- Each major directory should have a README.md
- Explain purpose, contents, and usage
- Link to relevant documentation

### Inline Comments
- Explain complex algorithms
- Document workarounds and gotchas
- Reference papers/sources for algorithms

## Anti-Patterns to Avoid

### Don't Repeat Yourself (DRY) Violations
```python
# Bad: Repeated code
aucs1 = []
for fid in range(1, 6):
    auc = evaluate(fid)
    aucs1.append(auc)

aucs2 = []
for fid in range(6, 10):
    auc = evaluate(fid)
    aucs2.append(auc)

# Good: Extract function
def evaluate_range(start, end):
    return [evaluate(fid) for fid in range(start, end)]

aucs1 = evaluate_range(1, 6)
aucs2 = evaluate_range(6, 10)
```

### Premature Abstraction
```python
# Bad: Over-engineered for single use
class AbstractEvaluatorFactory(ABC):
    @abstractmethod
    def create_evaluator(self): pass

class BBOBEvaluatorFactory(AbstractEvaluatorFactory):
    def create_evaluator(self):
        return BBOBEvaluator()

# Good: Simple direct implementation
def evaluate_bbob(solution):
    # Direct implementation
    pass
```

### Leaky Abstractions
```python
# Bad: Implementation details leak through API
def get_llm_client():
    return openai.OpenAI(api_key=...)  # Exposes OpenAI client

# Good: Hide implementation
class LLM:
    def query(self, messages):
        # Internal use of OpenAI client
        return response
```

## Summary

Follow these rules to maintain:
- **Consistency**: Code looks like it was written by one person
- **Readability**: New contributors can understand quickly
- **Maintainability**: Changes are easy and safe
- **Simplicity**: KISS principle always [[memory:2272829]]


