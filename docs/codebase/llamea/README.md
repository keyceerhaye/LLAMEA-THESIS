# llamea/ Package

## Purpose
Core package implementing the LLaMEA (Large Language Model Evolutionary Algorithm) framework.

## Overview
This package contains the main algorithm engine that combines evolutionary computation with Large Language Models for automated algorithm discovery and optimization.

## Package Structure

### Core Modules

#### `llamea.py`
**Main evolutionary algorithm engine**
- Orchestrates population initialization, evolution, and selection
- Manages interaction with LLM for code generation/mutation
- Implements niching strategies (fitness sharing, clearing)
- Handles parallel evaluation and logging
- **Entry point**: `LLaMEA` class

#### `llm.py`
**LLM provider abstraction layer**
- Interfaces with multiple LLM providers (OpenAI, Gemini, Ollama, DeepSeek)
- Extracts code, descriptions, and config spaces from LLM responses
- Handles retry logic and rate limiting
- Supports diff mode for incremental mutations
- **Key classes**: `LLM` (base), `OpenAI_LLM`, `Gemini_LLM`, `Ollama_LLM`, `Multi_LLM`, `Dummy_LLM`

#### `solution.py`
**Solution data structure**
- Represents individual candidate algorithms
- Stores code, fitness, feedback, genealogy
- Manages metadata and serialization
- Tracks parent-child relationships
- **Key class**: `Solution`

#### `utils.py`
**Utility functions**
- Unified diff patch application
- Code distance metrics (AST-based)
- Power law distributions for adaptive mutation
- Custom exceptions
- **Key functions**: `apply_unified_diff()`, `code_distance()`, `discrete_power_law_distribution()`

#### `loggers.py`
**Experiment logging**
- Creates timestamped experiment directories
- Logs conversations, code, and performance data
- Saves ConfigSpace definitions
- Maintains reproducibility
- **Key class**: `ExperimentLogger`

#### `__init__.py`
**Package exports**
- Exposes public API: `LLaMEA`, `Solution`, `ExperimentLogger`
- Exposes LLM classes
- Exposes utility functions and exceptions

### Subdirectory

#### `bbobalgs/`
Contains example black-box optimization algorithms:
- `ERADS_QuantumFluxUltraRefined.py` - Example generated algorithm
- Used for reference and testing

## Module Relationships

```
User Code
    ↓
LLaMEA (llamea.py)
    ↓
├── LLM (llm.py) ←→ Solution (solution.py)
│       ↓
│   LLM Providers (OpenAI/Gemini/Ollama)
│
├── ExperimentLogger (loggers.py)
│       ↓
│   Disk Storage (JSONL, code files)
│
└── Utilities (utils.py)
        ↓
    apply_unified_diff, code_distance, power_law
```

## Key Design Patterns

### Abstraction Layers
1. **LLM abstraction**: `LLM` base class with provider-specific implementations
2. **Evaluation abstraction**: User-defined fitness function
3. **Logging abstraction**: Decoupled from core algorithm

### Composition
- `LLaMEA` composes `LLM`, `ExperimentLogger`
- `LLM` creates `Solution` objects
- `Solution` is data-only, no complex behavior

### Strategy Pattern
- Niching strategies: sharing vs clearing
- Selection strategies: elitism (μ+λ) vs comma (μ,λ)
- Mutation operators: configurable prompts

## Data Flow Through Package

### Initialization
```
User → LLaMEA(llm, f, ...) 
    → LLaMEA.initialize()
    → LLM.sample_solution()
    → Solution created
    → f(Solution) evaluates
    → Population ready
```

### Evolution Loop
```
LLaMEA.run()
    → Select parents
    → Construct mutation prompts
    → LLM.sample_solution() for each offspring
    → Evaluate offspring
    → Select survivors (with optional niching)
    → Log generation
    → Repeat until budget exhausted
```

### Logging
```
LLM.query() → ExperimentLogger.log_conversation()
Solution → ExperimentLogger.log_code()
Solution → ExperimentLogger.log_individual()
Fitness → ExperimentLogger.log_aucs()
```

## Configuration Points

### LLaMEA Configuration
- Population sizes (`n_parents`, `n_offspring`)
- Evolutionary strategy (`elitism`)
- Budget (`budget`)
- Prompts (role, task, example, output format)
- Niching (`niching`, `niche_radius`, `distance_metric`)
- Parallelization (`max_workers`, `parallel_backend`)
- Logging (`log`, `experiment_name`)
- HPO (`HPO`)
- Diff mode (`diff_mode`)

### LLM Configuration
- Provider choice (OpenAI/Gemini/Ollama)
- Model selection
- API credentials
- Temperature/sampling parameters
- Custom regex patterns

### Logging Configuration
- Experiment naming
- Automatic vs manual logging
- Directory structure (fixed)

## Dependencies

### External
- `numpy` - Numerical operations
- `openai` - OpenAI API client
- `google-generativeai` - Gemini API client
- `ollama` - Ollama API client
- `ConfigSpace` - Hyperparameter configuration
- `joblib` - Parallel processing
- `jsonlines` - JSONL file format

### Internal
All modules depend on each other minimally:
- `llamea.py` imports all other modules
- `llm.py` imports `solution.py`, `utils.py`
- Other modules are relatively independent

## Extension Points

### Adding New LLM Provider
1. Subclass `LLM` in `llm.py`
2. Implement `query()` method
3. Handle provider-specific errors
4. Add to `__init__.py` exports

### Custom Distance Metrics
1. Define function: `metric(solution_a, solution_b) -> float`
2. Pass to `LLaMEA(distance_metric=metric)`

### Custom Mutation Operators
1. Define prompt strings
2. Pass to `LLaMEA(mutation_prompts=[...])`

### Custom Logging
1. Extend `ExperimentLogger` in `loggers.py`
2. Add custom log methods
3. Call from evaluation function or LLaMEA

## Testing Strategy
- Unit tests in `tests/` directory
- Mock LLM using `Dummy_LLM`
- Test each module independently
- Integration tests for full pipeline

## Performance Characteristics
- **Bottleneck**: LLM API calls (seconds per query)
- **Parallelizable**: Solution evaluation
- **Memory**: O(population_size * generations) for history
- **Disk**: Grows linearly with evaluations (logs)

## Common Issues

### LLM Failures
- Invalid/non-executable code → logged as error, fitness = worst_value
- Rate limits → automatic retry with backoff
- No code extracted → NoCodeException

### Evaluation Issues
- Infinite loops → timeout after `eval_timeout`
- Exceptions → caught, logged, fitness = worst_value
- Over budget → OverBudgetException

### Niching Issues
- Inappropriate radius → poor diversity or slow convergence
- Distance metric mismatch → ineffective niching
- Adaptive radius instability → fluctuating population diversity

## Best Practices

### For Users
1. Start with small populations and budget for testing
2. Use `Dummy_LLM` for pipeline testing before paying API costs
3. Set reasonable `eval_timeout` to prevent hangs
4. Monitor logs to debug prompt issues
5. Use diff_mode to reduce token usage after initialization

### For Developers
1. Keep modules loosely coupled
2. Add type hints for clarity
3. Handle all exceptions gracefully
4. Log generously for debugging
5. Test with multiple LLM providers

## Future Extensions
- Coevolutionary algorithms
- Multi-objective optimization
- Constraint handling
- Transfer learning across problems
- GPU-accelerated evaluation
- Distributed evaluation


