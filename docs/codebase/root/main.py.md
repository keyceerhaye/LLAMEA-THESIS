# main.py

## Purpose
Original simple script demonstrating basic usage of LLaMEA with custom `AlgorithmManager` wrapper for BBOB benchmark evaluation.

## Role in System
Legacy/reference implementation showing:
- Basic evolutionary loop without full LLaMEA framework
- Custom algorithm management and refinement
- BBOB benchmark evaluation pipeline
- IOH experimenter integration

## Key Components

### Script Flow
1. Load OpenAI API key from environment
2. Initialize `AlgorithmManager` with model and settings
3. For each iteration (0 to budget):
   - First iteration: Fetch initial algorithm from LLM
   - Subsequent: Refine algorithm based on previous performance
   - Extract and execute generated code
   - Evaluate on BBOB functions (24 functions, 3 instances, 3 reps each)
   - Calculate AUC scores
   - Log code and results

### Configuration
- `api_key` - From `OPENAI_API_KEY` environment variable
- `ai_model` - "gpt-4-turbo" (hardcoded)
- `openai_budget` - 100 iterations (hardcoded)
- `budget` - 10000 evaluations per algorithm
- BBOB: Functions 1-24, instances 1-3, 5D, 3 repetitions

### Evaluation Details
- Uses custom `aoc_logger` to track Area Over Convergence Curve
- Computes corrected AUC with `correct_aoc()` for early-stopped runs
- Tracks detailed AUCs by function groups:
  - Functions 1-5: Separable
  - Functions 6-9: Low/moderate conditioning
  - Functions 10-14: High conditioning, unimodal
  - Functions 15-19: Multi-modal, adequate structure
  - Functions 20-24: Multi-modal, weak structure

### Error Handling
- `NoCodeException` → AUC = 0.0
- General exceptions → AUC = 0.0, error stored in `algorithm_manager.last_error`
- `OverBudgetException` → Caught and ignored (algorithm used too many evaluations)

## Dependencies
- `os` - Environment variables
- `numpy` - Numerical operations
- `ioh` - IOH experimenter (BBOB functions, logging)
- `re` - Class name extraction
- `managers` - `AlgorithmManager`, `ExperimentLogger`
- `utils` - `OverBudgetException`, `NoCodeException`, `aoc_logger`, `correct_aoc`

## Data Flow
```
main.py
  ↓
AlgorithmManager.fetch_algorithm() or .refine_algorithm()
  ↓
OpenAI API → Code generation
  ↓
exec(new_algorithm, globals()) → Load algorithm class
  ↓
For each BBOB function/instance/rep:
  ↓
  algorithm(problem) → Optimization run
  ↓
  aoc_logger tracks performance
  ↓
correct_aoc() → Normalized AUC score
  ↓
Aggregate AUCs → Mean/Std
  ↓
ExperimentLogger.log_code(), .log_aucs()
  ↓
Next iteration with feedback
```

## Differences from LLaMEA Framework
- **No population**: Single algorithm evolved (1+1)-style
- **Custom manager**: Uses `AlgorithmManager` instead of `LLaMEA` class
- **Manual loop**: Explicit for-loop instead of `LLaMEA.run()`
- **Elitism option**: Controlled via `AlgorithmManager(elitism=True)`
- **Limited features**: No niching, HPO, diff mode, parallelization

## Risks and Quirks
- **Hardcoded settings**: Model, budget, dimensions not configurable
- **Global exec**: `exec(new_algorithm, globals())` can pollute namespace
- **Manual error handling**: More verbose than framework version
- **No parallelization**: Sequential evaluation only
- **Legacy code**: Superseded by `main-thesis.py` and `main-evolutionary.py`

## Usage
```bash
export OPENAI_API_KEY="your_key"
python main.py
```

## Output
- Experiment directory: `exp-{date}_{time}-{model}-elitism/`
- Code files: `code/try-{N}-{AlgorithmName}.py`
- AUC files: `try-{N}-aucs.txt`
- Conversation log: `conversationlog.txt`
- Console output: Algorithm name, mean AUC, std AUC per iteration

## Comparison to Other Main Scripts
- **main.py**: Original simple version, no CLI args
- **main-thesis.py**: Enhanced with argparse, flexible configuration
- **main-evolutionary.py**: Uses full LLaMEA framework with populations

## When to Use
- Quick testing without framework overhead
- Understanding basic LLM-evolution loop
- Reference for custom evaluation pipelines
- **Not recommended** for production use (use `main-evolutionary.py` instead)


