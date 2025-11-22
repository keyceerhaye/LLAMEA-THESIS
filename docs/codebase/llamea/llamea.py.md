# llamea.py

## Purpose
Core evolutionary algorithm implementation that integrates Large Language Models (LLMs) for automated algorithm generation and evolution.

## Role in System
This is the main engine of the LLaMEA framework. It orchestrates:
- Population initialization and management
- Evolutionary loop (selection, mutation via LLM, evaluation)
- Interaction with LLM for code generation/mutation
- Fitness evaluation and logging
- Niching strategies for diversity maintenance

## Key Classes

### `LLaMEA`
The main evolutionary algorithm class that drives the entire optimization process.

**Key Responsibilities:**
- Initialize population by querying LLM for algorithm implementations
- Run evolutionary loop: select parents → mutate via LLM → evaluate → select survivors
- Manage population diversity through niching (fitness sharing or clearing)
- Track best solution across generations
- Handle parallel evaluation of solutions
- Log experiments and conversations with LLM

**Important Methods:**
- `__init__()` - Configure algorithm parameters (population sizes, budget, LLM settings, niching)
- `run()` - Main evolutionary loop until budget exhausted
- `initialize()` - Generate initial population by querying LLM
- `evolve_solution()` - Mutate a solution by prompting LLM with feedback
- `evaluate_fitness()` - Execute generated code and measure performance
- `selection()` - Select survivors based on fitness (μ+λ or μ,λ strategy)
- `apply_niching()` - Apply fitness sharing or clearing for diversity
- `construct_prompt()` - Build LLM prompts with context (population, feedback, mutation operators)
- `optimize_task_prompt()` - Adaptively refine the task prompt (if enabled)

## Key Features

### Evolutionary Strategies
- **Elitism** (μ+λ): Parents + offspring compete for survival
- **Comma Strategy** (μ,λ): Only offspring compete for survival
- Configurable population sizes (`n_parents`, `n_offspring`)

### LLM Integration
- Prompts constructed with role, task description, examples, and output format
- Feedback loop: performance metrics → LLM → refined algorithm
- Adaptive mutation: controls mutation strength based on code size
- Adaptive prompts: co-evolves task description with solutions
- Diff mode: requests incremental patches instead of full code

### Niching for Diversity
- **Fitness Sharing**: Penalizes similar solutions in same niche
- **Clearing**: Only fittest in each niche survives
- Distance metric: AST-based code similarity by default
- Adaptive niche radius: adjusts based on population distances

### Hyperparameter Optimization (HPO)
- Optional integration with SMAC for in-the-loop parameter tuning
- LLM generates ConfigSpace alongside algorithm code
- Offloads numerical tuning from LLM to dedicated optimizer

### Evaluation Modes
- **Individual evaluation**: Each solution evaluated independently
- **Population evaluation**: Batch evaluation for expensive fitness functions

### Parallelization
- Uses `joblib.Parallel` for parallel solution evaluation
- Configurable workers and backend (loky, threading)
- Timeout handling for stuck evaluations

## Dependencies
- `numpy` - Numerical operations
- `ConfigSpace` - Hyperparameter configuration (when HPO enabled)
- `joblib` - Parallel execution
- Internal: `Solution`, `ExperimentLogger`, utility functions

## Data Flow
1. **Initialization Phase**: LLM generates n_parents initial algorithms
2. **Evolutionary Loop** (repeated until budget):
   - Sample offspring parents from population
   - Construct mutation prompts with feedback
   - LLM mutates each parent to create offspring
   - Evaluate offspring fitness (parallel or batch)
   - Select survivors (elitism or comma strategy)
   - Apply niching if enabled
   - Update best-so-far solution
3. **Output**: Return best solution found

## Configuration Parameters
- **Population**: `n_parents`, `n_offspring`, `elitism`
- **Budget**: `budget` (total individuals to generate)
- **Prompts**: `role_prompt`, `task_prompt`, `example_prompt`, `output_format_prompt`
- **Mutation**: `mutation_prompts`, `adaptive_mutation`, `adaptive_prompt`
- **Niching**: `niching` (None/"sharing"/"clearing"), `niche_radius`, `distance_metric`
- **Evaluation**: `eval_timeout`, `max_workers`, `parallel_backend`, `evaluate_population`
- **Logging**: `log`, `experiment_name`
- **HPO**: `HPO` flag
- **Diff Mode**: `diff_mode` flag

## Risks and Quirks
- **LLM failures**: May generate invalid/non-executable code → caught and logged as errors
- **Timeout handling**: Long-running evaluations killed after `eval_timeout`
- **Memory growth**: Large populations + long runs = many stored solutions
- **Parallel overhead**: Too many workers can cause slowdown
- **Niching sensitivity**: Niche radius greatly affects diversity vs convergence
- **Prompt engineering**: Results heavily depend on prompt quality
- **API costs**: Each LLM call costs money/tokens

## Performance Considerations
- Parallel evaluation speeds up expensive fitness functions
- Population evaluation mode reduces overhead for batch-friendly problems
- Diff mode reduces token usage for mutations
- Adaptive mutation controls exploration/exploitation balance
- Logging can be disabled for production speed

## Integration Points
- **LLM Interface**: Expects `llm.sample_solution()` and `llm.query()` methods
- **Evaluation Function**: `f(solution, logger) → solution` or `f(population, parents, logger) → (population, parents)`
- **Logger**: Logs conversations, code, and individual data to disk
- **Solution Class**: Container for code, fitness, metadata


