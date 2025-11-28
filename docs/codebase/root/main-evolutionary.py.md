# main-evolutionary.py

## Purpose
Production implementation using the full LLaMEA framework with true population-based evolutionary optimization on BBOB benchmarks.

## Role in System
Complete, feature-rich script demonstrating:
- Full LLaMEA framework usage
- Population-based evolution (μ+λ or μ,λ strategies)
- BBOB benchmark evaluation
- CLI configuration
- Batch population evaluation

## Key Features

### Full LLaMEA Integration
Uses official `LLaMEA` class from framework:
- Population management
- Parallel evaluation
- Niching (if configured)
- HPO support (if configured)
- Diff mode support
- Adaptive prompts/mutations

### Population-Based Evolution
True evolutionary algorithm with:
- Configurable parent population size
- Configurable offspring count per generation
- Elitism support (μ+λ strategy)
- Generational evolution

### Command-Line Interface
```
--api-key          API authentication
--base-url         Custom API endpoint
--max-tokens       Response token limit (default: 8192)
--model            LLM model (default: gemini-2.0-flash)
--experiment-name  Experiment identifier
--n-parents        Parent count (default: 4)
--n-offspring      Offspring count (default: 16)
--api-budget       Total API calls (default: 100)
--generations      Fixed generation count
--elitism          Enable elitism (default: True)
--eval-budget      Evaluation budget per alg (default: 10000)
```

### Population Evaluation Mode
Uses `evaluate_population=True` for batch evaluation:
- All offspring evaluated together
- Can share computation across algorithms
- More efficient for expensive fitness functions

## Architecture

### Evaluation Functions

#### `evaluate_single_solution(solution, explogger, eval_budget)`
Evaluates one algorithm on full BBOB suite:
- Extracts code and class name
- Executes algorithm
- Runs on 24 functions × 3 instances × 3 reps = 216 runs
- Returns (feedback, fitness, error)

**Error Handling:**
- Class not found → descriptive error
- Execution exception → captured and returned
- Over budget → caught and ignored

#### `bbob_population_evaluation_function(population, parents, explogger, eval_budget)`
Batch evaluates entire population:
- Loops through offspring population
- Calls `evaluate_single_solution()` for each
- Sets scores on Solution objects
- Returns (evaluated_population, parents)

**Signature:**
Required by `LLaMEA` when `evaluate_population=True`:
```python
f(population, parents=None, logger=None) -> (population, parents)
```

### LLM Setup

#### Custom Max Tokens Support
Monkey-patches `llm.query()` to inject `max_tokens`:
```python
def query_with_max_tokens(session_messages, ...):
    response = llm.client.chat.completions.create(
        ...,
        max_tokens=args.max_tokens
    )
```

#### Base URL Support
Updates client after LLM initialization:
```python
if base_url:
    llm.base_url = base_url
    llm._client_kwargs["base_url"] = base_url
    llm.client = openai.OpenAI(**llm._client_kwargs)
```

### Budget Calculation

#### Fixed Generations Mode
```python
--generations N
→ estimated_calls = n_parents + (N * n_offspring)
→ total_budget = estimated_calls
```

#### Budget-Driven Mode
```python
max_generations = (api_budget - n_parents) // n_offspring
actual_calls = n_parents + (max_generations * n_offspring)
total_budget = actual_calls
```

**LLaMEA's `budget` parameter:**
Total number of individuals to generate (not generations).

## Task Prompt
Customized for BBOB evaluation:
```
- Optimize BBOB 24 noiseless functions
- Dimension: 5 (fixed)
- Bounds: [-5.0, 5.0]
- Signature: __init__(self, budget), __call__(self, func)
- Budget: self.budget evaluations
```

## Data Flow
```
CLI args + .env → Configuration
  ↓
OpenAI_LLM(api_key, model)
  ↓
[if base_url] → Update client
[if max_tokens] → Patch query method
  ↓
evaluation_wrapper(eval_budget)
  ↓
LLaMEA(
  f=evaluation_wrapper,
  llm=llm,
  n_parents=4,
  n_offspring=16,
  budget=total_budget,
  evaluate_population=True
)
  ↓
llamea.run() → Evolutionary loop
  ↓
Best solution returned
```

## Comparison with Other Main Scripts

| Feature | main.py | main-thesis.py | main-evolutionary.py |
|---------|---------|----------------|----------------------|
| Framework | Custom | Custom | LLaMEA |
| Population | No (1+1) | No (1+1) | Yes (μ+λ) |
| CLI args | No | Yes | Yes |
| Parallelization | No | No | Yes (via LLaMEA) |
| Niching | No | No | Optional |
| HPO | No | No | Optional |
| Diff mode | No | No | Optional |
| Recommended | ❌ | ⚠️ | ✅ |

## Configuration Examples

### Basic Run
```bash
python main-evolutionary.py \
  --model gemini-2.0-flash \
  --api-budget 100
```

### Large Population
```bash
python main-evolutionary.py \
  --n-parents 10 \
  --n-offspring 30 \
  --api-budget 300
```

### Fixed Generations
```bash
python main-evolutionary.py \
  --n-parents 5 \
  --n-offspring 10 \
  --generations 20
# Will use 5 + 20*10 = 205 API calls
```

### Custom Provider
```bash
python main-evolutionary.py \
  --base-url https://api.aimlapi.com/v1 \
  --api-key your_key \
  --max-tokens 4096
```

## Dependencies
- `os`, `argparse` - Configuration
- `numpy` - Numerical operations
- `ioh` - BBOB benchmarks
- `openai` - API client (for patching)
- `time` - Retry delays
- `llamea` - LLaMEA framework
- `llamea.llm` - OpenAI_LLM
- `utils` - BBOB utilities
- `dotenv` (optional) - .env loading

## Output
- Experiment directory: `LLaMEA-{model}-{experiment_name}/`
- Structure follows `ExperimentLogger` from framework:
  - `conversationlog.jsonl` - Full LLM conversations
  - `log.jsonl` - Individual metadata per solution
  - `code/` - Generated algorithm files
  - `configspace/` - ConfigSpace definitions (if HPO)
  - Performance files

## Advantages over main-thesis.py
1. **True populations**: Multiple algorithms compete
2. **Parallelization**: Faster evaluation
3. **Niching**: Can be enabled for diversity
4. **HPO**: Can be enabled for parameter tuning
5. **Diff mode**: Token-efficient mutations
6. **Framework features**: Benefits from all LLaMEA improvements
7. **Less code**: Delegates to framework

## Limitations
- More complex setup (LLaMEA configuration)
- Higher API costs (more calls per generation)
- Requires understanding LLaMEA parameters
- Harder to debug (framework abstracts details)

## Performance Considerations
- **Parallel evaluation**: `max_workers=2` (conservative, can increase)
- **BBOB is slow**: 24 functions × 3 instances × 3 reps = 216 runs
- **Population size**: Larger = more API calls but better exploration
- **Generations**: More = longer runtime but better convergence

## Error Handling
- Missing API key → ValueError with instructions
- Algorithm execution errors → Caught, logged, fitness=0.0
- Class not found → Descriptive error message
- Over budget → OverBudgetException caught

## Best Practices

### For Quick Testing
```bash
python main-evolutionary.py \
  --n-parents 2 \
  --n-offspring 4 \
  --api-budget 10 \
  --eval-budget 1000
```

### For Production
```bash
python main-evolutionary.py \
  --n-parents 10 \
  --n-offspring 30 \
  --api-budget 300 \
  --eval-budget 10000 \
  --elitism \
  --experiment-name production_run_001
```

### For Research
Enable full LLaMEA features by modifying `LLaMEA()` call:
```python
llamea = LLaMEA(
    ...,
    niching="sharing",
    niche_radius=0.3,
    HPO=True,
    diff_mode=True,
    adaptive_mutation=True,
    adaptive_prompt=True
)
```

## Integration Points
- **LLaMEA framework**: Primary dependency
- **BBOB utilities**: From `utils.py`
- **OpenAI_LLM**: From `llamea.llm`
- **Environment config**: From .env files

## Future Enhancements
- Add niching flags to CLI
- Add HPO flag to CLI
- Add diff_mode flag to CLI
- Add adaptive flags to CLI
- Support for other benchmarks
- Multi-objective optimization
- Constraint handling

## When to Use
- ✅ Production evolutionary algorithm discovery
- ✅ Research experiments with populations
- ✅ Comparing evolutionary strategies
- ✅ Leveraging full LLaMEA capabilities
- ✅ Scalable experiments

## When NOT to Use
- ❌ Quick single-algorithm tests (use main-thesis.py)
- ❌ Non-BBOB problems (requires evaluation function changes)
- ❌ Very limited API budget (populations expensive)







