# examples/ Directory

## Purpose
Collection of example scripts demonstrating various LLaMEA use cases and features.

## Overview
This directory provides working examples progressing from minimal implementations to complex scenarios including black-box optimization, hyperparameter optimization, and AutoML.

## File Structure

### `minimum_example.py`
**Minimal LLaMEA demonstration**
- Simplest possible usage
- Dummy evaluation function (random fitness)
- Uses Dummy_LLM for testing (no API costs)
- Single parent, single offspring
- Budget of 5 evaluations
- HPO enabled (for demonstration)
- **Purpose:** Understanding basic API without external dependencies
- **Runtime:** Seconds
- **API Cost:** $0 (uses Dummy_LLM)

### `simple_benchmark.py`
**Basic BBOB benchmark example** (not shown, inferred from pattern)
- Standard BBOB evaluation
- Simple configuration
- Good starting point for benchmarking

### `black-box-optimization.py`
**Standard LLaMEA on BBOB**
- Full BBOB test suite (24 functions)
- Real LLM integration (GPT-4 or GPT-3.5)
- (1+1) evolutionary strategy
- IOH experimenter logging
- AOCC evaluation metric
- **Purpose:** Benchmark algorithm discovery
- **Runtime:** Hours (depends on budget)
- **API Cost:** $$$ (100+ LLM calls)

### `black-box-opt-with-HPO.py`
**LLaMEA-HPO (with hyperparameter optimization)**
- BBOB evaluation
- SMAC3 integration for HPO
- LLM generates algorithm + ConfigSpace
- SMAC optimizes numerical parameters
- LLM focuses on structural improvements
- **Purpose:** Offload numerical tuning to dedicated optimizer
- **Runtime:** Hours to days
- **API Cost:** $$$$ (LLM + many SMAC evaluations)

### `automl_example.py`
**AutoML with LLaMEA**
- Generates machine learning pipelines
- Evaluates on breast cancer dataset (sklearn)
- Accuracy on hold-out test set
- Demonstrates LLaMEA beyond optimization algorithms
- **Purpose:** Show generality of framework
- **Runtime:** Minutes to hours
- **API Cost:** $$ (smaller budget typically)

## Complexity Progression

### Level 1: Understanding API
```
minimum_example.py
  ↓
Purpose: Learn LLaMEA API
Features: Dummy evaluation, Dummy LLM
Use case: Testing, debugging, learning
```

### Level 2: Real Benchmarks
```
simple_benchmark.py → black-box-optimization.py
  ↓
Purpose: Algorithm discovery
Features: BBOB, real LLM, IOH logging
Use case: Research, benchmarking
```

### Level 3: Advanced Features
```
black-box-opt-with-HPO.py
  ↓
Purpose: Hybrid LLM + HPO
Features: SMAC integration, ConfigSpace
Use case: State-of-the-art performance
```

### Level 4: Beyond Optimization
```
automl_example.py
  ↓
Purpose: General algorithm generation
Features: ML pipelines, sklearn
Use case: AutoML, meta-learning
```

## Common Patterns

### Basic LLaMEA Setup
```python
from llamea import LLaMEA, OpenAI_LLM

llm = OpenAI_LLM(api_key=api_key, model="gpt-4-turbo")

llamea = LLaMEA(
    f=evaluation_function,
    llm=llm,
    n_parents=5,
    n_offspring=5,
    budget=100,
    task_prompt="Your task description...",
    experiment_name="my_experiment"
)

best_solution = llamea.run()
```

### Evaluation Function Pattern
```python
def evaluate(solution, logger=None):
    # Extract code
    code = solution.code
    algorithm_name = solution.name
    
    # Execute code
    exec(code, globals())
    algorithm = globals()[algorithm_name](budget)
    
    # Run algorithm
    result = algorithm(problem)
    
    # Calculate fitness
    fitness = compute_fitness(result)
    
    # Set scores
    solution.set_scores(fitness, feedback="Performance: ...", error="")
    
    return solution
```

### BBOB Evaluation Pattern
```python
from ioh import get_problem, logger
from utils import aoc_logger, correct_aoc

def evaluate_bbob(solution, explogger=None):
    # Setup
    budget = 10000
    logger = aoc_logger(budget, upper=1e2)
    aucs = []
    
    # Run on all BBOB functions
    for fid in range(1, 25):
        for iid in [1, 2, 3]:
            problem = get_problem(fid, iid, 5)
            problem.attach_logger(logger)
            
            for rep in range(3):
                algorithm = AlgorithmClass(budget)
                algorithm(problem)
                
                auc = correct_aoc(problem, logger, budget)
                aucs.append(auc)
                
                logger.reset(problem)
                problem.reset()
    
    # Aggregate
    mean_auc = np.mean(aucs)
    solution.set_scores(mean_auc, f"AUC: {mean_auc:.4f}")
    
    return solution
```

## Feature Matrix

| Example | LLM | BBOB | HPO | AutoML | IOH | SMAC | Complexity |
|---------|-----|------|-----|--------|-----|------|------------|
| minimum_example.py | Dummy | ❌ | ✅* | ❌ | ❌ | ❌ | ⭐ |
| simple_benchmark.py | Real | ✅ | ❌ | ❌ | ✅ | ❌ | ⭐⭐ |
| black-box-optimization.py | Real | ✅ | ❌ | ❌ | ✅ | ❌ | ⭐⭐⭐ |
| black-box-opt-with-HPO.py | Real | ✅ | ✅ | ❌ | ✅ | ✅ | ⭐⭐⭐⭐ |
| automl_example.py | Real | ❌ | ❌ | ✅ | ❌ | ❌ | ⭐⭐⭐ |

*HPO enabled but not actually used (demonstration only)

## Dependencies by Example

### minimum_example.py
- `llamea` (LLaMEA, Dummy_LLM, OpenAI_LLM, Gemini_LLM)
- `numpy` (random fitness)
- `re` (class name extraction)
- **Optional:** OpenAI/Gemini API key (if not using Dummy)

### black-box-optimization.py
- `llamea` (full framework)
- `ioh` (BBOB functions, IOH logging)
- `numpy` (numerical operations)
- OpenAI/Gemini API key (required)

### black-box-opt-with-HPO.py
- `llamea` (with HPO support)
- `ioh` (BBOB functions)
- `smac` (SMAC3 optimizer)
- `ConfigSpace` (hyperparameter spaces)
- `numpy`
- OpenAI/Gemini API key (required)

### automl_example.py
- `llamea`
- `scikit-learn` (datasets, models, metrics)
- `numpy`, `pandas`
- OpenAI/Gemini API key (required)

## Usage Instructions

### Running minimum_example.py
```bash
# No API key needed
python examples/minimum_example.py

# Or with real LLM (edit file to uncomment)
export OPENAI_API_KEY="sk-..."
# Uncomment llm = OpenAI_LLM(...) in file
python examples/minimum_example.py
```

### Running black-box-optimization.py
```bash
export OPENAI_API_KEY="sk-..."
uv run python examples/black-box-optimization.py

# Expect: Long runtime, high API costs
# Output: exp-{date}-{model}-bbob/ directory
```

### Running black-box-opt-with-HPO.py
```bash
export OPENAI_API_KEY="sk-..."
uv run python examples/black-box-opt-with-HPO.py

# Expect: Very long runtime, very high API costs
# Output: Experiment directory with HPO results
```

### Running automl_example.py
```bash
export OPENAI_API_KEY="sk-..."
uv run python examples/automl_example.py

# Expect: Moderate runtime, moderate API costs
# Output: Experiment directory with ML pipelines
```

## Customization Points

### Changing LLM Model
```python
# In any example:
llm = OpenAI_LLM(api_key=api_key, model="gpt-3.5-turbo")  # Cheaper
llm = OpenAI_LLM(api_key=api_key, model="gpt-4o")         # Better
llm = Gemini_LLM(api_key=api_key, model="gemini-2.0-flash")
```

### Changing Budget
```python
llamea = LLaMEA(
    ...,
    budget=200,  # More iterations
    n_parents=10,  # Larger population
    n_offspring=20
)
```

### Changing Task Prompt
```python
task_prompt = """
Your custom task description here.
Specify:
- Input/output format
- Constraints
- Example code
- Evaluation criteria
"""
```

### Adding Niching
```python
llamea = LLaMEA(
    ...,
    niching="sharing",
    niche_radius=0.3,
    adaptive_niche_radius=True
)
```

### Adding Diff Mode
```python
llamea = LLaMEA(
    ...,
    diff_mode=True  # LLM generates patches instead of full code
)
```

## Expected Outputs

### minimum_example.py
```
Initializing first population
Started evolutionary loop, best so far: {fitness}
Generation 1, best so far: {fitness}
...
Solution(name='RandomSearch', fitness=0.xxx)
```

### black-box-optimization.py
```
exp-{date}_{time}-gpt-4-turbo-bbob/
├── conversationlog.jsonl
├── log.jsonl
├── code/
│   ├── try-0-Algorithm1.py
│   ├── try-1-Algorithm2.py
│   └── ...
└── try-{N}-aucs.txt
```

### automl_example.py
```
exp-{date}_{time}-{model}-automl/
├── conversationlog.jsonl
├── log.jsonl
└── code/
    └── try-{N}-MLPipeline.py  # Generated ML pipelines
```

## Learning Path

### For Beginners
1. Read `minimum_example.py` thoroughly
2. Modify task_prompt, see effect
3. Run with Dummy_LLM (free)
4. Switch to real LLM with small budget

### For Researchers
1. Start with `black-box-optimization.py`
2. Understand BBOB evaluation pipeline
3. Experiment with population sizes
4. Try `black-box-opt-with-HPO.py` for best results

### For ML Practitioners
1. Study `automl_example.py`
2. Adapt for your dataset
3. Modify task_prompt for your ML task
4. Integrate custom evaluation metrics

## Troubleshooting

### "No API key found"
```bash
export OPENAI_API_KEY="sk-..."
# or
export GEMINI_API_KEY="..."
```

### "Module not found: ioh"
```bash
uv sync --group examples
# or
pip install ioh
```

### "Module not found: smac"
```bash
uv sync --group examples
# or
pip install smac
```

### "Out of memory"
- Reduce budget
- Reduce eval_budget
- Reduce population sizes
- Use smaller model (gpt-3.5-turbo)

### "API rate limit exceeded"
- Wait for rate limit reset
- Use smaller budget
- Add delays between calls
- Upgrade API tier

## Best Practices

### Development
1. Test with Dummy_LLM first
2. Use small budgets during development
3. Verify evaluation function works standalone
4. Check log files for debugging

### Production
1. Use real LLM with appropriate model
2. Set reasonable budgets (100-200)
3. Enable logging for reproducibility
4. Archive experiment directories
5. Monitor API costs

### Research
1. Run multiple seeds for statistics
2. Compare with baselines
3. Document all hyperparameters
4. Use IOH experimenter for standardization
5. Archive code with git hash

## Performance Tips

### Reducing Runtime
- Smaller eval_budget for BBOB
- Fewer repetitions (1 instead of 3)
- Fewer BBOB instances (1 instead of 3)
- Parallel evaluation (use LLaMEA framework features)

### Reducing API Costs
- Use gpt-3.5-turbo instead of gpt-4
- Enable diff_mode
- Reduce budget
- Use Dummy_LLM for testing

### Improving Results
- Larger budgets
- Better prompts (more examples, clearer constraints)
- Enable niching for diversity
- Enable HPO for numerical parameters
- Larger populations

## Common Modifications

### Using Custom Dataset (AutoML)
```python
# In automl_example.py
from sklearn.datasets import load_your_dataset
X, y = load_your_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
```

### Using Custom Problem (Optimization)
```python
# Define your evaluation function
def my_problem(x):
    return objective_function(x)

# Adapt evaluation function
def evaluate(solution, logger=None):
    ...
    result = algorithm(my_problem)
    ...
```

### Saving Best Algorithm
```python
best = llamea.run()

# Save to file
with open("best_algorithm.py", "w") as f:
    f.write(best.code)

# Load and use later
exec(open("best_algorithm.py").read())
algorithm = BestAlgorithm(budget=10000)
```

## Extension Examples

### Multi-Objective Optimization
```python
def evaluate_multi_objective(solution, logger=None):
    ...
    fitness_vector = [obj1, obj2, obj3]
    solution.set_scores(fitness_vector, ...)
    return solution

# Would require modifying selection logic
```

### Constraint Handling
```python
def evaluate_with_constraints(solution, logger=None):
    ...
    violations = check_constraints(result)
    if violations > 0:
        fitness = penalty(fitness, violations)
    ...
```

### Transfer Learning
```python
# Use best algorithm from one problem as starting point
best_from_problem_A = ...

task_prompt = f"""
Based on this successful algorithm:
{best_from_problem_A.code}

Adapt it for this new problem:
{problem_B_description}
"""
```

## References
- [BBOB Documentation](https://coco.gforge.inria.fr/downloads/download16.00/bbobdocfunctions.pdf)
- [IOH Experimenter](https://iohprofiler.github.io/)
- [SMAC3 Documentation](https://automl.github.io/SMAC3/)
- [LLaMEA Paper](https://ieeexplore.ieee.org/document/10752628)


