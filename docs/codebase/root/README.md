# Root-Level Files

## Purpose
Entry point scripts and support modules for running LLaMEA experiments on BBOB benchmarks.

## Overview
This directory contains executable scripts for different experiment configurations and supporting utility modules. Represents both legacy implementations and modern framework usage.

## File Structure

### Main Execution Scripts

#### `main.py`
**Legacy simple script**
- Single-algorithm (1+1) evolution
- Hardcoded configuration
- Custom AlgorithmManager
- No CLI arguments
- **Status:** Superseded by main-thesis.py

#### `main-thesis.py`
**Enhanced thesis script with CLI**
- Single-algorithm evolution
- Full CLI argument support
- Environment variable support (.env)
- Custom API endpoints
- Detailed feedback options
- **Status:** Recommended for simple experiments

#### `main-evolutionary.py`
**Production framework-based script**
- True population evolution (μ+λ)
- Full LLaMEA framework integration
- Population-based evaluation
- All framework features available
- **Status:** Recommended for production

### Supporting Modules

#### `managers.py`
**Custom manager classes for legacy scripts**
- `AlgorithmManager` - LLM interaction wrapper
- `ExperimentLogger` - Simple experiment logging
- Prompt engineering
- Elitism tracking
- Detailed feedback
- **Used by:** main.py, main-thesis.py

#### `utils.py`
**BBOB-specific utilities**
- `aoc_logger` - Area Over Convergence tracking
- `budget_logger` - Budget enforcement
- `correct_aoc()` - Early-stop correction
- `OverBudgetException` - Budget violation
- `NoCodeException` - Code extraction failure
- **Used by:** All main scripts

## Script Comparison Matrix

| Feature | main.py | main-thesis.py | main-evolutionary.py |
|---------|---------|----------------|----------------------|
| **Evolution Style** | (1+1) single | (1+1) single | (μ+λ) population |
| **Framework** | Custom | Custom | LLaMEA |
| **CLI Args** | ❌ | ✅ | ✅ |
| **.env Support** | ❌ | ✅ | ✅ |
| **Custom Endpoints** | ❌ | ✅ | ✅ |
| **Parallelization** | ❌ | ❌ | ✅ |
| **Niching** | ❌ | ❌ | Optional |
| **HPO** | ❌ | ❌ | Optional |
| **Diff Mode** | ❌ | ❌ | Optional |
| **Logging** | Plain text | Plain text | JSONL |
| **Complexity** | Low | Medium | High |
| **Recommended** | ❌ | For simple | ✅ Production |

## Workflow Comparison

### main.py / main-thesis.py Flow
```
Start
  ↓
AlgorithmManager initialized
  ↓
For i in range(budget):
  ↓
  [i==0]: fetch_algorithm()
  [i>0]: refine_algorithm(feedback)
  ↓
  Extract code
  ↓
  exec(code) → Load class
  ↓
  For each BBOB function:
    ↓
    algorithm(problem) → Evaluate
    ↓
    Calculate AUC
  ↓
  Aggregate AUCs
  ↓
  Log results
  ↓
Next iteration
```

### main-evolutionary.py Flow
```
Start
  ↓
LLaMEA initialized with population
  ↓
llamea.run():
  ↓
  Initialize population (n_parents)
  ↓
  For generation in range(max_generations):
    ↓
    Select parents (random sampling)
    ↓
    Generate offspring (parallel)
    ↓
    Evaluate population (batch)
    ↓
    Select survivors (elitism or comma)
    ↓
    [Optional] Apply niching
    ↓
    Log generation
  ↓
Return best solution
```

## Configuration Approaches

### main.py (Hardcoded)
```python
api_key = os.getenv("OPENAI_API_KEY")
ai_model = "gpt-4-turbo"
openai_budget = 100
# No flexibility
```

### main-thesis.py (CLI + .env)
```bash
# .env file
OPENAI_API_KEY=sk-...
BASE_URL=https://api.aimlapi.com/v1

# CLI
python main-thesis.py \
  --model gemini-2.0-flash \
  --budget 100 \
  --elitism \
  --detailed-feedback
```

### main-evolutionary.py (CLI + Framework)
```bash
python main-evolutionary.py \
  --model gemini-2.0-flash \
  --n-parents 5 \
  --n-offspring 10 \
  --api-budget 150 \
  --elitism
```

## Module Dependencies

### managers.py Dependencies
```
managers.py
  ├── openai (API client)
  ├── re (code extraction)
  ├── datetime (timestamps)
  └── llamea.utils (NoCodeException)
```

### utils.py Dependencies
```
utils.py
  ├── numpy (numerical operations)
  ├── ioh.logger (AbstractLogger)
  └── ioh.LogInfo (evaluation state)
```

### main-*.py Dependencies
```
main scripts
  ├── managers (AlgorithmManager, ExperimentLogger)
  ├── utils (aoc_logger, correct_aoc, exceptions)
  ├── ioh (BBOB benchmarks)
  ├── numpy (numerical operations)
  └── [main-evolutionary only] llamea framework
```

## Common Usage Patterns

### Quick Test
```bash
# Simplest (hardcoded)
python main.py

# With configuration
python main-thesis.py --budget 5 --eval-budget 1000

# With populations
python main-evolutionary.py --n-parents 2 --n-offspring 4 --api-budget 10
```

### Production Run
```bash
python main-evolutionary.py \
  --model gpt-4-turbo \
  --n-parents 10 \
  --n-offspring 30 \
  --api-budget 300 \
  --eval-budget 10000 \
  --elitism \
  --experiment-name production_001
```

### Custom Provider
```bash
python main-thesis.py \
  --base-url https://custom-api.com/v1 \
  --api-key custom_key \
  --model custom-model \
  --max-tokens 4096
```

## Output Structure

### main.py / main-thesis.py
```
exp-{date}_{time}-{model}-{name}/
├── conversationlog.txt          # Plain text conversations
├── ioh/                          # IOH data (if enabled)
├── code/
│   └── try-{N}-{AlgName}.py    # Generated algorithms
└── try-{N}-aucs.txt             # Performance data
```

### main-evolutionary.py
```
LLaMEA-{model}-{name}/
├── conversationlog.jsonl        # JSONL conversations
├── log.jsonl                     # Individual metadata
├── code/
│   └── try-{N}-{AlgName}.py
├── configspace/                  # ConfigSpace (if HPO)
│   └── try-{N}-{AlgName}.py
└── {UUID}-aucs.txt              # Performance by solution ID
```

## Evaluation Pipeline (Common)

### BBOB Benchmark Setup
- **Functions:** 24 noiseless BBOB functions
- **Instances:** 3 per function (1, 2, 3)
- **Dimension:** 5 (fixed)
- **Bounds:** [-5.0, 5.0]
- **Repetitions:** 3 per instance
- **Total runs:** 24 × 3 × 3 = 216 per algorithm
- **Metric:** Area Under Convergence (AUC), higher is better

### Evaluation Code (Typical)
```python
budget = 10000
logger = aoc_logger(budget, upper=1e2)
aucs = []

for fid in range(1, 25):  # Functions
    for iid in [1, 2, 3]:  # Instances
        problem = get_problem(fid, iid, 5)
        problem.attach_logger(logger)
        
        for rep in range(3):  # Repetitions
            np.random.seed(rep)
            
            try:
                algorithm = AlgorithmClass(budget)
                algorithm(problem)
            except OverBudgetException:
                pass
            
            auc = correct_aoc(problem, logger, budget)
            aucs.append(auc)
            
            logger.reset(problem)
            problem.reset()

mean_auc = np.mean(aucs)
std_auc = np.std(aucs)
```

## Error Handling Strategy

### Common Errors
1. **No code extracted** → NoCodeException → fitness = 0.0
2. **Syntax error in code** → exec() fails → fitness = 0.0
3. **Algorithm timeout** → Caught by evaluation wrapper → fitness = 0.0
4. **Over budget** → OverBudgetException → Caught, continue with partial results
5. **API errors** → Propagate (let user handle)

### Error Recovery
- All scripts continue after individual evaluation failures
- Error messages logged
- Algorithms with errors get fitness = 0.0
- Next iteration provides error feedback to LLM

## Performance Considerations

### Bottlenecks
1. **LLM API calls:** Seconds per query (dominant cost)
2. **BBOB evaluation:** 216 runs × budget per algorithm
3. **Code execution:** exec() overhead
4. **Disk I/O:** Logging (minimal)

### Optimization Strategies
- **Parallelization:** Only in main-evolutionary.py (framework feature)
- **Reduce eval_budget:** Faster but noisier estimates
- **Fewer BBOB instances/reps:** Less robust evaluation
- **Diff mode:** Reduce token usage (framework feature)
- **Population evaluation:** Batch processing (framework feature)

## Integration Points

### With LLaMEA Framework
- **main-evolutionary.py:** Full integration
- **main.py, main-thesis.py:** External (use managers.py)

### With IOH Experimenter
- All scripts use IOH for BBOB benchmarks
- Custom loggers implement AbstractLogger interface

### With LLM Providers
- OpenAI API (default)
- Custom endpoints via base_url
- Gemini via API key
- Ollama (local) - not used in these scripts

## Best Practices

### For Beginners
1. Start with `main-thesis.py` for simplicity
2. Use small budgets for testing (--budget 5)
3. Test with Dummy_LLM before spending API credits
4. Monitor conversationlog to debug prompts

### For Researchers
1. Use `main-evolutionary.py` for publications
2. Enable detailed logging
3. Run multiple seeds for statistical significance
4. Archive experiment directories with git hash

### For Developers
1. Extend `main-evolutionary.py` for new features
2. Use LLaMEA framework features (niching, HPO, diff mode)
3. Add CLI flags for new parameters
4. Keep evaluation pipeline consistent with IOH standards

## Migration Guide

### From main.py to main-thesis.py
1. Replace hardcoded values with CLI args
2. Create .env file for secrets
3. No code changes to core logic

### From main-thesis.py to main-evolutionary.py
1. Replace AlgorithmManager with LLaMEA
2. Replace manual loop with llamea.run()
3. Adapt evaluation function signature
4. Configure population sizes
5. Benefits: parallelization, niching, HPO, diff mode

### Code Comparison
```python
# main-thesis.py
manager = AlgorithmManager(...)
for i in range(budget):
    code = manager.fetch_algorithm() if i==0 else manager.refine_algorithm(...)
    # evaluate...

# main-evolutionary.py
llm = OpenAI_LLM(...)
llamea = LLaMEA(f=eval_fn, llm=llm, n_parents=5, n_offspring=10, budget=150)
best = llamea.run()
```

## Future Work
- Merge scripts into single CLI with modes
- Add more benchmarks (TSP, AutoML, etc.)
- Add multi-objective support
- Add constraint handling
- Add coevolution support
- Improve error recovery strategies







