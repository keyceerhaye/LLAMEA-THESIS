# LLaMEA AI Context Summary

**Purpose:** High-density overview of LLaMEA thesis codebase optimized for LLM consumption during coding tasks.

## What This System Does

**LLaMEA** = Large Language Model Evolutionary Algorithm. Evolves Python algorithms by treating code as genome and using LLMs for generation/mutation. Core use case: automated discovery of optimization algorithms (tested on BBOB benchmarks).

**Key Innovation:** LLM generates code → Execute → Measure performance → Feed back to LLM → LLM mutates code → Repeat

## Critical Architecture Facts

### Core Components (5 files matter most)

```
llamea/llamea.py    - Main evolutionary engine (population, selection, evolution loop)
llamea/llm.py       - LLM provider abstraction (OpenAI/Gemini/Ollama/Custom)
llamea/solution.py  - Data structure for algorithm code + fitness + metadata
llamea/utils.py     - Utilities (diff patching, code distance, power law sampling)
llamea/loggers.py   - Experiment logging (JSONL conversations, code files, metrics)
```

### Data Flow

```
User → LLaMEA.run()
  ↓
Initialize: LLM generates n_parents algorithms
  ↓
Loop until budget exhausted:
  ├─ Select parents (random sampling)
  ├─ LLM.sample_solution() mutates each parent (parallel)
  ├─ Execute generated code via exec()
  ├─ Evaluate fitness (user-defined function)
  ├─ Select survivors (μ+λ elitism or μ,λ comma)
  ├─ [Optional] Apply niching (fitness sharing or clearing)
  └─ Log generation
  ↓
Return best solution
```

### Solution Object (Central Data Structure)

```python
Solution:
    id: UUID              # Unique identifier
    code: str             # Python source code
    name: str             # Class name
    fitness: float        # Performance score
    feedback: str         # Evaluation feedback for LLM
    error: str            # Error messages
    generation: int       # When created
    parent_ids: [UUID]    # Genealogy tracking
    configspace: ConfigSpace  # For HPO (optional)
```

## Execution Scripts (3 Tiers)

**main.py** - Legacy: Single algorithm (1+1), custom manager, hardcoded config
**main-thesis.py** - Enhanced: Single algorithm, CLI args, .env support, custom endpoints
**main-evolutionary.py** - Modern: True populations (μ+λ), full framework, all features

**Recommendation:** Edit main-evolutionary.py for new features. Others are legacy/reference.

## Key Design Patterns

### 1. Strategy Pattern Everywhere
- Niching: "sharing" | "clearing" | None
- Selection: elitism (μ+λ) | comma (μ,λ)
- Distance: AST-based | custom function

### 2. Dependency Injection
```python
LLaMEA(f=eval_fn, llm=llm_instance, logger=logger_instance)
# Not: LLaMEA() that creates its own dependencies
```

### 3. Composition Over Inheritance
```python
LLaMEA has-a LLM, has-a Logger
# Not: LLaMEA extends BaseEvolutionaryAlgorithm extends ...
```

## Common Editing Scenarios

### Adding New LLM Provider
1. Subclass `llamea/llm.py::LLM`
2. Implement `query(session_messages) -> str`
3. Handle provider-specific errors/retries
4. Export in `llamea/__init__.py`

### Adding New Evaluation Function
```python
def my_eval(solution: Solution, logger: ExperimentLogger) -> Solution:
    exec(solution.code, globals())
    algorithm = globals()[solution.name](budget)
    result = algorithm(my_problem)
    fitness = compute_fitness(result)
    solution.set_scores(fitness, feedback="...", error="")
    return solution
```

### Changing BBOB Evaluation
Edit: `utils.py` (root) functions: `aoc_logger`, `correct_aoc`, `OverBudgetException`
Used by: All main-*.py scripts

### Adding CLI Arguments
Edit: `main-thesis.py` or `main-evolutionary.py` argparse section
Pattern: `parser.add_argument('--name', type=X, default=Y, help="...")`

## Critical Invariants (Do NOT Break)

1. **Solution.code must be executable Python**: `exec(solution.code)` must not raise SyntaxError
2. **Fitness scoring required**: Always call `solution.set_scores(fitness, feedback, error)` after evaluation
3. **Parent IDs tracking**: `child.parent_ids = [parent.id]` for genealogy
4. **Generation increment**: `child.generation = parent.generation + 1`
5. **JSONL format**: One JSON object per line in logs (don't break parser)
6. **Budget semantics**: `budget` = total individuals to generate, not generations

## File Organization Rules

```
llamea/          - Core framework (never import from root or examples)
  ├─ No external dependencies except numpy, openai, gemini, ollama, joblib, ConfigSpace
  └─ Self-contained, installable package

root level/      - Execution scripts (can import from llamea/)
  ├─ managers.py   - Legacy helper for main.py/main-thesis.py
  └─ utils.py      - BBOB-specific utilities

examples/        - Standalone demos (can import from llamea/)
misc/            - Analysis tools (post-processing only)
logreader/       - Web app for log visualization
tests/           - Unit/integration tests
```

## Import Rules (Strict)

```python
# CORRECT: Lower level imports
from llamea import LLaMEA, OpenAI_LLM, Solution
from llamea.utils import code_distance, apply_unified_diff

# WRONG: Upper level imports (circular dependency)
# In llamea/llm.py: from managers import AlgorithmManager  ❌
# In llamea/llamea.py: from main import evaluate_bbob     ❌
```

## Configuration Patterns

### Framework Configuration (LLaMEA.__init__)
```python
LLaMEA(
    f=eval_fn,                    # Evaluation function
    llm=llm_instance,             # LLM provider
    n_parents=5,                  # Population size
    n_offspring=10,               # Offspring per generation
    budget=100,                   # Total individuals to generate
    niching="sharing",            # Diversity strategy
    niche_radius=0.3,             # Niche size
    HPO=False,                    # Enable hyperparameter optimization
    diff_mode=False,              # Generate patches instead of full code
    adaptive_mutation=False,      # Adaptive mutation strength
    adaptive_prompt=False,        # Co-evolve task prompt
    elitism=True,                 # μ+λ vs μ,λ strategy
    eval_timeout=3600,            # Per-evaluation timeout (seconds)
    max_workers=4,                # Parallel evaluators
    log=True,                     # Enable logging
    experiment_name="my_exp"      # Log directory suffix
)
```

### LLM Configuration
```python
OpenAI_LLM(api_key="...", model="gpt-4-turbo", temperature=0.8)
Gemini_LLM(api_key="...", model="gemini-2.0-flash")
Ollama_LLM(model="llama3.2")  # Local, no API key
```

## Error Handling Strategy

### Framework Catches Everything
```python
try:
    solution = llm.sample_solution(...)
    solution = evaluate_fitness(solution)
except Exception as e:
    solution.set_scores(worst_value, f"Error: {e}", repr(e))
    # Logs but continues
```

### User Evaluation Should Also Catch
```python
def my_eval(solution, logger):
    try:
        # Your evaluation code
        return solution
    except Exception as e:
        solution.set_scores(0.0, f"Eval failed: {e}", str(e))
        return solution
```

## Debugging Workflow

1. **Prompt issues**: Read `conversationlog.jsonl` in experiment directory
2. **Code generation**: Check `code/try-{N}-{AlgName}.py` files
3. **Fitness tracking**: Read `log.jsonl` (one solution per line)
4. **Live monitoring**: `python logreader/app.py --logfile path/to/conversationlog.jsonl`
5. **Performance analysis**: `try-{N}-aucs.txt` files contain raw scores

## Performance Characteristics

**Bottleneck:** LLM API calls (1-5 seconds each) - dominates all else
**Scaling:** O(budget) LLM calls, O(budget × eval_time) total runtime
**Memory:** O(budget) solutions stored in run_history
**Parallelization:** Only evaluation phase (LLM calls sequential within generation)

## Common Pitfalls

1. **Don't modify population during iteration**: Copy first if mutating
2. **Don't use `globals()` in eval without isolation**: Namespace pollution
3. **Don't hardcode API keys**: Use environment variables
4. **Don't forget to reset problem/logger between runs**: IOH requirement
5. **Don't exceed eval_budget**: Raises `OverBudgetException`
6. **Don't assume code is safe**: exec() has full Python permissions

## KISS Principle [[memory:2272829]]

**When in doubt, choose the simpler solution:**
- Simple function > Complex class hierarchy
- Direct implementation > Abstraction layer
- Explicit configuration > Magic defaults
- Copy-paste-adapt > Premature generalization

## Quick Reference: Where Logic Lives

| Concern | Location | Key Function/Class |
|---------|----------|-------------------|
| Evolution loop | llamea/llamea.py | `LLaMEA.run()` |
| LLM queries | llamea/llm.py | `LLM.query()`, `LLM.sample_solution()` |
| Code extraction | llamea/llm.py | `extract_algorithm_code()` |
| Diff patching | llamea/utils.py | `apply_unified_diff()` |
| Code distance | llamea/utils.py | `code_distance()` |
| Fitness sharing | llamea/llamea.py | `apply_niching()` |
| Selection | llamea/llamea.py | `selection()` |
| Logging | llamea/loggers.py | `ExperimentLogger` |
| BBOB eval | utils.py (root) | `aoc_logger`, `correct_aoc()` |
| Main execution | main-evolutionary.py | `main()` |

## Prompt Engineering (for LLM-Generated Code)

Framework constructs prompts with:
```
Role: "You are a computer scientist expert..."
Task: "Write an optimization algorithm for..."
Example: "class RandomSearch: ..."
Population Context: "Current algorithms: [{name: score}, ...]"
Feedback: "Algorithm X scored Y, improve it..."
Mutation: "Refine the strategy..." or "Change N% of code..."
Format: "Provide in format: # Description: ...\n# Code:\n```python\n...\n```"
```

**Output parsing expects:**
- Code in triple-backtick block (```python ... ```)
- Description line: `# Description: <text>`
- Class definition extractable via regex: `class\s*(\w+)\s*:`
- [If HPO] ConfigSpace as Python dict: `Space: {...}`

## Testing Approach

**Unit tests:** `tests/test_*.py` - Mock LLM with `Dummy_LLM`
**Integration:** Run with small budget (5-10) and Dummy_LLM
**Production:** Start with gpt-3.5-turbo (cheaper) before gpt-4

## Most Likely Edit Locations

1. **New evaluation benchmark**: Create new `eval_*.py`, adapt BBOB pattern
2. **New LLM provider**: Extend `llamea/llm.py`
3. **New niching strategy**: Add to `llamea/llamea.py::apply_niching()`
4. **CLI arguments**: Modify `main-thesis.py` or `main-evolutionary.py`
5. **Logging format**: Extend `llamea/loggers.py::ExperimentLogger`
6. **Prompt engineering**: Edit `role_prompt`, `task_prompt` in LLaMEA.__init__

## When to Read Full Documentation

- **Architecture**: `docs/architecture.md` - System design, data flow, scalability
- **Conventions**: `docs/rules.md` - Coding style, naming, patterns
- **Known issues**: `docs/bugs.md` - Bugs, limitations, workarounds
- **File details**: `docs/codebase/**/*.md` - Deep dives on specific files
- **Examples**: `docs/codebase/examples/README.md` - Usage patterns

## Emergency Quick Fixes

**Import error:** Check `llamea/__init__.py` exports, verify install: `pip install -e .`
**LLM timeout:** Increase `eval_timeout`, check network, verify API key
**Code won't execute:** Check `solution.error` field, read generated code file
**Out of memory:** Reduce `budget`, `n_parents`, `n_offspring`
**Slow performance:** Increase `max_workers`, reduce `eval_budget`, use faster model
**Bad algorithms:** Improve prompts (role/task/example), add more context

## Token Optimization for This Context

**Core files to load for editing:**
- Evaluation logic: `llamea/llamea.py` (~600 lines)
- LLM interface: `llamea/llm.py` (~540 lines)
- Execution script: `main-evolutionary.py` (~300 lines)

**Usually DON'T need:**
- `llamea/solution.py` - Simple data class
- `llamea/loggers.py` - Logging only
- `managers.py` - Legacy code
- `examples/` - Reference only
- `tests/` - Unless writing tests
- `misc/` - Post-processing only

**This summary = ~2K tokens. Full codebase ≈ 15K tokens. Read selectively.**

---

**TL;DR for AI:** This is an evolutionary algorithm that uses LLMs to generate/mutate Python code. Main loop: LLM generates → exec() → measure fitness → feedback to LLM → repeat. Edit `main-evolutionary.py` for experiments, `llamea/llamea.py` for algorithm logic, `llamea/llm.py` for LLM providers. Follow KISS principle. Most bugs are in prompt engineering, not code.


