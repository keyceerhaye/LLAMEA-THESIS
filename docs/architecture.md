# LLaMEA Architecture Design Document

## System Overview

**LLaMEA** (Large Language Model Evolutionary Algorithm) is a framework that combines evolutionary computation with large language models for automated algorithm discovery and optimization. The system treats algorithm code as the evolving genome, using LLMs to generate and mutate Python implementations.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  main-*.py   │  │  examples/   │  │  Custom      │     │
│  │  Scripts     │  │  Scripts     │  │  Scripts     │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
┌────────────────────────────┴─────────────────────────────────┐
│                     LLaMEA Framework                          │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              LLaMEA Core (llamea.py)                  │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │  │
│  │  │ Initialize │→ │  Evolve    │→ │  Select    │     │  │
│  │  │ Population │  │  Solutions │  │  Survivors │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘     │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                                 │
│         ┌───────────────────┼───────────────────┐            │
│         │                   │                   │            │
│  ┌──────▼──────┐     ┌──────▼──────┐    ┌──────▼──────┐   │
│  │LLM Layer    │     │Solution     │    │Logger       │   │
│  │(llm.py)     │     │(solution.py)│    │(loggers.py) │   │
│  └──────┬──────┘     └─────────────┘    └─────────────┘   │
│         │                                                    │
└─────────┼────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────┐
│                    External Services                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ OpenAI   │  │ Gemini   │  │  Ollama  │  │  Custom  │   │
│  │   API    │  │   API    │  │  Local   │  │   LLMs   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────────────────────────────────────────┘
          │
┌─────────▼────────────────────────────────────────────────────┐
│                   Evaluation Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  BBOB    │  │  IOH     │  │  Custom  │  │  AutoML  │   │
│  │Benchmark │  │Experiment│  │ Problems │  │   Tasks  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Core Components

#### LLaMEA Engine (`llamea.py`)
**Responsibilities:**
- Population initialization and management
- Evolutionary loop orchestration
- Parent selection and survival selection
- Niching strategy application
- Parallel evaluation coordination
- Best solution tracking

**Key Interfaces:**
```python
class LLaMEA:
    def __init__(f, llm, n_parents, n_offspring, budget, ...)
    def run() -> Solution
    def initialize() -> List[Solution]
    def evolve_solution(individual) -> Solution
    def selection(parents, offspring) -> List[Solution]
```

#### LLM Abstraction Layer (`llm.py`)
**Responsibilities:**
- Provider-agnostic LLM interface
- Code extraction from responses
- Retry logic and rate limit handling
- Diff mode support
- Conversation logging

**Key Interfaces:**
```python
class LLM(ABC):
    @abstractmethod
    def query(session: list) -> str
    def sample_solution(...) -> Solution
    def extract_algorithm_code(message) -> str
```

#### Solution Representation (`solution.py`)
**Responsibilities:**
- Algorithm code storage
- Fitness and feedback tracking
- Genealogy management
- Metadata storage
- Serialization

**Key Interfaces:**
```python
class Solution:
    code: str
    fitness: float
    feedback: str
    parent_ids: List[str]
    def set_scores(fitness, feedback, error)
    def copy() -> Solution
```

#### Experiment Logger (`loggers.py`)
**Responsibilities:**
- Experiment directory management
- Conversation logging (JSONL)
- Code file persistence
- Performance data storage
- Reproducibility tracking

**Key Interfaces:**
```python
class ExperimentLogger:
    def log_conversation(role, content)
    def log_population(population)
    def log_individual(individual)
    def log_code(attempt, name, code)
```

### 2. Data Flow

#### Initialization Phase
```
User → LLaMEA.__init__() → Configure parameters
                         ↓
                   LLaMEA.initialize()
                         ↓
        ┌────────────────┴────────────────┐
        │  For i in range(n_parents):     │
        │    Construct initial prompt      │
        │           ↓                      │
        │    LLM.sample_solution()         │
        │           ↓                      │
        │    Execute and evaluate          │
        │           ↓                      │
        │    Create Solution object        │
        └────────────────┬────────────────┘
                         ↓
                   Population ready
```

#### Evolution Loop
```
While budget not exhausted:
    │
    ├─ Select parents (random sampling)
    │       ↓
    ├─ For each parent:
    │   │
    │   ├─ Construct mutation prompt (with feedback)
    │   │       ↓
    │   ├─ LLM.sample_solution() [parallel]
    │   │       ↓
    │   ├─ Execute generated code
    │   │       ↓
    │   └─ Evaluate fitness
    │
    ├─ [Optional] Batch evaluation
    │       ↓
    ├─ Selection (elitism or comma)
    │       ↓
    ├─ [Optional] Apply niching
    │       ↓
    ├─ Update best solution
    │       ↓
    └─ Log generation
```

#### Evaluation Flow
```
Solution.code → exec() → Algorithm class
                              ↓
                        algorithm(problem)
                              ↓
                        ┌─────┴─────┐
                        │  Problem  │
                        │ Evaluates │
                        └─────┬─────┘
                              ↓
                        Performance metrics
                              ↓
                        Solution.set_scores()
```

### 3. State Management

#### Population State
```python
LLaMEA:
    population: List[Solution]      # Current generation
    run_history: List[Solution]     # All evaluations
    best_so_far: Solution           # Best found
    generation: int                 # Current generation number
```

#### Solution State
```python
Solution:
    id: UUID                        # Unique identifier
    code: str                       # Algorithm implementation
    fitness: float                  # Performance score
    generation: int                 # When created
    parent_ids: List[UUID]          # Genealogy
    feedback: str                   # Evaluation feedback
    error: str                      # Error messages
```

#### Logging State
```python
ExperimentLogger:
    dirname: str                    # Experiment directory
    attempt: int                    # Evaluation counter
    # Files: conversationlog.jsonl, log.jsonl, code/, configspace/
```

### 4. External Interfaces

#### LLM Provider Interface
```
LLaMEA → LLM wrapper → Provider SDK → API
                                        ↓
OpenAI API / Gemini API / Ollama / Custom
```

**Data Format:**
- Input: List of message dicts `[{"role": str, "content": str}, ...]`
- Output: String response containing code and metadata

#### Evaluation Interface
```python
# Single evaluation mode
def f(solution: Solution, logger: ExperimentLogger) -> Solution:
    # Execute, measure, set scores
    return solution

# Population evaluation mode
def f(population: List[Solution], 
      parents: List[Solution], 
      logger: ExperimentLogger) -> Tuple[List[Solution], List[Solution]]:
    # Batch evaluation
    return evaluated_population, updated_parents
```

#### Problem Interface
```
BBOB (IOH Experimenter):
    problem = get_problem(fid, iid, dim)
    problem(x) → fitness
    problem.reset()

Custom:
    def my_problem(x):
        return objective(x)
```

### 5. Build and Deployment Model

#### Package Structure
```
llamea/
├── pyproject.toml          # Project metadata, dependencies
├── uv.lock                 # Locked dependencies
├── llamea/                 # Core package
│   ├── __init__.py
│   ├── llamea.py
│   ├── llm.py
│   ├── solution.py
│   ├── loggers.py
│   └── utils.py
├── examples/               # Usage examples
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

#### Installation Modes
```bash
# PyPI installation (end users)
pip install llamea

# Development installation (contributors)
git clone https://github.com/XAI-liacs/LLaMEA.git
cd LLaMEA
uv sync --dev --group examples

# With extras
pip install llamea[llm-extras]  # For local LLM support
```

#### Dependency Management
- **Core**: numpy, openai, google-generativeai, ollama, joblib, ConfigSpace, jsonlines, ioh
- **Dev**: pytest, black, isort, pytest-cov, pytest-mock
- **Examples**: ioh, scikit-learn, seaborn, smac, levenshtein, flask, networkx, lizard

### 6. Scalability and Performance

#### Parallelization Strategy
```
LLaMEA uses joblib.Parallel for evaluation:

Generation N:
    ├─ Worker 1: Evaluate solution 1
    ├─ Worker 2: Evaluate solution 2
    ├─ Worker 3: Evaluate solution 3
    └─ Worker 4: Evaluate solution 4
         ↓
    Sync: All complete
         ↓
    Selection and logging
```

**Configuration:**
- `max_workers`: Number of parallel evaluators
- `parallel_backend`: "loky" (default), "threading", "multiprocessing"
- `eval_timeout`: Per-evaluation timeout

#### Performance Characteristics
```
Bottlenecks:
1. LLM API calls (seconds per query) - DOMINANT
2. Algorithm evaluation (depends on problem)
3. Code execution (exec overhead)
4. Disk I/O (logging)

Scaling factors:
- Population size: O(n_parents + n_offspring) per generation
- Budget: O(budget) total LLM calls
- Problem difficulty: O(eval_budget) per evaluation
```

#### Memory Management
```
Memory growth:
- run_history: O(budget) solutions stored
- Logging: O(budget) files on disk
- LLM context: O(generation) conversation history

Mitigation:
- Periodic garbage collection
- Compress old logs
- Limit conversation context
- Stream logs to disk
```

### 7. State Management Approaches

#### Stateless Components
- `Solution`: Pure data object
- `utils` functions: Pure functions
- LLM providers: Stateless API wrappers

#### Stateful Components
- `LLaMEA`: Maintains population and history
- `ExperimentLogger`: File system state
- LLM API clients: Connection state

#### Persistence Strategy
```
Runtime State:
    LLaMEA instance → Population, history

Disk Persistence:
    conversationlog.jsonl → All LLM interactions
    log.jsonl → All solution metadata
    code/ → All generated code
    {attempt}-aucs.txt → Performance data

Reproducibility:
    Random seeds + Conversation log → Fully reproducible
```

### 8. Key Design Patterns

#### Strategy Pattern
- Niching strategies: fitness sharing, clearing, none
- Selection strategies: elitism (μ+λ), comma (μ,λ)
- Distance metrics: AST-based, custom

#### Template Method Pattern
```python
LLaMEA.run():
    initialize()
    while not done:
        select_parents()
        mutate()      # Delegates to LLM
        evaluate()    # Delegates to user function
        select_survivors()
    return best
```

#### Factory Pattern
```python
def create_llm(provider, **kwargs):
    if provider == "openai":
        return OpenAI_LLM(**kwargs)
    elif provider == "gemini":
        return Gemini_LLM(**kwargs)
    # ...
```

#### Observer Pattern
- `ExperimentLogger` observes evolutionary events
- Logs conversations, generations, evaluations

### 9. Security Considerations

#### Code Execution Risk
```
exec(generated_code) poses risks:
- Arbitrary code execution
- File system access
- Network access
- System calls

Mitigation:
- Timeout enforcement
- Sandboxing (recommended but not implemented)
- Trust in LLM providers (rely on their safety measures)
```

#### API Key Management
```
Best practices:
- Environment variables (not hardcoded)
- .env files (gitignored)
- Key rotation
- Rate limiting awareness
```

#### Data Privacy
```
Conversation logs may contain:
- Problem-specific data
- Proprietary algorithms
- Performance metrics

Recommendation:
- Review logs before sharing
- Sanitize sensitive information
- Use private repositories
```

### 10. Known Architectural Limitations

#### Single-Machine Constraint
- No distributed evaluation
- Limited by single machine resources
- Workaround: Increase `max_workers`, use powerful machine

#### LLM API Dependency
- Requires internet connection (except Ollama)
- Subject to API rate limits
- Costs scale with usage
- Workaround: Cache responses, use local models

#### Code Execution Safety
- exec() inherently unsafe
- No sandboxing implemented
- Trust-based security model
- Workaround: Manual code review, restricted environments

#### Memory Accumulation
- run_history grows linearly with budget
- No automatic cleanup
- Can exhaust memory on very long runs
- Workaround: Periodic checkpointing and restart

#### Context Window Limits
- LLM context limited (varies by model)
- Long experiments may exceed context
- Conversation history grows unbounded
- Workaround: Truncate history, use diff mode

### 11. Extension Points

#### Custom LLM Providers
```python
class MyCustomLLM(LLM):
    def query(self, session):
        # Custom API logic
        return response
```

#### Custom Distance Metrics
```python
def my_distance(solution_a, solution_b):
    # Custom similarity metric
    return distance
```

#### Custom Evaluation
```python
def my_evaluation(solution, logger):
    # Custom fitness function
    solution.set_scores(fitness, feedback, error)
    return solution
```

#### Custom Logging
```python
class MyLogger(ExperimentLogger):
    def log_custom_data(self, data):
        # Custom logging
        pass
```

## Conclusion

LLaMEA's architecture prioritizes:
1. **Modularity**: Clear separation of concerns
2. **Extensibility**: Easy to add providers, metrics, problems
3. **Simplicity**: KISS principle throughout [[memory:2272829]]
4. **Reproducibility**: Comprehensive logging
5. **Flexibility**: Works with any LLM provider and problem type

The system successfully bridges evolutionary computation and LLM capabilities while maintaining a clean, understandable codebase.







