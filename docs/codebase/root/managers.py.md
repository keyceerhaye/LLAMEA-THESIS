# managers.py

## Purpose
Custom manager classes for legacy/thesis scripts (`main.py`, `main-thesis.py`) that provide simplified interface to OpenAI API without using full LLaMEA framework.

## Role in System
Bridges between simple scripts and LLM interaction:
- Handles OpenAI API calls
- Manages conversation history
- Extracts code and names from responses
- Tracks best algorithm (elitism)
- Provides detailed feedback options
- Logs conversations and code

## Key Classes

### `ExperimentLogger`
Simplified logger for non-framework scripts.

**Purpose:** Create experiment directories and log basic data.

**Key Methods:**
- `create_log_dir(name)` - Creates `exp-{date}_{time}-{name}/` with `ioh/`, `code/` subdirs
- `log_conversation(content)` - Appends to `conversationlog.txt` (plain text, not JSONL)
- `log_code(attempt, algorithm_name, code)` - Saves code to `code/try-{N}-{name}.py`
- `log_aucs(attempt, aucs)` - Saves AUC array to `try-{N}-aucs.txt`

**Differences from llamea.ExperimentLogger:**
- Plain text conversation log (not JSONL)
- No `log_individual()` method
- No ConfigSpace logging
- Simpler structure

### `AlgorithmManager`
Manages LLM interaction for algorithm generation and refinement.

**Purpose:** Encapsulates OpenAI API calls and prompt engineering for algorithm evolution.

**Key Attributes:**
- `client` - OpenAI client instance
- `ai_model` - Model identifier (e.g., "gpt-4-turbo", "gemini-2.0-flash")
- `max_tokens` - Optional token limit
- `elitism` - Flag to track and report best algorithm
- `detailed_feedback` - Flag to provide function-group performance
- `current_best_algorithm` - Code of best algorithm found
- `current_best_AOCC` - Fitness of best algorithm
- `last_algorithm` - Most recent LLM response
- `tried_algorithms` - Accumulated list of tried algorithms
- `last_error` - Most recent error message
- `role_prompt` - System prompt for LLM
- `init_prompt` - Initial task description
- `debug_mode` - Flag to read from file instead of API

**Key Methods:**

#### `__init__(api_key, logger, ai_model, elitism, detailed_feedback, base_url, max_tokens)`
Initialize manager with configuration.

**Features:**
- Accepts custom `base_url` for API endpoint
- Accepts `max_tokens` for response limiting
- Sets up OpenAI client
- Defines role and task prompts
- Initializes tracking variables

#### `fetch_algorithm()`
Request initial algorithm from LLM.

**Prompt Structure:**
- System: Role prompt
- User: Initial prompt (task + example + format)

**Returns:** LLM response text

**Side Effects:**
- Logs conversation to logger
- Can read from file if `debug_mode=True`

#### `refine_algorithm(auc_mean, auc_std, algorithm_name, detailed_aucs)`
Request algorithm refinement based on performance feedback.

**Inputs:**
- `auc_mean` - Average AOCC score
- `auc_std` - Standard deviation
- `algorithm_name` - Name of algorithm to refine
- `detailed_aucs` - List of 5 scores by function group

**Prompt Structure:**
- System: Role prompt
- User: Initial prompt
- User: Tried algorithms history
- Assistant: Last algorithm response
- [Optional] User: Detailed feedback by function group
- [Optional] User: Elitism prompt with best algorithm
- User: Refine prompt with performance

**Elitism Logic:**
Updates `current_best_algorithm` if new algorithm better.

**Detailed Feedback:**
Provides AOCC breakdown by BBOB function groups:
1. Separable (1-5)
2. Low/moderate conditioning (6-9)
3. High conditioning, unimodal (10-14)
4. Multi-modal, adequate structure (15-19)
5. Multi-modal, weak structure (20-24)

**Returns:** LLM response text

**Side Effects:**
- Updates `tried_algorithms` history
- Logs full conversation

#### `extract_algorithm_code(message)`
Extract Python code from LLM response.

**Pattern:** `` ```(?:python)?\n(.*?)\n``` ``

**Returns:** Code string

**Raises:** `NoCodeException` if no code block found

#### `extract_algorithm_name(message)`
Extract algorithm name from response.

**Pattern:** `` `#\s*Name:\s*(\\w*)` ``

**Returns:** Name string or empty string

**Note:** Often not used; class name extracted from code directly.

## Prompt Design

### Role Prompt
```
"You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box
optimization problems. Do not use hyped nature-inspired algorithms such as
Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are
generally not well performing."
```

**Purpose:** Set expertise level and discourage poor algorithm patterns.

### Initial Prompt
```
- Task: Optimize BBOB 24 noiseless functions
- Signature: __init__(self, budget), __call__(self, func)
- Bounds: [-5.0, 5.0]
- Dimension: 5 (fixed in code, not mentioned in prompt)
- Example: RandomSearch implementation
- Format: # Name: <classname>\n# Code: <code>
```

**Purpose:** Clear specification with working example.

### Refinement Prompt
```
"The last proposed algorithm {name} got an average AOCC of {mean},
and standard deviation of {std}. Either refine or redesign to improve
the algorithm."
```

**Purpose:** Feedback loop for evolution.

### Error Refinement Prompt
```
"The last proposed algorithm {name} got an error: {error},
an average AOCC of {mean}, and standard deviation of {std}.
Either refine or redesign to improve the algorithm."
```

**Purpose:** Signal failure mode to LLM.

### Detailed Feedback Prompt
```
"The mean AOCC score on Separable functions was {score1},
on functions with low/moderate conditioning {score2},
..."
```

**Purpose:** Guide LLM toward problem-specific improvements.

### Elitism Prompt
```
"The best so far proposed algorithm got an average AOCC of {best_score}
and the code was as follows:\n{best_code}"
```

**Purpose:** Remind LLM of best solution (elitism strategy).

## Configuration Options

### Base URL
Custom API endpoint for OpenAI-compatible providers:
```python
AlgorithmManager(
    ...,
    base_url="https://api.aimlapi.com/v1"
)
```

### Max Tokens
Limit response length:
```python
AlgorithmManager(
    ...,
    max_tokens=4096
)
```

### Elitism
Track and report best algorithm:
```python
AlgorithmManager(
    ...,
    elitism=True
)
```

### Detailed Feedback
Provide function-group performance:
```python
AlgorithmManager(
    ...,
    detailed_feedback=True
)
```

## Dependencies
- `os` - Not used directly (leftover import)
- `numpy` - Not used directly (leftover import)
- `re` - Regular expressions for extraction
- `openai` - OpenAI API client
- `datetime` - Timestamps for logging
- `llamea.utils` - NoCodeException

## Data Flow
```
main.py/main-thesis.py
  ↓
AlgorithmManager.fetch_algorithm() [first iteration]
  ↓
OpenAI API → Response
  ↓
extract_algorithm_code() → Code
  ↓
exec(code) → Evaluate → AUC
  ↓
AlgorithmManager.refine_algorithm(auc_mean, auc_std, ...)
  ↓
[Update elitism tracking]
[Build refinement prompt]
  ↓
OpenAI API → Response
  ↓
(repeat)
```

## Comparison with LLaMEA Framework

| Feature | AlgorithmManager | LLaMEA Framework |
|---------|------------------|------------------|
| Populations | No (single alg) | Yes (μ+λ) |
| Niching | No | Yes (sharing/clearing) |
| HPO | No | Yes (SMAC integration) |
| Diff mode | No | Yes |
| Parallelization | No | Yes (joblib) |
| Conversation log | Plain text | JSONL |
| Flexibility | Limited | Highly configurable |
| Complexity | Low | Medium |

**Use case:**
- AlgorithmManager: Simple (1+1)-style evolution
- LLaMEA: Full-featured evolutionary algorithm

## Risks and Quirks
- **No ConfigSpace**: Cannot handle HPO
- **Plain text logs**: Harder to parse than JSONL
- **Manual conversation tracking**: `tried_algorithms` string concatenation
- **Temperature hardcoded**: 0.8 (not configurable without editing)
- **No retry logic**: Relies on OpenAI client defaults
- **Global conversation history**: Can grow large and hit context limits

## Performance Considerations
- **Conversation context growth**: O(n) where n = iterations
- **Token usage**: Increases with longer history
- **No caching**: Every call includes full history

## Integration Points
- **main.py**: Original user
- **main-thesis.py**: Enhanced user
- **OpenAI API**: External dependency
- **ExperimentLogger**: Logging partner

## Migration Path to LLaMEA
To migrate from AlgorithmManager to LLaMEA:

1. Replace:
```python
manager = AlgorithmManager(api_key, logger, model, elitism, detailed_feedback)
for i in range(budget):
    if i == 0:
        message = manager.fetch_algorithm()
    else:
        message = manager.refine_algorithm(...)
    # evaluate...
```

2. With:
```python
llm = OpenAI_LLM(api_key, model)
llamea = LLaMEA(f=eval_fn, llm=llm, n_parents=1, n_offspring=1, budget=budget)
best = llamea.run()
```

## When to Use
- ✅ Simple (1+1)-style evolution
- ✅ Quick prototyping
- ✅ Understanding basic LLM-evolution
- ✅ Custom evaluation pipelines
- ✅ Thesis-specific experiments

## When NOT to Use
- ❌ Population-based evolution (use LLaMEA)
- ❌ Need niching/diversity (use LLaMEA)
- ❌ Need HPO (use LLaMEA)
- ❌ Need parallelization (use LLaMEA)
- ❌ Production systems (use LLaMEA)







