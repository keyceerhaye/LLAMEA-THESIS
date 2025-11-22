# loggers.py

## Purpose
Provides logging infrastructure for LLaMEA experiments, recording conversations, code, performance data, and population snapshots.

## Role in System
Centralized logging mechanism that:
- Creates experiment directories with timestamps
- Records LLM conversations (prompts and responses)
- Saves generated algorithm code files
- Logs ConfigSpace definitions for HPO
- Records individual solution metadata
- Saves performance metrics (AUCs, fitness)
- Maintains experiment reproducibility

## Key Classes

### `ExperimentLogger`
Main logging class for experiment data persistence.

**Initialization:**
- Creates timestamped directory: `exp-{MM-DD_HHMMSS}-{name}/`
- Creates subdirectories:
  - `configspace/` - HPO configuration spaces
  - `code/` - Generated algorithm source files

**Key Attributes:**
- `dirname` - Path to experiment log directory
- `attempt` - Counter for sequential logging

**Key Methods:**

#### `create_log_dir(name="")`
Creates experiment directory structure.
- **Format**: `exp-{date}_{time}-{name}`
- **Returns**: Directory name
- **Subdirs**: `configspace/`, `code/`

#### `log_conversation(role, content)`
Logs LLM conversation messages to JSONL file.
- **File**: `conversationlog.jsonl`
- **Format**: One JSON object per line
- **Fields**: role (client/model), time, content
- **Purpose**: Reproducing experiments, debugging prompts

#### `log_population(population)`
Logs entire population of solutions.
- For each solution:
  - Save code to `code/try-{attempt}-{name}.py`
  - Save configspace to `configspace/try-{attempt}-{name}.py`
  - Log individual metadata to `log.jsonl`
- Increments `attempt` counter

#### `log_individual(individual)`
Logs single solution metadata to JSONL.
- **File**: `log.jsonl`
- **Content**: `individual.to_dict()` serialized
- **Converts**: NumPy types to JSON-serializable

#### `log_code(attempt, algorithm_name, code)`
Saves algorithm source code to file.
- **Path**: `code/try-{attempt}-{algorithm_name}.py`
- **Updates**: `attempt` counter
- **Purpose**: Inspecting generated algorithms

#### `log_configspace(attempt, algorithm_name, config_space)`
Saves ConfigSpace definition to file.
- **Path**: `configspace/try-{attempt}-{algorithm_name}.py`
- **Format**: ConfigSpace JSON serialization
- **Fallback**: Writes error message if invalid

#### `log_aucs(attempt, aucs)`
Saves array of AUC scores to text file.
- **Path**: `try-{attempt}-aucs.txt`
- **Format**: NumPy savetxt (one value per line)
- **Purpose**: Post-experiment analysis

#### `set_attempt(attempt)`
Manually set the attempt counter.
- **Use case**: Custom numbering schemes

### Helper Function

#### `convert_to_serializable(data)`
Recursively converts NumPy types to JSON-compatible Python types.

**Conversions:**
- `np.integer` → `int`
- `np.floating` → `float`
- `np.bool_` → `bool`
- `np.ndarray` → `list`
- Recurses through dicts and lists

**Purpose:** Ensures all logged data is JSON-serializable.

## File Structure

### Experiment Directory Layout
```
exp-{date}_{time}-{name}/
├── conversationlog.jsonl     # LLM conversation history
├── log.jsonl                  # Solution metadata (one per line)
├── code/
│   ├── try-0-Algorithm1.py   # Generated code files
│   ├── try-1-Algorithm2.py
│   └── ...
├── configspace/
│   ├── try-0-Algorithm1.py   # ConfigSpace definitions
│   ├── try-1-Algorithm2.py
│   └── ...
└── try-{N}-aucs.txt          # Performance data files
```

### JSONL Format
Each line in `.jsonl` files is a complete JSON object:
```json
{"role": "client", "time": "2025-11-20 12:00:00.123", "content": "..."}
{"role": "gpt-4-turbo", "time": "2025-11-20 12:00:05.456", "content": "..."}
```

## Dependencies
- `os` - Directory creation
- `datetime` - Timestamps
- `jsonlines` - JSONL file handling
- `numpy` - Data serialization
- `ConfigSpace.read_and_write.json` - ConfigSpace serialization

## Data Flow
```
LLaMEA Run
    ↓
ExperimentLogger created → mkdir exp-{date}-{name}
    ↓
Initialization
    ↓
log_conversation() → conversationlog.jsonl
    ↓
LLM Response → Solution
    ↓
log_population() → log_code() + log_configspace() + log_individual()
    ↓
Evaluation → log_aucs()
    ↓
Next Generation (repeat)
```

## Usage Pattern

### Basic Setup
```python
logger = ExperimentLogger("my_experiment")
# Creates: exp-11-20_120000-my_experiment/
```

### LLaMEA Integration
```python
llamea = LLaMEA(f=eval_fn, llm=llm, log=True, experiment_name="test")
# Logger automatically attached to LLM
# All conversations logged
# All populations logged
```

### Manual Logging
```python
logger.log_code(0, "MyAlg", code_string)
logger.log_aucs(0, np.array([0.5, 0.6, 0.7]))
logger.log_individual(solution)
```

## Configuration
- **Automatic naming**: Uses model name and experiment name
- **No file rotation**: Append-only logging
- **No size limits**: Can grow indefinitely

## Error Handling
- Directory creation: OS errors propagate
- File writing: IO errors propagate
- ConfigSpace serialization: Catches exceptions, logs error message
- JSON serialization: Handles NumPy types gracefully

## Risks and Quirks
- **Name collisions**: Rare due to timestamp precision, but possible
- **Disk space**: No limits on log growth
- **Concurrency**: Not thread-safe (single-process assumption)
- **ConfigSpace errors**: Silently logs "Failed to extract config space"
- **Path separators**: Uses model name which may contain `/` or `:`

## Performance Considerations
- **File I/O**: Opens/closes file per log call (not buffered)
- **JSONL append**: Efficient for streaming data
- **NumPy savetxt**: Fast for numeric arrays
- **JSON serialization**: Overhead for large solutions

## Integration Points
- **LLaMEA**: Calls log_population() each generation
- **LLM**: Calls log_conversation() per query
- **Evaluation functions**: Can call log_aucs() or log_individual()
- **Analysis scripts**: Read JSONL and text files for post-processing

## Analysis Workflow
1. Run experiment → creates log directory
2. Read `conversationlog.jsonl` → inspect prompts/responses
3. Read `log.jsonl` → extract fitness trajectories
4. Read `try-*-aucs.txt` → aggregate performance
5. Inspect `code/*.py` → examine best algorithms

## Extension Points
To add custom logging:
1. Add method to `ExperimentLogger`
2. Call from LLaMEA or evaluation function
3. Use convert_to_serializable() for complex data
4. Follow naming pattern: `try-{attempt}-{descriptor}.{ext}`


