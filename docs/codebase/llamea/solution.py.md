# solution.py

## Purpose
Defines the `Solution` class, which represents a candidate algorithm (individual) in the evolutionary population.

## Role in System
Core data structure that encapsulates:
- Generated algorithm code
- Fitness scores and evaluation feedback
- Metadata and genealogy (parent relationships)
- Configuration spaces for HPO
- Task prompts used for generation

## Key Classes

### `Solution`
Represents a single candidate algorithm in the evolutionary process.

**Key Attributes:**
- `id` - Unique UUID for this solution
- `code` - Python source code of the algorithm
- `name` - Class name extracted from code
- `description` - Short textual description of algorithm
- `configspace` - Optional ConfigSpace for hyperparameter optimization
- `generation` - Which generation this solution belongs to
- `fitness` - Performance score (starts at -∞)
- `feedback` - Textual feedback from evaluation
- `error` - Error message if evaluation failed
- `parent_ids` - List of UUIDs of parent solutions
- `operator` - LLM mutation operator that created this solution
- `task_prompt` - Task description used to generate this
- `metadata` - Dictionary for arbitrary additional data

**Key Methods:**
- `set_scores(fitness, feedback, error)` - Update evaluation results
- `set_operator(operator)` - Record which mutation created this
- `add_metadata(key, value)` - Store custom metadata
- `get_metadata(key)` - Retrieve metadata value
- `get_summary()` - Concise string representation for prompts
- `copy()` - Create child solution with incremented generation
- `empty_copy()` - Create child with only genealogy preserved
- `to_dict()` - Convert to dictionary (for serialization)
- `to_json()` - Convert to JSON string

## Data Structure

### Genealogy Tracking
- Each solution has unique `id`
- `parent_ids` list tracks evolutionary lineage
- `generation` tracks depth in evolutionary tree
- Enables analysis of successful lineages

### Metadata Storage
- Flexible `metadata` dict for extensions
- Can store:
  - HPO results
  - Execution times
  - Detailed performance metrics
  - Algorithm-specific data

### ConfigSpace Integration
- `configspace` stores SMAC3 ConfigurationSpace
- Serialized to dict for JSON export
- Used by HPO evaluation pipeline

## Usage Patterns

### Creation
```python
solution = Solution(
    code="class MyAlg: ...",
    name="MyAlg",
    description="Novel algorithm using...",
    generation=0
)
```

### Evaluation
```python
solution.set_scores(
    fitness=0.85,
    feedback="Good performance on separable functions",
    error=""
)
```

### Mutation (Creating Offspring)
```python
child = parent.copy()
child.set_operator("Refine strategy")
# LLM mutates child.code
child.generation  # Automatically incremented
child.parent_ids  # Contains parent.id
```

### Serialization
```python
# For logging/storage
json_str = solution.to_json()
dict_obj = solution.to_dict()
```

## Dependencies
- `uuid` - Unique identifier generation
- `numpy` - Fitness initialization (-np.inf)
- `json` - JSON serialization
- `ConfigSpace` - Hyperparameter configuration (optional)

## Data Flow
```
LLM Response → extract code/description
              ↓
        Create Solution
              ↓
        Execute code → set_scores()
              ↓
        Selection → copy() → Mutation
              ↓
        Logger → to_dict() → JSON storage
```

## Risks and Quirks
- **UUID collisions**: Theoretically possible but extremely rare
- **ConfigSpace serialization**: May fail if ConfigSpace invalid → returns empty string
- **Fitness initialization**: Starts at -∞, requires explicit setting
- **Memory**: Long runs accumulate many solution objects
- **Metadata**: Unstructured dict can lead to inconsistencies

## Performance Considerations
- Lightweight object (mostly strings and primitives)
- Deep copy of metadata dict on `copy()`
- JSON serialization relatively fast
- UUID generation negligible cost

## Integration Points
- **LLaMEA**: Creates, evaluates, selects solutions
- **LLM**: Extracts code/description into Solution
- **Logger**: Serializes solutions to disk
- **Evaluation Function**: Receives solution, sets scores
- **HPO**: Extracts configspace, stores results

## Extension Pattern
To add custom fields:
1. Add to `__init__()` parameters
2. Add to `copy()` and `empty_copy()`
3. Add to `to_dict()` for serialization
4. Update summary format if needed


