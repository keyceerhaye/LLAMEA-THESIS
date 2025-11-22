# utils.py

## Purpose
Utility functions supporting the LLaMEA framework, including exception handling, unified diff application, distance metrics, and probability distributions.

## Role in System
Provides foundational utilities used throughout the codebase:
- Custom exceptions for error signaling
- Unified diff patch application (for diff_mode)
- Code distance metrics (for niching)
- Power law distributions (for adaptive mutation)
- Timeout handling for evaluation

## Key Components

### Exceptions

#### `NoCodeException`
Raised when LLM response doesn't contain extractable code block.
- Used by: `llm.py` extraction methods
- Caught by: `llamea.py` to handle failed generations

### Functions

#### `handle_timeout(signum, frame)`
Signal handler for timeout interruption.
- **Purpose**: Raises `TimeoutError` when evaluation exceeds limit
- **Usage**: Set as signal handler before timed operations
- **Platform**: Unix/Linux signal-based (not Windows-compatible)

#### `apply_unified_diff(text: str, diff: str) -> str`
Applies unified diff patch to source text using pure Python.

**Features:**
- Cross-platform (no shell dependencies)
- Handles missing headers (auto-adds if needed)
- Line-by-line hunk application
- Context line matching with flexibility

**Algorithm:**
1. Parse diff headers and locate hunks
2. Extract old_start, new_start, counts from hunk headers
3. For each hunk:
   - Apply deletions (skip lines)
   - Apply additions (insert lines)
   - Verify context lines
4. Reconstruct patched text

**Error Handling:**
- Invalid hunk header → ValueError
- Line mismatches → Flexible matching on stripped lines
- Missing newlines → Auto-append

**Used By:** `llm.py` when `diff_mode=True`

#### `discrete_power_law_distribution(n, beta) -> float`
Samples from discrete power law distribution for adaptive mutation strength.

**Source:** Doerr et al. (2017) Fast Genetic Algorithms

**Parameters:**
- `n` - Number of lines in code
- `beta` - Power law exponent (typically 1.5)

**Returns:** Fraction of code to mutate (0.0 to 0.5)

**Purpose:** Controls mutation rate adaptively:
- Larger code → sample from broader range
- Power law favors small mutations
- Prevents overly aggressive mutations

**Algorithm:**
1. Compute half_n = n/2
2. Calculate normalization constant C_β
3. Compute probabilities for each mutation size
4. Sample mutation size from distribution
5. Return as fraction of n

**Used By:** `llamea.py` in `construct_prompt()` when `adaptive_mutation=True`

#### `code_distance(a, b) -> float`
Computes dissimilarity between two code snippets using AST comparison.

**Parameters:**
- `a`, `b` - Either `Solution` objects or raw code strings

**Returns:** Distance in [0, 1], where:
- 0.0 = identical ASTs
- 1.0 = completely different (or parse error)

**Algorithm:**
1. Extract code from Solution objects or use strings directly
2. Parse both into Abstract Syntax Trees
3. Dump ASTs to string representations
4. Compute SequenceMatcher similarity ratio
5. Return 1 - similarity

**Advantages:**
- Ignores whitespace/formatting differences
- Captures semantic structure
- Relatively fast

**Limitations:**
- Parse errors → returns 1.0 (max distance)
- Doesn't understand algorithm semantics
- Sensitive to variable names

**Used By:** `llamea.py` for niching distance calculations

### Helper Function

#### `_apply_hunk(lines, hunk_lines, old_start, old_count) -> List[str]`
Internal helper for applying a single diff hunk.

**Algorithm:**
1. Keep lines before hunk unchanged
2. Process each hunk line:
   - ` ` (context) → keep line
   - `-` (deletion) → skip line
   - `+` (addition) → insert line
3. Append remaining lines after hunk

## Dependencies
- `ast` - Abstract Syntax Tree parsing
- `re` - Regular expressions for diff parsing
- `difflib.SequenceMatcher` - String similarity
- `numpy` - Random sampling for power law

## Data Flow

### Diff Application
```
Original Code + Unified Diff
        ↓
  Parse diff headers and hunks
        ↓
  Apply each hunk sequentially
        ↓
  Patched Code
```

### Distance Calculation
```
Solution A + Solution B
        ↓
  Extract code strings
        ↓
  Parse to ASTs
        ↓
  Dump AST strings
        ↓
  SequenceMatcher ratio
        ↓
  1 - ratio = distance
```

### Power Law Sampling
```
Code length (n) + Beta
        ↓
  Compute probabilities
        ↓
  Sample mutation size
        ↓
  Return as fraction
```

## Configuration
No explicit configuration. Functions use:
- Beta=1.5 recommended for power law
- Diff format must follow unified diff spec
- AST parsing requires valid Python syntax

## Error Handling
- **NoCodeException**: Explicitly raised, caught upstream
- **Diff errors**: ValueError on invalid format
- **Parse errors**: Return max distance (1.0)
- **Timeout**: Raises TimeoutError

## Risks and Quirks
- **Diff edge cases**: Line ending mismatches can cause failures
- **AST limitations**: Only works for syntactically valid Python
- **Power law**: Empty code (n=0) returns 0.05 as fallback
- **Signal handling**: Not Windows-compatible
- **Diff fuzzy matching**: May apply patches incorrectly on ambiguous hunks

## Performance Considerations
- **AST parsing**: O(n) where n = code length
- **SequenceMatcher**: O(n*m) where n, m = AST dump lengths
- **Diff application**: O(lines * hunks)
- **Power law**: O(n/2) for probability computation

## Platform Compatibility
- **Unix/Linux**: Full support (signal handling)
- **Windows**: No signal-based timeout (handle_timeout won't work)
- **Cross-platform**: Diff application, distance, power law

## Integration Points
- **llamea.py**: Uses all utilities
- **llm.py**: Uses NoCodeException, apply_unified_diff
- **solution.py**: Implicitly used via distance metric


