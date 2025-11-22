# Known Bugs and Technical Debt

## Overview
This document tracks known bugs, limitations, technical debt, and areas requiring improvement in the LLaMEA codebase.

## Critical Issues

### None Currently Identified
No critical bugs that prevent core functionality.

## Known Bugs

### Minor Issues

#### 1. Windows Timeout Handling Not Supported
**Location:** `llamea/utils.py` - `handle_timeout()`

**Issue:** Signal-based timeout uses Unix signals, not available on Windows.

**Impact:** 
- Windows users cannot use signal-based evaluation timeouts
- Fallback to timeout parameter in Parallel() (which works)

**Workaround:** Use `eval_timeout` parameter in LLaMEA (Joblib handles cross-platform)

**Fix Priority:** Low (workaround exists)

**Proposed Fix:** Implement threading-based timeout as alternative

#### 2. ConfigSpace Serialization Failures Silent
**Location:** `llamea/loggers.py` - `log_configspace()`

**Issue:** If ConfigSpace serialization fails, writes "Failed to extract config space" instead of raising error.

**Impact:**
- HPO experiments may have missing ConfigSpace logs
- Silent failures make debugging harder

**Workaround:** Check configspace/ directory for error messages

**Fix Priority:** Low (rare occurrence)

**Proposed Fix:** Add logging.warning() before writing error message

#### 3. Model Name with Special Characters Breaks Paths
**Location:** `llamea/loggers.py` - `create_log_dir()`

**Issue:** Model names containing `/` or `:` (e.g., "gpt-3.5:latest") can create invalid paths.

**Impact:**
- Directory creation may fail
- Inconsistent log directory names

**Workaround:** Use `experiment_name` parameter, sanitize model names

**Fix Priority:** Medium

**Proposed Fix:** Sanitize model name: `model.replace("/", "_").replace(":", "_")`

#### 4. Conversation Context Can Exceed LLM Limits
**Location:** `managers.py` - `refine_algorithm()`

**Issue:** Accumulates `tried_algorithms` string indefinitely, can exceed context window.

**Impact:**
- Long experiments may fail with context length errors
- API costs increase with conversation length

**Workaround:** Restart with new session periodically

**Fix Priority:** Medium

**Proposed Fix:** Implement sliding window or truncation strategy

## Technical Debt

### Code Quality

#### 1. Duplicate Exception Definitions
**Location:** `llamea/utils.py`, `utils.py` (root)

**Issue:** `NoCodeException` and `OverBudgetException` defined in multiple places.

**Impact:** Confusion about import paths, potential inconsistencies

**Fix Priority:** Low

**Proposed Fix:** Consolidate in `llamea/utils.py`, import from there

#### 2. Global exec() Usage
**Location:** All main scripts, examples

**Issue:** `exec(algorithm_code, globals())` pollutes global namespace.

**Impact:**
- Namespace collisions possible
- Difficult to clean up
- Security concerns

**Fix Priority:** Medium

**Proposed Fix:** Use isolated namespace: `exec(code, isolated_globals)`

#### 3. Hardcoded BBOB Configuration
**Location:** `main.py`, `main-thesis.py`, `main-evolutionary.py`

**Issue:** BBOB evaluation parameters hardcoded (functions 1-24, instances 1-3, reps 3).

**Impact:** Inflexible evaluation pipeline

**Fix Priority:** Low

**Proposed Fix:** Extract to configuration or command-line arguments

#### 4. Inconsistent Logging Formats
**Location:** `llamea/loggers.py`, `managers.py`

**Issue:** Framework uses JSONL, legacy scripts use plain text.

**Impact:** Inconsistent log parsing

**Fix Priority:** Low

**Proposed Fix:** Migrate legacy scripts to JSONL

### Architecture

#### 1. No Sandboxing for Code Execution
**Location:** All evaluation functions

**Issue:** Generated code executes with full Python permissions.

**Impact:**
- Security risk
- Can access file system, network, etc.
- Trust-based security model

**Fix Priority:** High (for production use)

**Proposed Fix:** Implement RestrictedPython or Docker-based sandboxing

#### 2. No Distributed Evaluation Support
**Location:** `llamea/llamea.py`

**Issue:** Only single-machine parallelization via Joblib.

**Impact:**
- Cannot scale to clusters
- Limited by single machine resources

**Fix Priority:** Low (current use cases fit single machine)

**Proposed Fix:** Add Dask or Ray backend for distributed execution

#### 3. Memory Accumulation in Long Runs
**Location:** `llamea/llamea.py` - `run_history`

**Issue:** All evaluated solutions stored in memory.

**Impact:**
- Memory growth O(budget)
- Very long runs (budget > 10,000) may exhaust memory

**Fix Priority:** Low (typical budgets < 1,000)

**Proposed Fix:** Implement checkpointing and memory-mapped storage

#### 4. No Checkpoint/Resume Functionality
**Location:** Framework-wide

**Issue:** Cannot save and resume experiments.

**Impact:**
- Interrupted experiments lost
- Difficult to continue from specific generation

**Fix Priority:** Medium

**Proposed Fix:** Add `save_checkpoint()` and `load_checkpoint()` methods

### Testing

#### 1. Limited Test Coverage for Edge Cases
**Location:** `tests/`

**Issue:** Main functionality tested, but edge cases undercovered.

**Examples:**
- Empty populations
- Invalid diff formats
- Concurrent API rate limits
- Extreme niche radii

**Fix Priority:** Medium

**Proposed Fix:** Expand test suite, add property-based tests

#### 2. Integration Tests Missing
**Location:** `tests/`

**Issue:** Mostly unit tests, few end-to-end tests.

**Impact:** Component interactions not fully tested

**Fix Priority:** Low

**Proposed Fix:** Add integration tests in `tests/integration/`

#### 3. Mock LLM Not Realistic
**Location:** `llamea/llm.py` - `Dummy_LLM`

**Issue:** Always returns same hardcoded response, doesn't test variations.

**Impact:** Limited testing of prompt engineering

**Fix Priority:** Low

**Proposed Fix:** Implement configurable Dummy_LLM with response variations

## Performance Issues

### 1. Sequential LLM Calls Dominate Runtime
**Location:** `llamea/llamea.py` - `evolve_solution()`

**Issue:** LLM calls made sequentially even with parallel evaluation.

**Impact:** Major bottleneck (seconds per call)

**Fix Priority:** High (for large populations)

**Proposed Fix:** Batch LLM calls, use async API if available

**Status:** Partially addressed (parallel evaluation, but serial LLM queries)

### 2. Repeated AST Parsing for Distance
**Location:** `llamea/utils.py` - `code_distance()`

**Issue:** AST parsed on every distance calculation, no caching.

**Impact:** O(n²) AST parsing for niching

**Fix Priority:** Low (AST parsing fast)

**Proposed Fix:** Cache parsed ASTs with LRU cache

### 3. Excessive Disk I/O
**Location:** `llamea/loggers.py`

**Issue:** File open/close on every log call.

**Impact:** Slow for high-frequency logging

**Fix Priority:** Low (not a bottleneck)

**Proposed Fix:** Buffer logs, periodic flush

## Usability Issues

### 1. Error Messages Not Always Actionable
**Location:** Various

**Examples:**
- "An exception occurred: ..." (vague)
- LLM API errors not always caught gracefully

**Fix Priority:** Medium

**Proposed Fix:** Improve error messages with context and solutions

### 2. Progress Tracking Minimal
**Location:** `llamea/llamea.py`

**Issue:** Only prints generation number, no ETA or detailed progress.

**Impact:** User doesn't know how long to wait

**Fix Priority:** Low

**Proposed Fix:** Add tqdm progress bars, ETA calculations

### 3. Configuration Validation Missing
**Location:** `llamea/llamea.py` - `__init__()`

**Issue:** Invalid configurations (e.g., negative budget) not caught early.

**Impact:** Confusing errors later in execution

**Fix Priority:** Medium

**Proposed Fix:** Validate configuration in __init__()

### 4. Documentation Scattered
**Location:** Multiple places

**Issue:** Documentation in README, docs/, docstrings, comments.

**Impact:** Hard to find complete information

**Fix Priority:** Low (now addressed with this documentation suite)

**Proposed Fix:** Centralize in docs/ (in progress)

## Risky Areas of Code

### 1. Unified Diff Application
**Location:** `llamea/utils.py` - `apply_unified_diff()`

**Risk Level:** Medium

**Issues:**
- Complex line-by-line parsing
- Fragile to malformed diffs
- Edge cases with line endings

**Mitigation:** Extensive testing, fallback to full code if diff fails

### 2. Code Extraction Regex
**Location:** `llamea/llm.py` - `extract_algorithm_code()`

**Risk Level:** Medium

**Issues:**
- Regex may not handle all markdown variants
- LLM output format can vary
- False positives/negatives possible

**Mitigation:** Clear output format instructions in prompts

### 3. Parallel Evaluation
**Location:** `llamea/llamea.py` - `evolve_solution()`, `initialize()`

**Risk Level:** Medium

**Issues:**
- Timeout handling varies by backend
- Pickling failures with complex objects
- Resource exhaustion with large populations

**Mitigation:** Conservative max_workers, timeout buffers

### 4. Dynamic Code Execution
**Location:** All evaluation functions

**Risk Level:** High

**Issues:**
- Arbitrary code execution
- No sandboxing
- Security implications

**Mitigation:** Trust LLM providers, manual code review, isolated environments

## Unimplemented / Partial Features

### 1. Multi-Objective Optimization
**Status:** Not implemented

**Priority:** Low

**Description:** Framework assumes single fitness value, but could support Pareto fronts.

### 2. Constraint Handling
**Status:** Not implemented

**Priority:** Low

**Description:** No built-in constraint violation handling.

### 3. Transfer Learning
**Status:** Not implemented

**Priority:** Medium

**Description:** Cannot reuse knowledge from one problem to another.

### 4. Coevolution
**Status:** Not implemented

**Priority:** Low

**Description:** Only single population, could support competitive/cooperative coevolution.

### 5. Adaptive Prompt Already Implemented
**Status:** Implemented but underdocumented

**Priority:** Low

**Description:** `adaptive_prompt=True` works but lacks detailed documentation.

### 6. Population Evaluation Mode
**Status:** Implemented but underused

**Priority:** Low

**Description:** `evaluate_population=True` works but most examples use individual evaluation.

## Platform-Specific Issues

### Windows
- Signal-based timeout not supported
- Path separators in model names problematic
- Joblib backend "loky" may be slower

### MacOS
- No known specific issues

### Linux
- No known specific issues

## Dependency Issues

### 1. ConfigSpace API Changes
**Risk:** ConfigSpace updates may break compatibility

**Mitigation:** Pin version in pyproject.toml

### 2. OpenAI API Changes
**Risk:** OpenAI client updates may change API

**Mitigation:** Pin major version, monitor changelog

### 3. IOH Version Compatibility
**Risk:** IOH updates may change BBOB interface

**Mitigation:** Pin version for reproducibility

## Feature Requests / Wishlist

### High Priority
1. Checkpoint/resume functionality
2. Better progress tracking
3. Configuration validation
4. Improved error messages

### Medium Priority
1. Distributed evaluation support
2. Transfer learning
3. Batched LLM calls
4. Multi-objective optimization

### Low Priority
1. Coevolution
2. GUI for experiment monitoring
3. Automatic hyperparameter tuning for framework itself
4. Integration with MLOps tools

## Reporting New Issues

To report bugs or issues:

1. Check this document first to see if issue is known
2. Open GitHub issue with:
   - LLaMEA version
   - Python version
   - Operating system
   - Minimal reproducible example
   - Expected vs actual behavior
   - Relevant log output
3. Label appropriately: bug, enhancement, question, etc.

## Contributing Fixes

To fix issues:

1. Reference this document in PR description
2. Add tests for the fix
3. Update this document to mark issue as fixed
4. Follow coding conventions in `rules.md`

## Maintenance Schedule

This document should be reviewed:
- After each release
- When new bugs discovered
- Quarterly maintenance reviews
- Before major version bumps

Last updated: 2025-11-20


