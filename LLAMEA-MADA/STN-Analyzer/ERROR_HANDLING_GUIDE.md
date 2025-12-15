# Error Handling and Debugging Guide

## Overview

The STN Analyzer has been enhanced with comprehensive error handling and detailed progress logging to help identify where the analysis gets stuck when processing large experiments.

## What's Been Added

### 1. **Enhanced Progress Logging**

Every major operation now includes detailed logging:

- Loading offspring records (with count every 100 records)
- Loading fitness scores (progress every 50 files)
- Linking code files (progress every 50 files)
- Computing metrics (step-by-step logging)
- Creating visualizations (each visualization logged)

### 2. **Error Recovery**

The analyzer now continues processing even if some steps fail:

- If visualization fails, metrics are still saved
- If some files can't be parsed, processing continues
- Each error is logged with type and message

### 3. **Timeout Detection**

For potentially slow operations:

- Large graph detection (automatically uses approximate methods)
- Path finding with progress updates
- Cycle detection in lineage tracing

### 4. **Debug Mode Script**

A new `analyze_experiment.py` script provides:

- Step-by-step timing information
- Clear indication of which step is running
- Full error tracebacks if something fails
- Summary of time spent in each phase

## How to Use

### Option 1: Simple Batch Script (Windows)

```bash
cd STN-Analyzer
run_analysis.bat "..\exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
```

### Option 2: Python Debug Script

```bash
cd STN-Analyzer
python analyze_experiment.py "../exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
```

With interactive HTML visualization:

```bash
python analyze_experiment.py "../exp-directory" --interactive
```

### Option 3: Original Script (Still Works)

```bash
python stn_analyzer.py --exp-dir "../exp-directory"
```

## What to Look For

When the analyzer runs, you'll see detailed output like:

```
================================================================================
STEP 1: LOADING DATA
================================================================================

Initializing data loader...
  Loading offspring log from: experiment.jsonl
  Found 1234 records to process
  Processed 100 offspring records...
  Processed 200 offspring records...
  ...
  Successfully processed 1234 records (0 errors)

✓ STEP 1 COMPLETE (took 5.2s)
  Loaded 1234 nodes and 1180 edges
```

### If It Gets Stuck

**Look at the last message printed** - this tells you exactly where it stopped:

1. **"Counting lines in offspring log..."** - File might be corrupted or too large
2. **"Loading fitness scores..."** - Check if AUC files are accessible
3. **"Linking code files..."** - May have too many code files
4. **"Computing trajectory analysis..."** - Graph might be too complex
5. **"Identifying successful lineages..."** - Lineage tracing might have cycles
6. **"Computing layout..."** - Visualization layout computation might be stuck

### Common Issues and Solutions

#### Issue: Stuck on "Identifying successful lineages"

**Solution:** The graph might have cycles. The new code detects this and prints warnings:

```
Warning: Cycle detected in lineage for node mada_000123
```

#### Issue: Stuck on "Computing layout"

**Solution:** Large graphs can take time. The new code prints progress:

```
Computing hierarchical layout for 1000 nodes...
Graphviz layout failed, using custom tree layout
```

#### Issue: Out of memory

**Solution:** For very large experiments (>5000 nodes), the analyzer now:

- Uses approximate methods for path finding
- Limits lineage depth to 1000 nodes
- Shows progress every N operations

## Error Output Examples

### Good Run

```
✓ STEP 1 COMPLETE (took 5.2s)
✓ STEP 2 COMPLETE (took 1.1s)
✓ STEP 3 COMPLETE (took 8.3s)
✓ STEP 4 COMPLETE (took 45.2s)
✓ STEP 5 COMPLETE (took 2.1s)
```

### Partial Failure (Still Produces Results)

```
✓ STEP 3 COMPLETE (took 8.3s)
[Step 4/6] Creating visualizations
    Generating professional static STN...
    [ERROR] Failed to create professional static STN: MemoryError
    [OK] STN full: stn_full.png
    [OK] Fitness trajectory: fitness_trajectory.png
[WARNING] Error creating visualizations: 1 visualization failed
✓ STEP 5 COMPLETE (took 2.1s)
```

### Complete Failure

```
[Step 3/6] Calculating STN metrics
  Computing trajectory analysis...
    Checked 1000 root-leaf paths...
    Checked 2000 root-leaf paths...
[ERROR] Failed to calculate metrics: KeyboardInterrupt
Full traceback:
...
```

## Performance Tips

For very large experiments:

1. **Skip interactive HTML** (it's the slowest visualization):

   ```bash
   python analyze_experiment.py "experiment" --no-interactive
   ```

2. **Use the standard analyzer** (skips debug overhead):

   ```bash
   python stn_analyzer.py --exp-dir "experiment" --quiet
   ```

3. **Monitor system resources** while running to identify bottlenecks

## Logging Details

All major components now log:

### Data Loader

- Number of records/files found
- Progress every 50-100 items
- Errors with line numbers
- Summary of loaded data

### Graph Builder

- Node and edge counts
- Build time

### Metrics Calculator

- Each metric type being computed
- Large graph detection
- Approximate method usage
- Lineage tracing progress

### Visualizer

- Each visualization being created
- Layout algorithm being used
- Success/failure for each viz
- File paths for outputs

## Debugging Failed Runs

If the analyzer crashes or hangs:

1. **Check the last printed line** - shows where it stopped
2. **Look for error messages** - shows what went wrong
3. **Check system resources** - might be out of memory
4. **Try with a smaller experiment** - validate the analyzer works
5. **Review the traceback** - shows the full error chain

## Getting Help

If you encounter issues:

1. Save the complete console output to a file:

   ```bash
   python analyze_experiment.py "experiment" > analysis_log.txt 2>&1
   ```

2. Look for the last completed step and first error
3. Check if any warnings appeared before the error
4. Verify the experiment directory structure is correct

## File Structure Check

The analyzer expects:

```
experiment-directory/
├── mada_offspring.jsonl      (required - lineage data)
├── try-*-aucs.txt            (optional - fitness scores)
├── code/
│   └── try-*.py              (optional - algorithm code)
└── bandit_snapshots.jsonl    (optional - bandit history)
```

If `mada_offspring.jsonl` is missing, the analyzer will try to reconstruct from AUC files only.
