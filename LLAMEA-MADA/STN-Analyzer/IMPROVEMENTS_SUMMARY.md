# STN Analyzer Error Handling Improvements

## Summary

The STN Analyzer has been significantly enhanced with comprehensive error handling and detailed progress logging to help identify where processing gets stuck, especially with large experiments.

## Changes Made

### 1. Data Loader (`stn/data_loader.py`)

#### Enhanced `_load_offspring_log()`:

- ✅ Added line counting with error handling
- ✅ Progress logging every 100 records
- ✅ Individual try-catch for each record
- ✅ Error counter and summary
- ✅ Full traceback on critical errors

#### Enhanced `_load_fitness_scores()`:

- ✅ File search error handling
- ✅ Progress logging every 50 files
- ✅ Error counter with sample error messages
- ✅ Summary of loaded vs skipped files

#### Enhanced `_link_code_files()`:

- ✅ Directory existence check
- ✅ File search error handling
- ✅ Progress logging every 50 files
- ✅ Error counter and summary

#### Enhanced `_load_bandit_history()`:

- ✅ File existence logging
- ✅ Line-by-line error handling
- ✅ Count of loaded snapshots

### 2. Main Analyzer (`stn_analyzer.py`)

#### Step 1 (Load Data):

- ✅ Detailed step logging
- ✅ Error type identification
- ✅ Full traceback on failure
- ✅ Timing information

#### Step 2 (Build Graph):

- ✅ Progress logging
- ✅ Error handling with traceback
- ✅ Timing information

#### Step 3 (Calculate Metrics):

- ✅ Individual metric logging
- ✅ Step-by-step progress
- ✅ Error handling for entire step
- ✅ Timing information

#### Step 4 (Create Visualizations):

- ✅ Directory creation with logging
- ✅ Error handling continues processing
- ✅ Count of successful visualizations
- ✅ Timing information

#### Steps 5 & 6 (Save Results):

- ✅ Individual file save logging
- ✅ Warning on failure (not fatal)
- ✅ Timing information

### 3. Metrics Calculator (`stn/metrics.py`)

#### Enhanced `trajectory_analysis()`:

- ✅ Progress logging for path computation
- ✅ Large graph detection (>10,000 root-leaf pairs)
- ✅ Approximate method for large graphs
- ✅ Progress every 1000 paths
- ✅ Complete error recovery
- ✅ Default values on failure

#### Enhanced `identify_successful_lineages()`:

- ✅ Fitness threshold logging
- ✅ Cycle detection in lineages
- ✅ Maximum depth limit (1000 nodes)
- ✅ Progress every 50 lineages
- ✅ Error recovery per lineage
- ✅ Summary statistics

### 4. Visualizer (`stn/visualizer.py`)

#### Enhanced `_compute_layout()`:

- ✅ Layout algorithm logging
- ✅ Fallback strategy (graphviz → tree → spring → circular)
- ✅ Error messages for each failure
- ✅ Node count logging

#### Enhanced `generate_all_visualizations()`:

- ✅ Try-catch for each visualization
- ✅ Detailed error messages
- ✅ Continue on failure
- ✅ Individual visualization logging

## New Tools

### 1. `analyze_experiment.py`

A debug-focused script that provides:

- Clear step-by-step progress
- Timing for each major phase
- Visual separators between steps
- Comprehensive error reporting
- Final summary with timing breakdown

### 2. `run_analysis.bat`

Windows batch script for easy execution:

- Simple command-line interface
- Directory validation
- Error checking
- Pause on completion

### 3. `ERROR_HANDLING_GUIDE.md`

Complete documentation including:

- What was added
- How to use each tool
- Common issues and solutions
- Error output examples
- Debugging tips

## Key Features

### Progressive Error Handling

The analyzer now uses a "graceful degradation" approach:

1. **Continue on non-critical errors** (e.g., single file parse failure)
2. **Report but proceed** (e.g., visualization failure)
3. **Stop only on critical errors** (e.g., no data loaded)

### Performance Optimizations

For large experiments:

1. **Approximate methods** for expensive operations
2. **Progress indicators** so you know it's working
3. **Early detection** of problematic patterns
4. **Memory-conscious** algorithms

### Detailed Logging

Every operation now logs:

- What it's doing
- How many items to process
- Progress updates
- Success/failure status
- Error details with types

## Usage Examples

### Quick Analysis

```bash
cd STN-Analyzer
python analyze_experiment.py "../your-experiment-directory"
```

### With Interactive HTML

```bash
python analyze_experiment.py "../your-experiment-directory" --interactive
```

### Using Batch File (Windows)

```bash
cd STN-Analyzer
run_analysis.bat "..\your-experiment-directory"
```

### Original Tool (Still Works)

```bash
python stn_analyzer.py --exp-dir "../your-experiment-directory"
```

## Expected Output

### Successful Run

```
================================================================================
STEP 1: LOADING DATA
================================================================================

[Step 1/6] Loading experiment data from: experiment-dir
  Starting data load...
  Loading offspring log from: experiment-dir/mada_offspring.jsonl
  Counting lines in offspring log...
  Found 1234 records to process
  Processed 100 offspring records...
  ...
  Successfully processed 1234 records (0 errors)
[OK] Loaded 1234 nodes and 1180 edges

✓ STEP 1 COMPLETE (took 5.2s)
  Loaded 1234 nodes and 1180 edges

================================================================================
STEP 2: BUILDING GRAPH
================================================================================
...
```

### Handling Errors

```
[Step 4/6] Creating visualizations
    Generating professional static STN...
       Computing hierarchical layout for 1000 nodes...
       Trying graphviz dot layout...
       Graphviz layout failed (ModuleNotFoundError), using custom tree layout
       Using custom tree layout based on parent-child relationships
    [OK] STN static (professional): stn_static.png
    ...
    Generating interactive HTML...
    [ERROR] Failed to create interactive HTML: MemoryError: out of memory
[WARNING] Error creating visualizations: MemoryError
  Generated 5/6 visualizations

✓ STEP 4 COMPLETE (took 45.2s)
```

## Troubleshooting

### If Analysis Gets Stuck

1. **Check the last printed line** - tells you where it stopped
2. **Look for progress numbers** - shows if it's making progress
3. **Monitor system resources** - might be resource constrained
4. **Try keyboard interrupt (Ctrl+C)** - will show current operation

### Common Hang Points (Now Fixed)

1. ✅ **Loading large offspring logs** → Progress every 100 records
2. ✅ **Loading many AUC files** → Progress every 50 files
3. ✅ **Finding paths in large graphs** → Approximate methods used
4. ✅ **Tracing lineages with cycles** → Cycle detection added
5. ✅ **Computing graph layouts** → Fallback strategies added

## Benefits

### Before Enhancement

- Silent failures (no idea what went wrong)
- Hangs with no feedback (is it working?)
- Complete failure on any error
- No way to debug large experiments

### After Enhancement

- ✅ Detailed error messages with types
- ✅ Progress indicators for long operations
- ✅ Graceful degradation (partial results)
- ✅ Clear indication of where it stopped
- ✅ Timing information for optimization
- ✅ Debug tools for investigation

## Testing Recommendations

For your specific experiment:

```bash
cd STN-Analyzer

# Run with debug script to see detailed progress
python analyze_experiment.py "..\exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
```

Watch for:

1. How many records are loaded
2. Which step takes the longest
3. Any error messages
4. Final timing breakdown

## Future Enhancements (If Needed)

If you still encounter issues, we can add:

- Timeout limits for operations
- Checkpoint/resume functionality
- Memory usage monitoring
- Parallel processing for visualizations
- Incremental processing for very large experiments

## Questions to Consider

As you run the analyzer, note:

1. **Where does it stop?** (if at all)
2. **How long does each step take?**
3. **Are there any warnings?**
4. **Does it complete successfully?**
5. **Which visualizations fail?** (if any)

This information will help us further optimize the analyzer if needed.
