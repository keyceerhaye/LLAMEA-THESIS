# STN Analyzer - Enhanced Error Handling & Debugging

## 🎯 Quick Fix for Your Issue

Your experiment is getting stuck during analysis. I've added comprehensive error handling so you can **see exactly where and why** it stops.

## 🚀 How to Use Right Now

### Option 1: Debug Script (Best for Troubleshooting)

Open PowerShell and run:

```powershell
cd "C:\Users\Kukoy\Documents\MADA-LLAMEA-IMPLEMENTATION\LLAMEA-THESIS\LLAMEA-MADA\STN-Analyzer"

python analyze_experiment.py "..\exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
```

### Option 2: Batch File (Easy Double-Click)

1. Open `STN-Analyzer` folder in File Explorer
2. Right-click `run_analysis.bat` → Edit
3. Change the path to your experiment
4. Save and double-click to run

### Option 3: Original Tool (Still Works)

```powershell
python stn_analyzer.py --exp-dir "..\exp-12-13_091444-google-gemini-2.5-flash-mada-v2-experiment-evolutionary-elitism(4,16(4+16, 0.9discount, 0.8-0.4cosine))"
```

## 📊 What You'll See

### Normal Progress Output

```
================================================================================
STEP 1: LOADING DATA
================================================================================

[Step 1/6] Loading experiment data from: exp-12-13_091444...
  Initializing data loader...
  Loading offspring log from: .../mada_offspring.jsonl
  Counting lines in offspring log...
  Found 150 records to process

  Processed 100 offspring records...
  Successfully processed 150 records (0 errors)

  Searching for AUC files in: ...
  Found 150 AUC files
  Loaded fitness for 50 nodes...
  Loaded fitness for 100 nodes...
  Loaded fitness for 150 nodes (0 files skipped)

✓ STEP 1 COMPLETE (took 5.2s)
  Loaded 150 nodes and 145 edges

================================================================================
STEP 2: BUILDING GRAPH
================================================================================
[Step 2/6] Building STN graph from 150 nodes and 145 edges
  Building graph structure...
[OK] Built STN graph: 150 nodes, 145 edges

✓ STEP 2 COMPLETE (took 1.1s)

[... continues through all 6 steps ...]

================================================================================
ANALYSIS COMPLETE!
================================================================================

Total time: 65.8s
  Step 1 (Load data):      5.2s
  Step 2 (Build graph):    1.1s
  Step 3 (Metrics):        8.3s
  Step 4 (Visualizations): 45.2s  ← Usually the slowest
  Step 5 (Save results):   2.1s

Results saved to: stn_outputs/exp-12-13_091444...
```

### If Something Goes Wrong

```
[Step 3/6] Calculating STN metrics
  Computing trajectory analysis...
    Computed paths from 10/50 roots...
    Computed paths from 20/50 roots...
    Checked 1000 root-leaf paths...
    Checked 2000 root-leaf paths...
    [STUCK OR SLOW HERE]
```

**The last line tells you exactly where the problem is!**

## 🔍 What Changed

### Before (Your Problem)

- ❌ Analyzer gets stuck silently
- ❌ No idea where it stopped
- ❌ Can't debug the issue
- ❌ Have to kill and restart

### After (Fixed)

- ✅ Detailed progress at every step
- ✅ Shows exactly where it stops
- ✅ Error messages with context
- ✅ Timing information
- ✅ Graceful degradation (partial results)

## 📁 New Files Created

| File                      | Purpose                                                 |
| ------------------------- | ------------------------------------------------------- |
| `analyze_experiment.py`   | **Main debug tool** - Use this to see detailed progress |
| `run_analysis.bat`        | Windows batch script for easy execution                 |
| `CHANGES_SUMMARY.md`      | Quick summary of what changed (read this first!)        |
| `ERROR_HANDLING_GUIDE.md` | Complete troubleshooting guide                          |
| `IMPROVEMENTS_SUMMARY.md` | Technical details of all changes                        |

## 🛠️ Modified Files

All core components now have enhanced error handling:

- ✅ `stn/data_loader.py` - Progress logging for file operations
- ✅ `stn/metrics.py` - Cycle detection, large graph handling
- ✅ `stn/visualizer.py` - Layout fallbacks, error recovery
- ✅ `stn_analyzer.py` - Step-by-step logging
- ✅ `QUICK_START.md` - Updated with troubleshooting info

## 🎯 Key Features

### 1. Progress Indicators

Every long operation shows progress:

```
Processed 100 offspring records...
Loaded fitness for 50 nodes...
Computed paths from 10/50 roots...
Checked 1000 root-leaf paths...
```

### 2. Detailed Error Messages

```
[ERROR] Failed to create interactive HTML: MemoryError
Warning: Cycle detected in lineage for node mada_000123
Large graph detected (50 roots, 200 leaves), using approximate longest lineage
```

### 3. Timing Breakdown

See which step is slow:

```
Step 1 (Load data):      5.2s  ✓ Fast
Step 2 (Build graph):    1.1s  ✓ Fast
Step 3 (Metrics):        8.3s  ✓ Reasonable
Step 4 (Visualizations): 45.2s ← Might be slow for large graphs
Step 5 (Save results):   2.1s  ✓ Fast
```

### 4. Graceful Degradation

If one part fails, the rest continues:

```
[OK] STN full: stn_full.png
[ERROR] Failed to create interactive HTML: MemoryError
[OK] Fitness trajectory: fitness_trajectory.png
[OK] Operator analysis: operator_analysis.png
```

## 🐛 Common Issues & Solutions

### Issue 1: Stuck on "Loading offspring log"

```
Counting lines in offspring log...
[STUCK]
```

**Cause:** Very large file or file read error  
**Solution:** Check file size, try opening in text editor

### Issue 2: Stuck on "Checking root-leaf paths"

```
Checked 5000 root-leaf paths...
Checked 10000 root-leaf paths...
[VERY SLOW]
```

**Cause:** Complex graph with many possible paths  
**Solution:** This is normal for large graphs - wait or interrupt with Ctrl+C

### Issue 3: Stuck on "Computing layout"

```
Computing hierarchical layout for 1000 nodes...
[STUCK]
```

**Cause:** Layout algorithm is computationally expensive  
**Solution:** Can take 5-10 minutes for large graphs - be patient

### Issue 4: Out of Memory

```
[ERROR] MemoryError
```

**Cause:** Graph too large for available RAM  
**Solution:** Close other programs, or run on machine with more RAM

## 📝 What to Do Now

### Step 1: Run the Analyzer

Use the debug script:

```powershell
cd STN-Analyzer
python analyze_experiment.py "..\your-experiment-directory"
```

### Step 2: Watch the Output

Pay attention to:

1. **Where does it stop?** (last message printed)
2. **How long does each step take?** (timing info)
3. **Any warnings or errors?** (error messages)

### Step 3: Report Back

If it still has issues, share:

- The **last 10-20 lines** of output
- The **timing information** (if it completes)
- Any **error messages** that appear

## 📚 Documentation

| Document                    | When to Read                     |
| --------------------------- | -------------------------------- |
| **CHANGES_SUMMARY.md**      | Read this first - quick overview |
| **ERROR_HANDLING_GUIDE.md** | Full troubleshooting guide       |
| **IMPROVEMENTS_SUMMARY.md** | Technical details of changes     |
| **QUICK_START.md**          | General usage guide              |

## 💡 Pro Tips

1. **Save output to file:**

   ```powershell
   python analyze_experiment.py "experiment" > analysis_log.txt 2>&1
   ```

2. **Interrupt if stuck:**

   - Press `Ctrl+C` to stop
   - The output shows where it was stuck
   - Partial results might still be saved

3. **Skip interactive HTML** (fastest):

   ```powershell
   python stn_analyzer.py --exp-dir "experiment"
   # Don't use --interactive flag
   ```

4. **Check system resources:**
   - Open Task Manager (Ctrl+Shift+Esc)
   - Monitor CPU and Memory while running
   - Close other programs if needed

## ✨ Summary

You now have:

- ✅ Detailed progress logging throughout the analyzer
- ✅ Clear error messages when things go wrong
- ✅ Timing information to identify bottlenecks
- ✅ Tools to debug and troubleshoot
- ✅ Graceful handling of errors
- ✅ Ability to see exactly where it stops

**Try running the analyzer now and you'll immediately see where the issue is!**
