# STN Analyzer - Quick Reference Cheat Sheet

## 🚀 Run Analysis (Pick One)

### Option 1: Debug Script (Shows Everything)

```powershell
cd STN-Analyzer
python analyze_experiment.py "..\your-experiment-folder"
```

### Option 2: Batch File (Easy)

```powershell
cd STN-Analyzer
run_analysis.bat "..\your-experiment-folder"
```

### Option 3: Original Tool

```powershell
cd STN-Analyzer
python stn_analyzer.py --exp-dir "..\your-experiment-folder"
```

## 📊 What the Steps Do

| Step                 | What It Does             | Typical Time | Can Be Slow?        |
| -------------------- | ------------------------ | ------------ | ------------------- |
| 1. Load Data         | Read JSONL and AUC files | 5-10s        | ⚠️ If 1000+ files   |
| 2. Build Graph       | Create network structure | 1-2s         | ❌ Always fast      |
| 3. Calculate Metrics | Compute statistics       | 5-15s        | ⚠️ If complex graph |
| 4. Visualizations    | Generate PNG/HTML files  | 20-60s       | ✅ Usually slowest  |
| 5. Save Results      | Write JSON files         | 1-3s         | ❌ Always fast      |

## 🔍 Reading the Output

### ✅ Success Looks Like:

```
✓ STEP 1 COMPLETE (took 5.2s)
✓ STEP 2 COMPLETE (took 1.1s)
✓ STEP 3 COMPLETE (took 8.3s)
✓ STEP 4 COMPLETE (took 45.2s)
✓ STEP 5 COMPLETE (took 2.1s)

ANALYSIS COMPLETE!
Total time: 61.9s
```

### ❌ Problem Looks Like:

```
[Step 3/6] Calculating STN metrics
  Computing trajectory analysis...
    Checked 5000 root-leaf paths...
    [STUCK HERE - NO MORE OUTPUT]
```

**→ The last line tells you where it stopped!**

## 🐛 Quick Troubleshooting

| If It Says...                 | Problem               | Solution                             |
| ----------------------------- | --------------------- | ------------------------------------ |
| "Counting lines..." [stuck]   | Large file            | Check file size, be patient          |
| "Checked N paths..." [slow]   | Complex graph         | Normal - wait or Ctrl+C              |
| "Computing layout..." [stuck] | Layout algorithm slow | Can take 5-10 min                    |
| "MemoryError"                 | Out of RAM            | Close programs or use bigger machine |
| "FileNotFoundError"           | Missing file          | Check experiment directory structure |

## 🎯 Expected File Structure

```
your-experiment/
├── mada_offspring.jsonl      ← Required!
├── try-0-aucs.txt
├── try-1-aucs.txt
├── ...
└── code/
    ├── try-0-Algorithm.py
    └── ...
```

## 💾 Where Results Are Saved

```
STN-Analyzer/
└── stn_outputs/
    └── your-experiment/
        ├── stn_metrics.json           ← Main metrics
        ├── generation_stats.json      ← Per-generation data
        ├── successful_lineages.json   ← Top lineages
        ├── stn_graph.json             ← Graph structure
        ├── stn_static.png             ← Professional viz
        ├── stn_full.png               ← Full network
        ├── fitness_trajectory.png     ← Fitness over time
        ├── operator_analysis.png      ← Operator comparison
        ├── diversity_metrics.png      ← Diversity plots
        └── stn_interactive.html       ← Interactive viz
```

## ⚡ Quick Commands

### Save output to file:

```powershell
python analyze_experiment.py "experiment" > log.txt 2>&1
```

### Stop if stuck:

```
Press: Ctrl+C
```

### Skip interactive HTML (faster):

```powershell
python stn_analyzer.py --exp-dir "experiment"
```

### Check only specific experiment:

```powershell
python analyze_experiment.py "experiment-name-here"
```

## 📚 Full Documentation

| File                       | Purpose                  |
| -------------------------- | ------------------------ |
| `README_ERROR_HANDLING.md` | Main guide - start here! |
| `CHANGES_SUMMARY.md`       | What changed & why       |
| `ERROR_HANDLING_GUIDE.md`  | Full troubleshooting     |
| `IMPROVEMENTS_SUMMARY.md`  | Technical details        |

## 🎓 Understanding Progress Messages

### Data Loading:

```
Processed 100 offspring records...     ← Reading JSONL
Loaded fitness for 50 nodes...         ← Reading AUC files
Linked 50 code files...                ← Reading Python files
```

### Metrics:

```
Computed paths from 10/50 roots...     ← Path finding
Checked 1000 root-leaf paths...        ← Finding longest paths
Traced 50/100 lineages...              ← Lineage tracing
```

### Visualization:

```
Computing hierarchical layout...        ← Positioning nodes
Trying graphviz dot layout...          ← Layout algorithm
Generating professional static STN...  ← Creating PNG
```

## 🆘 Emergency Help

If completely stuck:

1. Read last line of output ← **Tells you where**
2. Check timing info ← **Tells you what's slow**
3. Look for errors ← **Tells you what's wrong**

Then:

1. Check `ERROR_HANDLING_GUIDE.md` for your specific issue
2. Or share the last 20 lines of output for help

---

**Remember:** The debug script (`analyze_experiment.py`) shows you EXACTLY where it stops!
