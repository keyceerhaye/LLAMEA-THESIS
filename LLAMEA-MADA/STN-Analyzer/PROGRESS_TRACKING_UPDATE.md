# STN-Analyzer Progress Tracking Update

## Summary

Added comprehensive progress tracking to the STN-Analyzer to provide real-time feedback on analysis duration and completion status.

## Changes Made

### 1. Added `tqdm` Progress Bars

#### **data_loader.py**

- Added progress bar for loading offspring records from `mada_offspring.jsonl`
- Added progress bar for loading fitness scores from AUC files
- Added progress bar for linking code files to nodes

#### **stn_analyzer.py**

- Added main progress bar showing 6 major analysis steps:
  1. Loading data
  2. Building graph
  3. Calculating metrics
  4. Creating visualizations
  5. Saving metrics
  6. Saving lineages

#### **visualizer.py**

- Added progress bar for generating visualizations (6 visualizations total):
  1. Full STN plot
  2. Fitness trajectory
  3. Operator analysis
  4. Diversity metrics
  5. Lineage tree
  6. Interactive HTML

### 2. Updated Dependencies

- Added `tqdm>=4.65` to `requirements.txt`

## Example Output

```
============================================================
STN ANALYZER FOR MADA-LLAMEA
Search Trajectory Network Analysis Tool
============================================================

============================================================
Analyzing: exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)
============================================================
Analysis Progress:   0%|          | 0/6 [00:00<?, ?step/s]

Loading offspring records: 100%|██████████| 96/96 [00:00<00:00, 18856.10record/s]
Loading fitness scores: 100%|██████████| 100/100 [00:02<00:00, 40.18file/s]
Linking code files: 100%|██████████| 100/100 [00:00<00:00, 117257.59file/s]

[OK] Loaded 99 nodes and 129 edges
[OK] Built STN graph

--- Basic Metrics ---
  Nodes: 99
  Edges: 129
  DAG: False
  Connected Components: 3
  Max Depth: 8

--- Operator Analysis ---
  Counts: {'crossover': 66, 'refine': 41, 'mutation': 22, 'init': 3}
  Most Effective: refine
    mutation: 22 uses, 22.7% success, μ_imp=-0.1558
    crossover: 66 uses, 33.3% success, μ_imp=-0.0623
    refine: 41 uses, 46.3% success, μ_imp=-0.0479
    init: 3 uses, 0.0% success, μ_imp=0.0000

Creating visualizations: 100%|██████████| 6/6 [00:10<00:00, 1.82s/viz]

    [OK] STN full: stn_full.png
    [OK] Fitness trajectory: fitness_trajectory.png
    [OK] Operator analysis: operator_analysis.png
    [OK] Diversity metrics: diversity_metrics.png
    [OK] Lineage tree: lineage_tree.png
    [OK] Interactive HTML: stn_interactive.html
```

## Results from Test Run

### Experiment: `exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)`

**Total Analysis Time: ~13 seconds**

#### Timing Breakdown:

- **Step 1 (Loading data)**: ~2.6 seconds
  - Loading 96 offspring records
  - Loading 100 fitness score files
  - Linking 100 code files
- **Step 2 (Building graph)**: <1 second
  - Created graph with 99 nodes and 129 edges
- **Step 3 (Calculating metrics)**: <1 second
  - Basic metrics, operator analysis, trajectory analysis
  - Fitness landscape, diversity analysis, bandit analysis
- **Step 4 (Creating visualizations)**: ~10 seconds
  - Generated 6 visualizations (PNG + HTML)
- **Steps 5-6 (Saving outputs)**: <1 second

#### Generated Files (Total: 10 files, ~1.9 MB):

- `stn_full.png` (1.1 MB) - Complete STN visualization
- `fitness_trajectory.png` (92 KB) - Fitness over generations
- `operator_analysis.png` (115 KB) - Operator effectiveness
- `diversity_metrics.png` (94 KB) - Diversity over time
- `lineage_tree.png` (263 KB) - Successful lineages
- `stn_interactive.html` (87 KB) - Interactive network
- `stn_metrics.json` (6.5 KB) - All metrics
- `generation_stats.json` (1.1 KB) - Per-generation stats
- `successful_lineages.json` (7.4 KB) - Top lineages
- `stn_graph.json` (105 KB) - Graph data

## Benefits

1. **Time Estimation**: Users can see estimated completion time for each step
2. **Progress Visibility**: Real-time progress bars show exactly what's happening
3. **Performance Metrics**: Shows processing speed (records/sec, files/sec, etc.)
4. **Better UX**: Users know the tool is working and approximately how long to wait
5. **Debugging**: Easier to identify bottlenecks in the analysis pipeline

## Usage

```bash
# Single experiment analysis with progress tracking
python stn_analyzer.py --exp-dir "path/to/experiment" --interactive

# Batch analysis (progress shown for each experiment)
python stn_analyzer.py --batch "path/to/experiments/folder"

# Quiet mode (disables progress bars)
python stn_analyzer.py --exp-dir "path/to/experiment" --quiet
```

## Technical Details

### Progress Bar Libraries

- Uses `tqdm` library for cross-platform progress bars
- Works in Windows PowerShell, CMD, and Unix terminals
- Automatically adapts to terminal width
- Shows percentage, bar, count, rate, and ETA

### Implementation Pattern

```python
from tqdm import tqdm

# For iterating over items
for item in tqdm(items, desc="Processing", unit="item"):
    process(item)

# For tracking major steps
with tqdm(total=steps, desc="Progress", unit="step") as pbar:
    do_step_1()
    pbar.update(1)
    do_step_2()
    pbar.update(1)
```

## Future Enhancements

Potential improvements for progress tracking:

1. Add ETA for total analysis time
2. Progress bars for batch analysis of multiple experiments
3. Memory usage monitoring
4. Parallel processing with multiple progress bars
5. Save timing statistics to metrics file
6. Web-based progress dashboard for long-running analyses

## Testing

Successfully tested on:

- **OS**: Windows 10
- **Python**: 3.11
- **Experiment**: exp-12-13_012735 (99 nodes, 129 edges)
- **Duration**: ~13 seconds total
- **All outputs**: Generated successfully

---

**Date**: December 14, 2025
**Version**: 1.0.0
**Status**: ✅ Complete and tested








