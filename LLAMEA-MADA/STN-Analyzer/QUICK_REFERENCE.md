# STN Analyzer - Quick Reference

## What Changed?

I added **progress tracking** to the STN-Analyzer so you can see exactly how long the analysis takes and what's currently happening.

## Progress Bars Added

### 1. Main Analysis Steps (6 steps)

Shows overall progress through the analysis pipeline:

- ✓ Step 1/6: Loading data
- ✓ Step 2/6: Building graph
- ✓ Step 3/6: Calculating metrics
- ✓ Step 4/6: Creating visualizations
- ✓ Step 5/6: Saving metrics
- ✓ Step 6/6: Saving lineages

### 2. Data Loading (detailed sub-steps)

- Loading offspring records (shows: X/Y records)
- Loading fitness scores (shows: X/Y files)
- Linking code files (shows: X/Y files)

### 3. Visualization Generation (6 visualizations)

- Full STN plot
- Fitness trajectory
- Operator analysis
- Diversity metrics
- Lineage tree
- Interactive HTML

## Example Output

```
Analysis Progress:  50%|█████     | 3/6 [00:02<00:07, 2.59s/step]

Loading offspring records: 100%|██████████| 96/96 [00:00<00:00, 18856.10record/s]
Loading fitness scores: 100%|██████████| 100/100 [00:02<00:00, 40.18file/s]

Creating visualizations:  67%|██████▋   | 4/6 [00:09<00:03, 1.87s/viz]
```

## How Long Does It Take?

Based on your experiment (exp-12-13_012735):

- **Total time**: ~13 seconds
- **Loading data**: ~2.6 seconds (96 records, 100 files)
- **Building graph**: <1 second (99 nodes, 129 edges)
- **Calculating metrics**: <1 second
- **Creating visualizations**: ~10 seconds (6 visualizations)
- **Saving outputs**: <1 second

**Tip**: The visualization step takes the longest because it generates high-quality PNG images.

## What Gets Created?

### Output Directory Structure:

```
stn_outputs/
└── exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)/
    ├── stn_full.png (1.1 MB)              # Complete network graph
    ├── fitness_trajectory.png (92 KB)      # Fitness over time
    ├── operator_analysis.png (115 KB)      # Which operators work best
    ├── diversity_metrics.png (94 KB)       # Diversity tracking
    ├── lineage_tree.png (263 KB)          # Evolution of best solutions
    ├── stn_interactive.html (87 KB)       # Interactive explorer
    ├── stn_metrics.json (6.5 KB)          # All numerical metrics
    ├── generation_stats.json (1.1 KB)     # Per-generation details
    ├── successful_lineages.json (7.4 KB)  # Best algorithm lineages
    └── stn_graph.json (105 KB)            # Raw graph data
```

## Key Metrics from Your Experiment

### Basic Stats:

- **Nodes**: 99 algorithms
- **Edges**: 129 transitions
- **Generations**: 8 levels deep
- **Connected Components**: 3 separate lineages

### Operator Effectiveness:

- **Best operator**: `refine` (46.3% success rate)
- **Most used**: `crossover` (66 uses)
- **Mutation**: 22 uses (22.7% success)

### Quality:

- **Best fitness**: 0.6086 (algorithm mada_000089)
- **Mean fitness**: 0.3981
- **Error rate**: 6.1%
- **Improvement trend**: ✅ Yes (fitness improves over generations)

### Diversity:

- **Mean NN-Distance**: 0.0791 (good diversity maintained)
- **Mean Alpha**: 0.6000

### Search Behavior:

- **Roots**: 3 starting algorithms
- **Leaves**: 77 terminal algorithms
- **Avg path length**: 4.16 generations
- **Branching factor**: 5.86 (high exploration)
- **Dead-end ratio**: 77.8% (many explorations didn't continue)

### Successful Lineages:

- Found **10 successful lineages** leading to top-performing algorithms

## Usage Examples

### Analyze single experiment (what you just did):

```bash
cd "c:\Users\Kukoy\Documents\MADA-LLAMEA-IMPLEMENTATION\LLAMEA-THESIS\LLAMEA-MADA"
python "STN-Analyzer/stn_analyzer.py" --exp-dir "exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)" --interactive
```

### Analyze without interactive HTML (faster):

```bash
python "STN-Analyzer/stn_analyzer.py" --exp-dir "path/to/experiment"
```

### Batch analyze all experiments in a folder:

```bash
python "STN-Analyzer/stn_analyzer.py" --batch "."
```

### Quiet mode (minimal output):

```bash
python "STN-Analyzer/stn_analyzer.py" --exp-dir "path/to/experiment" --quiet
```

## Next Steps

1. **View the visualizations**:

   - Open `stn_full.png` to see the complete network
   - Open `stn_interactive.html` in a browser for interactive exploration
   - Check `operator_analysis.png` to understand which operators work best

2. **Analyze the metrics**:

   - Open `stn_metrics.json` for comprehensive numerical analysis
   - Check `successful_lineages.json` to see the evolution of best algorithms

3. **Compare experiments**:
   - Run the analyzer on multiple experiments
   - Compare their metrics to see which configuration works best

## Troubleshooting

### Progress bars not showing?

- Make sure you're not using `--quiet` flag
- Some terminal emulators may not support ANSI escape codes

### Analysis taking too long?

- Skip interactive HTML: don't use `--interactive` flag
- Expected times:
  - Small experiments (<50 algorithms): 5-10 seconds
  - Medium experiments (50-200 algorithms): 10-20 seconds
  - Large experiments (>200 algorithms): 20-40 seconds

### Missing dependencies?

```bash
pip install tqdm
# or install all dependencies:
pip install -r STN-Analyzer/requirements.txt
```

---

**Enjoy the new progress tracking! 🎉**

You can now see exactly how long your STN analysis takes and what's happening at each step.








