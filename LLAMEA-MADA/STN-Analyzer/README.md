# STN Analyzer for MADA-LLAMEA

**Search Trajectory Networks (STN) Analysis Tool**

A standalone tool for analyzing and visualizing the evolutionary behavior of MADA-LLAMEA experiments using Search Trajectory Networks.

## Overview

This tool implements STN analysis based on the research paper:

> "Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics" (Applied Soft Computing, 2020)

### Key Features

- **Post-hoc Analysis**: Works with existing experiment results—no new experiments required
- **Standalone Design**: Completely separate from the main MADA-LLAMEA codebase
- **Comprehensive Metrics**: Basic graph, operator, trajectory, fitness, and diversity analysis
- **Rich Visualizations**: Static plots (PNG) and interactive HTML visualizations
- **Batch Processing**: Analyze multiple experiments at once with comparison summaries

## Installation

```bash
# Navigate to the STN-Analyzer directory
cd STN-Analyzer

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Analyze a Single Experiment

```bash
python stn_analyzer.py --exp-dir "../exp-12-14_100247-google-gemini-2.5-flash-mada-v2-experiment-evolutionary"
```

### Generate Interactive HTML Visualization

```bash
python stn_analyzer.py --exp-dir "../exp-folder" --interactive
```

### Batch Analyze All Experiments

```bash
python stn_analyzer.py --batch ".." --output-dir analysis_results
```

## Data Requirements

The analyzer reads data from MADA-LLAMEA experiment directories:

```
experiment_dir/
├── mada_offspring.jsonl     # PRIMARY: Parent-child relationships, fitness, operators
├── bandit_snapshots.jsonl   # Optional: D-TS bandit state history
├── try-X-aucs.txt           # Fitness scores per algorithm
├── code/
│   ├── try-0-AlgorithmName.py
│   └── ...
└── conversationlog.txt      # Optional: LLM conversation log
```

### Key Data Fields (from mada_offspring.jsonl)

| Field               | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `attempt`           | Algorithm attempt number                              |
| `operator`          | How algorithm was created (mutation/crossover/refine) |
| `fitness`           | Mean AOCC score                                       |
| `parent_ids`        | List of parent algorithm IDs                          |
| `generation`        | Generation number                                     |
| `nn_dist`           | Nearest-neighbor distance (behavioral diversity)      |
| `alpha`             | Diversity weight parameter                            |
| `raw_reward`        | Raw bandit reward                                     |
| `normalized_reward` | Normalized bandit reward                              |
| `fitness_delta`     | Fitness improvement over parent                       |
| `diversity_bonus`   | Diversity component of reward                         |

## Output Structure

```
stn_outputs/
├── experiment_name/
│   ├── stn_full.png              # Complete network visualization
│   ├── fitness_trajectory.png    # Fitness over generations
│   ├── operator_analysis.png     # Operator effectiveness
│   ├── diversity_metrics.png     # MADA diversity analysis
│   ├── lineage_tree.png          # Successful lineages
│   ├── stn_interactive.html      # Interactive network (if --interactive)
│   ├── stn_metrics.json          # All computed metrics
│   ├── stn_graph.json            # Graph data for reuse
│   ├── generation_stats.json     # Per-generation statistics
│   └── successful_lineages.json  # Top successful algorithm lineages
├── batch_summary.json            # Batch analysis summary
└── comparison_summary.json       # Cross-experiment comparison
```

## Metrics Reference

### Basic Metrics

- `num_nodes`: Total algorithms discovered
- `num_edges`: Parent-child relationships
- `density`: Graph density
- `is_dag`: Whether graph is acyclic
- `num_connected_components`: Separate evolution lines
- `max_depth`: Deepest lineage

### Operator Analysis

- `operator_counts`: Usage frequency per operator
- `success_rate`: Proportion producing fitness improvement
- `mean_improvement`: Average fitness delta
- `mean_fitness_produced`: Average fitness of offspring

### Trajectory Analysis

- `num_roots`: Starting algorithms (init)
- `num_leaves`: Terminal algorithms
- `avg_path_length`: Average lineage length
- `branching_factor`: Average children per node
- `dead_end_ratio`: Proportion of terminal nodes

### Fitness Landscape

- `fitness_mean/std/max/min`: Distribution statistics
- `best_node`: Highest fitness algorithm
- `fitness_improvement_trend`: Whether later generations improve
- `error_rate`: Proportion with execution errors

### Diversity Analysis (MADA-specific)

- `mean_nn_dist`: Average behavioral diversity
- `nn_dist_over_generations`: Diversity trend
- `alpha_over_generations`: Diversity weight decay
- `mean_diversity_bonus`: Contribution to rewards

## Visualization Guide

### STN Full Network

- **Node Size**: Proportional to fitness
- **Node Color**: Generation (viridis colormap)
- **Edge Color**: Operator type (red=mutation, teal=crossover, blue=refine)

### Fitness Trajectory

- Left panel: Scatter plot of fitness vs generation
- Right panel: Box plot of fitness by operator

### Operator Analysis

- Usage counts, success rates, mean improvement, mean fitness

### Interactive HTML

- Hover for detailed node/edge information
- Drag nodes to explore network structure
- Physics simulation for organic layout

## API Usage

```python
from stn.data_loader import ExperimentDataLoader
from stn.graph_builder import STNGraphBuilder
from stn.metrics import STNMetrics
from stn.visualizer import STNVisualizer
from pathlib import Path

# Load experiment data
loader = ExperimentDataLoader(Path("experiment_dir"))
nodes, edges = loader.load()

# Build graph
builder = STNGraphBuilder()
graph = builder.build_graph(nodes, edges)

# Calculate metrics
metrics = STNMetrics(graph)
basic = metrics.basic_metrics()
operators = metrics.operator_analysis()
fitness = metrics.fitness_landscape_analysis()
lineages = metrics.identify_successful_lineages()

# Generate visualizations
viz = STNVisualizer(graph, Path("outputs"))
viz.plot_full_stn()
viz.plot_fitness_trajectory()
viz.create_interactive_html()

# Print summary
metrics.print_summary()
```

## STN Theory Background

Search Trajectory Networks represent metaheuristic search as a directed graph:

1. **Nodes** = Unique solutions/algorithms discovered
2. **Edges** = Transitions between solutions (parent → offspring)
3. **Edge Attributes** = Operator used, fitness change

### Insights from STN Analysis

- **Operator Effectiveness**: Which operators drive improvement?
- **Convergence Patterns**: How does search converge to solutions?
- **Diversity Dynamics**: Is sufficient exploration maintained?
- **Lineage Success**: What paths lead to top solutions?
- **Dead Ends**: Where does search stagnate?

## Troubleshooting

### No data loaded

- Ensure `mada_offspring.jsonl` exists in experiment directory
- Check that experiment completed successfully

### Visualization errors

- Install matplotlib: `pip install matplotlib`
- Install pyvis for interactive: `pip install pyvis`

### Interactive HTML not opening

- Open the `.html` file directly in a web browser
- Check file path in console output

## License

Part of the MADA-LLAMEA project.

## Citation

If you use this tool, please cite:

```bibtex
@article{stn2020,
  title={Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics},
  journal={Applied Soft Computing},
  year={2020}
}
```








