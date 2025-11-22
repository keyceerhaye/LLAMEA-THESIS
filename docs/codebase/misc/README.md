# misc/ Directory

## Purpose
Collection of miscellaneous utility scripts for analysis, visualization, and post-processing of LLaMEA experiments.

## Overview
Contains tools for analyzing experimental results, processing code, visualizing evolutionary graphs, and working with IOH data. These are support tools not part of the core framework.

## Key Files

### `__init__.py`
Empty module initializer.

### `ast.py`
Abstract Syntax Tree (AST) utilities for Python code analysis.
- Parse and analyze generated algorithm code
- Extract structural features
- Compute code complexity metrics
- **Use case:** Understanding algorithm structure

### `python_ast_analysis.py`
Extended AST analysis specific to Python algorithms.
- Detailed code structure analysis
- Function/class extraction
- Dependency analysis
- **Use case:** Algorithm characterization research

### `iohrun.py`
Utilities for running IOH experimenter evaluations.
- Batch IOH experiment execution
- Result aggregation
- Performance analysis helpers
- **Use case:** Systematic benchmarking

### `plot_aucs.py`
Visualization tools for AUC (Area Under Curve) data.
- Plot AUC trajectories over generations
- Compare multiple algorithms
- Statistical visualization (mean, std, confidence intervals)
- **Use case:** Result visualization for papers/presentations

### `transform_to_stn.py`
Transforms code/data to STN (Semantic Tree Network) format.
- Code representation transformation
- Semantic analysis
- Graph-based code representation
- **Use case:** Advanced code analysis, machine learning on code

### `visualize_graphs.py`
Graph visualization tools.
- Evolutionary tree visualization
- Population diversity plots
- Genealogy graphs
- Niching visualization
- **Use case:** Understanding evolutionary dynamics

### `utils.py`
General utility functions for miscellaneous tasks.
- File I/O helpers
- Data processing
- Common operations not fitting elsewhere
- **Use case:** Support for other misc/ scripts

## Typical Usage

### Analyzing Experiment Results
```python
# After running experiment
from misc.plot_aucs import plot_auc_trajectory

aucs = read_experiment_aucs("exp-{date}-{name}/")
plot_auc_trajectory(aucs, save_path="results.png")
```

### Code Structure Analysis
```python
from misc.ast import analyze_code_structure

with open("exp-{date}/code/try-0-BestAlg.py") as f:
    code = f.read()

analysis = analyze_code_structure(code)
print(f"Complexity: {analysis['complexity']}")
print(f"Functions: {analysis['functions']}")
```

### Visualizing Evolution
```python
from misc.visualize_graphs import plot_genealogy

log_data = read_jsonl("exp-{date}/log.jsonl")
plot_genealogy(log_data, save_path="evolution.png")
```

## Dependencies
- `ast` (builtin) - Python AST parsing
- `networkx` - Graph structures and algorithms
- `matplotlib` / `seaborn` - Visualization
- `numpy`, `pandas` - Data processing
- `lizard` - Code complexity analysis
- `ioh` - IOH experimenter integration

## Integration Points
- Reads experiment logs from main scripts
- Processes IOH data files
- Visualizes LLaMEA evolution dynamics
- Supports research analysis workflows

## Status
**Note:** These are research/analysis scripts, not production code. May require adaptation for specific use cases.

## When to Use
- ✅ Post-experiment analysis
- ✅ Paper/presentation visualizations
- ✅ Understanding evolutionary dynamics
- ✅ Algorithm characterization
- ✅ Research explorations

## When NOT to Use
- ❌ During LLaMEA runtime (separate concerns)
- ❌ Production systems (research-quality code)
- ❌ Real-time monitoring (use logreader/ app instead)


