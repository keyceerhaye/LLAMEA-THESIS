# STN (Search Trajectory Networks) Implementation Plan for MADA-LLAMEA

## Executive Summary

This document outlines a comprehensive plan to implement Search Trajectory Networks (STN) visualization and analysis for the MADA-LLAMEA algorithm discovery system. **The implementation will be a separate standalone program** that works with existing experiment results without modifying the main codebase.

---

## 1. What is STN?

Search Trajectory Networks (STN) are graph-based representations that capture and visualize the behavior of metaheuristic algorithms. Based on the research paper "Search trajectory networks: A tool for analysing and visualising the behaviour of metaheuristics":

### Key Concepts:

- **Nodes**: Represent distinct solutions/algorithms discovered during the search
- **Edges**: Represent transitions between solutions (parent → offspring relationships)
- **Edge Weights**: Can represent fitness improvement, operator used, or frequency
- **Network Properties**: Provide insights into search dynamics, diversity, and convergence

### STN Benefits for MADA-LLAMEA:

1. Visualize how algorithms evolve over generations
2. Identify which operators (mutation, crossover, refine) lead to improvements
3. Detect evolutionary dead-ends vs. productive search paths
4. Compare search behaviors across different LLM configurations
5. Understand algorithm family trees and lineage

---

## 2. Can STN Work with Existing Experiment Results?

### ✅ **YES - STN can work entirely with existing results!**

Your experiment data already contains all necessary information:

### Available Data in Experiment Results:

```
exp-12-14_100247-google-gemini-2.5-flash-mada-v2-experiment-evolutionary/
├── code/
│   ├── try-0-AdaptiveGaussianMixtureModelSearch.py    # Algorithm code
│   ├── try-1-AdaptiveGaussianSearch.py
│   └── ...
├── mada_offspring.jsonl    # ← KEY: Contains parent-child relationships
├── try-0-aucs.txt          # Fitness scores
├── try-1-aucs.txt
└── conversationlog.txt
```

### Data Structure in `mada_offspring.jsonl`:

```json
{
  "attempt": 4,
  "operator": "mutation", // Operator used (mutation/crossover/refine)
  "fitness": 0.0, // Offspring fitness
  "parent_fitness": 0.229, // Parent fitness
  "parent_ids": ["mada_000003"], // Parent algorithm ID(s)
  "generation": 1, // Generation number
  "nn_dist": 1.0, // Nearest neighbor distance (diversity)
  "alpha": 0.8, // Alpha parameter
  "raw_reward": -1.0,
  "normalized_reward": -1.0,
  "fitness_delta": 0.0,
  "diversity_bonus": 0.0,
  "theta_sampled": 0.328
}
```

### What We Need to Extract:

1. **Nodes**: Each unique algorithm (identified by attempt/mada_id)
2. **Edges**: Parent → Child relationships from `parent_ids`
3. **Node Attributes**: Fitness, generation, operator used
4. **Edge Attributes**: Fitness improvement, operator type

---

## 3. Implementation Architecture

### 3.1 Standalone Program Structure

```
STN-Analyzer/
├── stn_analyzer.py           # Main entry point
├── stn/
│   ├── __init__.py
│   ├── data_loader.py        # Load experiment results
│   ├── graph_builder.py      # Build NetworkX graphs
│   ├── metrics.py            # STN metrics and analysis
│   ├── visualizer.py         # Graph visualization
│   └── code_features.py      # Optional: Extract code features
├── config/
│   └── stn_config.yaml       # Configuration
├── outputs/                  # Generated visualizations
├── requirements.txt
└── README.md
```

### 3.2 Core Dependencies

```txt
# requirements.txt
networkx>=3.0
matplotlib>=3.7
seaborn>=0.12
pandas>=2.0
numpy>=1.24
plotly>=5.15          # Interactive visualizations
pyvis>=0.3            # Interactive network visualization
graphviz>=0.20        # Layout algorithms
scikit-learn>=1.3     # For dimensionality reduction
pyyaml>=6.0
```

---

## 4. Implementation Phases

### Phase 1: Data Loading & Graph Construction (Core)

**Goal**: Parse experiment results and build STN graphs

```python
# stn/data_loader.py

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class AlgorithmNode:
    """Represents a single algorithm/solution in the STN."""
    node_id: str
    attempt: int
    generation: int
    fitness: float
    operator: str  # 'mutation', 'crossover', 'refine', 'initial'
    parent_ids: List[str]
    code_file: Optional[Path] = None

    # Additional attributes
    nn_dist: float = 0.0
    fitness_delta: float = 0.0

@dataclass
class STNEdge:
    """Represents a transition in the search trajectory."""
    source: str
    target: str
    operator: str
    fitness_improvement: float
    generation_step: int

class ExperimentDataLoader:
    """Load and parse MADA-LLAMEA experiment data."""

    def __init__(self, experiment_dir: Path):
        self.exp_dir = Path(experiment_dir)
        self.nodes: Dict[str, AlgorithmNode] = {}
        self.edges: List[STNEdge] = []

    def load(self) -> Tuple[Dict[str, AlgorithmNode], List[STNEdge]]:
        """Load all experiment data and return nodes and edges."""
        self._load_offspring_log()
        self._load_fitness_scores()
        self._link_code_files()
        return self.nodes, self.edges

    def _load_offspring_log(self):
        """Parse mada_offspring.jsonl for parent-child relationships."""
        jsonl_path = self.exp_dir / "mada_offspring.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"No offspring log found: {jsonl_path}")

        with open(jsonl_path, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                self._process_offspring_record(record)

    def _process_offspring_record(self, record: dict):
        """Process a single offspring record into node and edges."""
        node_id = f"mada_{record['attempt']:06d}"

        node = AlgorithmNode(
            node_id=node_id,
            attempt=record['attempt'],
            generation=record.get('generation', 0),
            fitness=record.get('fitness', 0.0),
            operator=record.get('operator', 'unknown'),
            parent_ids=record.get('parent_ids', []),
            nn_dist=record.get('nn_dist', 0.0),
            fitness_delta=record.get('fitness_delta', 0.0)
        )
        self.nodes[node_id] = node

        # Create edges from parents to this node
        parent_fitness = record.get('parent_fitness', 0.0)
        for parent_id in node.parent_ids:
            edge = STNEdge(
                source=parent_id,
                target=node_id,
                operator=node.operator,
                fitness_improvement=node.fitness - parent_fitness,
                generation_step=node.generation
            )
            self.edges.append(edge)

    def _load_fitness_scores(self):
        """Load detailed fitness scores from AUC files."""
        for auc_file in self.exp_dir.glob("try-*-aucs.txt"):
            attempt = int(auc_file.stem.split('-')[1])
            node_id = f"mada_{attempt:06d}"
            if node_id in self.nodes:
                # Parse AUC file for detailed fitness
                self.nodes[node_id].fitness = self._parse_auc_file(auc_file)

    def _parse_auc_file(self, auc_file: Path) -> float:
        """Parse AUC file to get fitness score."""
        try:
            with open(auc_file, 'r') as f:
                content = f.read().strip()
                # Parse based on your AUC file format
                return float(content.split()[-1])
        except:
            return 0.0

    def _link_code_files(self):
        """Link algorithm code files to nodes."""
        code_dir = self.exp_dir / "code"
        if code_dir.exists():
            for code_file in code_dir.glob("try-*.py"):
                attempt = int(code_file.stem.split('-')[1])
                node_id = f"mada_{attempt:06d}"
                if node_id in self.nodes:
                    self.nodes[node_id].code_file = code_file
```

### Phase 2: NetworkX Graph Builder

```python
# stn/graph_builder.py

import networkx as nx
from typing import Dict, List
from .data_loader import AlgorithmNode, STNEdge

class STNGraphBuilder:
    """Build NetworkX graphs from MADA experiment data."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(
        self,
        nodes: Dict[str, AlgorithmNode],
        edges: List[STNEdge]
    ) -> nx.DiGraph:
        """Build complete STN graph."""

        # Add nodes with attributes
        for node_id, node in nodes.items():
            self.graph.add_node(
                node_id,
                attempt=node.attempt,
                generation=node.generation,
                fitness=node.fitness,
                operator=node.operator,
                nn_dist=node.nn_dist,
                code_file=str(node.code_file) if node.code_file else None
            )

        # Add edges with attributes
        for edge in edges:
            if edge.source in self.graph and edge.target in self.graph:
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    operator=edge.operator,
                    fitness_improvement=edge.fitness_improvement,
                    generation_step=edge.generation_step
                )

        return self.graph

    def filter_by_operator(self, operator: str) -> nx.DiGraph:
        """Create subgraph for specific operator."""
        edges_to_keep = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get('operator') == operator
        ]
        return self.graph.edge_subgraph(edges_to_keep).copy()

    def get_improvement_paths(self) -> List[List[str]]:
        """Find paths where fitness consistently improves."""
        improvement_paths = []
        for source in self.graph.nodes():
            for target in self.graph.nodes():
                if source != target:
                    try:
                        paths = nx.all_simple_paths(self.graph, source, target)
                        for path in paths:
                            if self._is_improvement_path(path):
                                improvement_paths.append(path)
                    except nx.NetworkXNoPath:
                        continue
        return improvement_paths

    def _is_improvement_path(self, path: List[str]) -> bool:
        """Check if path shows consistent fitness improvement."""
        fitnesses = [self.graph.nodes[n]['fitness'] for n in path]
        return all(fitnesses[i] <= fitnesses[i+1] for i in range(len(fitnesses)-1))
```

### Phase 3: STN Metrics & Analysis

```python
# stn/metrics.py

import networkx as nx
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple

class STNMetrics:
    """Calculate STN metrics for analysis."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def basic_metrics(self) -> Dict:
        """Calculate basic graph metrics."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]),
        }

    def operator_analysis(self) -> Dict:
        """Analyze contribution of each operator."""
        operators = Counter()
        improvements = {op: [] for op in ['mutation', 'crossover', 'refine']}

        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            operators[op] += 1
            if op in improvements:
                improvements[op].append(data.get('fitness_improvement', 0))

        stats = {}
        for op, imps in improvements.items():
            if imps:
                stats[f'{op}_count'] = len(imps)
                stats[f'{op}_mean_improvement'] = np.mean(imps)
                stats[f'{op}_success_rate'] = sum(1 for i in imps if i > 0) / len(imps)

        return {'operator_counts': dict(operators), **stats}

    def trajectory_analysis(self) -> Dict:
        """Analyze search trajectory patterns."""
        # Find root nodes (no incoming edges)
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        # Find leaf nodes (no outgoing edges / terminal solutions)
        leaves = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

        # Calculate path lengths from roots
        path_lengths = []
        for root in roots:
            lengths = nx.single_source_shortest_path_length(self.graph, root)
            path_lengths.extend(lengths.values())

        return {
            'num_roots': len(roots),
            'num_leaves': len(leaves),
            'avg_path_length': np.mean(path_lengths) if path_lengths else 0,
            'max_path_length': max(path_lengths) if path_lengths else 0,
            'branching_factor': self.graph.number_of_edges() / max(len(roots), 1),
        }

    def fitness_landscape_analysis(self) -> Dict:
        """Analyze fitness distribution and progression."""
        fitnesses = [self.graph.nodes[n]['fitness'] for n in self.graph.nodes()]
        generations = [self.graph.nodes[n]['generation'] for n in self.graph.nodes()]

        # Fitness progression per generation
        gen_fitness = {}
        for node in self.graph.nodes():
            gen = self.graph.nodes[node]['generation']
            fit = self.graph.nodes[node]['fitness']
            if gen not in gen_fitness:
                gen_fitness[gen] = []
            gen_fitness[gen].append(fit)

        gen_stats = {
            gen: {
                'mean': np.mean(fits),
                'max': max(fits),
                'min': min(fits)
            }
            for gen, fits in gen_fitness.items()
        }

        return {
            'fitness_mean': np.mean(fitnesses),
            'fitness_std': np.std(fitnesses),
            'fitness_max': max(fitnesses),
            'best_node': max(self.graph.nodes(), key=lambda n: self.graph.nodes[n]['fitness']),
            'generation_stats': gen_stats,
        }

    def identify_successful_lineages(self) -> List[List[str]]:
        """Identify lineages that led to high-fitness solutions."""
        # Find top 10% solutions
        fitnesses = {n: self.graph.nodes[n]['fitness'] for n in self.graph.nodes()}
        threshold = np.percentile(list(fitnesses.values()), 90)
        top_nodes = [n for n, f in fitnesses.items() if f >= threshold]

        lineages = []
        for node in top_nodes:
            lineage = self._trace_lineage(node)
            lineages.append(lineage)

        return lineages

    def _trace_lineage(self, node: str) -> List[str]:
        """Trace ancestry of a node back to root."""
        lineage = [node]
        current = node
        while True:
            predecessors = list(self.graph.predecessors(current))
            if not predecessors:
                break
            # Take first parent (for simplicity)
            current = predecessors[0]
            lineage.insert(0, current)
        return lineage
```

### Phase 4: Visualization

```python
# stn/visualizer.py

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyvis.network import Network
import plotly.graph_objects as go
from typing import Optional, Dict
from pathlib import Path

class STNVisualizer:
    """Visualize Search Trajectory Networks."""

    OPERATOR_COLORS = {
        'mutation': '#FF6B6B',      # Red
        'crossover': '#4ECDC4',     # Teal
        'refine': '#45B7D1',        # Blue
        'initial': '#95E1D3',       # Light green
        'unknown': '#DDD'
    }

    def __init__(self, graph: nx.DiGraph, output_dir: Path):
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_full_stn(self, filename: str = "stn_full.png"):
        """Create static visualization of full STN."""
        plt.figure(figsize=(16, 12))

        # Layout
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Node sizes based on fitness
        fitnesses = [self.graph.nodes[n]['fitness'] for n in self.graph.nodes()]
        max_fit = max(fitnesses) if fitnesses else 1
        node_sizes = [300 + 500 * (self.graph.nodes[n]['fitness'] / max_fit)
                      for n in self.graph.nodes()]

        # Node colors based on generation
        generations = [self.graph.nodes[n]['generation'] for n in self.graph.nodes()]

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=generations,
            cmap=plt.cm.viridis,
            alpha=0.8
        )

        # Draw edges colored by operator
        for operator, color in self.OPERATOR_COLORS.items():
            edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                     if d.get('operator') == operator]
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=edges,
                edge_color=color,
                alpha=0.6,
                arrows=True,
                arrowsize=10,
                label=operator
            )

        # Add legend
        plt.legend(scatterpoints=1, frameon=True, loc='upper left')
        plt.title("Search Trajectory Network (MADA-LLAMEA)")
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis),
                     label='Generation', shrink=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        return self.output_dir / filename

    def plot_fitness_trajectory(self, filename: str = "fitness_trajectory.png"):
        """Plot fitness over generations with operator coloring."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter plot: Generation vs Fitness
        ax1 = axes[0]
        for node in self.graph.nodes():
            gen = self.graph.nodes[node]['generation']
            fit = self.graph.nodes[node]['fitness']
            op = self.graph.nodes[node]['operator']
            color = self.OPERATOR_COLORS.get(op, '#999')
            ax1.scatter(gen, fit, c=color, s=50, alpha=0.7)

        # Add legend for operators
        for op, color in self.OPERATOR_COLORS.items():
            ax1.scatter([], [], c=color, label=op, s=50)
        ax1.legend(loc='lower right')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness by Generation & Operator')
        ax1.grid(True, alpha=0.3)

        # Box plot: Fitness distribution by operator
        ax2 = axes[1]
        operator_fitness = {}
        for node in self.graph.nodes():
            op = self.graph.nodes[node]['operator']
            fit = self.graph.nodes[node]['fitness']
            if op not in operator_fitness:
                operator_fitness[op] = []
            operator_fitness[op].append(fit)

        positions = list(range(len(operator_fitness)))
        bp = ax2.boxplot(
            list(operator_fitness.values()),
            positions=positions,
            patch_artist=True
        )
        for i, (op, patch) in enumerate(zip(operator_fitness.keys(), bp['boxes'])):
            patch.set_facecolor(self.OPERATOR_COLORS.get(op, '#999'))

        ax2.set_xticks(positions)
        ax2.set_xticklabels(list(operator_fitness.keys()))
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness Distribution by Operator')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        return self.output_dir / filename

    def create_interactive_html(self, filename: str = "stn_interactive.html"):
        """Create interactive HTML visualization using pyvis."""
        net = Network(
            height="800px",
            width="100%",
            directed=True,
            bgcolor="#222222",
            font_color="white"
        )

        # Configure physics
        net.toggle_physics(True)
        net.set_options("""
        {
            "nodes": {
                "font": {"size": 12}
            },
            "edges": {
                "smooth": {"type": "curvedCW", "roundness": 0.2}
            },
            "physics": {
                "barnesHut": {"gravitationalConstant": -3000}
            }
        }
        """)

        # Add nodes
        fitnesses = [self.graph.nodes[n]['fitness'] for n in self.graph.nodes()]
        max_fit = max(fitnesses) if fitnesses else 1

        for node in self.graph.nodes():
            fit = self.graph.nodes[node]['fitness']
            gen = self.graph.nodes[node]['generation']
            op = self.graph.nodes[node]['operator']

            size = 10 + 30 * (fit / max_fit)
            color = self.OPERATOR_COLORS.get(op, '#999')

            title = f"ID: {node}<br>Fitness: {fit:.4f}<br>Gen: {gen}<br>Op: {op}"

            net.add_node(
                node,
                label=f"{node[-4:]}",
                title=title,
                size=size,
                color=color
            )

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            imp = data.get('fitness_improvement', 0)
            color = self.OPERATOR_COLORS.get(op, '#999')

            title = f"Operator: {op}<br>Improvement: {imp:.4f}"

            net.add_edge(u, v, title=title, color=color)

        output_path = self.output_dir / filename
        net.save_graph(str(output_path))

        return output_path

    def plot_lineage_tree(
        self,
        lineages: List[List[str]],
        filename: str = "lineage_tree.png"
    ):
        """Visualize successful lineages as a tree."""
        # Create subgraph with only lineage nodes and edges
        lineage_nodes = set()
        for lineage in lineages:
            lineage_nodes.update(lineage)

        subgraph = self.graph.subgraph(lineage_nodes).copy()

        plt.figure(figsize=(14, 10))

        # Use hierarchical layout
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)

        # Draw
        fitnesses = [subgraph.nodes[n]['fitness'] for n in subgraph.nodes()]

        nx.draw(
            subgraph, pos,
            with_labels=True,
            node_color=fitnesses,
            cmap=plt.cm.RdYlGn,
            node_size=500,
            font_size=8,
            arrows=True,
            arrowsize=15,
            alpha=0.8
        )

        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn),
                     label='Fitness', shrink=0.5)
        plt.title("Successful Algorithm Lineages")

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

        return self.output_dir / filename
```

### Phase 5: Main Entry Point

```python
# stn_analyzer.py

#!/usr/bin/env python3
"""
STN Analyzer for MADA-LLAMEA Experiments

A standalone tool for analyzing and visualizing Search Trajectory Networks
from MADA-LLAMEA experiment results.

Usage:
    python stn_analyzer.py --exp-dir path/to/experiment
    python stn_analyzer.py --exp-dir path/to/experiment --interactive
    python stn_analyzer.py --batch path/to/experiments/folder
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

from stn.data_loader import ExperimentDataLoader
from stn.graph_builder import STNGraphBuilder
from stn.metrics import STNMetrics
from stn.visualizer import STNVisualizer


def analyze_experiment(exp_dir: Path, output_dir: Path, interactive: bool = False):
    """Analyze a single experiment and generate STN visualizations."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {exp_dir.name}")
    print('='*60)

    # Load data
    loader = ExperimentDataLoader(exp_dir)
    try:
        nodes, edges = loader.load()
        print(f"✓ Loaded {len(nodes)} nodes and {len(edges)} edges")
    except FileNotFoundError as e:
        print(f"✗ Skipping: {e}")
        return None

    if not nodes:
        print("✗ No valid data found")
        return None

    # Build graph
    builder = STNGraphBuilder()
    graph = builder.build_graph(nodes, edges)
    print(f"✓ Built STN graph")

    # Calculate metrics
    metrics = STNMetrics(graph)
    basic = metrics.basic_metrics()
    operator_stats = metrics.operator_analysis()
    trajectory_stats = metrics.trajectory_analysis()
    fitness_stats = metrics.fitness_landscape_analysis()
    lineages = metrics.identify_successful_lineages()

    print(f"\n--- Basic Metrics ---")
    for key, val in basic.items():
        print(f"  {key}: {val}")

    print(f"\n--- Operator Analysis ---")
    for key, val in operator_stats.items():
        if isinstance(val, dict):
            print(f"  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\n--- Trajectory Analysis ---")
    for key, val in trajectory_stats.items():
        print(f"  {key}: {val:.2f}" if isinstance(val, float) else f"  {key}: {val}")

    print(f"\n--- Fitness Analysis ---")
    print(f"  Best solution: {fitness_stats['best_node']} (fitness: {fitness_stats['fitness_max']:.4f})")
    print(f"  Mean fitness: {fitness_stats['fitness_mean']:.4f}")
    print(f"  Found {len(lineages)} successful lineages")

    # Create visualizations
    exp_output = output_dir / exp_dir.name
    viz = STNVisualizer(graph, exp_output)

    stn_plot = viz.plot_full_stn()
    print(f"\n✓ Saved: {stn_plot}")

    fitness_plot = viz.plot_fitness_trajectory()
    print(f"✓ Saved: {fitness_plot}")

    if lineages:
        lineage_plot = viz.plot_lineage_tree(lineages[:5])  # Top 5 lineages
        print(f"✓ Saved: {lineage_plot}")

    if interactive:
        html_viz = viz.create_interactive_html()
        print(f"✓ Saved interactive: {html_viz}")

    # Save metrics to JSON
    all_metrics = {
        'basic': basic,
        'operator': operator_stats,
        'trajectory': trajectory_stats,
        'fitness': {k: v for k, v in fitness_stats.items() if k != 'generation_stats'},
        'best_node': fitness_stats['best_node'],
        'num_successful_lineages': len(lineages)
    }

    metrics_file = exp_output / "stn_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"✓ Saved: {metrics_file}")

    return all_metrics


def batch_analyze(experiments_dir: Path, output_dir: Path, interactive: bool = False):
    """Analyze multiple experiments."""
    exp_dirs = [d for d in experiments_dir.iterdir()
                if d.is_dir() and d.name.startswith('exp-')]

    print(f"Found {len(exp_dirs)} experiment directories")

    all_results = {}
    for exp_dir in sorted(exp_dirs):
        result = analyze_experiment(exp_dir, output_dir, interactive)
        if result:
            all_results[exp_dir.name] = result

    # Summary report
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Batch analysis complete. Analyzed {len(all_results)} experiments.")
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="STN Analyzer for MADA-LLAMEA Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python stn_analyzer.py --exp-dir exp-12-14_100247-google-gemini-2.5-flash-mada-v2-experiment-evolutionary

  # Analyze with interactive HTML output
  python stn_analyzer.py --exp-dir path/to/exp --interactive

  # Batch analyze all experiments in a folder
  python stn_analyzer.py --batch LLAMEA-THESIS/LLAMEA-MADA
        """
    )

    parser.add_argument(
        '--exp-dir',
        type=Path,
        help='Path to single experiment directory'
    )
    parser.add_argument(
        '--batch',
        type=Path,
        help='Path to folder containing multiple experiment directories'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('stn_outputs'),
        help='Output directory for visualizations (default: stn_outputs)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive HTML visualizations'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.exp_dir:
        analyze_experiment(args.exp_dir, args.output_dir, args.interactive)
    elif args.batch:
        batch_analyze(args.batch, args.output_dir, args.interactive)
    else:
        parser.print_help()
        print("\nError: Please specify --exp-dir or --batch")


if __name__ == "__main__":
    main()
```

---

## 5. Running Without New Experiments

### The Key Question: Can STN work with existing results only?

**YES!** Here's why:

### Data Already Available:

1. ✅ **Algorithm IDs** - from attempt numbers and mada_000XXX format
2. ✅ **Parent-Child relationships** - from `parent_ids` in mada_offspring.jsonl
3. ✅ **Fitness values** - from fitness field and try-X-aucs.txt files
4. ✅ **Operator used** - mutation, crossover, refine tracked per offspring
5. ✅ **Generation info** - tracked in offspring records
6. ✅ **Code files** - available in code/ subdirectory

### What You DON'T Need to Re-run:

- No need to re-execute algorithms
- No need for new LLM calls
- No need for new benchmark evaluations

### Optional Enhancements (Requiring Code Analysis):

If you want richer STN visualizations, you could optionally:

- Extract AST features from code files (complexity, structure)
- Compute code similarity between algorithms
- These are POST-HOC analyses, not new experiments

---

## 6. Extended Features (Future Work)

### 6.1 Code Similarity Analysis

```python
# stn/code_features.py - Optional extension

import ast
from difflib import SequenceMatcher

def calculate_code_similarity(code1: str, code2: str) -> float:
    """Calculate similarity between two algorithm codes."""
    return SequenceMatcher(None, code1, code2).ratio()

def extract_code_features(code_path: Path) -> Dict:
    """Extract structural features from algorithm code."""
    with open(code_path, 'r') as f:
        code = f.read()

    tree = ast.parse(code)

    return {
        'num_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
        'num_classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
        'num_loops': len([n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]),
        'code_length': len(code),
        'num_numpy_calls': code.count('np.'),
    }
```

### 6.2 Comparative Analysis Across Experiments

```python
def compare_experiments(exp_dirs: List[Path]) -> Dict:
    """Compare STN metrics across multiple experiments."""
    results = {}
    for exp_dir in exp_dirs:
        loader = ExperimentDataLoader(exp_dir)
        nodes, edges = loader.load()
        builder = STNGraphBuilder()
        graph = builder.build_graph(nodes, edges)
        metrics = STNMetrics(graph)
        results[exp_dir.name] = metrics.basic_metrics()
    return results
```

### 6.3 Operator Effectiveness Over Time

```python
def analyze_operator_evolution(graph: nx.DiGraph) -> pd.DataFrame:
    """Track operator effectiveness across generations."""
    data = []
    for u, v, attr in graph.edges(data=True):
        data.append({
            'generation': graph.nodes[v]['generation'],
            'operator': attr['operator'],
            'improvement': attr['fitness_improvement'],
        })
    return pd.DataFrame(data)
```

---

## 7. Quick Start Commands

```bash
# 1. Create the STN analyzer directory
mkdir STN-Analyzer
cd STN-Analyzer

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install networkx matplotlib seaborn pandas numpy plotly pyvis pyyaml scikit-learn

# 4. Create the package structure
mkdir stn config outputs

# 5. Run analysis on existing experiment
python stn_analyzer.py --exp-dir "..\LLAMEA-THESIS\LLAMEA-MADA\exp-12-14_100247-google-gemini-2.5-flash-mada-v2-experiment-evolutionary" --interactive
```

---

## 8. Summary

| Aspect                      | Status                    |
| --------------------------- | ------------------------- |
| Works with existing results | ✅ Yes                    |
| Requires new experiments    | ❌ No                     |
| Separate from main code     | ✅ Standalone program     |
| Modifies main programs      | ❌ No modifications       |
| Uses mada_offspring.jsonl   | ✅ Primary data source    |
| Generates visualizations    | ✅ Static + Interactive   |
| Provides metrics            | ✅ Comprehensive analysis |

The STN implementation is designed to be a **post-hoc analysis tool** that reads existing experiment results and generates insightful visualizations and metrics about the algorithm discovery process, without interfering with the main MADA-LLAMEA implementation.








