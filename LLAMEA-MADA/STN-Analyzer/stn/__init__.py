"""
STN (Search Trajectory Networks) Package for MADA-LLAMEA Analysis

This package provides tools for analyzing and visualizing Search Trajectory Networks
from MADA-LLAMEA experiment results.

Based on: "Search trajectory networks: A tool for analysing and visualising 
          the behaviour of metaheuristics" (Applied Soft Computing, 2020)

Components:
- data_loader: Load and parse experiment results
- graph_builder: Build NetworkX graphs from algorithm lineages
- metrics: Calculate STN metrics and analysis
- visualizer: Create static and interactive visualizations
"""

from .data_loader import (
    AlgorithmNode,
    STNEdge,
    ExperimentDataLoader,
)
from .graph_builder import STNGraphBuilder
from .metrics import STNMetrics
from .visualizer import STNVisualizer

__version__ = "1.0.0"
__all__ = [
    "AlgorithmNode",
    "STNEdge",
    "ExperimentDataLoader",
    "STNGraphBuilder",
    "STNMetrics",
    "STNVisualizer",
]








