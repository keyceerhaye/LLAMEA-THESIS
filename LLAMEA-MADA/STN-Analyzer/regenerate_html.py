#!/usr/bin/env python3
"""
Quick script to regenerate the interactive HTML with improved visualization
"""

import sys
from pathlib import Path

# Add parent directory to path to import stn modules
sys.path.insert(0, str(Path(__file__).parent))

from stn.data_loader import ExperimentDataLoader
from stn.graph_builder import STNGraphBuilder
from stn.visualizer import STNVisualizer

def regenerate_html(exp_dir: str):
    """Regenerate the interactive HTML for an experiment."""
    
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        return
    
    print(f"\nRegenerating interactive HTML for: {exp_path.name}")
    print("=" * 80)
    
    # Find output directory
    output_dir = exp_path.parent / 'stn_outputs' / exp_path.name
    
    # Load data
    print("\nLoading experiment data...")
    loader = ExperimentDataLoader(exp_path)
    nodes_dict, edges_list = loader.load()
    
    if not nodes_dict:
        print("No data found in experiment directory")
        return
    
    print(f"   Loaded {len(nodes_dict)} nodes, {len(edges_list)} edges")
    
    # Build graph
    print("\nBuilding search trajectory network...")
    builder = STNGraphBuilder()
    graph = builder.from_data(nodes_dict, edges_list)
    
    print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create visualizer
    print("\nCreating enhanced interactive visualization...")
    viz = STNVisualizer(graph, output_dir)
    
    # Generate HTML
    html_path = viz.create_interactive_html(filename="stn_interactive.html")
    
    if html_path:
        print(f"\nInteractive HTML created: {html_path}")
        print(f"\nOpen in browser: file:///{html_path.absolute()}")
        print("\n" + "=" * 80)
        print("The visualization now includes:")
        print("   * Comprehensive legend with operator colors")
        print("   * Network statistics panel")
        print("   * Operator performance metrics")
        print("   * Enhanced tooltips with detailed algorithm info")
        print("   * Interactive controls (zoom, pan, physics toggle)")
        print("   * Node size based on fitness")
        print("   * Edge width based on improvement magnitude")
        print("=" * 80)
    else:
        print("Failed to generate HTML")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python regenerate_html.py <experiment_directory>")
        print("\nExample:")
        print("  python regenerate_html.py ../stn_outputs/exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    regenerate_html(exp_dir)








