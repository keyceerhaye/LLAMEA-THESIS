#!/usr/bin/env python3
"""
Quick script to regenerate the interactive HTML from existing STN graph JSON
"""

import sys
import json
import networkx as nx
from pathlib import Path

# Add parent directory to path to import stn modules
sys.path.insert(0, str(Path(__file__).parent))

from stn.visualizer import STNVisualizer

def regenerate_html_from_json(output_dir: str):
    """Regenerate the interactive HTML from existing STN graph JSON."""
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    print(f"\nRegenerating interactive HTML for: {output_path.name}")
    print("=" * 80)
    
    # Load graph from JSON
    graph_file = output_path / 'stn_graph.json'
    
    if not graph_file.exists():
        print(f"Error: Graph file not found: {graph_file}")
        return
    
    print("\nLoading graph from JSON...")
    with open(graph_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    # Create NetworkX graph from JSON
    graph = nx.node_link_graph(graph_data, directed=True, edges="edges")
    
    print(f"   Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create visualizer
    print("\nCreating enhanced interactive visualization...")
    viz = STNVisualizer(graph, output_path)
    
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
        print("Usage: python regenerate_html_from_json.py <output_directory>")
        print("\nExample:")
        print('  python regenerate_html_from_json.py "../stn_outputs/exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)"')
        sys.exit(1)
    
    output_dir = sys.argv[1]
    regenerate_html_from_json(output_dir)








