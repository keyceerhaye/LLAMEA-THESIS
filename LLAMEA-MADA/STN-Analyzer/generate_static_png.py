#!/usr/bin/env python3
"""
Generate professional static PNG visualization from existing STN graph JSON
"""

import sys
import json
import networkx as nx
from pathlib import Path

# Add parent directory to path to import stn modules
sys.path.insert(0, str(Path(__file__).parent))

from stn.visualizer import STNVisualizer

def generate_static_png(output_dir: str):
    """Generate professional static PNG from existing STN graph JSON."""
    
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    print(f"\nGenerating professional static PNG for: {output_path.name}")
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
    print("\nCreating professional static PNG visualization...")
    viz = STNVisualizer(graph, output_path)
    
    # Generate static PNG
    png_path = viz.plot_static_stn_professional(filename="stn_static.png")
    
    if png_path:
        print(f"\nProfessional static PNG created: {png_path}")
        print("\n" + "=" * 80)
        print("The static visualization includes:")
        print("   * High resolution (300 DPI) for publication quality")
        print("   * Enhanced color scheme:")
        print("     - INIT: Gold (most vibrant!)")
        print("     - MUTATION: Deep Pink (highly distinct)")
        print("     - CROSSOVER: Dark Turquoise (clear separation)")
        print("     - REFINE: Medium Purple (unique)")
        print("   * Comprehensive legend with network statistics")
        print("   * Operator performance metrics")
        print("   * Clear visual encodings")
        print("   * Professional layout suitable for presentations")
        print("=" * 80)
    else:
        print("Failed to generate static PNG")
    
    # Also regenerate interactive HTML with new colors
    print("\nRegenerating interactive HTML with new colors...")
    html_path = viz.create_interactive_html(filename="stn_interactive.html")
    
    if html_path:
        print(f"Interactive HTML updated: {html_path}")
        print("\nBoth visualizations now use the enhanced color scheme!")
    else:
        print("Failed to regenerate HTML")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_static_png.py <output_directory>")
        print("\nExample:")
        print('  python generate_static_png.py "../stn_outputs/exp-12-13_012735-google-gemini-2.5-flash-mada-v2-experiment-evolutionary(4+16, 0.9discount, 0.8-0.4cosine)"')
        sys.exit(1)
    
    output_dir = sys.argv[1]
    generate_static_png(output_dir)
