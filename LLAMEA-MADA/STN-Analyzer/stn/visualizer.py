"""
STN Visualizer for MADA-LLAMEA Experiments

Provides comprehensive visualization capabilities for Search Trajectory Networks:
- Static plots using matplotlib
- Interactive HTML visualizations using pyvis
- Fitness trajectory plots
- Operator analysis charts
- Lineage tree visualizations

Color Scheme:
- Mutation: Red (#FF6B6B)
- Crossover: Teal (#4ECDC4)
- Refine: Blue (#45B7D1)
- Init: Light Green (#95E1D3)
- Unknown: Gray (#DDD)
"""

import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from tqdm import tqdm

# Import visualization libraries with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Static plots disabled.")

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False
    print("Warning: pyvis not available. Interactive HTML visualization disabled.")


class STNVisualizer:
    """
    Visualize Search Trajectory Networks from MADA-LLAMEA experiments.
    
    This class provides multiple visualization methods:
    - plot_full_stn: Complete network visualization
    - plot_fitness_trajectory: Fitness over generations
    - plot_operator_analysis: Operator effectiveness comparison
    - plot_lineage_tree: Successful lineage visualization
    - create_interactive_html: Interactive HTML visualization
    
    Usage:
        viz = STNVisualizer(graph, output_dir=Path("outputs"))
        viz.plot_full_stn()
        viz.create_interactive_html()
    """
    
    # Operator color scheme - Enhanced for clarity
    OPERATOR_COLORS = {
        'mutation': '#FF1493',      # Deep Pink (highly distinct)
        'crossover': '#00CED1',     # Dark Turquoise (clear separation)
        'refine': '#9370DB',        # Medium Purple (unique)
        'init': '#FFD700',          # Gold (most vibrant!)
        'unknown': '#CCCCCC',       # Gray
    }
    
    # Generation colormap
    GENERATION_CMAP = 'viridis'
    
    # Fitness colormap (for success visualization)
    FITNESS_CMAP = 'RdYlGn'
    
    def __init__(self, graph: nx.DiGraph, output_dir: Path):
        """
        Initialize visualizer with a graph and output directory.
        
        Args:
            graph: NetworkX DiGraph from STNGraphBuilder
            output_dir: Directory to save generated visualizations
        """
        self.graph = graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Precompute some values for consistent visualization
        self._fitnesses = [self.graph.nodes[n].get('fitness', 0.0) 
                         for n in self.graph.nodes()]
        self._max_fitness = max(self._fitnesses) if self._fitnesses else 1.0
        self._min_fitness = min(self._fitnesses) if self._fitnesses else 0.0
        
        self._generations = [self.graph.nodes[n].get('generation', 0)
                            for n in self.graph.nodes()]
        self._max_generation = max(self._generations) if self._generations else 0
    
    def plot_static_stn_professional(
        self,
        filename: str = "stn_static.png",
        figsize: Tuple[int, int] = (20, 14),
        dpi: int = 300,
    ) -> Optional[Path]:
        """
        Create professional static PNG visualization with comprehensive legend.
        
        This is the publication-quality version with:
        - High resolution (300 DPI)
        - Clear operator colors
        - Comprehensive legend
        - Statistics panel
        - Professional layout
        
        Args:
            filename: Output filename
            figsize: Figure size in inches
            dpi: Output resolution (300 for publication quality)
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return None
        
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, skipping visualization")
            return None
        
        # Create figure with gridspec for layout
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(1, 10, wspace=0.3)
        
        # Main graph axis
        ax_main = fig.add_subplot(gs[0, :7])
        
        # Legend/stats axis
        ax_legend = fig.add_subplot(gs[0, 7:])
        ax_legend.axis('off')
        
        # Compute layout - use hierarchical to show structure clearly
        pos = self._compute_layout("hierarchical")
        
        # Node sizes based on fitness
        node_sizes = []
        for n in self.graph.nodes():
            fit = self.graph.nodes[n].get('fitness', 0.0)
            if self._max_fitness > 0:
                relative_fit = fit / self._max_fitness
            else:
                relative_fit = 0.5
            node_sizes.append(200 + 800 * relative_fit)
        
        # Node colors based on operator (not generation)
        node_colors = []
        for n in self.graph.nodes():
            op = self.graph.nodes[n].get('operator', 'unknown')
            color = self.OPERATOR_COLORS.get(op, '#CCCCCC')
            node_colors.append(color)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.9,
            edgecolors='white',
            linewidths=2,
            ax=ax_main,
        )
        
        # Draw edges colored by operator
        for operator, color in self.OPERATOR_COLORS.items():
            edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                    if d.get('operator') == operator]
            if edges:
                # Edge widths based on improvement
                widths = []
                for u, v in edges:
                    edge_data = self.graph.get_edge_data(u, v)
                    imp = edge_data.get('fitness_improvement', 0) if edge_data else 0
                    widths.append(1 + min(abs(imp) * 20, 5))
                
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=edges,
                    edge_color=color,
                    alpha=0.7,
                    arrows=True,
                    arrowsize=15,
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.1',
                    width=widths,
                    ax=ax_main,
                )
        
        # Add labels for high-fitness nodes
        top_nodes = sorted(self.graph.nodes(), 
                          key=lambda n: self.graph.nodes[n].get('fitness', 0),
                          reverse=True)[:10]
        labels = {n: n[-4:] for n in top_nodes}
        nx.draw_networkx_labels(self.graph, pos, labels, 
                               font_size=9, font_weight='bold',
                               font_color='black',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', 
                                       edgecolor='none',
                                       alpha=0.8),
                               ax=ax_main)
        
        ax_main.set_title('MADA Search Trajectory Network', 
                         fontsize=20, fontweight='bold', pad=20)
        ax_main.axis('off')
        
        # Create comprehensive legend panel
        self._add_professional_legend(ax_legend)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def _add_professional_legend(self, ax):
        """Add comprehensive legend and statistics to axis."""
        # Calculate statistics
        stats = self._calculate_graph_stats()
        
        y_pos = 0.95
        line_height = 0.045
        
        # Title
        ax.text(0.5, y_pos, 'Network Legend & Statistics', 
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F4F8', 
                        edgecolor='#4A90E2', linewidth=2))
        y_pos -= line_height * 2
        
        # Network Statistics
        ax.text(0.05, y_pos, 'Network Statistics', 
               ha='left', va='top', fontsize=12, fontweight='bold')
        y_pos -= line_height
        
        stat_items = [
            f"Total Algorithms: {stats['total_nodes']}",
            f"Total Transitions: {stats['total_edges']}",
            f"Generations: {stats['max_generation'] + 1}",
            f"Best Fitness: {stats['max_fitness']:.4f}",
            f"Avg Fitness: {stats['avg_fitness']:.4f}",
            f"Avg Diversity (NN-Dist): {stats['avg_nn_dist']:.4f}",
        ]
        
        for item in stat_items:
            ax.text(0.1, y_pos, item, ha='left', va='top', 
                   fontsize=10, family='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.7))
            y_pos -= line_height * 0.9
        
        y_pos -= line_height * 0.5
        
        # Operator Legend
        ax.text(0.05, y_pos, 'Operator Colors', 
               ha='left', va='top', fontsize=12, fontweight='bold')
        y_pos -= line_height
        
        operator_info = [
            ('init', 'INIT', 'Initial Population'),
            ('mutation', 'MUTATION', 'Code Modification'),
            ('crossover', 'CROSSOVER', 'Code Combination'),
            ('refine', 'REFINE', 'LLM Improvement'),
        ]
        
        for op_key, op_name, op_desc in operator_info:
            color = self.OPERATOR_COLORS[op_key]
            count = stats['operator_counts'].get(op_key, 0)
            
            # Color box
            ax.add_patch(plt.Rectangle((0.1, y_pos - 0.015), 0.08, 0.03,
                                      facecolor=color, edgecolor='black', linewidth=1))
            
            # Text
            ax.text(0.2, y_pos, f"{op_name} ({count})", 
                   ha='left', va='top', fontsize=10, fontweight='bold')
            ax.text(0.2, y_pos - line_height * 0.5, op_desc, 
                   ha='left', va='top', fontsize=8, style='italic', color='#666')
            
            y_pos -= line_height * 1.3
        
        y_pos -= line_height * 0.3
        
        # Operator Performance
        ax.text(0.05, y_pos, 'Operator Performance', 
               ha='left', va='top', fontsize=12, fontweight='bold')
        y_pos -= line_height
        
        for op in ['mutation', 'crossover', 'refine']:
            if op in stats['operator_success_rates']:
                success = stats['operator_success_rates'][op] * 100
                avg_fit = stats['operator_avg_fitness'][op]
                color = self.OPERATOR_COLORS[op]
                
                ax.text(0.1, y_pos, f"{op.upper()}", 
                       ha='left', va='top', fontsize=9, 
                       fontweight='bold', color=color)
                ax.text(0.1, y_pos - line_height * 0.55, 
                       f"Success: {success:.1f}% | Avg Fit: {avg_fit:.4f}", 
                       ha='left', va='top', fontsize=8, family='monospace')
                y_pos -= line_height * 1.1
        
        y_pos -= line_height * 0.5
        
        # Visual Encodings
        ax.text(0.05, y_pos, 'Visual Encoding', 
               ha='left', va='top', fontsize=12, fontweight='bold')
        y_pos -= line_height
        
        encoding_items = [
            "• Node Color = Operator Type",
            "• Node Size = Fitness (larger = better)",
            "• Edge Color = Operator Type",
            "• Edge Width = Improvement Magnitude",
            "• Labels = Top 10 Algorithms",
        ]
        
        for item in encoding_items:
            ax.text(0.1, y_pos, item, ha='left', va='top', fontsize=9)
            y_pos -= line_height * 0.7
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def plot_full_stn(
        self,
        filename: str = "stn_full.png",
        figsize: Tuple[int, int] = (16, 12),
        node_size_scale: float = 500,
        show_labels: bool = False,
        layout: str = "spring",
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Create static visualization of the full STN.
        
        Args:
            filename: Output filename
            figsize: Figure size in inches
            node_size_scale: Base node size (scaled by fitness)
            show_labels: Whether to show node labels
            layout: Layout algorithm ('spring', 'kamada_kawai', 'hierarchical')
            dpi: Output resolution
            
        Returns:
            Path to saved figure, or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available")
            return None
        
        if self.graph.number_of_nodes() == 0:
            print("Graph is empty, skipping visualization")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Compute layout
        pos = self._compute_layout(layout)
        
        # Node sizes based on fitness
        node_sizes = []
        for n in self.graph.nodes():
            fit = self.graph.nodes[n].get('fitness', 0.0)
            if self._max_fitness > 0:
                relative_fit = fit / self._max_fitness
            else:
                relative_fit = 0.5
            node_sizes.append(100 + node_size_scale * relative_fit)
        
        # Node colors based on generation
        node_colors = [self.graph.nodes[n].get('generation', 0) 
                      for n in self.graph.nodes()]
        
        # Draw nodes
        nodes_collection = nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.8,
            ax=ax,
        )
        
        # Draw edges colored by operator
        for operator, color in self.OPERATOR_COLORS.items():
            edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                    if d.get('operator') == operator]
            if edges:
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edgelist=edges,
                    edge_color=color,
                    alpha=0.6,
                    arrows=True,
                    arrowsize=10,
                    connectionstyle='arc3,rad=0.1',
                    ax=ax,
                )
        
        # Labels
        if show_labels:
            labels = {n: n[-4:] for n in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels, font_size=6, ax=ax)
        
        # Legend for operators
        legend_elements = [
            Line2D([0], [0], color=color, linewidth=3, label=op.capitalize())
            for op, color in self.OPERATOR_COLORS.items()
            if any(d.get('operator') == op for _, _, d in self.graph.edges(data=True))
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
        
        # Colorbar for generations
        if nodes_collection:
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.viridis,
                norm=plt.Normalize(0, self._max_generation)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.5, label='Generation')
        
        # Title and styling
        ax.set_title(f"Search Trajectory Network\n"
                    f"({self.graph.number_of_nodes()} nodes, "
                    f"{self.graph.number_of_edges()} edges)",
                    fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def _compute_layout(self, layout: str) -> Dict[str, Tuple[float, float]]:
        """Compute node positions using specified layout algorithm."""
        print(f"   Computing {layout} layout for {self.graph.number_of_nodes()} nodes...")
        
        if layout == "hierarchical":
            # Try graphviz hierarchical layout first (best for trees)
            try:
                print("   Trying graphviz dot layout...")
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog='dot')
                print("   Successfully computed graphviz layout")
                return pos
            except Exception as e:
                print(f"   Graphviz layout failed ({type(e).__name__}), using custom tree layout")
            
            # Use custom tree layout with proper parent-child positioning
            print("   Using custom tree layout based on parent-child relationships")
            return self._compute_tree_layout()
        
        elif layout == "kamada_kawai":
            try:
                print("   Computing Kamada-Kawai layout...")
                return nx.kamada_kawai_layout(self.graph)
            except Exception as e:
                print(f"   Kamada-Kawai failed ({type(e).__name__}), using spring layout")
                return nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        else:  # spring layout
            print("   Computing spring layout...")
            try:
                return nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
            except Exception as e:
                print(f"   Error computing spring layout: {e}")
                # Fallback to simple circular layout
                return nx.circular_layout(self.graph)
    
    def _compute_tree_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Compute tree layout that shows parent-child relationships clearly.
        Uses a layered approach where each generation is a layer.
        """
        from collections import deque
        
        # Find root nodes (nodes with no predecessors)
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        if not roots:
            # If no roots found, use nodes from generation 0
            gen_0 = [n for n in self.graph.nodes() 
                    if self.graph.nodes[n].get('generation', 0) == 0]
            roots = gen_0 if gen_0 else [list(self.graph.nodes())[0]]
        
        # Build tree structure using BFS
        pos = {}
        visited = set()
        node_to_level = {}
        level_nodes = defaultdict(list)
        
        # BFS to assign levels
        queue = deque([(root, 0) for root in roots])
        
        while queue:
            node, level = queue.popleft()
            if node in visited:
                continue
            
            visited.add(node)
            node_to_level[node] = level
            level_nodes[level].append(node)
            
            # Add children
            children = list(self.graph.successors(node))
            # Sort children by fitness for better visual ordering
            children.sort(key=lambda n: self.graph.nodes[n].get('fitness', 0), reverse=True)
            
            for child in children:
                if child not in visited:
                    queue.append((child, level + 1))
        
        # Handle disconnected nodes
        for node in self.graph.nodes():
            if node not in visited:
                gen = self.graph.nodes[node].get('generation', 0)
                node_to_level[node] = gen
                level_nodes[gen].append(node)
                visited.add(node)
        
        # Position nodes using Reingold-Tilford-like algorithm
        # Calculate positions for each level
        for level in sorted(level_nodes.keys()):
            nodes = level_nodes[level]
            n_nodes = len(nodes)
            
            if n_nodes == 1:
                # Single node: center it
                pos[nodes[0]] = (0, -level * 1.5)
            else:
                # Multiple nodes: distribute based on parent positions if possible
                for i, node in enumerate(nodes):
                    # Try to position near parent
                    parents = list(self.graph.predecessors(node))
                    
                    if parents and parents[0] in pos:
                        # Position near parent
                        parent_x = pos[parents[0]][0]
                        # Spread children around parent
                        siblings = [n for n in nodes if list(self.graph.predecessors(n)) == parents]
                        if node in siblings:
                            sib_idx = siblings.index(node)
                            n_sibs = len(siblings)
                            offset = (sib_idx - (n_sibs - 1) / 2.0) * 0.8
                            x = parent_x + offset
                        else:
                            x = parent_x
                    else:
                        # No parent positioned yet: spread evenly
                        x = (i - (n_nodes - 1) / 2.0) * (10.0 / n_nodes)
                    
                    pos[node] = (x, -level * 1.5)
        
        # Adjust x positions to reduce overlap
        max_iterations = 5
        for _ in range(max_iterations):
            moved = False
            for level in sorted(level_nodes.keys()):
                nodes = level_nodes[level]
                sorted_nodes = sorted(nodes, key=lambda n: pos[n][0])
                
                for i in range(len(sorted_nodes) - 1):
                    n1, n2 = sorted_nodes[i], sorted_nodes[i + 1]
                    x1, y1 = pos[n1]
                    x2, y2 = pos[n2]
                    
                    min_dist = 0.6
                    if abs(x2 - x1) < min_dist:
                        # Push nodes apart
                        mid = (x1 + x2) / 2
                        pos[n1] = (mid - min_dist / 2, y1)
                        pos[n2] = (mid + min_dist / 2, y2)
                        moved = True
            
            if not moved:
                break
        
        return pos
    
    def plot_fitness_trajectory(
        self,
        filename: str = "fitness_trajectory.png",
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Plot fitness over generations with operator coloring.
        
        Creates a two-panel figure:
        - Left: Scatter plot of fitness vs generation
        - Right: Box plot of fitness distribution by operator
        
        Args:
            filename: Output filename
            figsize: Figure size in inches
            dpi: Output resolution
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if self.graph.number_of_nodes() == 0:
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: Generation vs Fitness scatter
        ax1 = axes[0]
        
        for node in self.graph.nodes():
            gen = self.graph.nodes[node].get('generation', 0)
            fit = self.graph.nodes[node].get('fitness', 0.0)
            op = self.graph.nodes[node].get('operator', 'unknown')
            color = self.OPERATOR_COLORS.get(op, '#999999')
            ax1.scatter(gen, fit, c=color, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        
        # Best fitness line per generation
        gen_max = defaultdict(float)
        for node in self.graph.nodes():
            gen = self.graph.nodes[node].get('generation', 0)
            fit = self.graph.nodes[node].get('fitness', 0.0)
            gen_max[gen] = max(gen_max[gen], fit)
        
        if gen_max:
            gens = sorted(gen_max.keys())
            maxes = [gen_max[g] for g in gens]
            ax1.plot(gens, maxes, 'k--', alpha=0.5, linewidth=2, label='Best per gen')
        
        # Legend for operators
        for op, color in self.OPERATOR_COLORS.items():
            count = sum(1 for n in self.graph.nodes() 
                       if self.graph.nodes[n].get('operator') == op)
            if count > 0:
                ax1.scatter([], [], c=color, label=f'{op.capitalize()} ({count})', s=50)
        
        ax1.legend(loc='lower right', fontsize=9)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness (AOCC)', fontsize=11)
        ax1.set_title('Fitness by Generation & Operator', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, self._max_generation + 0.5)
        
        # Panel 2: Fitness distribution by operator
        ax2 = axes[1]
        
        operator_fitness = defaultdict(list)
        for node in self.graph.nodes():
            op = self.graph.nodes[node].get('operator', 'unknown')
            fit = self.graph.nodes[node].get('fitness', 0.0)
            operator_fitness[op].append(fit)
        
        # Filter to operators with data
        ops_with_data = [op for op in self.OPERATOR_COLORS.keys() 
                        if operator_fitness[op]]
        
        if ops_with_data:
            positions = list(range(len(ops_with_data)))
            data = [operator_fitness[op] for op in ops_with_data]
            
            bp = ax2.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
            
            for i, (op, patch) in enumerate(zip(ops_with_data, bp['boxes'])):
                color = self.OPERATOR_COLORS.get(op, '#999999')
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Color median lines
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            
            ax2.set_xticks(positions)
            ax2.set_xticklabels([op.capitalize() for op in ops_with_data], fontsize=10)
        
        ax2.set_ylabel('Fitness (AOCC)', fontsize=11)
        ax2.set_title('Fitness Distribution by Operator', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def plot_operator_analysis(
        self,
        filename: str = "operator_analysis.png",
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Create comprehensive operator analysis visualization.
        
        Four-panel figure:
        - Usage count
        - Success rate
        - Mean improvement
        - Fitness produced
        
        Args:
            filename: Output filename
            figsize: Figure size
            dpi: Output resolution
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Collect operator statistics
        op_counts = defaultdict(int)
        op_improvements = defaultdict(list)
        op_fitness = defaultdict(list)
        
        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            op_counts[op] += 1
            op_improvements[op].append(data.get('fitness_improvement', 0.0))
            op_fitness[op].append(self.graph.nodes[v].get('fitness', 0.0))
        
        # Add init nodes
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('operator') == 'init':
                op_counts['init'] += 1
                op_fitness['init'].append(self.graph.nodes[node].get('fitness', 0.0))
        
        # Filter operators with data
        ops = [op for op in ['init', 'mutation', 'crossover', 'refine'] 
               if op_counts[op] > 0]
        
        if not ops:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        colors = [self.OPERATOR_COLORS.get(op, '#999') for op in ops]
        
        # Panel 1: Usage counts
        ax1 = axes[0, 0]
        counts = [op_counts[op] for op in ops]
        bars = ax1.bar(ops, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Operator Usage', fontsize=12)
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10)
        ax1.set_xticklabels([op.capitalize() for op in ops])
        
        # Panel 2: Success rate
        ax2 = axes[0, 1]
        success_rates = []
        for op in ops:
            imps = op_improvements.get(op, [])
            if imps:
                rate = sum(1 for i in imps if i > 0) / len(imps)
            else:
                rate = 0.0
            success_rates.append(rate)
        
        bars = ax2.bar(ops, [r * 100 for r in success_rates], color=colors, alpha=0.8,
                      edgecolor='white', linewidth=2)
        ax2.set_ylabel('Success Rate (%)', fontsize=11)
        ax2.set_title('Operator Success Rate\n(% producing improvement)', fontsize=12)
        ax2.set_ylim(0, 100)
        for bar, rate in zip(bars, success_rates):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.0%}', ha='center', va='bottom', fontsize=10)
        ax2.set_xticklabels([op.capitalize() for op in ops])
        
        # Panel 3: Mean improvement
        ax3 = axes[1, 0]
        mean_imps = []
        for op in ops:
            imps = op_improvements.get(op, [0])
            mean_imps.append(np.mean(imps) if imps else 0)
        
        colors_imp = []
        for op, imp in zip(ops, mean_imps):
            if op == 'init':
                colors_imp.append(self.OPERATOR_COLORS['init'])
            else:
                colors_imp.append('green' if imp > 0 else 'red')
        
        bars = ax3.bar(ops, mean_imps, color=colors_imp, alpha=0.8,
                      edgecolor='white', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('Mean Improvement', fontsize=11)
        ax3.set_title('Average Fitness Improvement', fontsize=12)
        for bar, imp in zip(bars, mean_imps):
            y = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    y + (0.005 if y >= 0 else -0.015),
                    f'{imp:.4f}', ha='center', 
                    va='bottom' if y >= 0 else 'top', fontsize=10)
        ax3.set_xticklabels([op.capitalize() for op in ops])
        
        # Panel 4: Mean fitness produced
        ax4 = axes[1, 1]
        mean_fits = []
        for op in ops:
            fits = op_fitness.get(op, [0])
            mean_fits.append(np.mean(fits) if fits else 0)
        
        bars = ax4.bar(ops, mean_fits, color=colors, alpha=0.8,
                      edgecolor='white', linewidth=2)
        ax4.set_ylabel('Mean Fitness (AOCC)', fontsize=11)
        ax4.set_title('Mean Fitness of Produced Algorithms', fontsize=12)
        for bar, fit in zip(bars, mean_fits):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{fit:.4f}', ha='center', va='bottom', fontsize=10)
        ax4.set_xticklabels([op.capitalize() for op in ops])
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def plot_lineage_tree(
        self,
        lineages: List[List[str]],
        filename: str = "lineage_tree.png",
        max_lineages: int = 5,
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Visualize successful lineages as a tree.
        
        Args:
            lineages: List of lineage lists (from identify_successful_lineages)
            filename: Output filename
            max_lineages: Maximum number of lineages to show
            figsize: Figure size
            dpi: Output resolution
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        if not lineages:
            print("No lineages to visualize")
            return None
        
        # Limit to max_lineages
        lineages_to_show = lineages[:max_lineages]
        
        # Collect all nodes from lineages
        lineage_nodes = set()
        for lin in lineages_to_show:
            if isinstance(lin, dict):
                lineage_nodes.update(lin.get('lineage', []))
            else:
                lineage_nodes.update(lin)
        
        if not lineage_nodes:
            return None
        
        # Create subgraph
        subgraph = self.graph.subgraph(lineage_nodes).copy()
        
        if subgraph.number_of_nodes() == 0:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        pos = nx.spring_layout(subgraph, k=3, iterations=100, seed=42)
        
        # Node colors based on fitness
        node_colors = [subgraph.nodes[n].get('fitness', 0.0) for n in subgraph.nodes()]
        
        # Node sizes based on whether they're endpoints
        node_sizes = []
        for n in subgraph.nodes():
            if subgraph.in_degree(n) == 0 or subgraph.out_degree(n) == 0:
                node_sizes.append(600)  # Larger for start/end
            else:
                node_sizes.append(300)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            cmap=plt.cm.RdYlGn,
            node_size=node_sizes,
            alpha=0.9,
            ax=ax,
        )
        
        # Draw edges
        for operator, color in self.OPERATOR_COLORS.items():
            edges = [(u, v) for u, v, d in subgraph.edges(data=True)
                    if d.get('operator') == operator]
            if edges:
                nx.draw_networkx_edges(
                    subgraph, pos,
                    edgelist=edges,
                    edge_color=color,
                    width=2,
                    arrows=True,
                    arrowsize=15,
                    alpha=0.8,
                    ax=ax,
                )
        
        # Labels (show last 4 chars of ID)
        labels = {n: n[-4:] for n in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.RdYlGn,
            norm=plt.Normalize(vmin=min(node_colors) if node_colors else 0,
                              vmax=max(node_colors) if node_colors else 1)
        )
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, label='Fitness')
        
        # Legend
        legend_elements = [
            Line2D([0], [0], color=color, linewidth=3, label=op.capitalize())
            for op, color in self.OPERATOR_COLORS.items()
            if any(d.get('operator') == op for _, _, d in subgraph.edges(data=True))
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_title(f"Top {len(lineages_to_show)} Successful Algorithm Lineages", fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def plot_diversity_metrics(
        self,
        filename: str = "diversity_metrics.png",
        figsize: Tuple[int, int] = (14, 6),
        dpi: int = 150,
    ) -> Optional[Path]:
        """
        Plot MADA diversity-related metrics over generations.
        
        Args:
            filename: Output filename
            figsize: Figure size
            dpi: Output resolution
            
        Returns:
            Path to saved figure
        """
        if not HAS_MATPLOTLIB:
            return None
        
        # Collect diversity data
        gen_nn_dist = defaultdict(list)
        gen_alpha = defaultdict(list)
        gen_div_bonus = defaultdict(list)
        
        for node in self.graph.nodes():
            gen = self.graph.nodes[node].get('generation', 0)
            nn_dist = self.graph.nodes[node].get('nn_dist', 0.0)
            alpha = self.graph.nodes[node].get('alpha', 0.0)
            div_bonus = self.graph.nodes[node].get('diversity_bonus', 0.0)
            
            # Clip NN-distance to [0, 1] range for consistency with baseline LLAMEA
            nn_dist = max(0.0, min(1.0, nn_dist))
            
            if nn_dist >= 0:  # Include zero values after clipping
                gen_nn_dist[gen].append(nn_dist)
            if alpha > 0:
                gen_alpha[gen].append(alpha)
            if div_bonus >= 0:  # Include zero values
                gen_div_bonus[gen].append(div_bonus)
        
        if not gen_nn_dist and not gen_alpha:
            print("No diversity data available")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Panel 1: NN-Dist over generations
        ax1 = axes[0]
        if gen_nn_dist:
            gens = sorted(gen_nn_dist.keys())
            means = [np.mean(gen_nn_dist[g]) for g in gens]
            stds = [np.std(gen_nn_dist[g]) for g in gens]
            
            # Convert generations to API calls (assuming 20 calls per generation for 6 generations = 100 total)
            api_calls = [(g-1) * 20 + 10 for g in gens]  # Center each generation's API calls
            
            ax1.plot(api_calls, means, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
            # Clip standard deviation bands to [0, 1] to prevent negative values
            lower_bounds = [max(0.0, m - s) for m, s in zip(means, stds)]
            upper_bounds = [min(1.0, m + s) for m, s in zip(means, stds)]
            ax1.fill_between(api_calls, lower_bounds, upper_bounds, color='#4ECDC4', alpha=0.2)
        
        ax1.set_xlabel('API Calls', fontsize=11)
        ax1.set_ylabel('NN-Distance', fontsize=11)
        ax1.set_title('Behavioral Diversity (NN-Dist) Over API Calls', fontsize=12)
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 1)  # Enforce [0, 1] range
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Alpha and diversity bonus
        ax2 = axes[1]
        
        if gen_alpha:
            gens = sorted(gen_alpha.keys())
            alpha_means = [np.mean(gen_alpha[g]) for g in gens]
            # Convert generations to API calls
            api_calls = [(g-1) * 20 + 10 for g in gens]
            ax2.plot(api_calls, alpha_means, 'o-', color='#FF6B6B', 
                    linewidth=2, markersize=8, label='α (diversity weight)')
        
        if gen_div_bonus:
            gens = sorted(gen_div_bonus.keys())
            bonus_means = [np.mean(gen_div_bonus[g]) for g in gens]
            # Convert generations to API calls
            api_calls = [(g-1) * 20 + 10 for g in gens]
            ax2.plot(api_calls, bonus_means, 's--', color='#45B7D1',
                    linewidth=2, markersize=6, label='Diversity Bonus')
        
        ax2.set_xlabel('API Calls', fontsize=11)
        ax2.set_ylabel('Value', fontsize=11)
        ax2.set_title('MADA Parameters Over API Calls', fontsize=12)
        ax2.set_xlim(0, 100)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        return output_path
    
    def create_interactive_html(
        self,
        filename: str = "stn_interactive.html",
        height: str = "800px",
        width: str = "100%",
        bgcolor: str = "#1a1a1a",
        physics: bool = True,
    ) -> Optional[Path]:
        """
        Create enhanced interactive HTML visualization with comprehensive legend and controls.
        
        Args:
            filename: Output filename
            height: Height of visualization
            width: Width of visualization
            bgcolor: Background color
            physics: Enable physics simulation
            
        Returns:
            Path to saved HTML file
        """
        if not HAS_PYVIS:
            print("pyvis not available")
            return None
        
        if self.graph.number_of_nodes() == 0:
            return None
        
        # Calculate statistics for display
        stats = self._calculate_graph_stats()
        
        # Create pyvis network
        net = Network(
            height=height,
            width=width,
            directed=True,
            bgcolor=bgcolor,
            font_color="white",
        )
        
        # Configure physics
        net.toggle_physics(physics)
        net.set_options("""
        {
            "nodes": {
                "font": {"size": 14, "face": "Arial", "color": "#ffffff"},
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)", "size": 10}
            },
            "edges": {
                "smooth": {"type": "curvedCW", "roundness": 0.2},
                "arrows": {"to": {"enabled": true, "scaleFactor": 1}},
                "shadow": {"enabled": false}
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -5000,
                    "centralGravity": 0.5,
                    "springLength": 150,
                    "springConstant": 0.04,
                    "damping": 0.09,
                    "avoidOverlap": 0.2
                },
                "minVelocity": 0.75,
                "stabilization": {
                    "iterations": 200,
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "navigationButtons": true,
                "keyboard": true,
                "zoomView": true,
                "dragView": true
            }
        }
        """)
        
        # Add nodes with enhanced information
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            fit = data.get('fitness', 0.0)
            gen = data.get('generation', 0)
            op = data.get('operator', 'unknown')
            name = data.get('algorithm_name', node[-6:])
            error = data.get('error', '')
            
            # Node size based on fitness
            if self._max_fitness > 0:
                size = 15 + 35 * (fit / self._max_fitness)
            else:
                size = 25
            
            # Node color based on operator
            color = self.OPERATOR_COLORS.get(op, '#999999')
            
            # Enhanced tooltip with all details
            title = self._create_node_tooltip(node, data, fit, gen, op, name, error)
            
            net.add_node(
                node,
                label=node[-4:],
                title=title,
                size=size,
                color=color,
                borderWidth=2,
                borderWidthSelected=4,
            )
        
        # Add edges with enhanced information
        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            imp = data.get('fitness_improvement', 0)
            
            color = self.OPERATOR_COLORS.get(op, '#999999')
            
            # Edge width based on improvement magnitude
            width = 1 + min(abs(imp) * 10, 4)
            
            # Enhanced edge tooltip
            title = (
                f"<b style='color: {color}'>Operator:</b> {op.upper()}<br>"
                f"<b>Fitness Change:</b> {imp:+.4f}<br>"
                f"<b>From:</b> {self.graph.nodes[u].get('algorithm_name', u[-6:])}<br>"
                f"<b>To:</b> {self.graph.nodes[v].get('algorithm_name', v[-6:])}<br>"
            )
            
            net.add_edge(u, v, title=title, color=color, width=width)
        
        # Save the base HTML
        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        
        # Enhance the HTML with custom legend and controls
        self._enhance_html_with_legend(output_path, stats)
        
        return output_path
    
    def _calculate_graph_stats(self) -> Dict[str, Any]:
        """Calculate statistics about the graph for display."""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'max_generation': self._max_generation,
            'max_fitness': self._max_fitness,
            'min_fitness': self._min_fitness,
            'avg_fitness': np.mean(self._fitnesses) if self._fitnesses else 0,
        }
        
        # Operator counts
        op_counts = defaultdict(int)
        op_fitness = defaultdict(list)
        for node in self.graph.nodes():
            op = self.graph.nodes[node].get('operator', 'unknown')
            fit = self.graph.nodes[node].get('fitness', 0.0)
            op_counts[op] += 1
            op_fitness[op].append(fit)
        
        stats['operator_counts'] = dict(op_counts)
        stats['operator_avg_fitness'] = {
            op: np.mean(fits) for op, fits in op_fitness.items()
        }
        
        # Diversity metrics
        nn_dists = [self.graph.nodes[n].get('nn_dist', 0) 
                   for n in self.graph.nodes()]
        nn_dists = [d for d in nn_dists if d > 0]
        if nn_dists:
            stats['avg_nn_dist'] = np.mean(nn_dists)
            stats['max_nn_dist'] = max(nn_dists)
        else:
            stats['avg_nn_dist'] = 0
            stats['max_nn_dist'] = 0
        
        # Success rate per operator
        op_improvements = defaultdict(list)
        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            imp = data.get('fitness_improvement', 0)
            op_improvements[op].append(imp)
        
        stats['operator_success_rates'] = {}
        for op, imps in op_improvements.items():
            if imps:
                success = sum(1 for i in imps if i > 0) / len(imps)
                stats['operator_success_rates'][op] = success
        
        return stats
    
    def _create_node_tooltip(self, node: str, data: Dict, fit: float, 
                            gen: int, op: str, name: str, error: str) -> str:
        """Create enhanced HTML tooltip for node."""
        color = self.OPERATOR_COLORS.get(op, '#999999')
        
        html = f"""
        <div style='font-family: Arial, sans-serif; padding: 5px; max-width: 400px;'>
            <h3 style='margin: 0 0 10px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 5px;'>
                Algorithm {node[-4:]}
            </h3>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr style='background-color: rgba(255,255,255,0.05);'>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Name:</td>
                    <td style='padding: 5px; color: #fff;'>{name}</td>
                </tr>
                <tr>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Fitness (AOCC):</td>
                    <td style='padding: 5px; color: #0f0; font-weight: bold;'>{fit:.4f}</td>
                </tr>
                <tr style='background-color: rgba(255,255,255,0.05);'>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Generation:</td>
                    <td style='padding: 5px; color: #fff;'>{gen}</td>
                </tr>
                <tr>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Operator:</td>
                    <td style='padding: 5px; color: {color}; font-weight: bold;'>{op.upper()}</td>
                </tr>
        """
        
        # Add diversity metrics if available
        nn_dist = data.get('nn_dist', 0)
        if nn_dist > 0:
            html += f"""
                <tr style='background-color: rgba(255,255,255,0.05);'>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>NN-Distance:</td>
                    <td style='padding: 5px; color: #4ECDC4;'>{nn_dist:.4f}</td>
                </tr>
            """
        
        alpha = data.get('alpha', 0)
        if alpha > 0:
            html += f"""
                <tr>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Alpha (α):</td>
                    <td style='padding: 5px; color: #FF6B6B;'>{alpha:.4f}</td>
                </tr>
            """
        
        div_bonus = data.get('diversity_bonus', 0)
        if div_bonus > 0:
            html += f"""
                <tr style='background-color: rgba(255,255,255,0.05);'>
                    <td style='padding: 5px; font-weight: bold; color: #aaa;'>Diversity Bonus:</td>
                    <td style='padding: 5px; color: #45B7D1;'>{div_bonus:.4f}</td>
                </tr>
            """
        
        if error:
            html += f"""
                <tr>
                    <td colspan='2' style='padding: 5px; color: #f00; font-style: italic;'>
                        <b>Error:</b> {error[:100]}...
                    </td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
        
        return html
    
    def _enhance_html_with_legend(self, html_path: Path, stats: Dict[str, Any]):
        """Add comprehensive legend and information panels to the HTML."""
        
        # Read the generated HTML
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create the legend and info panel HTML
        legend_html = f"""
        <!-- Custom Legend and Info Panel -->
        <style>
            #info-panel {{
                position: fixed;
                top: 20px;
                right: 20px;
                width: 350px;
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                border: 2px solid #4ECDC4;
                border-radius: 10px;
                padding: 20px;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 13px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
                z-index: 1000;
                max-height: 90vh;
                overflow-y: auto;
            }}
            
            #info-panel h2 {{
                margin: 0 0 15px 0;
                color: #4ECDC4;
                font-size: 20px;
                border-bottom: 2px solid #4ECDC4;
                padding-bottom: 8px;
                text-align: center;
            }}
            
            #info-panel h3 {{
                margin: 15px 0 10px 0;
                color: #95E1D3;
                font-size: 15px;
                border-bottom: 1px solid #45B7D1;
                padding-bottom: 5px;
            }}
            
            .legend-item {{
                display: flex;
                align-items: center;
                margin: 8px 0;
                padding: 5px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
            }}
            
            .legend-color {{
                width: 30px;
                height: 20px;
                border-radius: 3px;
                margin-right: 12px;
                border: 1px solid rgba(255, 255, 255, 0.3);
            }}
            
            .legend-line {{
                width: 30px;
                height: 3px;
                margin-right: 12px;
                border-radius: 2px;
            }}
            
            .stat-row {{
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                padding: 5px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 3px;
            }}
            
            .stat-label {{
                color: #bbb;
                font-weight: bold;
            }}
            
            .stat-value {{
                color: #4ECDC4;
                font-weight: bold;
            }}
            
            .control-button {{
                background: #45B7D1;
                border: none;
                color: white;
                padding: 8px 15px;
                margin: 5px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 12px;
                transition: all 0.3s;
            }}
            
            .control-button:hover {{
                background: #4ECDC4;
                transform: scale(1.05);
            }}
            
            #toggle-panel {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: #45B7D1;
                border: none;
                color: white;
                padding: 10px 15px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                z-index: 1001;
                display: none;
            }}
            
            #toggle-panel:hover {{
                background: #4ECDC4;
            }}
            
            .size-legend {{
                display: flex;
                align-items: center;
                margin: 10px 0;
            }}
            
            .size-example {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #4ECDC4;
                margin: 0 5px;
                display: inline-block;
            }}
        </style>
        
        <button id="toggle-panel" onclick="toggleInfoPanel()">Show Info</button>
        
        <div id="info-panel">
            <h2>🔍 MADA Search Network</h2>
            
            <h3>📊 Network Statistics</h3>
            <div class="stat-row">
                <span class="stat-label">Total Algorithms:</span>
                <span class="stat-value">{stats['total_nodes']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Total Transitions:</span>
                <span class="stat-value">{stats['total_edges']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Generations:</span>
                <span class="stat-value">{stats['max_generation'] + 1}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Best Fitness:</span>
                <span class="stat-value">{stats['max_fitness']:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Fitness:</span>
                <span class="stat-value">{stats['avg_fitness']:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg NN-Distance:</span>
                <span class="stat-value">{stats['avg_nn_dist']:.4f}</span>
            </div>
            
            <h3>🎨 Operator Colors</h3>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {self.OPERATOR_COLORS['init']};"></div>
                <span><b>INIT</b> - Initial population ({stats['operator_counts'].get('init', 0)})</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {self.OPERATOR_COLORS['mutation']};"></div>
                <span><b>MUTATION</b> - Code modification ({stats['operator_counts'].get('mutation', 0)})</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {self.OPERATOR_COLORS['crossover']};"></div>
                <span><b>CROSSOVER</b> - Code combination ({stats['operator_counts'].get('crossover', 0)})</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: {self.OPERATOR_COLORS['refine']};"></div>
                <span><b>REFINE</b> - LLM improvement ({stats['operator_counts'].get('refine', 0)})</span>
            </div>
            
            <h3>📈 Operator Performance</h3>
        """
        
        # Add operator success rates
        for op in ['mutation', 'crossover', 'refine']:
            if op in stats['operator_success_rates']:
                success_rate = stats['operator_success_rates'][op] * 100
                avg_fit = stats['operator_avg_fitness'].get(op, 0)
                color = self.OPERATOR_COLORS[op]
                legend_html += f"""
            <div class="stat-row">
                <span class="stat-label" style="color: {color};">{op.capitalize()}:</span>
                <span class="stat-value">{success_rate:.1f}% success | Avg: {avg_fit:.4f}</span>
            </div>
                """
        
        legend_html += f"""
            <h3>📏 Node Size Legend</h3>
            <div class="size-legend">
                <div class="size-example" style="width: 15px; height: 15px;"></div>
                <span style="flex: 1;">Low Fitness</span>
            </div>
            <div class="size-legend">
                <div class="size-example" style="width: 30px; height: 30px;"></div>
                <span style="flex: 1;">Medium Fitness</span>
            </div>
            <div class="size-legend">
                <div class="size-example" style="width: 50px; height: 50px;"></div>
                <span style="flex: 1;">High Fitness</span>
            </div>
            
            <h3>⚡ Edge Width Legend</h3>
            <div class="legend-item">
                <div class="legend-line" style="background-color: #fff; width: 30px; height: 1px;"></div>
                <span>Small improvement</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background-color: #fff; width: 30px; height: 3px;"></div>
                <span>Medium improvement</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background-color: #fff; width: 30px; height: 5px;"></div>
                <span>Large improvement</span>
            </div>
            
            <h3>🎮 Controls</h3>
            <div style="text-align: center; margin-top: 10px;">
                <div>🖱️ <b>Click & Drag</b>: Move view</div>
                <div>🔍 <b>Scroll</b>: Zoom in/out</div>
                <div>👆 <b>Click Node</b>: View details</div>
                <div>⌨️ <b>Keyboard</b>: Arrow keys to pan</div>
            </div>
            
            <div style="text-align: center; margin-top: 15px;">
                <button class="control-button" onclick="network.fit()">🎯 Fit All</button>
                <button class="control-button" onclick="togglePhysics()">⚡ Toggle Physics</button>
            </div>
            
            <div style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 5px; font-size: 11px;">
                <b>💡 Tip:</b> Hover over nodes and edges for detailed information. Node size indicates fitness, color indicates operator type.
            </div>
        </div>
        
        <script>
            let panelVisible = true;
            let physicsEnabled = true;
            
            function toggleInfoPanel() {{
                const panel = document.getElementById('info-panel');
                const button = document.getElementById('toggle-panel');
                panelVisible = !panelVisible;
                
                if (panelVisible) {{
                    panel.style.display = 'block';
                    button.style.display = 'none';
                }} else {{
                    panel.style.display = 'none';
                    button.style.display = 'block';
                }}
            }}
            
            function togglePhysics() {{
                physicsEnabled = !physicsEnabled;
                network.setOptions({{
                    physics: {{enabled: physicsEnabled}}
                }});
            }}
            
            // Add close button to panel
            document.addEventListener('DOMContentLoaded', function() {{
                const panel = document.getElementById('info-panel');
                const closeBtn = document.createElement('button');
                closeBtn.innerHTML = '✕';
                closeBtn.style.cssText = 'position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.2); border: none; color: white; width: 25px; height: 25px; border-radius: 50%; cursor: pointer; font-size: 16px;';
                closeBtn.onclick = toggleInfoPanel;
                panel.insertBefore(closeBtn, panel.firstChild);
            }});
        </script>
        """
        
        # Insert the legend before the closing body tag
        html_content = html_content.replace('</body>', legend_html + '</body>')
        
        # Write the enhanced HTML
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_all_visualizations(
        self,
        lineages: Optional[List[Dict]] = None,
        interactive: bool = True,
    ) -> Dict[str, Optional[Path]]:
        """
        Generate all available visualizations.
        
        Args:
            lineages: Optional list of successful lineages
            interactive: Whether to generate interactive HTML
            
        Returns:
            Dictionary mapping visualization names to output paths
        """
        outputs = {}
        
        print("  Generating visualizations...")
        
        # Determine number of visualizations to generate
        viz_count = 5 + (1 if lineages else 0) + (1 if interactive else 0)
        
        with tqdm(total=viz_count, desc="Creating visualizations", unit="viz") as pbar:
            # Professional Static STN (NEW!)
            pbar.set_description("Professional static STN")
            try:
                print("    Generating professional static STN...")
                outputs['stn_static'] = self.plot_static_stn_professional()
                if outputs['stn_static']:
                    print(f"    [OK] STN static (professional): {outputs['stn_static'].name}")
                else:
                    print(f"    [SKIP] STN static (professional) - skipped")
            except Exception as e:
                print(f"    [ERROR] Failed to create professional static STN: {type(e).__name__}: {e}")
                outputs['stn_static'] = None
            pbar.update(1)
            
            # Full STN (original)
            pbar.set_description("Full STN plot")
            try:
                print("    Generating full STN plot...")
                outputs['stn_full'] = self.plot_full_stn()
                if outputs['stn_full']:
                    print(f"    [OK] STN full: {outputs['stn_full'].name}")
                else:
                    print(f"    [SKIP] STN full - skipped")
            except Exception as e:
                print(f"    [ERROR] Failed to create full STN: {type(e).__name__}: {e}")
                outputs['stn_full'] = None
            pbar.update(1)
            
            # Fitness trajectory
            pbar.set_description("Fitness trajectory")
            try:
                print("    Generating fitness trajectory...")
                outputs['fitness_trajectory'] = self.plot_fitness_trajectory()
                if outputs['fitness_trajectory']:
                    print(f"    [OK] Fitness trajectory: {outputs['fitness_trajectory'].name}")
                else:
                    print(f"    [SKIP] Fitness trajectory - skipped")
            except Exception as e:
                print(f"    [ERROR] Failed to create fitness trajectory: {type(e).__name__}: {e}")
                outputs['fitness_trajectory'] = None
            pbar.update(1)
            
            # Operator analysis
            pbar.set_description("Operator analysis")
            try:
                print("    Generating operator analysis...")
                outputs['operator_analysis'] = self.plot_operator_analysis()
                if outputs['operator_analysis']:
                    print(f"    [OK] Operator analysis: {outputs['operator_analysis'].name}")
                else:
                    print(f"    [SKIP] Operator analysis - skipped")
            except Exception as e:
                print(f"    [ERROR] Failed to create operator analysis: {type(e).__name__}: {e}")
                outputs['operator_analysis'] = None
            pbar.update(1)
            
            # Diversity metrics
            pbar.set_description("Diversity metrics")
            try:
                print("    Generating diversity metrics...")
                outputs['diversity_metrics'] = self.plot_diversity_metrics()
                if outputs['diversity_metrics']:
                    print(f"    [OK] Diversity metrics: {outputs['diversity_metrics'].name}")
                else:
                    print(f"    [SKIP] Diversity metrics - skipped")
            except Exception as e:
                print(f"    [ERROR] Failed to create diversity metrics: {type(e).__name__}: {e}")
                outputs['diversity_metrics'] = None
            pbar.update(1)
            
            # Lineage tree
            if lineages:
                pbar.set_description("Lineage tree")
                try:
                    print("    Generating lineage tree...")
                    outputs['lineage_tree'] = self.plot_lineage_tree(lineages)
                    if outputs['lineage_tree']:
                        print(f"    [OK] Lineage tree: {outputs['lineage_tree'].name}")
                    else:
                        print(f"    [SKIP] Lineage tree - skipped")
                except Exception as e:
                    print(f"    [ERROR] Failed to create lineage tree: {type(e).__name__}: {e}")
                    outputs['lineage_tree'] = None
                pbar.update(1)
            
            # Interactive HTML
            if interactive:
                pbar.set_description("Interactive HTML")
                try:
                    print("    Generating interactive HTML...")
                    outputs['interactive'] = self.create_interactive_html()
                    if outputs['interactive']:
                        print(f"    [OK] Interactive HTML: {outputs['interactive'].name}")
                    else:
                        print(f"    [SKIP] Interactive HTML - skipped")
                except Exception as e:
                    print(f"    [ERROR] Failed to create interactive HTML: {type(e).__name__}: {e}")
                    outputs['interactive'] = None
                pbar.update(1)
        
        return outputs


if __name__ == "__main__":
    print("STN Visualizer module - run stn_analyzer.py for full analysis")

