"""
STN Metrics for MADA-LLAMEA Experiments

Provides comprehensive metrics and analysis for Search Trajectory Networks.
Based on STN theory from metaheuristic analysis literature.

Metrics Categories:
1. Basic Graph Metrics: nodes, edges, density, connectivity
2. Operator Analysis: effectiveness and success rates per operator
3. Trajectory Analysis: path lengths, branching, exploration patterns
4. Fitness Landscape: distribution, progression, peaks
5. Lineage Analysis: successful paths, dead ends, convergence

Reference:
"Search trajectory networks: A tool for analysing and visualising 
 the behaviour of metaheuristics"
"""

import networkx as nx
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class OperatorStats:
    """Statistics for a single operator."""
    name: str
    count: int = 0
    improvements: List[float] = None
    success_rate: float = 0.0
    mean_improvement: float = 0.0
    std_improvement: float = 0.0
    max_improvement: float = 0.0
    mean_fitness_produced: float = 0.0
    
    def __post_init__(self):
        if self.improvements is None:
            self.improvements = []


class STNMetrics:
    """
    Calculate comprehensive STN metrics for MADA-LLAMEA experiments.
    
    This class provides methods to analyze:
    - Graph structure and connectivity
    - Operator effectiveness (mutation, crossover, refine)
    - Search trajectory patterns
    - Fitness landscape characteristics
    - Algorithm lineages and evolution
    
    Usage:
        metrics = STNMetrics(graph)
        basic = metrics.basic_metrics()
        operators = metrics.operator_analysis()
        trajectory = metrics.trajectory_analysis()
    """
    
    OPERATOR_TYPES = ['mutation', 'crossover', 'refine', 'init', 'unknown']
    
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize metrics calculator with a graph.
        
        Args:
            graph: NetworkX DiGraph from STNGraphBuilder
        """
        self.graph = graph
        self._cache: Dict[str, Any] = {}
    
    def basic_metrics(self) -> Dict[str, Any]:
        """
        Calculate basic graph metrics.
        
        Returns:
            Dictionary with:
                - num_nodes: Total number of algorithms
                - num_edges: Total number of transitions
                - density: Graph density (edges / possible edges)
                - is_dag: Whether graph is a directed acyclic graph
                - num_connected_components: Number of weakly connected components
                - avg_degree: Average node degree
                - avg_in_degree: Average incoming edges per node
                - avg_out_degree: Average outgoing edges per node
                - max_depth: Maximum depth from any root
        """
        if 'basic' in self._cache:
            return self._cache['basic']
        
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        
        if n_nodes == 0:
            return {
                'num_nodes': 0,
                'num_edges': 0,
                'density': 0.0,
                'is_dag': True,
                'num_connected_components': 0,
                'avg_degree': 0.0,
                'avg_in_degree': 0.0,
                'avg_out_degree': 0.0,
                'max_depth': 0,
            }
        
        # Calculate depths
        depths = self._compute_depths()
        max_depth = max(depths.values()) if depths else 0
        
        result = {
            'num_nodes': n_nodes,
            'num_edges': n_edges,
            'density': nx.density(self.graph),
            'is_dag': nx.is_directed_acyclic_graph(self.graph),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': np.mean([d for n, d in self.graph.degree()]) if n_nodes > 0 else 0.0,
            'avg_in_degree': np.mean([d for n, d in self.graph.in_degree()]) if n_nodes > 0 else 0.0,
            'avg_out_degree': np.mean([d for n, d in self.graph.out_degree()]) if n_nodes > 0 else 0.0,
            'max_depth': max_depth,
        }
        
        self._cache['basic'] = result
        return result
    
    def _compute_depths(self) -> Dict[str, int]:
        """Compute depth for each node (distance from nearest root)."""
        depths = {}
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        for root in roots:
            queue = [(root, 0)]
            visited = set()
            while queue:
                node, depth = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                if node not in depths or depths[node] > depth:
                    depths[node] = depth
                for successor in self.graph.successors(node):
                    if successor not in visited:
                        queue.append((successor, depth + 1))
        
        # Handle disconnected nodes
        for node in self.graph.nodes():
            if node not in depths:
                depths[node] = 0
        
        return depths
    
    def operator_analysis(self) -> Dict[str, Any]:
        """
        Analyze contribution and effectiveness of each operator.
        
        Returns:
            Dictionary with:
                - operator_counts: Count of transitions per operator
                - per_operator: Detailed stats per operator (OperatorStats)
                - most_effective: Operator with highest success rate
                - highest_avg_improvement: Operator with highest mean improvement
                - operator_fitness_distributions: Fitness distributions by operator
        """
        if 'operator' in self._cache:
            return self._cache['operator']
        
        operator_counts = Counter()
        improvements = defaultdict(list)
        fitnesses_produced = defaultdict(list)
        
        for u, v, data in self.graph.edges(data=True):
            op = data.get('operator', 'unknown')
            operator_counts[op] += 1
            improvements[op].append(data.get('fitness_improvement', 0.0))
            fitnesses_produced[op].append(self.graph.nodes[v].get('fitness', 0.0))
        
        # Also count nodes by operator (for init nodes without edges)
        for node in self.graph.nodes():
            op = self.graph.nodes[node].get('operator', 'unknown')
            if op == 'init':
                operator_counts['init'] += 1
                fitnesses_produced['init'].append(
                    self.graph.nodes[node].get('fitness', 0.0)
                )
        
        # Calculate per-operator statistics
        per_operator = {}
        for op in self.OPERATOR_TYPES:
            imps = improvements.get(op, [])
            fits = fitnesses_produced.get(op, [])
            
            if imps:
                success_rate = sum(1 for i in imps if i > 0) / len(imps)
                mean_imp = float(np.mean(imps))
                std_imp = float(np.std(imps))
                max_imp = float(max(imps))
            else:
                success_rate = 0.0
                mean_imp = 0.0
                std_imp = 0.0
                max_imp = 0.0
            
            mean_fit = float(np.mean(fits)) if fits else 0.0
            
            per_operator[op] = {
                'count': operator_counts.get(op, 0),
                'success_rate': success_rate,
                'mean_improvement': mean_imp,
                'std_improvement': std_imp,
                'max_improvement': max_imp,
                'mean_fitness_produced': mean_fit,
            }
        
        # Determine best operators
        ops_with_data = [op for op in self.OPERATOR_TYPES 
                        if per_operator[op]['count'] > 0 and op != 'init']
        
        most_effective = max(ops_with_data, 
                            key=lambda o: per_operator[o]['success_rate']) if ops_with_data else None
        highest_avg = max(ops_with_data,
                         key=lambda o: per_operator[o]['mean_improvement']) if ops_with_data else None
        
        result = {
            'operator_counts': dict(operator_counts),
            'per_operator': per_operator,
            'most_effective': most_effective,
            'highest_avg_improvement': highest_avg,
        }
        
        self._cache['operator'] = result
        return result
    
    def trajectory_analysis(self) -> Dict[str, Any]:
        """
        Analyze search trajectory patterns.
        
        Returns:
            Dictionary with:
                - num_roots: Number of starting points (init algorithms)
                - num_leaves: Number of terminal nodes
                - avg_path_length: Average path length from roots
                - max_path_length: Maximum path length
                - branching_factor: Average number of children per node
                - dead_end_ratio: Ratio of leaves to total nodes
                - exploration_depth: Standard deviation of depths
                - convergence_ratio: Ratio of nodes in longest lineage to total
        """
        if 'trajectory' in self._cache:
            return self._cache['trajectory']
        
        n_nodes = self.graph.number_of_nodes()
        if n_nodes == 0:
            return {
                'num_roots': 0,
                'num_leaves': 0,
                'avg_path_length': 0.0,
                'max_path_length': 0,
                'branching_factor': 0.0,
                'dead_end_ratio': 0.0,
                'exploration_depth': 0.0,
                'convergence_ratio': 0.0,
            }
        
        try:
            # Find roots and leaves
            roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
            leaves = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
            
            # Calculate path lengths from all roots
            all_path_lengths = []
            for i, root in enumerate(roots):
                try:
                    lengths = nx.single_source_shortest_path_length(self.graph, root)
                    all_path_lengths.extend(lengths.values())
                    if (i + 1) % 10 == 0:
                        print(f"    Computed paths from {i + 1}/{len(roots)} roots...")
                except Exception as e:
                    print(f"    Warning: Error computing paths from root {root}: {e}")
            
            avg_path = float(np.mean(all_path_lengths)) if all_path_lengths else 0.0
            max_path = max(all_path_lengths) if all_path_lengths else 0
            
            # Calculate branching factor (average out-degree excluding leaves)
            out_degrees = [d for n, d in self.graph.out_degree() if d > 0]
            branching = float(np.mean(out_degrees)) if out_degrees else 0.0
            
            # Calculate exploration depth (variation in depths)
            depths = self._compute_depths()
            depth_std = float(np.std(list(depths.values()))) if depths else 0.0
            
            # Find longest lineage (limit search for large graphs)
            longest_lineage = 0
            if len(roots) * len(leaves) > 10000:
                print(f"    Large graph detected ({len(roots)} roots, {len(leaves)} leaves), using approximate longest lineage")
                # Use approximate method for large graphs
                longest_lineage = max_path + 1
            else:
                checked = 0
                for root in roots:
                    for leaf in leaves:
                        try:
                            path_length = nx.shortest_path_length(self.graph, root, leaf)
                            longest_lineage = max(longest_lineage, path_length + 1)
                            checked += 1
                            if checked % 1000 == 0:
                                print(f"    Checked {checked} root-leaf paths...")
                        except nx.NetworkXNoPath:
                            pass
                        except Exception as e:
                            print(f"    Warning: Error finding path from {root} to {leaf}: {e}")
            
            result = {
                'num_roots': len(roots),
                'num_leaves': len(leaves),
                'avg_path_length': avg_path,
                'max_path_length': max_path,
                'branching_factor': branching,
                'dead_end_ratio': len(leaves) / n_nodes if n_nodes > 0 else 0.0,
                'exploration_depth': depth_std,
                'convergence_ratio': longest_lineage / n_nodes if n_nodes > 0 else 0.0,
            }
            
            self._cache['trajectory'] = result
            return result
            
        except Exception as e:
            print(f"    Error in trajectory analysis: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Return default values
            return {
                'num_roots': 0,
                'num_leaves': 0,
                'avg_path_length': 0.0,
                'max_path_length': 0,
                'branching_factor': 0.0,
                'dead_end_ratio': 0.0,
                'exploration_depth': 0.0,
                'convergence_ratio': 0.0,
            }
    
    def fitness_landscape_analysis(self) -> Dict[str, Any]:
        """
        Analyze fitness distribution and progression.
        
        Returns:
            Dictionary with:
                - fitness_mean: Mean fitness across all nodes
                - fitness_std: Standard deviation of fitness
                - fitness_max: Maximum fitness
                - fitness_min: Minimum fitness (excluding errors)
                - best_node: ID of highest-fitness node
                - generation_stats: Per-generation fitness statistics
                - fitness_improvement_trend: Whether fitness improves over generations
                - num_fitness_peaks: Number of local optima
                - error_rate: Proportion of nodes with errors
        """
        if 'fitness' in self._cache:
            return self._cache['fitness']
        
        n_nodes = self.graph.number_of_nodes()
        if n_nodes == 0:
            return {
                'fitness_mean': 0.0,
                'fitness_std': 0.0,
                'fitness_max': 0.0,
                'fitness_min': 0.0,
                'best_node': None,
                'generation_stats': {},
                'fitness_improvement_trend': False,
                'num_fitness_peaks': 0,
                'error_rate': 0.0,
            }
        
        # Collect fitness values
        fitnesses = []
        generations = []
        errors = 0
        
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            fit = data.get('fitness', 0.0)
            gen = data.get('generation', 0)
            error = data.get('error', '')
            
            fitnesses.append(fit)
            generations.append(gen)
            if error:
                errors += 1
        
        # Per-generation statistics
        gen_fitness = defaultdict(list)
        for node in self.graph.nodes():
            gen = self.graph.nodes[node].get('generation', 0)
            fit = self.graph.nodes[node].get('fitness', 0.0)
            gen_fitness[gen].append(fit)
        
        generation_stats = {}
        for gen in sorted(gen_fitness.keys()):
            fits = gen_fitness[gen]
            generation_stats[gen] = {
                'mean': float(np.mean(fits)),
                'max': float(max(fits)),
                'min': float(min(fits)),
                'std': float(np.std(fits)),
                'count': len(fits),
            }
        
        # Check for improvement trend
        gen_means = [generation_stats[g]['mean'] for g in sorted(generation_stats.keys())]
        improvement_trend = False
        if len(gen_means) >= 2:
            # Trend is positive if later generations have higher mean fitness
            first_half = np.mean(gen_means[:len(gen_means)//2])
            second_half = np.mean(gen_means[len(gen_means)//2:])
            improvement_trend = second_half > first_half
        
        # Find fitness peaks (local optima - nodes with no improving successors)
        peaks = []
        for node in self.graph.nodes():
            node_fit = self.graph.nodes[node].get('fitness', 0.0)
            successors = list(self.graph.successors(node))
            
            if successors:
                successor_fits = [self.graph.nodes[s].get('fitness', 0.0) for s in successors]
                if all(sf <= node_fit for sf in successor_fits):
                    peaks.append(node)
            elif self.graph.in_degree(node) > 0:  # Leaf that's not a root
                peaks.append(node)
        
        # Best node
        best_node = max(self.graph.nodes(), 
                       key=lambda n: self.graph.nodes[n].get('fitness', 0.0))
        
        result = {
            'fitness_mean': float(np.mean(fitnesses)),
            'fitness_std': float(np.std(fitnesses)),
            'fitness_max': float(max(fitnesses)),
            'fitness_min': float(min(f for f in fitnesses if f > 0)) if any(f > 0 for f in fitnesses) else 0.0,
            'best_node': best_node,
            'best_node_fitness': self.graph.nodes[best_node].get('fitness', 0.0),
            'generation_stats': generation_stats,
            'fitness_improvement_trend': improvement_trend,
            'num_fitness_peaks': len(peaks),
            'error_rate': errors / n_nodes if n_nodes > 0 else 0.0,
        }
        
        self._cache['fitness'] = result
        return result
    
    def identify_successful_lineages(self, top_percentile: float = 0.9) -> List[Dict[str, Any]]:
        """
        Identify lineages that led to high-fitness solutions.
        
        Args:
            top_percentile: Fitness percentile to consider as "successful"
            
        Returns:
            List of lineage dictionaries with:
                - lineage: List of node IDs from root to top node
                - final_fitness: Fitness of the terminal node
                - fitness_progression: Fitness at each step
                - operators_used: Sequence of operators in the lineage
                - total_improvement: Fitness gain from start to end
        """
        n_nodes = self.graph.number_of_nodes()
        if n_nodes == 0:
            return []
        
        try:
            # Get fitness threshold
            fitnesses = [self.graph.nodes[n].get('fitness', 0.0) for n in self.graph.nodes()]
            threshold = np.percentile(fitnesses, top_percentile * 100)
            
            # Find nodes above threshold
            top_nodes = [n for n in self.graph.nodes() 
                        if self.graph.nodes[n].get('fitness', 0.0) >= threshold]
            
            print(f"    Identified {len(top_nodes)} high-fitness nodes (threshold: {threshold:.4f})")
            
            lineages = []
            seen_lineages = set()  # Avoid duplicates
            max_depth = 0  # Track maximum lineage depth to detect infinite loops
            
            for idx, top_node in enumerate(top_nodes):
                try:
                    # Trace back to root
                    lineage = [top_node]
                    current = top_node
                    visited = {current}  # Prevent infinite loops
                    
                    while len(lineage) < 1000:  # Safety limit
                        predecessors = list(self.graph.predecessors(current))
                        if not predecessors:
                            break
                        # Take first parent (could also take highest fitness parent)
                        parent = predecessors[0]
                        
                        # Check for cycles
                        if parent in visited:
                            print(f"    Warning: Cycle detected in lineage for node {top_node}")
                            break
                        
                        visited.add(parent)
                        lineage.insert(0, parent)
                        current = parent
                    
                    max_depth = max(max_depth, len(lineage))
                    
                    # Create hashable key
                    lineage_key = tuple(lineage)
                    if lineage_key in seen_lineages:
                        continue
                    seen_lineages.add(lineage_key)
                    
                    # Collect lineage info
                    fitness_progression = [self.graph.nodes[n].get('fitness', 0.0) for n in lineage]
                    operators_used = []
                    for i in range(len(lineage) - 1):
                        edge_data = self.graph.get_edge_data(lineage[i], lineage[i+1])
                        if edge_data:
                            operators_used.append(edge_data.get('operator', 'unknown'))
                        else:
                            operators_used.append('unknown')
                    
                    lineages.append({
                        'lineage': lineage,
                        'final_fitness': fitness_progression[-1],
                        'fitness_progression': fitness_progression,
                        'operators_used': operators_used,
                        'total_improvement': fitness_progression[-1] - fitness_progression[0],
                        'length': len(lineage),
                    })
                    
                    if (idx + 1) % 50 == 0:
                        print(f"    Traced {idx + 1}/{len(top_nodes)} lineages...")
                    
                except Exception as e:
                    print(f"    Warning: Error tracing lineage for node {top_node}: {e}")
                    continue
            
            # Sort by final fitness (highest first)
            lineages.sort(key=lambda x: x['final_fitness'], reverse=True)
            
            print(f"    Found {len(lineages)} unique lineages (max depth: {max_depth})")
            
            return lineages
            
        except Exception as e:
            print(f"    Error identifying lineages: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def diversity_analysis(self) -> Dict[str, Any]:
        """
        Analyze behavioral diversity based on MADA metrics.
        
        Returns:
            Dictionary with:
                - mean_nn_dist: Average nearest-neighbor distance
                - nn_dist_over_generations: NN-dist trend per generation
                - diversity_bonus_contribution: Average diversity bonus in rewards
                - alpha_progression: Alpha values over generations (if available)
        """
        if 'diversity' in self._cache:
            return self._cache['diversity']
        
        nn_dists = []
        diversity_bonuses = []
        alphas = []
        gen_nn_dist = defaultdict(list)
        gen_alpha = defaultdict(list)
        
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            nn_dist = data.get('nn_dist', 0.0)
            div_bonus = data.get('diversity_bonus', 0.0)
            alpha = data.get('alpha', 0.0)
            gen = data.get('generation', 0)
            
            # Clip NN-distance to [0, 1] range for consistency with baseline LLAMEA
            nn_dist = max(0.0, min(1.0, nn_dist))
            
            if nn_dist >= 0:  # Include zero values after clipping
                nn_dists.append(nn_dist)
                gen_nn_dist[gen].append(nn_dist)
            if div_bonus >= 0:  # Include zero values
                diversity_bonuses.append(div_bonus)
            if alpha > 0:
                alphas.append(alpha)
                gen_alpha[gen].append(alpha)
        
        # Per-generation NN-dist
        nn_dist_over_gens = {}
        for gen in sorted(gen_nn_dist.keys()):
            nn_dist_over_gens[gen] = float(np.mean(gen_nn_dist[gen]))
        
        # Per-generation alpha
        alpha_over_gens = {}
        for gen in sorted(gen_alpha.keys()):
            alpha_over_gens[gen] = float(np.mean(gen_alpha[gen]))
        
        result = {
            'mean_nn_dist': float(np.mean(nn_dists)) if nn_dists else 0.0,
            'std_nn_dist': float(np.std(nn_dists)) if nn_dists else 0.0,
            'nn_dist_over_generations': nn_dist_over_gens,
            'mean_diversity_bonus': float(np.mean(diversity_bonuses)) if diversity_bonuses else 0.0,
            'alpha_over_generations': alpha_over_gens,
            'mean_alpha': float(np.mean(alphas)) if alphas else 0.0,
        }
        
        self._cache['diversity'] = result
        return result
    
    def bandit_analysis(self) -> Dict[str, Any]:
        """
        Analyze bandit-related metrics (D-TS behavior).
        
        Returns:
            Dictionary with:
                - theta_sampled_stats: Statistics on sampled theta values
                - reward_stats: Statistics on rewards
                - operator_selection_pattern: How operators were selected over time
        """
        thetas = defaultdict(list)
        rewards = defaultdict(list)
        gen_operators = defaultdict(list)
        
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            op = data.get('operator', 'unknown')
            theta = data.get('theta_sampled', 0.0)
            reward = data.get('normalized_reward', 0.0)
            gen = data.get('generation', 0)
            
            if theta != 0.0:
                thetas[op].append(theta)
            if reward != 0.0:
                rewards[op].append(reward)
            if op not in ['init', 'unknown']:
                gen_operators[gen].append(op)
        
        # Operator selection pattern over generations
        selection_pattern = {}
        for gen in sorted(gen_operators.keys()):
            ops = gen_operators[gen]
            selection_pattern[gen] = dict(Counter(ops))
        
        # Per-operator theta/reward stats
        theta_stats = {}
        reward_stats = {}
        for op in self.OPERATOR_TYPES:
            if thetas[op]:
                theta_stats[op] = {
                    'mean': float(np.mean(thetas[op])),
                    'std': float(np.std(thetas[op])),
                    'min': float(min(thetas[op])),
                    'max': float(max(thetas[op])),
                }
            if rewards[op]:
                reward_stats[op] = {
                    'mean': float(np.mean(rewards[op])),
                    'std': float(np.std(rewards[op])),
                    'total': float(sum(rewards[op])),
                }
        
        return {
            'theta_stats': theta_stats,
            'reward_stats': reward_stats,
            'selection_pattern': selection_pattern,
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics in a single dictionary.
        
        Returns:
            Comprehensive metrics dictionary
        """
        return {
            'basic': self.basic_metrics(),
            'operator': self.operator_analysis(),
            'trajectory': self.trajectory_analysis(),
            'fitness': self.fitness_landscape_analysis(),
            'diversity': self.diversity_analysis(),
            'bandit': self.bandit_analysis(),
        }
    
    def print_summary(self) -> None:
        """Print a human-readable summary of key metrics."""
        basic = self.basic_metrics()
        ops = self.operator_analysis()
        traj = self.trajectory_analysis()
        fit = self.fitness_landscape_analysis()
        
        print("\n" + "="*60)
        print("STN METRICS SUMMARY")
        print("="*60)
        
        print("\n--- Basic Metrics ---")
        print(f"  Nodes: {basic['num_nodes']}")
        print(f"  Edges: {basic['num_edges']}")
        print(f"  DAG: {basic['is_dag']}")
        print(f"  Components: {basic['num_connected_components']}")
        print(f"  Max Depth: {basic['max_depth']}")
        
        print("\n--- Operator Analysis ---")
        print(f"  Counts: {ops['operator_counts']}")
        print(f"  Most Effective: {ops['most_effective']}")
        for op, stats in ops['per_operator'].items():
            if stats['count'] > 0:
                print(f"    {op}: {stats['count']} uses, "
                      f"{stats['success_rate']:.1%} success, "
                      f"μ_imp={stats['mean_improvement']:.4f}")
        
        print("\n--- Trajectory Analysis ---")
        print(f"  Roots: {traj['num_roots']}")
        print(f"  Leaves: {traj['num_leaves']}")
        print(f"  Avg Path Length: {traj['avg_path_length']:.2f}")
        print(f"  Branching Factor: {traj['branching_factor']:.2f}")
        
        print("\n--- Fitness Landscape ---")
        print(f"  Mean: {fit['fitness_mean']:.4f}")
        print(f"  Max: {fit['fitness_max']:.4f}")
        print(f"  Best Node: {fit['best_node']}")
        print(f"  Improvement Trend: {fit['fitness_improvement_trend']}")
        print(f"  Error Rate: {fit['error_rate']:.1%}")


def compare_experiments(
    experiments: Dict[str, nx.DiGraph]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiments: Dictionary mapping experiment names to graphs
        
    Returns:
        Dictionary with per-experiment metrics
    """
    results = {}
    
    for name, graph in experiments.items():
        metrics = STNMetrics(graph)
        results[name] = metrics.get_all_metrics()
    
    return results


if __name__ == "__main__":
    # Test metrics
    from .data_loader import ExperimentDataLoader
    from .graph_builder import STNGraphBuilder
    from pathlib import Path
    import sys
    
    if len(sys.argv) > 1:
        exp_path = Path(sys.argv[1])
        loader = ExperimentDataLoader(exp_path)
        nodes, edges = loader.load()
        
        builder = STNGraphBuilder()
        graph = builder.build_graph(nodes, edges)
        
        metrics = STNMetrics(graph)
        metrics.print_summary()

