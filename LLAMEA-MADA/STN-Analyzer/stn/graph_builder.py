"""
STN Graph Builder for MADA-LLAMEA Experiments

Builds NetworkX directed graphs from algorithm nodes and edges.
Provides filtering, subgraph extraction, and path analysis capabilities.

Graph Structure:
- Nodes: Algorithms with attributes (fitness, generation, operator, etc.)
- Edges: Parent-child relationships with attributes (operator, improvement, etc.)

Based on STN (Search Trajectory Networks) theory from:
"Search trajectory networks: A tool for analysing and visualising 
 the behaviour of metaheuristics"
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from .data_loader import AlgorithmNode, STNEdge


class STNGraphBuilder:
    """
    Build NetworkX graphs from MADA experiment data.
    
    This class constructs directed graphs where:
    - Each node represents an algorithm discovered during the search
    - Each edge represents a parent-child relationship (search transition)
    
    The resulting graph can be analyzed for:
    - Search trajectory patterns
    - Operator effectiveness
    - Algorithm lineages
    - Fitness landscapes
    
    Usage:
        builder = STNGraphBuilder()
        graph = builder.build_graph(nodes, edges)
        mutation_subgraph = builder.filter_by_operator('mutation')
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, AlgorithmNode] = {}
        self._edges: List[STNEdge] = []
    
    def build_graph(
        self,
        nodes: Dict[str, AlgorithmNode],
        edges: List[STNEdge]
    ) -> nx.DiGraph:
        """
        Build complete STN graph from nodes and edges.
        
        Args:
            nodes: Dictionary mapping node_id to AlgorithmNode
            edges: List of STNEdge objects
            
        Returns:
            NetworkX DiGraph with all node and edge attributes
        """
        self._nodes = nodes
        self._edges = edges
        self.graph = nx.DiGraph()
        
        # Add nodes with all attributes
        for node_id, node in nodes.items():
            self.graph.add_node(
                node_id,
                attempt=node.attempt,
                generation=node.generation,
                fitness=node.fitness,
                fitness_std=node.fitness_std,
                operator=node.operator,
                nn_dist=node.nn_dist,
                fitness_delta=node.fitness_delta,
                raw_reward=node.raw_reward,
                normalized_reward=node.normalized_reward,
                diversity_bonus=node.diversity_bonus,
                alpha=node.alpha,
                theta_sampled=node.theta_sampled,
                algorithm_name=node.algorithm_name,
                code_file=str(node.code_file) if node.code_file else None,
                error=node.error,
                parent_ids=node.parent_ids,
            )
        
        # Add edges with all attributes
        for edge in edges:
            # Only add edge if both source and target exist
            if edge.source in self.graph and edge.target in self.graph:
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    operator=edge.operator,
                    fitness_improvement=edge.fitness_improvement,
                    generation_step=edge.generation_step,
                    weight=edge.weight,
                )
        
        return self.graph
    
    def filter_by_operator(self, operator: str) -> nx.DiGraph:
        """
        Create subgraph containing only edges with specified operator.
        
        Args:
            operator: Operator type to filter ('mutation', 'crossover', 'refine')
            
        Returns:
            Subgraph with only edges of the specified operator type
        """
        edges_to_keep = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if d.get('operator') == operator
        ]
        
        if not edges_to_keep:
            return nx.DiGraph()
        
        return self.graph.edge_subgraph(edges_to_keep).copy()
    
    def filter_by_generation(
        self,
        min_gen: int = 0,
        max_gen: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Create subgraph containing only nodes within generation range.
        
        Args:
            min_gen: Minimum generation (inclusive)
            max_gen: Maximum generation (inclusive), or None for no limit
            
        Returns:
            Subgraph with only nodes in the specified generation range
        """
        nodes_to_keep = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n]['generation'] >= min_gen and
               (max_gen is None or self.graph.nodes[n]['generation'] <= max_gen)
        ]
        
        return self.graph.subgraph(nodes_to_keep).copy()
    
    def filter_by_fitness(
        self,
        min_fitness: float = 0.0,
        max_fitness: Optional[float] = None
    ) -> nx.DiGraph:
        """
        Create subgraph containing only nodes within fitness range.
        
        Args:
            min_fitness: Minimum fitness (inclusive)
            max_fitness: Maximum fitness (inclusive), or None for no limit
            
        Returns:
            Subgraph with only nodes in the specified fitness range
        """
        nodes_to_keep = [
            n for n in self.graph.nodes()
            if self.graph.nodes[n]['fitness'] >= min_fitness and
               (max_fitness is None or self.graph.nodes[n]['fitness'] <= max_fitness)
        ]
        
        return self.graph.subgraph(nodes_to_keep).copy()
    
    def get_improvement_paths(self, min_length: int = 2) -> List[List[str]]:
        """
        Find paths where fitness consistently improves.
        
        An improvement path is a sequence of nodes where each subsequent
        node has higher fitness than its predecessor.
        
        Args:
            min_length: Minimum path length to include
            
        Returns:
            List of node ID lists representing improvement paths
        """
        improvement_paths = []
        
        # Find all leaf nodes (no outgoing edges)
        leaves = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        
        # Find all root nodes (no incoming edges)
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        # For each root, find paths to leaves that show improvement
        for root in roots:
            for leaf in leaves:
                if root == leaf:
                    continue
                try:
                    paths = list(nx.all_simple_paths(self.graph, root, leaf, cutoff=20))
                    for path in paths:
                        if len(path) >= min_length and self._is_improvement_path(path):
                            improvement_paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        # Sort by path length (longest first) and fitness improvement
        improvement_paths.sort(key=lambda p: (
            len(p),
            self.graph.nodes[p[-1]]['fitness'] - self.graph.nodes[p[0]]['fitness']
        ), reverse=True)
        
        return improvement_paths
    
    def _is_improvement_path(self, path: List[str]) -> bool:
        """
        Check if path shows consistent fitness improvement.
        
        Args:
            path: List of node IDs
            
        Returns:
            True if fitness increases along the path
        """
        if len(path) < 2:
            return False
        
        fitnesses = [self.graph.nodes[n]['fitness'] for n in path]
        
        # Allow for ties (non-decreasing) but require at least some improvement
        has_improvement = False
        for i in range(len(fitnesses) - 1):
            if fitnesses[i] > fitnesses[i + 1]:
                return False  # Fitness decreased
            if fitnesses[i] < fitnesses[i + 1]:
                has_improvement = True
        
        return has_improvement
    
    def get_lineage(self, node_id: str, direction: str = "ancestors") -> List[str]:
        """
        Get the lineage of a node (ancestors or descendants).
        
        Args:
            node_id: ID of the node to trace lineage from
            direction: "ancestors" for parents, "descendants" for children
            
        Returns:
            List of node IDs in lineage order
        """
        if node_id not in self.graph:
            return []
        
        lineage = [node_id]
        
        if direction == "ancestors":
            # Trace back through parents
            current = node_id
            visited = {current}
            while True:
                predecessors = list(self.graph.predecessors(current))
                if not predecessors:
                    break
                # Take highest fitness parent if multiple
                parent = max(predecessors, 
                            key=lambda n: self.graph.nodes[n].get('fitness', 0))
                if parent in visited:
                    break
                visited.add(parent)
                lineage.insert(0, parent)
                current = parent
        
        elif direction == "descendants":
            # Trace forward through children
            current = node_id
            visited = {current}
            while True:
                successors = list(self.graph.successors(current))
                if not successors:
                    break
                # Take highest fitness child if multiple
                child = max(successors,
                           key=lambda n: self.graph.nodes[n].get('fitness', 0))
                if child in visited:
                    break
                visited.add(child)
                lineage.append(child)
                current = child
        
        return lineage
    
    def get_all_lineages_to_top(self, top_n: int = 5) -> List[List[str]]:
        """
        Get lineages leading to the top N highest-fitness nodes.
        
        Args:
            top_n: Number of top nodes to trace lineages for
            
        Returns:
            List of lineage lists (from root to top node)
        """
        # Get top N nodes by fitness
        nodes_by_fitness = sorted(
            self.graph.nodes(),
            key=lambda n: self.graph.nodes[n].get('fitness', 0),
            reverse=True
        )[:top_n]
        
        lineages = []
        for node_id in nodes_by_fitness:
            lineage = self.get_lineage(node_id, direction="ancestors")
            lineages.append(lineage)
        
        return lineages
    
    def get_connected_components(self) -> List[Set[str]]:
        """
        Get weakly connected components of the graph.
        
        Returns:
            List of sets, each containing node IDs in a connected component
        """
        return list(nx.weakly_connected_components(self.graph))
    
    def get_roots(self) -> List[str]:
        """
        Get all root nodes (nodes with no incoming edges).
        
        Returns:
            List of root node IDs
        """
        return [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
    
    def get_leaves(self) -> List[str]:
        """
        Get all leaf nodes (nodes with no outgoing edges).
        
        Returns:
            List of leaf node IDs
        """
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
    
    def get_best_node(self) -> Optional[str]:
        """
        Get the node with highest fitness.
        
        Returns:
            Node ID of best node, or None if graph is empty
        """
        if not self.graph.nodes():
            return None
        
        return max(
            self.graph.nodes(),
            key=lambda n: self.graph.nodes[n].get('fitness', 0)
        )
    
    def compute_node_depths(self) -> Dict[str, int]:
        """
        Compute depth (distance from nearest root) for each node.
        
        Returns:
            Dictionary mapping node_id to depth
        """
        depths = {}
        roots = self.get_roots()
        
        for root in roots:
            # BFS from this root
            queue = [(root, 0)]
            while queue:
                node, depth = queue.pop(0)
                if node not in depths or depths[node] > depth:
                    depths[node] = depth
                    for successor in self.graph.successors(node):
                        queue.append((successor, depth + 1))
        
        # Handle any nodes not reached (disconnected)
        for node in self.graph.nodes():
            if node not in depths:
                depths[node] = 0
        
        return depths
    
    def get_operator_subgraphs(self) -> Dict[str, nx.DiGraph]:
        """
        Get separate subgraphs for each operator type.
        
        Returns:
            Dictionary mapping operator names to their subgraphs
        """
        operators = set()
        for _, _, data in self.graph.edges(data=True):
            operators.add(data.get('operator', 'unknown'))
        
        return {op: self.filter_by_operator(op) for op in operators}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": [
                {"id": n, **self.graph.nodes[n]}
                for n in self.graph.nodes()
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ],
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> nx.DiGraph:
        """
        Create graph from dictionary representation.
        
        Args:
            data: Dictionary with 'nodes' and 'edges' keys
            
        Returns:
            NetworkX DiGraph
        """
        graph = nx.DiGraph()
        
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            graph.add_node(node_id, **node_data)
        
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            graph.add_edge(source, target, **edge_data)
        
        return graph


def merge_graphs(graphs: List[nx.DiGraph], prefix_names: bool = True) -> nx.DiGraph:
    """
    Merge multiple STN graphs into one.
    
    Useful for comparing runs or creating aggregate visualizations.
    
    Args:
        graphs: List of NetworkX DiGraphs to merge
        prefix_names: If True, prefix node IDs with graph index
        
    Returns:
        Merged NetworkX DiGraph
    """
    merged = nx.DiGraph()
    
    for i, graph in enumerate(graphs):
        prefix = f"g{i}_" if prefix_names else ""
        
        for node, data in graph.nodes(data=True):
            new_id = f"{prefix}{node}"
            merged.add_node(new_id, graph_index=i, **data)
        
        for u, v, data in graph.edges(data=True):
            new_u = f"{prefix}{u}"
            new_v = f"{prefix}{v}"
            merged.add_edge(new_u, new_v, graph_index=i, **data)
    
    return merged


if __name__ == "__main__":
    # Test the graph builder
    from .data_loader import ExperimentDataLoader
    from pathlib import Path
    import sys
    
    if len(sys.argv) > 1:
        exp_path = Path(sys.argv[1])
        loader = ExperimentDataLoader(exp_path)
        nodes, edges = loader.load()
        
        builder = STNGraphBuilder()
        graph = builder.build_graph(nodes, edges)
        
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Roots: {len(builder.get_roots())}")
        print(f"Leaves: {len(builder.get_leaves())}")
        print(f"Best node: {builder.get_best_node()}")
        print(f"Components: {len(builder.get_connected_components())}")








