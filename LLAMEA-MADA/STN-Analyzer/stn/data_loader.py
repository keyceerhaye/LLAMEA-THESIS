"""
STN Data Loader for MADA-LLAMEA Experiments

Loads and parses experiment results from MADA-LLAMEA to build STN graphs.
Reads mada_offspring.jsonl files which contain parent-child relationships,
operator information, and fitness scores.

Data Sources:
- mada_offspring.jsonl: Primary data source with lineage information
- try-X-aucs.txt: Detailed fitness scores (AUC values)
- code/try-X-*.py: Algorithm source code files
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm


@dataclass
class AlgorithmNode:
    """
    Represents a single algorithm/solution in the STN.
    
    Each node corresponds to one algorithm generated during the MADA-LLAMEA run.
    The node captures both the algorithm's identity and its evaluation results.
    
    Attributes:
        node_id: Unique identifier (e.g., 'mada_000001')
        attempt: The attempt/try number (0-indexed)
        generation: Which generation this algorithm belongs to
        fitness: Mean AOCC (Area Over Convergence Curve) score
        fitness_std: Standard deviation of AOCC scores
        operator: How this algorithm was created ('init', 'mutation', 'crossover', 'refine')
        parent_ids: List of parent algorithm IDs
        code_file: Path to the algorithm's source code
        algorithm_name: Class name of the algorithm
        description: Short description of the algorithm
        nn_dist: Nearest-neighbor distance (behavioral diversity)
        fitness_delta: Fitness improvement over parent
        raw_reward: Raw bandit reward
        normalized_reward: Normalized bandit reward
        diversity_bonus: Diversity component of reward
        error: Error message if evaluation failed
        detailed_aucs: Per-function-group AUC scores [sep, multi, weak, misc, multi]
    """
    node_id: str
    attempt: int
    generation: int = 0
    fitness: float = 0.0
    fitness_std: float = 0.0
    operator: str = "unknown"
    parent_ids: List[str] = field(default_factory=list)
    code_file: Optional[Path] = None
    algorithm_name: str = ""
    description: str = ""
    
    # MADA-specific attributes
    nn_dist: float = 0.0
    fitness_delta: float = 0.0
    raw_reward: float = 0.0
    normalized_reward: float = 0.0
    diversity_bonus: float = 0.0
    alpha: float = 0.0
    theta_sampled: float = 0.0
    error: str = ""
    
    # Detailed fitness
    detailed_aucs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, AlgorithmNode):
            return self.node_id == other.node_id
        return False


@dataclass
class STNEdge:
    """
    Represents a transition in the search trajectory.
    
    Each edge represents a parent-child relationship between algorithms,
    capturing how the search progressed from one solution to another.
    
    Attributes:
        source: Parent algorithm ID
        target: Child algorithm ID
        operator: Operator used to create child ('mutation', 'crossover', 'refine')
        fitness_improvement: Change in fitness from parent to child
        generation_step: Generation when this transition occurred
        weight: Edge weight for visualization (default 1.0)
    """
    source: str
    target: str
    operator: str
    fitness_improvement: float
    generation_step: int
    weight: float = 1.0


class ExperimentDataLoader:
    """
    Load and parse MADA-LLAMEA experiment data for STN analysis.
    
    This class handles reading experiment directories and extracting
    all necessary information to build Search Trajectory Networks.
    
    Expected directory structure:
        experiment_dir/
        ├── mada_offspring.jsonl    # Primary data source
        ├── bandit_snapshots.jsonl  # Optional bandit state history
        ├── try-X-aucs.txt          # AUC scores per algorithm
        ├── code/
        │   ├── try-0-AlgorithmName.py
        │   └── ...
        └── conversationlog.txt     # Optional conversation log
    
    Usage:
        loader = ExperimentDataLoader(Path("exp-12-14_100247-..."))
        nodes, edges = loader.load()
    """
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize the loader with an experiment directory.
        
        Args:
            experiment_dir: Path to the experiment directory
        """
        self.exp_dir = Path(experiment_dir)
        self.nodes: Dict[str, AlgorithmNode] = {}
        self.edges: List[STNEdge] = []
        self.bandit_history: List[Dict] = []
        
        # Metadata
        self.experiment_name = self.exp_dir.name
        self.model_name = self._extract_model_name()
    
    def _extract_model_name(self) -> str:
        """Extract model name from experiment directory name."""
        # Format: exp-MM-DD_HHMMSS-model-name-experiment-type
        name = self.exp_dir.name
        parts = name.split('-')
        if len(parts) >= 4:
            # Skip exp, date, time and join model parts
            model_parts = parts[3:]
            # Remove common suffixes
            for suffix in ['mada', 'v2', 'experiment', 'evolutionary', 'elitism']:
                if suffix in model_parts:
                    idx = model_parts.index(suffix)
                    model_parts = model_parts[:idx]
                    break
            return '-'.join(model_parts)
        return "unknown"
    
    def load(self) -> Tuple[Dict[str, AlgorithmNode], List[STNEdge]]:
        """
        Load all experiment data and return nodes and edges.
        
        Returns:
            Tuple of (nodes_dict, edges_list):
                - nodes_dict: Dictionary mapping node_id to AlgorithmNode
                - edges_list: List of STNEdge objects
        
        Raises:
            FileNotFoundError: If mada_offspring.jsonl doesn't exist
        """
        self._load_offspring_log()
        self._load_fitness_scores()
        self._link_code_files()
        self._load_bandit_history()
        
        return self.nodes, self.edges
    
    def _load_offspring_log(self) -> None:
        """
        Parse mada_offspring.jsonl for parent-child relationships.
        
        This is the primary data source containing:
        - Operator used (mutation/crossover/refine)
        - Parent IDs
        - Fitness and reward information
        - Generation number
        """
        jsonl_path = self.exp_dir / "mada_offspring.jsonl"
        
        if not jsonl_path.exists():
            # Try to construct from AUC files only
            print(f"  Note: No mada_offspring.jsonl found, constructing from AUC files")
            self._construct_from_aucs()
            return
        
        print(f"  Loading offspring log from: {jsonl_path}")
        
        try:
            # Count total lines first for progress bar
            print(f"  Counting lines in offspring log...")
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for line in f if line.strip())
            print(f"  Found {total_lines} records to process")
        except Exception as e:
            print(f"  Error counting lines: {e}")
            total_lines = None
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                pbar = tqdm(total=total_lines, desc="Loading offspring records", unit="record") if total_lines else None
                processed = 0
                errors = 0
                
                for line_num, line in enumerate(f, 1):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        record = json.loads(line)
                        self._process_offspring_record(record)
                        processed += 1
                        
                        # Log progress every 100 records
                        if processed % 100 == 0:
                            print(f"  Processed {processed} offspring records...")
                        
                    except json.JSONDecodeError as e:
                        errors += 1
                        print(f"  Warning: Invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        errors += 1
                        print(f"  Warning: Error processing line {line_num}: {type(e).__name__}: {e}")
                    
                    if pbar:
                        pbar.update(1)
                
                if pbar:
                    pbar.close()
                
                print(f"  Successfully processed {processed} records ({errors} errors)")
                
        except Exception as e:
            print(f"  Error reading offspring log: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_offspring_record(self, record: Dict[str, Any]) -> None:
        """
        Process a single offspring record into node and edges.
        
        Args:
            record: Dictionary from mada_offspring.jsonl containing:
                - attempt: int
                - operator: str
                - fitness: float
                - parent_fitness: float
                - parent_ids: List[str]
                - generation: int
                - nn_dist: float
                - alpha: float
                - raw_reward: float
                - normalized_reward: float
                - fitness_delta: float
                - diversity_bonus: float
                - theta_sampled: float
                - error: str (optional)
        """
        attempt = record.get('attempt', 0)
        node_id = f"mada_{attempt:06d}"
        
        # Get parent IDs (handle both string and list formats)
        parent_ids = record.get('parent_ids', [])
        if isinstance(parent_ids, str):
            parent_ids = [parent_ids] if parent_ids else []
        
        # Create node
        node = AlgorithmNode(
            node_id=node_id,
            attempt=attempt,
            generation=record.get('generation', 0),
            fitness=float(record.get('fitness', 0.0)),
            operator=record.get('operator', 'unknown'),
            parent_ids=parent_ids,
            nn_dist=float(record.get('nn_dist', 0.0)),
            fitness_delta=float(record.get('fitness_delta', 0.0)),
            raw_reward=float(record.get('raw_reward', 0.0)),
            normalized_reward=float(record.get('normalized_reward', 0.0)),
            diversity_bonus=float(record.get('diversity_bonus', 0.0)),
            alpha=float(record.get('alpha', 0.0)),
            theta_sampled=float(record.get('theta_sampled', 0.0)),
            error=record.get('error', ''),
        )
        self.nodes[node_id] = node
        
        # Create edges from parents to this node
        parent_fitness = float(record.get('parent_fitness', 0.0))
        for parent_id in parent_ids:
            # Ensure parent_id exists
            if parent_id not in self.nodes:
                # Create placeholder parent node if it doesn't exist yet
                parent_attempt = self._extract_attempt_from_id(parent_id)
                if parent_attempt is not None:
                    parent_node = AlgorithmNode(
                        node_id=parent_id,
                        attempt=parent_attempt,
                        generation=max(0, node.generation - 1),
                        fitness=parent_fitness,
                        operator='init',
                    )
                    self.nodes[parent_id] = parent_node
            
            edge = STNEdge(
                source=parent_id,
                target=node_id,
                operator=node.operator,
                fitness_improvement=node.fitness - parent_fitness,
                generation_step=node.generation,
            )
            self.edges.append(edge)
    
    def _extract_attempt_from_id(self, node_id: str) -> Optional[int]:
        """Extract attempt number from node ID."""
        match = re.match(r'mada_(\d+)', node_id)
        if match:
            return int(match.group(1))
        return None
    
    def _construct_from_aucs(self) -> None:
        """
        Construct basic nodes from AUC files when mada_offspring.jsonl is missing.
        
        This is a fallback for older experiment formats or baseline experiments.
        No edge information will be available.
        """
        for auc_file in self.exp_dir.glob("try-*-aucs.txt"):
            try:
                stem = auc_file.stem  # try-X-aucs
                parts = stem.split('-')
                if len(parts) >= 2:
                    attempt = int(parts[1])
                    node_id = f"mada_{attempt:06d}"
                    
                    fitness = self._parse_auc_file(auc_file)
                    
                    node = AlgorithmNode(
                        node_id=node_id,
                        attempt=attempt,
                        generation=0,  # Unknown without offspring log
                        fitness=fitness,
                        operator='unknown',
                    )
                    self.nodes[node_id] = node
            except (ValueError, IndexError) as e:
                print(f"  Warning: Could not parse {auc_file.name}: {e}")
    
    def _load_fitness_scores(self) -> None:
        """
        Load detailed fitness scores from AUC files.
        
        Updates nodes with:
        - More accurate fitness values
        - Standard deviation of fitness
        - Detailed per-group AUC scores
        """
        print(f"  Searching for AUC files in: {self.exp_dir}")
        try:
            auc_files = list(self.exp_dir.glob("try-*-aucs.txt"))
            print(f"  Found {len(auc_files)} AUC files")
        except Exception as e:
            print(f"  Error searching for AUC files: {type(e).__name__}: {e}")
            return
        
        if not auc_files:
            print(f"  No AUC files found")
            return
        
        loaded = 0
        errors = 0
        
        for auc_file in tqdm(auc_files, desc="Loading fitness scores", unit="file"):
            try:
                stem = auc_file.stem
                parts = stem.split('-')
                if len(parts) >= 2:
                    attempt = int(parts[1])
                    node_id = f"mada_{attempt:06d}"
                    
                    if node_id in self.nodes:
                        fitness, std, aucs = self._parse_auc_file_detailed(auc_file)
                        self.nodes[node_id].fitness = fitness
                        self.nodes[node_id].fitness_std = std
                        # Store detailed AUCs so we can derive diversity metrics when
                        # MADA-specific nn_dist/alpha/diversity_bonus are not available
                        if aucs:
                            self.nodes[node_id].detailed_aucs = aucs
                        loaded += 1
                        
                        # Log progress every 50 files
                        if loaded % 50 == 0:
                            print(f"  Loaded fitness for {loaded} nodes...")
                    
            except (ValueError, IndexError) as e:
                errors += 1
                if errors < 5:  # Only show first few errors
                    print(f"  Warning: Could not parse {auc_file.name}: {e}")
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  Warning: Error loading {auc_file.name}: {type(e).__name__}: {e}")
        
        print(f"  Loaded fitness for {loaded} nodes ({errors} files skipped)")

        # If this experiment has no MADA-specific diversity info (nn_dist),
        # approximate behavioral diversity directly from detailed AUC vectors.
        # This allows baseline LLAMEA runs (no bandit snapshots, no offspring log)
        # to still get meaningful diversity metrics.
        self._compute_behavioral_diversity_from_aucs()
    
    def _parse_auc_file(self, auc_file: Path) -> float:
        """
        Parse AUC file to get mean fitness score.
        
        Args:
            auc_file: Path to try-X-aucs.txt file
            
        Returns:
            float: Mean AUC score, or 0.0 if parsing fails
        """
        try:
            with open(auc_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                if lines:
                    values = [float(v) for v in lines if v]
                    return float(np.mean(values)) if values else 0.0
        except Exception:
            pass
        return 0.0
    
    def _parse_auc_file_detailed(self, auc_file: Path) -> Tuple[float, float, List[float]]:
        """
        Parse AUC file for mean, std, and all values.
        
        Args:
            auc_file: Path to try-X-aucs.txt file
            
        Returns:
            Tuple of (mean, std, values_list)
        """
        try:
            with open(auc_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                if lines:
                    values = [float(v) for v in lines if v]
                    if values:
                        return float(np.mean(values)), float(np.std(values)), values
        except Exception:
            pass
        return 0.0, 0.0, []

    def _compute_behavioral_diversity_from_aucs(self) -> None:
        """
        Derive a nearest-neighbour diversity estimate from detailed AUC vectors.
        
        This is a fallback primarily for baseline LLAMEA experiments where
        MADA-specific diversity signals (nn_dist, alpha, diversity_bonus)
        are not available because there is no mada_offspring.jsonl.
        
        For each node with a non-empty detailed_aucs vector, we compute:
            nn_dist = 1 - cosine_similarity(to its nearest neighbour)
        clipped to [0, 1] for consistency with MADA runs.
        
        We only run this heuristic if *none* of the nodes currently has
        a positive nn_dist, to avoid overwriting real MADA diversity data.
        """
        if not self.nodes:
            return

        # If any node already has non-zero nn_dist, assume this is a MADA run
        # with proper diversity logging and do nothing.
        if any(getattr(n, "nn_dist", 0.0) > 0.0 for n in self.nodes.values()):
            return

        node_ids: List[str] = []
        vectors: List[np.ndarray] = []

        # First collect all valid vectors with their lengths
        lengths: Dict[int, int] = {}
        raw_entries: List[Tuple[str, np.ndarray]] = []

        for node in self.nodes.values():
            aucs = getattr(node, "detailed_aucs", None)
            if not aucs:
                continue
            try:
                vec = np.array(aucs, dtype=float)
                if vec.size == 0 or not np.all(np.isfinite(vec)):
                    continue
                raw_entries.append((node.node_id, vec))
                lengths[vec.size] = lengths.get(vec.size, 0) + 1
            except Exception:
                continue

        if len(raw_entries) < 2:
            # Need at least two distinct algorithms to define a NN distance
            return

        # Use the most common vector length to avoid shape mismatches
        mode_len = max(lengths.items(), key=lambda kv: kv[1])[0]

        for node_id, vec in raw_entries:
            if vec.size != mode_len:
                # Skip odd-length vectors (e.g., different logging formats)
                continue
            node_ids.append(node_id)
            vectors.append(vec)

        if len(vectors) < 2:
            return

        # Stack to (N, D) with consistent D
        mat = np.vstack(vectors)
        # Normalise for cosine similarity; guard against zero-norm
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_norm = mat / norms

        # Cosine similarity matrix then convert to distance
        sim_matrix = mat_norm @ mat_norm.T
        np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
        dist_matrix = 1.0 - sim_matrix  # in [0, 2]

        for i, node_id in enumerate(node_ids):
            # Exclude self-distance
            dists = dist_matrix[i].copy()
            dists[i] = np.inf
            finite_mask = np.isfinite(dists)
            if not finite_mask.any():
                continue
            nn = float(dists[finite_mask].min())
            # Clip to [0, 1] to keep same scale as MADA nn_dist
            nn = max(0.0, min(1.0, nn))
            if node_id in self.nodes:
                self.nodes[node_id].nn_dist = nn
    
    def _link_code_files(self) -> None:
        """
        Link algorithm code files to nodes.
        
        Also extracts algorithm names from file names and code.
        """
        code_dir = self.exp_dir / "code"
        if not code_dir.exists():
            print(f"  No code directory found at: {code_dir}")
            return
        
        print(f"  Searching for code files in: {code_dir}")
        try:
            code_files = list(code_dir.glob("try-*.py"))
            print(f"  Found {len(code_files)} code files")
        except Exception as e:
            print(f"  Error searching for code files: {type(e).__name__}: {e}")
            return
        
        if not code_files:
            print(f"  No code files found")
            return
        
        linked = 0
        errors = 0
        
        for code_file in tqdm(code_files, desc="Linking code files", unit="file"):
            try:
                stem = code_file.stem  # try-X-AlgorithmName
                parts = stem.split('-', 2)  # Split into [try, X, AlgorithmName]
                if len(parts) >= 2:
                    attempt = int(parts[1])
                    node_id = f"mada_{attempt:06d}"
                    
                    if node_id in self.nodes:
                        self.nodes[node_id].code_file = code_file
                        if len(parts) >= 3:
                            self.nodes[node_id].algorithm_name = parts[2]
                        linked += 1
                        
                        # Log progress every 50 files
                        if linked % 50 == 0:
                            print(f"  Linked {linked} code files...")
                    
            except (ValueError, IndexError) as e:
                errors += 1
                if errors < 5:
                    print(f"  Warning: Could not parse {code_file.name}: {e}")
            except Exception as e:
                errors += 1
                if errors < 5:
                    print(f"  Warning: Error linking {code_file.name}: {type(e).__name__}: {e}")
        
        print(f"  Linked {linked} code files ({errors} files skipped)")
    
    def _load_bandit_history(self) -> None:
        """Load bandit snapshot history if available."""
        bandit_file = self.exp_dir / "bandit_snapshots.jsonl"
        if not bandit_file.exists():
            print(f"  No bandit snapshot file found")
            return
        
        print(f"  Loading bandit history from: {bandit_file}")
        
        try:
            with open(bandit_file, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            self.bandit_history.append(record)
                            line_count += 1
                        except json.JSONDecodeError as e:
                            print(f"  Warning: Invalid JSON in bandit snapshot line {line_count}: {e}")
                
                print(f"  Loaded {line_count} bandit snapshots")
        except Exception as e:
            print(f"  Warning: Could not load bandit history: {type(e).__name__}: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the experiment.
        
        Returns:
            Dictionary with experiment metadata and statistics
        """
        if not self.nodes:
            return {"error": "No data loaded"}
        
        fitnesses = [n.fitness for n in self.nodes.values()]
        generations = [n.generation for n in self.nodes.values()]
        operators = [n.operator for n in self.nodes.values()]
        
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_generations": max(generations) if generations else 0,
            "fitness_mean": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "fitness_max": max(fitnesses) if fitnesses else 0.0,
            "fitness_min": min(fitnesses) if fitnesses else 0.0,
            "operator_counts": {op: operators.count(op) for op in set(operators)},
        }


def load_multiple_experiments(base_dir: Path) -> Dict[str, Tuple[Dict[str, AlgorithmNode], List[STNEdge]]]:
    """
    Load multiple experiments from a base directory.
    
    Args:
        base_dir: Directory containing experiment folders
        
    Returns:
        Dictionary mapping experiment names to (nodes, edges) tuples
    """
    results = {}
    exp_dirs = [d for d in base_dir.iterdir() 
                if d.is_dir() and d.name.startswith('exp-')]
    
    for exp_dir in sorted(exp_dirs):
        try:
            loader = ExperimentDataLoader(exp_dir)
            nodes, edges = loader.load()
            if nodes:
                results[exp_dir.name] = (nodes, edges)
        except Exception as e:
            print(f"  Warning: Could not load {exp_dir.name}: {e}")
    
    return results


if __name__ == "__main__":
    # Test the loader
    import sys
    if len(sys.argv) > 1:
        exp_path = Path(sys.argv[1])
        loader = ExperimentDataLoader(exp_path)
        nodes, edges = loader.load()
        print(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
        print(f"Summary: {loader.get_experiment_summary()}")

