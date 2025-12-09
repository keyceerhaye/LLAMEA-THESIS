"""
MADA-LLAMEA Components

Core components for the Multi-Adaptive Diverse Algorithm framework:
- TraceCollector: Captures optimization trajectories
- RewardNormalizer: Running standardization for bandit rewards
- AlphaScheduler: Diversity weight decay scheduler
- calculate_nn_dist: Nearest-neighbor distance for behavioral diversity

Based on MADA5.0.md Design Document
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ==============================================================================
# TRACE COLLECTION
# ==============================================================================

class OverBudgetException(Exception):
    """Raised when evaluation budget is exceeded."""
    pass


class TraceCollector:
    """
    Wrapper around IOH problem to collect best-so-far trace during optimization.
    
    This enables MADA behavioral diversity calculation by tracking the
    convergence trajectory of each algorithm run.
    
    Usage:
        collector = TraceCollector(problem, budget=10000)
        algorithm(collector)  # Run algorithm with collector as fitness function
        trace = collector.get_trace()  # Get best-so-far at each step
    """
    
    def __init__(self, problem, budget: int):
        """
        Args:
            problem: IOH problem instance (or any callable with .bounds attribute)
            budget: Maximum number of evaluations
        """
        self.problem = problem
        self.budget = budget
        self.trace: List[float] = []
        self.best_so_far = float('inf')
        self.eval_count = 0
        
        # Copy bounds from problem for compatibility
        self.bounds = problem.bounds
    
    def __call__(self, x) -> float:
        """
        Evaluate x on the problem and record best-so-far.
        
        Args:
            x: Solution vector to evaluate
            
        Returns:
            float: Fitness value
            
        Raises:
            OverBudgetException: If budget is exceeded
        """
        if self.eval_count >= self.budget:
            raise OverBudgetException(f"Budget {self.budget} exceeded at eval {self.eval_count}")
        
        f = self.problem(x)
        self.eval_count += 1
        
        # Update best-so-far (minimization)
        if f < self.best_so_far:
            self.best_so_far = f
        
        self.trace.append(self.best_so_far)
        
        return f
    
    def get_trace(self) -> List[float]:
        """
        Get the trace vector, padded to budget length.
        
        Returns:
            list: Best-so-far fitness at each evaluation step, length = budget
        """
        trace = self.trace.copy()
        
        # Pad with final value if run ended early
        if len(trace) < self.budget:
            final_value = trace[-1] if trace else float('inf')
            trace.extend([final_value] * (self.budget - len(trace)))
        
        return trace[:self.budget]
    
    def reset(self):
        """Reset collector for a new run."""
        self.trace = []
        self.best_so_far = float('inf')
        self.eval_count = 0


def aggregate_traces(traces: List[List[float]], budget: int) -> List[float]:
    """
    Aggregate multiple traces into a single representative trace.
    Uses element-wise median for robustness to outliers.
    
    Args:
        traces: List of trace lists from different runs
        budget: Expected trace length
        
    Returns:
        list: Aggregated trace of length budget
    """
    if not traces:
        return [0.0] * budget
    
    # Ensure all traces are same length
    padded = []
    for t in traces:
        if len(t) < budget:
            final_val = t[-1] if t else 0.0
            t = t + [final_val] * (budget - len(t))
        padded.append(t[:budget])
    
    # Element-wise median (robust to outliers)
    arr = np.array(padded, dtype=float)
    aggregated = np.median(arr, axis=0).tolist()
    
    return aggregated


# ==============================================================================
# BEHAVIORAL DIVERSITY (NN-DIST)
# ==============================================================================

def calculate_nn_dist(
    new_trace: List[float],
    history_traces: List[List[float]],
    global_min: Optional[float] = None,
    global_max: Optional[float] = None
) -> float:
    """
    Calculate nearest-neighbor distance for behavioral diversity.
    
    This measures how different a new algorithm's convergence behavior is
    from all previously seen algorithms. Higher distance = more novel behavior.
    
    Mathematical Definition (from MADA5.0):
        d(x_new, x_hist) = sqrt(sum((x_new[t] - x_hist[t])^2))
        DiversityScore = min(d(x_new, h) for h in history)
    
    Args:
        new_trace: Convergence trace of new algorithm [best_so_far_0, ..., best_so_far_B]
        history_traces: Traces of previously evaluated algorithms
        global_min: Minimum fitness seen across all runs (for normalization)
        global_max: Maximum fitness seen across all runs (for normalization)
    
    Returns:
        float: Nearest-neighbor distance in [0, ~1] range (normalized)
    """
    # Handle empty history case - maximum novelty for first algorithm
    if not history_traces:
        return 1.0
    
    new_arr = np.array(new_trace, dtype=float)
    
    # Determine normalization range
    if global_min is not None and global_max is not None:
        norm_min, norm_max = global_min, global_max
    else:
        # Auto-compute from new trace
        norm_min, norm_max = np.min(new_arr), np.max(new_arr)
    
    # Normalize new trace to [0, 1]
    range_val = norm_max - norm_min
    if range_val > 1e-10:
        new_normalized = (new_arr - norm_min) / range_val
    else:
        new_normalized = np.zeros_like(new_arr)
    
    min_distance = float('inf')
    
    for hist_trace in history_traces:
        hist_arr = np.array(hist_trace, dtype=float)
        
        # Normalize history trace
        if global_min is not None and global_max is not None:
            if range_val > 1e-10:
                hist_normalized = (hist_arr - norm_min) / range_val
            else:
                hist_normalized = np.zeros_like(hist_arr)
        else:
            hist_min, hist_max = np.min(hist_arr), np.max(hist_arr)
            hist_range = hist_max - hist_min
            if hist_range > 1e-10:
                hist_normalized = (hist_arr - hist_min) / hist_range
            else:
                hist_normalized = np.zeros_like(hist_arr)
        
        # Handle length mismatch (use shorter length)
        min_len = min(len(new_normalized), len(hist_normalized))
        new_trimmed = new_normalized[:min_len]
        hist_trimmed = hist_normalized[:min_len]
        
        # Euclidean distance
        distance = np.sqrt(np.sum((new_trimmed - hist_trimmed) ** 2))
        
        # Normalize by sqrt(budget) to get approximately [0, 1] range
        if min_len > 0:
            distance = distance / np.sqrt(min_len)
        else:
            distance = 0.0
        
        min_distance = min(min_distance, distance)
    
    return min_distance if min_distance != float('inf') else 1.0


# ==============================================================================
# REWARD NORMALIZATION
# ==============================================================================

class RewardNormalizer:
    """
    Running standardization for bandit rewards.
    
    Bandits like D-TS assume rewards follow a stable distribution.
    This class applies running standardization: R_final = (R - μ) / σ
    
    Uses Welford's online algorithm for numerical stability.
    
    From MADA5.0 Section 2.1:
        R_final = (R_t - μ_R) / σ_R
    """
    
    def __init__(self, warmup_period: int = 5):
        """
        Args:
            warmup_period: Number of rewards to collect before normalizing.
                          During warmup, raw rewards are returned.
        """
        self.warmup_period = warmup_period
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean (Welford)
        self.rewards_history: List[float] = []
    
    def update(self, reward: float) -> None:
        """
        Update running statistics with new reward (Welford's algorithm).
        
        Args:
            reward: Raw reward value
        """
        self.rewards_history.append(reward)
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        delta2 = reward - self.mean
        self.M2 += delta * delta2
    
    @property
    def variance(self) -> float:
        """Current variance estimate (sample variance)."""
        if self.n < 2:
            return 1.0  # Default variance during warmup
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        """Current standard deviation estimate."""
        return np.sqrt(self.variance)
    
    def normalize(self, reward: float) -> float:
        """
        Normalize a reward using running statistics.
        
        Args:
            reward: Raw reward value
            
        Returns:
            float: Normalized reward (z-score), or raw reward during warmup
        """
        # Update stats first
        self.update(reward)
        
        # During warmup, return raw reward (not enough data for stable stats)
        if self.n < self.warmup_period:
            return reward
        
        # Z-score normalization
        std = self.std
        if std < 1e-10:
            return 0.0  # Avoid division by zero
        
        return (reward - self.mean) / std
    
    def get_stats(self) -> Dict:
        """Get current statistics for logging."""
        return {
            'n': self.n,
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
        }
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.rewards_history = []


# ==============================================================================
# ALPHA SCHEDULER
# ==============================================================================

class AlphaScheduler:
    """
    Scheduler for the diversity weight parameter α.
    
    The importance of diversity should fade as the run progresses:
    - Start: High α → Explore diverse behaviors
    - End: Low/zero α → Exploit best-performing strategies
    
    From MADA5.0 Section 2.2:
        α_t = α_start × (1 - t/T_max)
    
    Supports multiple scheduling strategies:
    - linear: Linear decay from start to end
    - exponential: Exponential decay with configurable rate
    - cosine: Cosine annealing (smoother transitions)
    - constant: No decay (fixed alpha)
    """
    
    VALID_SCHEDULES = ['linear', 'exponential', 'cosine', 'constant']
    
    def __init__(
        self,
        alpha_start: float = 0.5,
        alpha_end: float = 0.0,
        t_max: int = 100,
        schedule: str = 'linear'
    ):
        """
        Args:
            alpha_start: Initial diversity weight (default 0.5)
            alpha_end: Final diversity weight (default 0.0)
            t_max: Total number of generations/steps
            schedule: Decay schedule type ('linear', 'exponential', 'cosine', 'constant')
        """
        if schedule not in self.VALID_SCHEDULES:
            raise ValueError(f"Invalid schedule '{schedule}'. Must be one of {self.VALID_SCHEDULES}")
        
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.t_max = max(1, t_max)  # Avoid division by zero
        self.schedule = schedule
    
    def get_alpha(self, t: int) -> float:
        """
        Get the diversity weight for step t.
        
        Args:
            t: Current step/generation (0-indexed)
            
        Returns:
            float: Current alpha value
        """
        if self.schedule == 'constant':
            return self.alpha_start
        
        # Clamp t to [0, t_max]
        t = max(0, min(t, self.t_max))
        progress = t / self.t_max
        
        if self.schedule == 'linear':
            # α(t) = α_start + (α_end - α_start) × (t/T_max)
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
            
        elif self.schedule == 'exponential':
            # α(t) = α_start × exp(-3 × t/T_max) (decay constant = 3)
            decay_rate = 3.0
            alpha = self.alpha_start * np.exp(-decay_rate * progress)
            alpha = max(alpha, self.alpha_end)
            
        elif self.schedule == 'cosine':
            # Cosine annealing: smoother transition
            # α(t) = α_end + 0.5 × (α_start - α_end) × (1 + cos(π × t/T_max))
            alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (1 + np.cos(np.pi * progress))
            
        else:
            alpha = self.alpha_start
        
        return alpha
    
    def __repr__(self) -> str:
        return f"AlphaScheduler(start={self.alpha_start}, end={self.alpha_end}, t_max={self.t_max}, schedule='{self.schedule}')"


# ==============================================================================
# COMPOSITE REWARD
# ==============================================================================

def compute_composite_reward(
    child_fitness: float,
    parent_fitness: float,
    nn_dist: float,
    alpha: float,
    error: bool = False,
    clamp: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute the MADA composite reward combining exploitation and exploration.
    
    Formula from MADA5.0:
        R_t = (Fit_child - Fit_parent) + α × NN-Dist
    
    Args:
        child_fitness: Fitness of offspring algorithm (higher is better)
        parent_fitness: Fitness of parent (or best parent)
        nn_dist: Nearest-neighbor distance (behavioral diversity) in [0, 1]
        alpha: Diversity weight from AlphaScheduler
        error: Whether the offspring had an error during evaluation
        clamp: Maximum absolute reward value for numerical stability
        
    Returns:
        Tuple of (composite_reward, fitness_delta, diversity_bonus)
    """
    if error:
        # Penalize errors heavily
        return -clamp, 0.0, 0.0
    
    # Exploitation signal: fitness improvement
    fitness_delta = child_fitness - parent_fitness
    
    # Exploration signal: behavioral diversity bonus
    diversity_bonus = alpha * nn_dist
    
    # Composite reward
    raw_reward = fitness_delta + diversity_bonus
    
    # Clamp for numerical stability
    reward = max(-clamp, min(clamp, raw_reward))
    
    # Small positive signal for ties (prevents bandit stagnation)
    if abs(reward) < 1e-6:
        reward = 0.01
    
    return reward, fitness_delta, diversity_bonus


# ==============================================================================
# TESTING / VALIDATION
# ==============================================================================

def _test_components():
    """Quick validation of MADA components."""
    print("Testing MADA Components...")
    
    # Test AlphaScheduler
    scheduler = AlphaScheduler(alpha_start=1.0, alpha_end=0.0, t_max=10, schedule='linear')
    alphas = [scheduler.get_alpha(t) for t in range(11)]
    assert abs(alphas[0] - 1.0) < 0.01, f"Alpha start failed: {alphas[0]}"
    assert abs(alphas[10] - 0.0) < 0.01, f"Alpha end failed: {alphas[10]}"
    print("  ✓ AlphaScheduler: linear decay works")
    
    # Test RewardNormalizer
    normalizer = RewardNormalizer(warmup_period=3)
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5]
    normalized = [normalizer.normalize(r) for r in rewards]
    assert normalizer.n == 5, f"Count failed: {normalizer.n}"
    assert abs(normalizer.mean - 0.3) < 0.01, f"Mean failed: {normalizer.mean}"
    print("  ✓ RewardNormalizer: running stats work")
    
    # Test calculate_nn_dist
    trace1 = [1.0, 0.9, 0.8, 0.7, 0.6]
    trace2 = [1.0, 0.8, 0.6, 0.4, 0.2]
    trace3 = [1.0, 0.9, 0.8, 0.7, 0.6]  # Same as trace1
    
    # First trace should have max diversity
    dist1 = calculate_nn_dist(trace1, [])
    assert dist1 == 1.0, f"Empty history should return 1.0: {dist1}"
    
    # Different trace should have positive distance
    dist2 = calculate_nn_dist(trace2, [trace1])
    assert 0 < dist2 < 1, f"Different trace distance: {dist2}"
    
    # Same trace should have zero distance
    dist3 = calculate_nn_dist(trace3, [trace1])
    assert dist3 < 0.01, f"Same trace should have ~0 distance: {dist3}"
    print("  ✓ calculate_nn_dist: diversity calculation works")
    
    # Test composite reward
    reward, fit_d, div_b = compute_composite_reward(
        child_fitness=0.8,
        parent_fitness=0.6,
        nn_dist=0.5,
        alpha=0.4,
        error=False
    )
    expected_fit_delta = 0.2
    expected_div_bonus = 0.2  # 0.4 * 0.5
    expected_total = 0.4
    assert abs(fit_d - expected_fit_delta) < 0.01, f"Fitness delta: {fit_d}"
    assert abs(div_b - expected_div_bonus) < 0.01, f"Diversity bonus: {div_b}"
    assert abs(reward - expected_total) < 0.01, f"Total reward: {reward}"
    print("  ✓ compute_composite_reward: formula correct")
    
    print("\n✅ All MADA component tests passed!")


if __name__ == "__main__":
    _test_components()

