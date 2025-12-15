# NN-Dist Implementation & Behavioral Diversity Analysis

This document describes the complete implementation of **Nearest-Neighbor Distance (NN-Dist)** for measuring behavioral diversity in MADA-LLAMEA, including the calculation method and the STN benchmark analysis.

---

## Table of Contents

1. [Overview](#overview)
2. [Trace Computation](#trace-computation)
3. [NN-Dist Calculation Method](#nn-dist-calculation-method)
4. [Integration in MADA Pipeline](#integration-in-mada-pipeline)
5. [STN Behavioral Diversity Analysis](#stn-behavioral-diversity-analysis)
6. [Complete Data Flow](#complete-data-flow)

---

## Overview

**NN-Dist (Nearest-Neighbor Distance)** quantifies how different a new algorithm's optimization behavior is from all previously evaluated algorithms. It operates on **convergence traces** (best-so-far trajectories) rather than solution values.

### Key Concepts

- **Behavioral Space**: Algorithms are represented by their convergence traces
- **Diversity Metric**: Minimum Euclidean distance to historical traces
- **Range**: [0, ~1] where higher values = more novel behavior
- **Use Case**: Encourages exploration of diverse search strategies in evolutionary algorithm design

---

## Trace Computation

Before NN-Dist can be calculated, we need to capture the **convergence trace** of each algorithm. A trace represents the optimization trajectory as a sequence of best-so-far fitness values at each evaluation step.

### What is a Trace?

A **convergence trace** (or **trajectory**) is a vector recording the best fitness found by an algorithm at each function evaluation:

```
trace = [f_best(1), f_best(2), f_best(3), ..., f_best(budget)]
```

Where `f_best(t)` is the best fitness value found by the algorithm up to evaluation `t`.

**Example**: Algorithm running for 10,000 evaluations (minimization problem)

```python
trace = [
    5.234,   # Best after eval 1
    5.234,   # No improvement at eval 2
    4.891,   # New best at eval 3
    4.891,   # No improvement at eval 4
    3.567,   # New best at eval 5
    3.567,   # No improvement at eval 6
    ...
    0.123    # Final best at eval 10,000
]
```

**Key Properties:**

- **Length**: Fixed at budget size (e.g., 10,000 evaluations)
- **Monotonic**: Non-increasing for minimization (best-so-far never gets worse)
- **Behavioral Signature**: Different search strategies produce different trajectory shapes
- **Problem-Independent**: Traces characterize _how_ an algorithm searches, not just final performance

---

### Why Traces for Diversity?

Traditional diversity metrics (e.g., genotype distance, phenotype distance) only capture _what_ solutions are found. **Behavioral diversity** based on traces captures _how_ the algorithm searches:

- **Fast Exploiter**: Steep improvement early, then plateaus
- **Steady Improver**: Consistent gradual improvement throughout
- **Late Bloomer**: Slow start, rapid improvement later
- **Explorer**: Many small improvements, explores broadly

Two algorithms with similar final fitness can have very different traces → high behavioral diversity!

---

### TraceCollector Implementation

**File**: `mada_components.py` (lines 25-100)

```python
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
        self.best_so_far = float('inf')  # For minimization
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

        # Evaluate on actual problem
        f = self.problem(x)
        self.eval_count += 1

        # Update best-so-far (minimization)
        if f < self.best_so_far:
            self.best_so_far = f

        # Record current best
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
```

---

### How Traces are Collected in Practice

**File**: `main-thesis-mada-v2.py` (lines 850-911)

Each algorithm is evaluated on **24 BBOB functions × 3 instances × 3 repetitions = 216 runs**. A trace is collected for each run.

```python
def evaluate_solution(algorithm_class, eval_budget=10000):
    """
    Evaluate an algorithm and collect traces from all runs.
    """
    aucs = []
    all_traces = []  # Store all 216 traces

    # Loop over 24 BBOB functions
    for fid in np.arange(1, 25):
        # Loop over 3 instances per function
        for iid in [1, 2, 3]:
            problem = get_problem(fid, iid, dimension=5)

            # Loop over 3 repetitions
            for rep in range(3):
                np.random.seed(rep)

                # Create trace collector wrapper
                trace_collector = TraceCollector(problem, eval_budget)

                try:
                    # Initialize algorithm
                    algorithm = algorithm_class(eval_budget)

                    # Set algorithm attributes
                    lb, ub = problem.bounds.lb, problem.bounds.ub
                    algorithm.lb = lb
                    algorithm.ub = ub
                    algorithm.bounds = (lb, ub)

                    # Run algorithm (calls trace_collector for each evaluation)
                    algorithm(trace_collector)

                except OverBudgetException:
                    # Normal termination when budget exhausted
                    pass

                # Compute AUC for this run
                auc = correct_aoc(problem, logger, eval_budget)
                aucs.append(auc)

                # Collect trace for this run
                run_trace = trace_collector.get_trace()
                all_traces.append(run_trace)  # <-- Stored here

                # Reset for next run
                logger.reset(problem)
                problem.reset()

    # Aggregate all 216 traces into single representative trace
    aggregated_trace = aggregate_traces(all_traces, eval_budget)

    return aucs, aggregated_trace
```

---

### Trace Aggregation

Since each algorithm runs on 216 different problems, we aggregate all individual traces into a **single representative trace** for NN-Dist calculation.

**File**: `mada_components.py` (lines 103-130)

```python
def aggregate_traces(traces: List[List[float]], budget: int) -> List[float]:
    """
    Aggregate multiple traces into a single representative trace.
    Uses element-wise median for robustness to outliers.

    Args:
        traces: List of trace lists from different runs (e.g., 216 traces)
        budget: Expected trace length (e.g., 10,000)

    Returns:
        list: Aggregated trace of length budget
    """
    if not traces:
        return [0.0] * budget

    # Ensure all traces are same length
    padded = []
    for t in traces:
        if len(t) < budget:
            # Pad with final value if algorithm terminated early
            final_val = t[-1] if t else 0.0
            t = t + [final_val] * (budget - len(t))
        padded.append(t[:budget])

    # Element-wise median (robust to outliers)
    arr = np.array(padded, dtype=float)
    aggregated = np.median(arr, axis=0).tolist()

    return aggregated
```

**Why Median?**

- **Robustness**: Median is less sensitive to outliers than mean
- **Element-wise**: Computed independently at each evaluation step
- **Representation**: Provides a typical convergence behavior across diverse problems

---

### Trace Example

**Scenario**: Algorithm A evaluated on 3 problems

```python
# Individual traces (10 evaluations each)
trace_1 = [5.2, 5.2, 4.8, 4.8, 3.5, 3.5, 2.1, 2.1, 1.8, 1.8]
trace_2 = [8.1, 7.3, 6.5, 5.9, 5.2, 4.8, 4.3, 3.9, 3.6, 3.2]
trace_3 = [6.7, 6.7, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]

# Aggregated trace (element-wise median)
aggregated = [6.7, 6.7, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5]
#             ^^^  ^^^  ^^^  ^^^  ^^^  ^^^  ^^^  ^^^  ^^^  ^^^
#            median at each evaluation step
```

**Result**: The aggregated trace represents the "typical" convergence behavior of Algorithm A across different problems.

---

### Visualization: Different Algorithms Produce Different Traces

```
Fitness
   │
 1 │  ╔════════════════════════════════  Algorithm A (slow converger)
   │  ║
   │  ║
 0.8│  ║  ╔═══════════════════  Algorithm B (steady improver)
   │  ║  ║
   │  ║  ║     ╔═════  Algorithm C (fast exploiter)
 0.6│  ║  ║     ║
   │  ║  ║     ║
   │  ║  ║     ╚═══════════════  Algorithm C plateaus
 0.4│  ║  ║
   │  ║  ║
   │  ║  ╚════════════════════════  Algorithm B plateaus
 0.2│  ║
   │  ╚══════════════════════════════════  Algorithm A plateaus
   │
 0  └──────────────────────────────────────────────────────────
    0        2500       5000       7500      10000
                    Evaluations
```

**Key Insight**: Different convergence patterns = different behavioral signatures = high NN-Dist between algorithms

---

## NN-Dist Calculation Method

### Implementation Location

**File**: `mada_components.py`

**Function**: `calculate_nn_dist()`

### Algorithm

```python
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
```

### Step-by-Step Breakdown

#### Step 1: Handle Empty History

```python
if not history_traces:
    return 1.0  # Maximum diversity for first algorithm
```

**Rationale**: The first evaluated algorithm has no comparison baseline, so it receives the maximum diversity score.

#### Step 2: Trace Normalization

```python
# Normalize new trace to [0, 1]
range_val = norm_max - norm_min
if range_val > 1e-10:
    new_normalized = (new_arr - norm_min) / range_val
```

**Why Normalize?**

- Makes traces from different fitness scales comparable
- Uses global min/max across ALL evaluations for consistency
- Prevents scale bias (e.g., fitness in [0, 1] vs [0, 1000])

#### Step 3: Compute Distances to All Historical Traces

```python
for hist_trace in history_traces:
    # Normalize history trace
    hist_normalized = (hist_arr - norm_min) / range_val

    # Euclidean distance
    distance = np.sqrt(np.sum((new_trimmed - hist_trimmed) ** 2))

    # Normalize by sqrt(budget)
    distance = distance / np.sqrt(min_len)

    min_distance = min(min_distance, distance)
```

**Key Details:**

- **Euclidean Distance**: Standard L2 norm in trace space
- **Length Handling**: Trims to shorter length if traces differ
- **Sqrt Normalization**: Divides by √(budget) to scale to [0, ~1]

#### Step 4: Return Nearest-Neighbor Distance

```python
return min_distance  # Minimum distance to any historical trace
```

**Interpretation:**

- `nn_dist = 0.0`: Identical behavior to a previous algorithm
- `nn_dist = 0.5`: Moderately novel behavior
- `nn_dist = 1.0`: Extremely novel behavior (or first algorithm)

---

## Integration in MADA Pipeline

### Data Flow in Evolutionary Loop

**File**: `main-thesis-mada-v2.py`

```python
# During offspring evaluation (lines 1185-1228):

# 1. Evaluate algorithm and collect trace
trace = trace_collector.collect_trace(algorithm, problem)

# 2. Update global fitness bounds
if trace:
    global_min_fitness = min(global_min_fitness, min(trace))
    global_max_fitness = max(global_max_fitness, max(trace))

# 3. Calculate NN-Dist
if trace and history_traces:
    nn_dist = calculate_nn_dist(
        trace,
        history_traces,
        global_min_fitness,
        global_max_fitness
    )
else:
    nn_dist = 1.0

print(f"  NN-Dist: {nn_dist:.4f}")

# 4. Use in composite reward
reward, fitness_delta, diversity_bonus = compute_composite_reward(
    child_fitness=child.fitness,
    parent_fitness=parent_fitness_for_reward,
    nn_dist=nn_dist,
    alpha=current_alpha,
    error=bool(error),
    clamp=args.reward_clamp
)

# 5. Add trace to history for future comparisons
if trace:
    history_traces.append(trace)

# 6. Log NN-Dist with offspring record
offspring_record = {
    'operator': child.operator,
    'fitness': child.fitness,
    'nn_dist': nn_dist,  # <-- Logged to mada_offspring.jsonl
    'alpha': current_alpha,
    'raw_reward': reward,
    'normalized_reward': normalized_reward,
    'fitness_delta': fitness_delta,
    'diversity_bonus': diversity_bonus,
    'generation': generation,
    # ... other fields
}
log_mada_offspring(explogger, api_calls, offspring_record)
```

### Composite Reward Formula

```python
def compute_composite_reward(
    child_fitness: float,
    parent_fitness: float,
    nn_dist: float,
    alpha: float,
    error: bool = False,
    clamp: float = 10.0
) -> Tuple[float, float, float]:
    """
    Composite reward: R = Δfitness + α × NN-Dist

    Returns:
        (reward, fitness_delta, diversity_bonus)
    """
    if error:
        return -1.0, 0.0, 0.0

    fitness_delta = child_fitness - parent_fitness
    diversity_bonus = alpha * nn_dist

    reward = fitness_delta + diversity_bonus

    # Clamp to prevent extreme rewards
    if clamp > 0:
        reward = np.clip(reward, -clamp, clamp)

    return reward, fitness_delta, diversity_bonus
```

**Key Points:**

- `fitness_delta`: Exploitation signal (improve on parent)
- `diversity_bonus`: Exploration signal (try novel behaviors)
- `alpha`: Controls exploration-exploitation tradeoff (annealed from 0.8 → 0.4)

---

## STN Behavioral Diversity Analysis

The **STN-Analyzer** reads the logged `nn_dist` values from `mada_offspring.jsonl` and performs post-hoc analysis of behavioral diversity trends over generations.

### Data Loading

**File**: `STN-Analyzer/stn/data_loader.py`

```python
class AlgorithmNode:
    """
    Represents an evaluated algorithm in the STN.

    Attributes:
        nn_dist: Nearest-neighbor distance (behavioral diversity)
        generation: Generation when evaluated
        fitness: Mean AOCC score
        alpha: Diversity weight at evaluation time
        diversity_bonus: Diversity component of reward
        # ... other fields
    """
    nn_dist: float = 0.0
    generation: int = 0
    alpha: float = 0.0
    diversity_bonus: float = 0.0
    # ...

def _parse_mada_record(self, record: Dict) -> None:
    """
    Parse a single offspring record from mada_offspring.jsonl.
    """
    node = AlgorithmNode(
        node_id=f"mada_{attempt:06d}",
        generation=record.get('generation', 0),
        fitness=float(record.get('fitness', 0.0)),
        nn_dist=float(record.get('nn_dist', 0.0)),  # <-- Loaded here
        alpha=float(record.get('alpha', 0.0)),
        diversity_bonus=float(record.get('diversity_bonus', 0.0)),
        # ...
    )
    self.nodes[node_id] = node
```

### Diversity Analysis Over Generations

**File**: `STN-Analyzer/stn/metrics.py`

```python
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
    gen_nn_dist = defaultdict(list)
    gen_alpha = defaultdict(list)

    # Collect NN-Dist values by generation
    for node in self.graph.nodes():
        data = self.graph.nodes[node]
        nn_dist = data.get('nn_dist', 0.0)
        alpha = data.get('alpha', 0.0)
        gen = data.get('generation', 0)

        if nn_dist > 0:
            nn_dists.append(nn_dist)
            gen_nn_dist[gen].append(nn_dist)  # Group by generation

        if alpha > 0:
            gen_alpha[gen].append(alpha)

    # Compute per-generation mean NN-Dist
    nn_dist_over_gens = {}
    for gen in sorted(gen_nn_dist.keys()):
        nn_dist_over_gens[gen] = float(np.mean(gen_nn_dist[gen]))

    # Compute per-generation mean alpha
    alpha_over_gens = {}
    for gen in sorted(gen_alpha.keys()):
        alpha_over_gens[gen] = float(np.mean(gen_alpha[gen]))

    result = {
        'mean_nn_dist': float(np.mean(nn_dists)) if nn_dists else 0.0,
        'std_nn_dist': float(np.std(nn_dists)) if nn_dists else 0.0,
        'nn_dist_over_generations': nn_dist_over_gens,  # <-- Key output
        'mean_diversity_bonus': float(np.mean(diversity_bonuses)) if diversity_bonuses else 0.0,
        'alpha_over_generations': alpha_over_gens,
        'mean_alpha': float(np.mean(alphas)) if alphas else 0.0,
    }

    self._cache['diversity'] = result
    return result
```

**Output Format:**

```json
{
  "diversity": {
    "mean_nn_dist": 0.0791,
    "std_nn_dist": 0.0523,
    "nn_dist_over_generations": {
      "0": 0.0124,
      "1": 0.0076,
      "2": 0.0154,
      "3": 0.0628,
      "4": 0.1645,
      "5": 0.1823,
      "6": 0.1621
    },
    "alpha_over_generations": {
      "0": 0.8,
      "1": 0.8,
      "2": 0.7,
      "3": 0.6,
      "4": 0.5,
      "5": 0.4,
      "6": 0.4
    }
  }
}
```

### Visualization

**File**: `STN-Analyzer/stn/visualizer.py`

```python
def plot_diversity_metrics(
    self,
    filename: str = "diversity_metrics.png",
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150
) -> Optional[Path]:
    """
    Plot MADA diversity-related metrics over generations.

    Creates a 2-panel figure:
    - Left: NN-Distance over generations (with std deviation bands)
    - Right: Alpha (α) values over generations

    Returns:
        Path to saved PNG file
    """
    if not HAS_MATPLOTLIB:
        return None

    # Collect diversity data
    gen_nn_dist = defaultdict(list)
    gen_alpha = defaultdict(list)

    for node in self.graph.nodes():
        gen = self.graph.nodes[node].get('generation', 0)
        nn_dist = self.graph.nodes[node].get('nn_dist', 0.0)
        alpha = self.graph.nodes[node].get('alpha', 0.0)

        if nn_dist > 0:
            gen_nn_dist[gen].append(nn_dist)
        if alpha > 0:
            gen_alpha[gen].append(alpha)

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

        # Plot mean with standard deviation bands
        ax1.plot(gens, means, 'o-', color='#4ECDC4', linewidth=2, markersize=8)
        ax1.fill_between(
            gens,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            color='#4ECDC4', alpha=0.2
        )

    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('NN-Distance', fontsize=11)
    ax1.set_title('Behavioral Diversity (NN-Dist) Over Generations', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Alpha over generations
    ax2 = axes[1]
    if gen_alpha:
        gens = sorted(gen_alpha.keys())
        alpha_means = [np.mean(gen_alpha[g]) for g in gens]

        ax2.plot(gens, alpha_means, 'o-', color='#FF6B6B', linewidth=2, markersize=8)

    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Alpha (α)', fontsize=11)
    ax2.set_title('Diversity Weight (α) Over Generations', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = self.output_dir / filename
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    return output_path
```

**Output**: `diversity_metrics.png`

**Interpretation Example:**

| Generation | NN-Distance | Alpha (α) | Interpretation                     |
| ---------- | ----------- | --------- | ---------------------------------- |
| 0-1        | 0.007-0.012 | 0.8       | Low diversity (exploitation phase) |
| 2-3        | 0.015-0.063 | 0.7-0.6   | Moderate diversity (transition)    |
| 4-6        | 0.160-0.182 | 0.5-0.4   | High diversity (exploration phase) |

**Key Findings:**

- **11× diversity increase**: From 0.015 (early) → 0.165 (late)
- **α-annealing effect**: As α decreases, NN-Dist increases (more exploration)
- **Effective exploration-exploitation balance**: Early generations exploit, later generations explore

---

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     MADA-LLAMEA EXPERIMENT                           │
│                     (main-thesis-mada-v2.py)                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 1. Algorithm Evaluation: Run on 216 benchmarks         │
    │    - 24 BBOB functions × 3 instances × 3 reps          │
    │    - Each run wrapped with TraceCollector              │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 2. TraceCollector: Capture convergence traces          │
    │    For each evaluation call:                           │
    │      - Evaluate x on problem: f = problem(x)           │
    │      - Update best_so_far if f < best_so_far           │
    │      - Append best_so_far to trace                     │
    │    Result: 216 traces, each length = budget (10,000)   │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 3. aggregate_traces(): Combine 216 traces              │
    │    - Pad traces to same length if needed               │
    │    - Compute element-wise median across all traces     │
    │    Result: Single representative trace [10,000 values] │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 4. calculate_nn_dist(): Compute diversity              │
    │    - Normalize aggregated trace using global_min/max   │
    │    - Compute Euclidean distance to all history traces  │
    │    - Return minimum distance                           │
    │    Result: nn_dist = 0.0845                            │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 5. compute_composite_reward(): Use in reward           │
    │    reward = fitness_delta + α × nn_dist                │
    │    Example: 0.023 + 0.7 × 0.0845 = 0.0821              │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 6. log_mada_offspring(): Save to JSONL                 │
    │    File: mada_offspring.jsonl                          │
    │    {                                                    │
    │      "attempt": 57,                                     │
    │      "generation": 3,                                   │
    │      "nn_dist": 0.0845,                                 │
    │      "alpha": 0.7,                                      │
    │      "fitness": 0.4563,                                 │
    │      ...                                                 │
    │    }                                                    │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     STN-ANALYZER POST-PROCESSING                     │
│                     (STN-Analyzer/stn_analyzer.py)                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 7. ExperimentDataLoader: Parse JSONL                   │
    │    nodes[i].nn_dist = record['nn_dist']                │
    │    nodes[i].generation = record['generation']          │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 8. STNMetrics.diversity_analysis():                    │
    │    Group NN-Dist by generation                         │
    │    gen_nn_dist[gen] = [nn_dist values...]              │
    │    Compute mean per generation                         │
    │    Result: {0: 0.012, 1: 0.008, ..., 6: 0.162}         │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 9. STNVisualizer.plot_diversity_metrics():             │
    │    Create 2-panel plot:                                │
    │    - Left: NN-Distance over generations                │
    │    - Right: Alpha (α) over generations                 │
    │    Output: diversity_metrics.png                       │
    └────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────┐
    │ 10. Save metrics to JSON:                              │
    │    File: stn_metrics.json                              │
    │    {                                                    │
    │      "diversity": {                                     │
    │        "mean_nn_dist": 0.0791,                          │
    │        "nn_dist_over_generations": {...},               │
    │        "alpha_over_generations": {...}                  │
    │      }                                                   │
    │    }                                                    │
    └────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

### Trace Computation

1. **Purpose**: Capture convergence behavior as behavioral signature
2. **Method**: Best-so-far tracking at each evaluation (monotonic non-increasing)
3. **Aggregation**: Element-wise median across 216 runs for robustness
4. **Length**: Fixed at budget size (10,000 evaluations)

### NN-Dist Calculation

1. **Purpose**: Quantify behavioral diversity in trace space
2. **Method**: Normalized Euclidean distance to nearest historical trace
3. **Range**: [0, ~1] where higher = more novel
4. **Normalization**: Global min/max fitness + √(budget) scaling

### MADA Integration

1. **Composite Reward**: `R = Δfitness + α × NN-Dist`
2. **Exploration-Exploitation**: α anneals from 0.8 → 0.4 over generations
3. **Logging**: All NN-Dist values saved to `mada_offspring.jsonl`

### STN Benchmark Analysis

1. **Data Source**: Reads pre-computed NN-Dist from experiment logs
2. **Aggregation**: Groups by generation, computes mean and std
3. **Visualization**: `diversity_metrics.png` shows trends over time
4. **Insights**: Reveals 11× diversity increase in successful experiments

---

## Files Reference

| File                              | Purpose                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------ |
| `mada_components.py`              | Core components: `TraceCollector`, `aggregate_traces()`, `calculate_nn_dist()` |
| `main-thesis-mada-v2.py`          | MADA evolutionary loop with trace collection and NN-Dist integration           |
| `STN-Analyzer/stn/data_loader.py` | Load NN-Dist from experiment logs                                              |
| `STN-Analyzer/stn/metrics.py`     | Aggregate NN-Dist by generation                                                |
| `STN-Analyzer/stn/visualizer.py`  | Plot diversity metrics over time                                               |

---

## Example Usage

### Collect Traces During Experiment

```python
from mada_components import TraceCollector, aggregate_traces

# Evaluate algorithm on multiple problems
all_traces = []

for problem in benchmark_suite:
    # Wrap problem with trace collector
    trace_collector = TraceCollector(problem, budget=10000)

    # Run algorithm
    algorithm(trace_collector)

    # Get trace for this run
    run_trace = trace_collector.get_trace()
    all_traces.append(run_trace)

# Aggregate all traces into single representative trace
aggregated_trace = aggregate_traces(all_traces, budget=10000)
print(f"Aggregated trace length: {len(aggregated_trace)}")
```

### Calculate NN-Dist During Experiment

```python
from mada_components import calculate_nn_dist

# Initialize
history_traces = []
global_min_fitness = float('inf')
global_max_fitness = float('-inf')

# For each evaluated algorithm
for algorithm in population:
    trace = evaluate_and_collect_trace(algorithm)

    # Update global bounds
    global_min_fitness = min(global_min_fitness, min(trace))
    global_max_fitness = max(global_max_fitness, max(trace))

    # Calculate diversity
    nn_dist = calculate_nn_dist(
        trace,
        history_traces,
        global_min_fitness,
        global_max_fitness
    )

    print(f"NN-Dist: {nn_dist:.4f}")

    # Add to history
    history_traces.append(trace)
```

### Analyze NN-Dist from STN

```python
from stn.metrics import STNMetrics

# Load experiment data
metrics = STNMetrics(graph)

# Get diversity analysis
diversity = metrics.diversity_analysis()

print(f"Mean NN-Dist: {diversity['mean_nn_dist']:.4f}")
print("\nNN-Dist over generations:")
for gen, dist in diversity['nn_dist_over_generations'].items():
    print(f"  Gen {gen}: {dist:.4f}")

# Plot
visualizer.plot_diversity_metrics()
```

---

## References

- **MADA5.0.md**: Original design document
- **MADA-Pipeline.md**: Full pipeline description
- **STN-Analyzer/README.md**: STN analysis guide
- **Behaviour Space Analysis.pdf**: Theoretical background

---

_Last Updated: December 2024_
