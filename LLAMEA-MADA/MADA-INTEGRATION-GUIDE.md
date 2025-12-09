# MADA-LLAMEA Integration Guide

> **Based on:** MADA5.0.md Design Document  
> **Target Codebase:** `main-thesis-eoh-dts.py`, `managers.py`  
> **Date:** December 2024

---

## 1. Gap Analysis

### 1.1 Functions That Need Modification

| File                     | Function/Class                    | Current Behavior                          | Required Change                                                                                |
| ------------------------ | --------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `main-thesis-eoh-dts.py` | `evaluate_algorithm()`            | Returns `(aucs, detailed_aucs, error)`    | Return `(aucs, detailed_aucs, trace, error)` where `trace` is best-so-far fitness at each step |
| `main-thesis-eoh-dts.py` | `EoHOperatorDTS.update_bandit()`  | Uses only `fitness_delta` as reward       | Use composite reward: `R = ΔFitness + α × NN-Dist`                                             |
| `main-thesis-eoh-dts.py` | `run_eoh_dts_evolutionary_mode()` | No trace history tracking                 | Maintain `history_traces` list, calculate NN-Dist                                              |
| `main-thesis-eoh-dts.py` | `DiscountedThompsonSampler`       | Works correctly but reward not normalized | Add reward normalization before update                                                         |

### 1.2 New Classes/Functions to Create

| Component             | Type     | Description                                                          |
| --------------------- | -------- | -------------------------------------------------------------------- |
| `calculate_nn_dist()` | Function | Compute nearest-neighbor Euclidean distance for behavioral diversity |
| `RewardNormalizer`    | Class    | Running standardization of rewards (μ, σ rolling stats)              |
| `AlphaScheduler`      | Class    | Decaying exploration weight scheduler                                |
| `TraceCollector`      | Class    | Wrapper to collect optimization trace during evaluation              |

---

## 2. Implementation Code

### 2.1 Metric Engineering

#### 2.1.1 Modify `evaluate_algorithm()` to Return Trace Vector

**Location:** `main-thesis-eoh-dts.py`, line ~762

**Current Implementation:**

```python
def evaluate_algorithm(algorithm_code, algorithm_name, eval_budget):
    # ... returns (aucs, detailed_aucs, error)
```

**New Implementation:**

```python
def evaluate_algorithm_with_trace(algorithm_code, algorithm_name, eval_budget):
    """
    Evaluates a single algorithm on BBOB benchmark suite.
    Returns trace vector along with fitness for MADA behavioral diversity.

    Returns:
        tuple: (aucs, detailed_aucs, aggregated_trace, error)
            - aucs: List of AUC scores per function/instance/rep
            - detailed_aucs: [5] group-wise mean AUCs
            - aggregated_trace: List[float] of length eval_budget (best-so-far per step)
            - error: Error string or empty
    """
    try:
        safe_globals = {
            "lb": -5.0, "ub": 5.0, "bounds": (-5.0, 5.0),
            "learning_rate": 0.5, "local_search_prob": 0.1,
            "stagnation_multiplier": 1.0, "initial_pop": None,
            "pop_size": 50, "F": 0.5, "CR": 0.9,
            "archive_size_multiplier": 1.0, "eps": 1e-12,
        }

        _orig_np_choice = np.random.choice

        def _safe_choice(a, size=None, replace=False, p=None, axis=0, shuffle=True):
            try:
                if not replace and size is not None:
                    if len(a) < size:
                        replace = True
            except Exception:
                pass
            return _orig_np_choice(a, size=size, replace=replace, p=p)

        np.random.choice = _safe_choice
        exec_globals = {**globals(), **safe_globals}
        exec(algorithm_code, exec_globals)

        if algorithm_name not in exec_globals:
            return [], [0, 0, 0, 0, 0], [], f"Class {algorithm_name} not found"

        algorithm_class = exec_globals[algorithm_name]
        l2 = aoc_logger(eval_budget, upper=1e2, triggers=[logger.trigger.ALWAYS])

        aucs = []
        detail_aucs = []
        detailed_aucs = [0, 0, 0, 0, 0]

        # Trace collection: aggregate across all runs
        all_traces = []

        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)

                for rep in range(3):
                    np.random.seed(rep)

                    # Create trace collector wrapper
                    trace_collector = TraceCollector(problem, eval_budget)

                    try:
                        algorithm = algorithm_class(eval_budget)
                        lb, ub = problem.bounds.lb, problem.bounds.ub
                        for attr, val in [("lb", lb), ("ub", ub), ("bounds", (lb, ub)),
                                         ("pop", []), ("archive_x", []), ("archive_f", []),
                                         ("evals", 0), ("F", 0.5), ("CR", 0.9),
                                         ("archive_size_multiplier", 1.0), ("eps", 1e-12)]:
                            if not hasattr(algorithm, attr):
                                setattr(algorithm, attr, val)
                        if not hasattr(algorithm, "fitness"):
                            algorithm.fitness = lambda x: trace_collector(x)

                        # Run with trace collection
                        algorithm(trace_collector)

                    except OverBudgetException:
                        pass
                    except Exception as e:
                        return [], [0, 0, 0, 0, 0], [], str(e)

                    auc = correct_aoc(problem, l2, eval_budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)

                    # Collect trace (padded to eval_budget length)
                    run_trace = trace_collector.get_trace()
                    all_traces.append(run_trace)

                    l2.reset(problem)
                    problem.reset()

            # Group-wise detailed AUCs
            if fid == 5:
                detailed_aucs[0] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 9:
                detailed_aucs[1] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 14:
                detailed_aucs[2] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 19:
                detailed_aucs[3] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 24:
                detailed_aucs[4] = np.mean(detail_aucs); detail_aucs = []

        # Aggregate traces: element-wise mean across all runs
        aggregated_trace = _aggregate_traces(all_traces, eval_budget)

        return aucs, detailed_aucs, aggregated_trace, ""

    except KeyboardInterrupt:
        return [], [0, 0, 0, 0, 0], [], "KeyboardInterrupt"
    except Exception as e:
        return [], [0, 0, 0, 0, 0], [], str(e)
    finally:
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice
```

#### 2.1.2 TraceCollector Class

```python
class TraceCollector:
    """
    Wrapper around IOH problem to collect best-so-far trace during optimization.

    This enables MADA behavioral diversity calculation by tracking the
    convergence trajectory of each algorithm run.
    """

    def __init__(self, problem, budget):
        """
        Args:
            problem: IOH problem instance
            budget: Maximum number of evaluations
        """
        self.problem = problem
        self.budget = budget
        self.trace = []
        self.best_so_far = float('inf')
        self.eval_count = 0

        # Copy bounds from problem
        self.bounds = problem.bounds

    def __call__(self, x):
        """
        Evaluate x on the problem and record best-so-far.

        Args:
            x: Solution vector to evaluate

        Returns:
            float: Fitness value
        """
        if self.eval_count >= self.budget:
            raise OverBudgetException(f"Budget {self.budget} exceeded")

        f = self.problem(x)
        self.eval_count += 1

        # Update best-so-far (minimization)
        if f < self.best_so_far:
            self.best_so_far = f

        self.trace.append(self.best_so_far)

        return f

    def get_trace(self):
        """
        Get the trace vector, padded to budget length.

        Returns:
            list: Best-so-far fitness at each evaluation step
        """
        trace = self.trace.copy()

        # Pad with final value if run ended early
        if len(trace) < self.budget:
            final_value = trace[-1] if trace else float('inf')
            trace.extend([final_value] * (self.budget - len(trace)))

        return trace[:self.budget]

    def reset(self):
        """Reset for a new run."""
        self.trace = []
        self.best_so_far = float('inf')
        self.eval_count = 0


def _aggregate_traces(traces, budget):
    """
    Aggregate multiple traces into a single representative trace.
    Uses element-wise median for robustness.

    Args:
        traces: List of trace lists
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
            t = t + [t[-1] if t else 0.0] * (budget - len(t))
        padded.append(t[:budget])

    # Element-wise median (robust to outliers)
    arr = np.array(padded)
    aggregated = np.median(arr, axis=0).tolist()

    return aggregated
```

#### 2.1.3 NN-Dist Calculator (Behavioral Diversity)

```python
def calculate_nn_dist(new_trace, history_traces, global_min=None, global_max=None):
    """
    Calculate nearest-neighbor distance for behavioral diversity.

    This measures how different a new algorithm's convergence behavior is
    from all previously seen algorithms. Higher distance = more novel behavior.

    Mathematical Definition:
        d(x_new, x_history) = sqrt(sum((x_new[t] - x_history[t])^2))
        DiversityScore = min(d(x_new, h) for h in history)

    Args:
        new_trace: List[float] - Convergence trace of new algorithm
        history_traces: List[List[float]] - Traces of previously evaluated algorithms
        global_min: Optional minimum fitness seen (for normalization)
        global_max: Optional maximum fitness seen (for normalization)

    Returns:
        float: Nearest-neighbor distance (0.0 to ~sqrt(budget) normalized)
    """
    # Handle empty history case
    if not history_traces:
        return 1.0  # Maximum diversity for first algorithm

    new_arr = np.array(new_trace, dtype=float)

    # Normalize traces to [0, 1] range for fair comparison
    if global_min is not None and global_max is not None:
        range_val = global_max - global_min
        if range_val > 1e-10:
            new_arr = (new_arr - global_min) / range_val
        else:
            new_arr = np.zeros_like(new_arr)
    else:
        # Auto-normalize based on new trace
        trace_min, trace_max = np.min(new_arr), np.max(new_arr)
        range_val = trace_max - trace_min
        if range_val > 1e-10:
            new_arr = (new_arr - trace_min) / range_val
        else:
            new_arr = np.zeros_like(new_arr)

    min_distance = float('inf')

    for hist_trace in history_traces:
        hist_arr = np.array(hist_trace, dtype=float)

        # Normalize history trace
        if global_min is not None and global_max is not None:
            range_val = global_max - global_min
            if range_val > 1e-10:
                hist_arr = (hist_arr - global_min) / range_val
            else:
                hist_arr = np.zeros_like(hist_arr)
        else:
            hist_min, hist_max = np.min(hist_arr), np.max(hist_arr)
            range_val = hist_max - hist_min
            if range_val > 1e-10:
                hist_arr = (hist_arr - hist_min) / range_val
            else:
                hist_arr = np.zeros_like(hist_arr)

        # Handle length mismatch
        min_len = min(len(new_arr), len(hist_arr))
        new_trimmed = new_arr[:min_len]
        hist_trimmed = hist_arr[:min_len]

        # Euclidean distance
        distance = np.sqrt(np.sum((new_trimmed - hist_trimmed) ** 2))

        # Normalize by sqrt(budget) to get [0, 1] range approximately
        distance = distance / np.sqrt(min_len) if min_len > 0 else 0.0

        min_distance = min(min_distance, distance)

    return min_distance if min_distance != float('inf') else 1.0
```

---

### 2.2 The Reward System

#### 2.2.1 RewardNormalizer Class

```python
class RewardNormalizer:
    """
    Running standardization for bandit rewards.

    Bandits like D-TS assume rewards follow a stable distribution.
    This class applies running standardization: R_final = (R - μ) / σ

    Uses Welford's online algorithm for numerical stability.
    """

    def __init__(self, warmup_period=5):
        """
        Args:
            warmup_period: Number of rewards to collect before normalizing
        """
        self.warmup_period = warmup_period
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # Sum of squared differences from mean
        self.rewards_history = []

    def update(self, reward):
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
    def variance(self):
        """Current variance estimate."""
        if self.n < 2:
            return 1.0
        return self.M2 / (self.n - 1)

    @property
    def std(self):
        """Current standard deviation estimate."""
        return np.sqrt(self.variance)

    def normalize(self, reward):
        """
        Normalize a reward using running statistics.

        Args:
            reward: Raw reward value

        Returns:
            float: Normalized reward (z-score)
        """
        # Update stats first
        self.update(reward)

        # During warmup, return raw reward
        if self.n < self.warmup_period:
            return reward

        # Z-score normalization
        std = self.std
        if std < 1e-10:
            return 0.0  # Avoid division by zero

        return (reward - self.mean) / std

    def get_stats(self):
        """Get current statistics for logging."""
        return {
            'n': self.n,
            'mean': self.mean,
            'std': self.std,
            'variance': self.variance,
        }
```

#### 2.2.2 AlphaScheduler Class

```python
class AlphaScheduler:
    """
    Scheduler for the diversity weight parameter α.

    The importance of diversity should fade as the run progresses:
    - Start: High α → Explore diverse behaviors
    - End: Low/zero α → Exploit best-performing strategies

    Supports multiple scheduling strategies.
    """

    def __init__(
        self,
        alpha_start=0.5,
        alpha_end=0.0,
        t_max=100,
        schedule='linear'
    ):
        """
        Args:
            alpha_start: Initial diversity weight (default 0.5)
            alpha_end: Final diversity weight (default 0.0)
            t_max: Total number of generations/steps
            schedule: Decay schedule type ('linear', 'exponential', 'cosine', 'constant')
        """
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.t_max = t_max
        self.schedule = schedule

    def get_alpha(self, t):
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
        progress = t / self.t_max if self.t_max > 0 else 1.0

        if self.schedule == 'linear':
            # α(t) = α_start × (1 - t/T_max)
            alpha = self.alpha_start * (1 - progress) + self.alpha_end * progress

        elif self.schedule == 'exponential':
            # α(t) = α_start × exp(-3 × t/T_max)
            decay_rate = 3.0
            alpha = self.alpha_start * np.exp(-decay_rate * progress)
            alpha = max(alpha, self.alpha_end)

        elif self.schedule == 'cosine':
            # Cosine annealing: smoother transition
            alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (1 + np.cos(np.pi * progress))

        else:
            alpha = self.alpha_start

        return alpha

    def __repr__(self):
        return f"AlphaScheduler(start={self.alpha_start}, end={self.alpha_end}, schedule={self.schedule})"
```

#### 2.2.3 Composite Reward Function

```python
def compute_composite_reward(
    child_fitness,
    parent_fitness,
    nn_dist,
    alpha,
    error=False,
    clamp=1.0
):
    """
    Compute the MADA composite reward combining exploitation and exploration.

    Formula: R_t = (Fit_child - Fit_parent) + α × NN-Dist

    Args:
        child_fitness: Fitness of offspring algorithm
        parent_fitness: Fitness of parent (or best parent)
        nn_dist: Nearest-neighbor distance (behavioral diversity)
        alpha: Diversity weight from AlphaScheduler
        error: Whether the offspring had an error
        clamp: Maximum absolute reward value for stability

    Returns:
        float: Composite reward value
    """
    if error:
        # Penalize errors heavily
        return -clamp

    # Exploitation signal: fitness improvement
    fitness_delta = child_fitness - parent_fitness

    # Exploration signal: behavioral diversity
    diversity_bonus = alpha * nn_dist

    # Composite reward
    raw_reward = fitness_delta + diversity_bonus

    # Clamp for stability
    reward = max(-clamp, min(clamp, raw_reward))

    # Small positive signal for ties (prevents bandit stagnation)
    if abs(reward) < 1e-6:
        reward = 0.01

    return reward
```

---

### 2.3 The Bandit Class (Upgraded)

The existing `DiscountedThompsonSampler` is well-implemented. We only need minor modifications for reward normalization integration:

```python
@dataclass
class ArmStatistics:
    """Statistics for a single bandit arm."""
    name: str
    N: float = 1.0          # Discounted effective sample size
    mu_tilde: float = 0.0   # Discounted cumulative reward
    mu_hat: float = 0.0     # Posterior mean estimate
    tau: float = 1.0        # Posterior standard deviation
    pulls: int = 0          # Total pulls (logging)
    total_reward: float = 0.0  # Total undiscounted reward (logging)
    # NEW: Track diversity contributions
    total_diversity_reward: float = 0.0
    total_fitness_reward: float = 0.0


class DiscountedThompsonSamplerMADA(DiscountedThompsonSampler):
    """
    Extended D-TS for MADA with reward decomposition tracking.

    Inherits core D-TS logic but adds:
    - Reward component tracking (fitness vs diversity)
    - Integration with RewardNormalizer
    """

    def __init__(
        self,
        arms: List[str],
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 0.25,
        prior_mean: float = 0.0,
    ):
        super().__init__(arms, discount, tau_max, reward_variance, prior_mean)

        # Add diversity tracking to arm stats
        for stats in self.arm_stats.values():
            stats.total_diversity_reward = 0.0
            stats.total_fitness_reward = 0.0

    def update_with_decomposition(
        self,
        arm_name: str,
        normalized_reward: float,
        fitness_component: float,
        diversity_component: float
    ):
        """
        Update bandit with reward and track component contributions.

        Args:
            arm_name: Selected arm
            normalized_reward: Final normalized reward for bandit update
            fitness_component: Raw fitness delta (for logging)
            diversity_component: Raw diversity bonus (for logging)
        """
        # Standard D-TS update
        self.update(arm_name, normalized_reward)

        # Track decomposition for analysis
        if arm_name in self.arm_stats:
            self.arm_stats[arm_name].total_fitness_reward += fitness_component
            self.arm_stats[arm_name].total_diversity_reward += diversity_component

    def get_detailed_state(self) -> Dict:
        """Get detailed state including reward decomposition."""
        state = self.get_state_snapshot()
        for arm_name, stats in self.arm_stats.items():
            state[arm_name]['total_fitness_reward'] = stats.total_fitness_reward
            state[arm_name]['total_diversity_reward'] = stats.total_diversity_reward
        return state
```

---

### 2.4 Main Loop Refactoring

**Location:** `main-thesis-eoh-dts.py`, function `run_eoh_dts_evolutionary_mode()`

Here is the complete refactored main loop with MADA integration:

````python
def run_mada_evolutionary_mode(
    algorithm_manager,
    eoh_operator,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
):
    """
    MADA-LLAMEA: Population-based evolution with behavioral diversity reward.

    Key differences from standard EoH-DTS:
    1. Evaluation returns optimization traces
    2. Composite reward = fitness_delta + α × NN-Dist
    3. Reward normalization before bandit update
    4. Alpha decay schedule for exploration → exploitation
    """

    # ====================
    # MADA INITIALIZATION
    # ====================

    # History of optimization traces for diversity calculation
    history_traces = []

    # Global fitness bounds for trace normalization
    global_min_fitness = float('inf')
    global_max_fitness = float('-inf')

    # Reward normalizer for stable bandit updates
    reward_normalizer = RewardNormalizer(warmup_period=5)

    # Alpha scheduler: start high (explore), decay to zero (exploit)
    alpha_scheduler = AlphaScheduler(
        alpha_start=args.alpha_start if hasattr(args, 'alpha_start') else 0.5,
        alpha_end=args.alpha_end if hasattr(args, 'alpha_end') else 0.0,
        t_max=generations,
        schedule=args.alpha_schedule if hasattr(args, 'alpha_schedule') else 'linear'
    )

    population = []
    best_ever = None
    api_calls = 0
    generation = 0

    # ====================
    # PHASE 1: INITIALIZATION
    # ====================

    print(f"\n{'='*60}")
    print(f"MADA-LLAMEA INITIALIZATION: Generating {n_parents} parent algorithms")
    print('='*60)

    seen_hashes = set()
    for i in range(n_parents):
        if api_calls >= args.budget:
            break

        print(f"\nInitializing Parent {i+1}/{n_parents} (API call {api_calls+1})")

        try:
            retries = 0
            max_retries = 3
            while retries <= max_retries:
                message = algorithm_manager.fetch_algorithm()

                pattern = r"```(?:python)?\n(.*?)\n```"
                match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
                if not match:
                    retries += 1
                    continue
                algorithm_code = match.group(1)

                class_match = re.search(r"class\s+(\w+)", algorithm_code)
                if not class_match:
                    retries += 1
                    continue
                algorithm_name = class_match.group(1)

                code_hash = hashlib.sha256(algorithm_code.encode()).hexdigest()
                if code_hash in seen_hashes:
                    retries += 1
                    continue

                solution = EoHSolution(
                    code=algorithm_code,
                    name=algorithm_name,
                    description=algorithm_name,
                    generation=generation,
                )
                solution.operator = "init"

                print(f"  Evaluating {algorithm_name}...")

                # >>> MADA CHANGE: Use trace-returning evaluation <<<
                aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
                    algorithm_code, algorithm_name, args.eval_budget
                )

                if error:
                    solution.fitness = 0.0
                    solution.error = error
                    print(f"  Error: {error}")
                    retries += 1
                    if retries > max_retries:
                        break
                    continue
                else:
                    solution.fitness = float(np.mean(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    solution.trace = trace  # Store trace on solution
                    seen_hashes.add(code_hash)
                    print(f"  Fitness: {solution.fitness:.4f}")

                    # >>> MADA CHANGE: Update global bounds <<<
                    if trace:
                        global_min_fitness = min(global_min_fitness, min(trace))
                        global_max_fitness = max(global_max_fitness, max(trace))
                        history_traces.append(trace)

                population.append(solution)
                explogger.log_code(api_calls, algorithm_name, algorithm_code)
                explogger.log_aucs(api_calls, aucs if aucs else [0])
                api_calls += 1
                break

        except Exception as e:
            print(f"  Initialization error: {e}")
            import traceback
            traceback.print_exc()

    if not population:
        print("ERROR: Failed to initialize any individuals!")
        return

    population = selection(population, len(population), elitism=True)
    best_ever = population[0]
    print(f"\nInitialization complete. Best initial fitness: {best_ever.fitness:.4f}")

    # ====================
    # PHASE 2: EVOLUTIONARY LOOP
    # ====================

    for gen in range(1, generations + 1):
        if api_calls >= args.budget:
            print(f"\nBudget exhausted at generation {gen}")
            break

        generation = gen

        # >>> MADA CHANGE: Get current alpha <<<
        current_alpha = alpha_scheduler.get_alpha(generation)

        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{generations}")
        print(f"API Calls: {api_calls}/{args.budget}")
        print(f"Best so far: {best_ever.fitness:.4f} ({best_ever.name})")
        print(f"MADA α (diversity weight): {current_alpha:.3f}")

        # Show bandit state
        bandit_state = eoh_operator.get_bandit_state()
        probs = bandit_state['selection_probs']
        print(f"D-TS Probabilities: mutation={probs['mutation']:.1%}, crossover={probs['crossover']:.1%}")
        print('='*60)

        parents = selection(population, n_parents, elitism=args.elitism)
        best_parent_fitness = max((p.fitness for p in parents), default=0.0)

        eoh_operator.reset_generation_counts()
        offspring = []

        for i in range(n_offspring):
            if api_calls >= args.budget:
                break

            parent = random.choice(parents)
            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")

            try:
                # >>> STEP 1: Bandit selects operator <<<
                child, selection_info = eoh_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                )

                operator = selection_info['operator']
                print(f"  D-TS selected: {operator} (θ={selection_info['theta_sampled']:.3f})")
                print(f"  Evaluating {child.name}...")

                # >>> STEP 2: Evaluate with trace <<<
                aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
                    child.code, child.name, args.eval_budget
                )

                if error:
                    child.fitness = 0.0
                    child.error = error
                    child.trace = []
                    print(f"  Error: {error}")
                else:
                    child.fitness = float(np.mean(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    child.trace = trace
                    print(f"  Fitness: {child.fitness:.4f}")

                    # Update global bounds
                    if trace:
                        global_min_fitness = min(global_min_fitness, min(trace))
                        global_max_fitness = max(global_max_fitness, max(trace))

                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                        best_ever = child

                # >>> STEP 3: Calculate NN-Dist (behavioral diversity) <<<
                if trace and history_traces:
                    nn_dist = calculate_nn_dist(
                        trace,
                        history_traces,
                        global_min_fitness,
                        global_max_fitness
                    )
                else:
                    nn_dist = 1.0  # Max diversity if no history

                print(f"  NN-Dist (diversity): {nn_dist:.4f}")

                # >>> STEP 4: Compute composite reward <<<
                fitness_delta = child.fitness - best_parent_fitness if not error else 0.0
                diversity_bonus = current_alpha * nn_dist

                raw_reward = compute_composite_reward(
                    child_fitness=child.fitness,
                    parent_fitness=best_parent_fitness,
                    nn_dist=nn_dist,
                    alpha=current_alpha,
                    error=bool(error),
                    clamp=args.reward_clamp if hasattr(args, 'reward_clamp') else 1.0
                )

                # >>> STEP 5: Normalize reward <<<
                normalized_reward = reward_normalizer.normalize(raw_reward)

                print(f"  Reward: raw={raw_reward:+.4f}, normalized={normalized_reward:+.4f}")
                print(f"    (fitness_Δ={fitness_delta:+.4f}, diversity={diversity_bonus:+.4f})")

                # >>> STEP 6: Update bandit <<<
                eoh_operator.bandit.update(operator, normalized_reward)

                # Add trace to history
                if trace:
                    history_traces.append(trace)

                # Log offspring
                offspring_record = {
                    'operator': child.operator,
                    'fitness': child.fitness,
                    'parent_fitness': best_parent_fitness,
                    'nn_dist': nn_dist,
                    'alpha': current_alpha,
                    'raw_reward': raw_reward,
                    'normalized_reward': normalized_reward,
                    'fitness_delta': fitness_delta,
                    'diversity_bonus': diversity_bonus,
                    'theta_sampled': selection_info['theta_sampled'],
                    'parent_ids': child.parent_ids,
                    'generation': generation,
                    'error': child.error,
                }
                log_eoh_offspring(explogger, api_calls, offspring_record)

                offspring.append(child)
                explogger.log_code(api_calls, child.name, child.code)
                explogger.log_aucs(api_calls, aucs if aucs else [0])
                api_calls += 1

            except Exception as e:
                print(f"  Offspring generation error: {e}")
                import traceback
                traceback.print_exc()

        # Selection for next generation
        if args.elitism:
            combined = parents + offspring
            population = selection(combined, n_parents, elitism=True)
            print(f"\nSelection: (μ+λ) - Best {n_parents} from {len(combined)}")
        else:
            population = selection(offspring, n_parents, elitism=False)
            print(f"\nSelection: (μ,λ) - Best {n_parents} from {len(offspring)}")

        # Log bandit state
        bandit_state = eoh_operator.get_bandit_state()
        bandit_state['alpha'] = current_alpha
        bandit_state['reward_stats'] = reward_normalizer.get_stats()
        bandit_state['history_size'] = len(history_traces)
        log_bandit_snapshot(explogger, generation, bandit_state)

        stats = eoh_operator.get_operator_stats()
        print(f"Operator usage: {stats['mutation_count']} mutations, {stats['crossover_count']} crossovers")

    # ====================
    # FINAL SUMMARY
    # ====================

    print(f"\n{'='*60}")
    print("MADA-LLAMEA OPTIMIZATION COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations completed: {generation}")
    print(f"Best algorithm: {best_ever.name}")
    print(f"Best fitness: {best_ever.fitness:.4f}")
    print(f"Total traces collected: {len(history_traces)}")

    # Final bandit statistics
    final_state = eoh_operator.get_bandit_state()
    print(f"\nFinal D-TS Bandit Statistics:")
    for arm, s in final_state['arm_stats'].items():
        print(f"  {arm}: pulls={s['pulls']}, avg_reward={s['avg_reward']:.4f}, μ̂={s['mu_hat']:.3f}")

    # Reward normalizer stats
    reward_stats = reward_normalizer.get_stats()
    print(f"\nReward Statistics:")
    print(f"  Total rewards: {reward_stats['n']}")
    print(f"  Mean: {reward_stats['mean']:.4f}")
    print(f"  Std: {reward_stats['std']:.4f}")

    # Save best algorithm
    with open(f"{explogger.dirname}/BEST_ALGORITHM.py", "w") as f:
        f.write(f"# Best Algorithm: {best_ever.name}\n")
        f.write(f"# Fitness: {best_ever.fitness:.4f}\n")
        f.write(f"# Generation: {best_ever.generation}\n")
        f.write(f"# Operator: {best_ever.operator}\n\n")
        f.write(best_ever.code)
````

---

## 3. New Command-Line Arguments

Add these to the argument parser in `main()`:

```python
# MADA-specific Configuration
parser.add_argument('--alpha-start', type=float, default=0.5,
                   help='Initial diversity weight α (default: 0.5)')
parser.add_argument('--alpha-end', type=float, default=0.0,
                   help='Final diversity weight α (default: 0.0)')
parser.add_argument('--alpha-schedule', type=str, default='linear',
                   choices=['linear', 'exponential', 'cosine', 'constant'],
                   help='Alpha decay schedule (default: linear)')
parser.add_argument('--reward-clamp', type=float, default=1.0,
                   help='Clamp rewards to [-clamp, +clamp] (default: 1.0)')
parser.add_argument('--mada-mode', action='store_true',
                   help='Enable MADA behavioral diversity rewards')
```

---

## 4. Integration Checklist

### Phase 1: Metric Engineering

- [ ] Add `TraceCollector` class
- [ ] Add `_aggregate_traces()` function
- [ ] Create `evaluate_algorithm_with_trace()` function
- [ ] Add `calculate_nn_dist()` function
- [ ] Add `trace` attribute to `EoHSolution` class

### Phase 2: Reward System

- [ ] Add `RewardNormalizer` class
- [ ] Add `AlphaScheduler` class
- [ ] Add `compute_composite_reward()` function

### Phase 3: Bandit Integration

- [ ] (Optional) Create `DiscountedThompsonSamplerMADA` subclass
- [ ] Modify `EoHOperatorDTS.update_bandit()` to use composite rewards

### Phase 4: Main Loop

- [ ] Create `run_mada_evolutionary_mode()` function
- [ ] Add MADA command-line arguments
- [ ] Update `main()` to support MADA mode
- [ ] Update logging to include MADA metrics

### Phase 5: Testing

- [ ] Unit test `calculate_nn_dist()` with synthetic traces
- [ ] Unit test `RewardNormalizer` convergence
- [ ] Unit test `AlphaScheduler` decay curves
- [ ] Integration test with small budget (10 calls)
- [ ] Full experiment comparison: EoH-DTS vs MADA-LLAMEA

---

## 5. File Structure After Integration

```
LLAMEA-MADA/
├── main-thesis-mada.py          # New MADA-LLAMEA implementation
├── main-thesis-eoh-dts.py       # Original EoH-DTS (unchanged)
├── main-thesis-eoh.py           # Original EoH (unchanged)
├── managers.py                  # Unchanged
├── mada_components.py           # New: TraceCollector, RewardNormalizer, AlphaScheduler
├── diversity.py                 # New: calculate_nn_dist()
├── MADA5.0.md                   # Design document
├── MADA-INTEGRATION-GUIDE.md    # This file
└── experiments/
    └── mada_vs_dts/             # Comparison experiments
```

---

## 6. Usage Example

```bash
# Standard EoH-DTS (baseline)
python main-thesis-eoh-dts.py --evolutionary-mode --elitism --budget 50

# MADA-LLAMEA with default settings
python main-thesis-mada.py --evolutionary-mode --elitism --mada-mode --budget 50

# MADA with custom alpha schedule
python main-thesis-mada.py --evolutionary-mode --elitism --mada-mode \
    --alpha-start 0.7 --alpha-end 0.1 --alpha-schedule cosine --budget 100

# MADA with faster D-TS adaptation
python main-thesis-mada.py --evolutionary-mode --elitism --mada-mode \
    --discount 0.8 --alpha-start 0.5 --budget 100
```

---

## 7. Summary of Changes

| Component          | Standard LLaMEA     | MADA-LLAMEA                     |
| ------------------ | ------------------- | ------------------------------- |
| Operator Selection | Fixed / Random      | Adaptive (DS-TS)                |
| Evaluation         | Scalar Fitness      | Fitness + Trace Vector          |
| Feedback Signal    | Fitness Improvement | Composite (Fitness + Diversity) |
| Exploration        | Random Mutation     | Guided by NN-Dist Reward        |
| Reward Processing  | Raw delta           | Normalized with running stats   |
| Diversity Weight   | N/A                 | Decaying α scheduler            |
