# MADA-LLAMEA Implementation Plan

> **Objective:** Upgrade the standard LLaMEA framework into MADA-LLAMEA by replacing the static evolutionary loop with an adaptive bandit loop driven by a composite reward signal.

---

## Overview

| Component          | Standard LLaMEA     | MADA-LLAMEA                     |
| ------------------ | ------------------- | ------------------------------- |
| Operator Selection | Fixed / Random      | Adaptive (DS-TS)                |
| Evaluation         | Scalar Fitness      | Fitness + Trace Vector          |
| Feedback Signal    | Fitness Improvement | Composite (Fitness + Diversity) |
| Exploration        | Random Mutation     | Guided by NN-Dist Reward        |

---

## Phase 1: Metric Engineering (The "Sensors")

**Goal:** Enable the system to measure algorithmic diversity, not just fitness.

### 1.1 Implement Optimization Trace Logging

- [ ] **Task:** Modify the `evaluate()` function in the LLaMEA class

**Problem:** Standard LLaMEA only logs the final scalar fitness. We need the full history to calculate behavioral diversity.

**Changes Required:**

- Return a tuple `(fitness, trace)` instead of a single scalar
- `trace` is a list of floats (length = budget) representing the best-so-far fitness at each evaluation step

**Example:**

```python
def evaluate(self, algorithm):
    # ... run the algorithm ...
    trace = []  # Best-so-far fitness at each step
    for step in range(budget):
        # ... evaluation logic ...
        trace.append(best_fitness_so_far)

    return final_fitness, trace
```

---

### 1.2 Implement the NN-Dist Calculator

- [ ] **Task:** Create a function to calculate "Behavioral Diversity" of a new solution

**Mathematical Definition:**

$$d(x_{new}, x_{history}) = \sqrt{\sum_{t=1}^{B} (x_{new}[t] - x_{history}[t])^2}$$

**Implementation Steps:**

1. **Normalize:** Scale the input trace `x_new` and all traces in history to \[0, 1\] based on the global min/max fitness seen so far
2. **Distance:** Compute Euclidean distance between `x_new` and every trace in the current population (or archive)
3. **Minimize:** The `DiversityScore` is the minimum distance found (i.e., distance to the nearest neighbor)

**Example:**

```python
import numpy as np

def calculate_nn_dist(new_trace, history_traces, global_min, global_max):
    """Calculate nearest-neighbor distance for behavioral diversity."""
    if not history_traces:
        return 1.0  # Max diversity if no history

    # Normalize traces to [0, 1]
    def normalize(trace):
        return (np.array(trace) - global_min) / (global_max - global_min + 1e-10)

    new_normalized = normalize(new_trace)

    min_distance = float('inf')
    for hist_trace in history_traces:
        hist_normalized = normalize(hist_trace)
        distance = np.sqrt(np.sum((new_normalized - hist_normalized) ** 2))
        min_distance = min(min_distance, distance)

    return min_distance
```

---

## Phase 2: The Reward System (The "Fuel")

**Goal:** Redefine "Success" for the bandit to prevent stagnation.

### 2.1 Define the Composite Reward Function

- [ ] **Task:** Implement behavior-aware reward calculation

**Equation:**

$$R_t = \underbrace{(\text{Fit}_{child} - \text{Fit}_{parent})}_{\text{Exploitation Signal}} + \alpha \cdot \underbrace{(\text{NN-Dist})}_{\text{Exploration Signal}}$$

**Normalization Step (Crucial):**

Bandits like DS-TS assume rewards follow a stable distribution.

**Action:** Apply a running standardization to \(R_t\) before feeding it to the bandit:

$$R_{final} = \frac{R_t - \mu_R}{\sigma_R}$$

Where \(\mu_R\) and \(\sigma_R\) are the rolling mean and standard deviation of rewards seen so far.

**Example:**

```python
class RewardNormalizer:
    def __init__(self):
        self.rewards = []

    def normalize(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) < 2:
            return 0.0

        mu = np.mean(self.rewards)
        sigma = np.std(self.rewards) + 1e-10
        return (reward - mu) / sigma
```

---

### 2.2 Implement the Alpha (α) Scheduler

- [ ] **Task:** Create a scheduler for the diversity weight parameter

**Rationale:** The importance of diversity should fade as the run progresses to allow convergence.

**Options:**

| Strategy       | Formula                                                                  | Use Case        |
| -------------- | ------------------------------------------------------------------------ | --------------- |
| Static Alpha   | \(\alpha = 0.5\)                                                         | Simple baseline |
| Decaying Alpha | \(\alpha*t = \alpha*{start} \times \left(1 - \frac{t}{T\_{max}}\right)\) | Recommended     |

**Behavior:**

- **Start:** High diversity reward → Explore the map
- **End:** Zero diversity reward → Climb the peak

**Example:**

```python
class AlphaScheduler:
    def __init__(self, alpha_start=0.5, t_max=100):
        self.alpha_start = alpha_start
        self.t_max = t_max

    def get_alpha(self, t):
        """Decaying alpha schedule."""
        return self.alpha_start * (1 - t / self.t_max)
```

---

## Phase 3: The Adaptive Engine (The "Brain")

**Goal:** Integrate the Discounted Thompson Sampling (DS-TS) algorithm.

### 3.1 Initialize the Bandit

- [ ] **Task:** Create `DiscountedThompsonSampler` class based on Qi, Guo, & Zhu (2025)

**State:** For each operator \(k\) (e.g., Mutation, Crossover), maintain:

- \(\hat{\mu}\_k\): Estimated mean reward
- \(N_k\): Effective number of samples (confidence)

**Hyperparameter:** Set discount factor \(\gamma = 0.9\) (or similar)

**Example:**

```python
class DiscountedThompsonSampler:
    def __init__(self, operators, gamma=0.9, sigma=1.0):
        self.operators = operators
        self.gamma = gamma
        self.sigma = sigma

        # Initialize state for each operator
        self.mu = {op: 0.0 for op in operators}      # Estimated mean
        self.N = {op: 1.0 for op in operators}       # Effective samples
```

---

### 3.2 Implement the Sampling Step (Selection)

- [ ] **Task:** Implement arm selection via Thompson Sampling

At the start of every generation, the bandit decides which operator to use.

**Math:** Sample a value \(\theta_k\) for each operator:

$$\theta_k \sim \mathcal{N}\left(\hat{\mu}_k, \frac{4\sigma^2}{N_k}\right)$$

**Action:** Select the operator with the highest sampled \(\theta_k\).

**Example:**

```python
def select_arm(self):
    """Select operator using Thompson Sampling."""
    sampled_values = {}

    for op in self.operators:
        variance = (4 * self.sigma ** 2) / self.N[op]
        theta = np.random.normal(self.mu[op], np.sqrt(variance))
        sampled_values[op] = theta

    # Return operator with highest sampled value
    return max(sampled_values, key=sampled_values.get)
```

---

### 3.3 Implement the Update Step (Learning)

- [ ] **Task:** Implement discounted reward update

After the operator runs and we calculate \(R\_{final}\), update the bandit's memory.

**Math (Discounting):**

$$N_{new} = \gamma \cdot N_{old} + 1$$

$$\hat{\mu}_{new} = \frac{\gamma \cdot \hat{\mu}_{old} \cdot N_{old} + R_{final}}{N_{new}}$$

**Explanation:** The \(\gamma\) factor shrinks the old \(N\) and \(\hat{\mu}\), effectively "forgetting" past successes if they happened too long ago.

**Example:**

```python
def update(self, operator, reward):
    """Update bandit with discounted reward."""
    # Apply discount to old values
    old_N = self.N[operator]
    old_mu = self.mu[operator]

    # Update with new observation
    self.N[operator] = self.gamma * old_N + 1
    self.mu[operator] = (self.gamma * old_mu * old_N + reward) / self.N[operator]
```

---

## Phase 4: Integration into LLaMEA

**Goal:** Wire the components into the main execution loop.

### 4.1 Complete Integration

- [ ] **Task:** Modify the main LLaMEA loop

**Full Pseudo-Code:**

```python
# ==================== INITIALIZATION ====================
bandit = DiscountedThompsonSampler(
    operators=["Mutation", "Crossover"],
    gamma=0.9
)
alpha_scheduler = AlphaScheduler(alpha_start=0.5, t_max=max_generations)
reward_normalizer = RewardNormalizer()
history_traces = []

global_min_fitness = float('inf')
global_max_fitness = float('-inf')

# ==================== MAIN LOOP ====================
generation = 0
while not done:
    # 1. Bandit chooses Operator (Phase 3.2)
    operator = bandit.select_arm()

    # 2. Generate & Evaluate (Phase 1.1)
    offspring = LLM.generate(parent, operator)
    fitness, trace = evaluate_with_trace(offspring)

    # Update global bounds for normalization
    global_min_fitness = min(global_min_fitness, min(trace))
    global_max_fitness = max(global_max_fitness, max(trace))

    # 3. Calculate Components (Phase 1.2)
    fit_gain = max(0, fitness - parent.fitness)
    div_score = calculate_nn_dist(
        trace,
        history_traces,
        global_min_fitness,
        global_max_fitness
    )

    # 4. Composite Reward (Phase 2.1 & 2.2)
    alpha = alpha_scheduler.get_alpha(generation)
    reward = fit_gain + (alpha * div_score)
    normalized_reward = reward_normalizer.normalize(reward)

    # 5. Teach the Bandit (Phase 3.3)
    bandit.update(operator, normalized_reward)

    # 6. Standard Evolution Updates
    history_traces.append(trace)
    if fitness > parent.fitness:
        parent = offspring

    generation += 1
```

---

## Implementation Checklist

### Phase 1: Metric Engineering

- [ ] 1.1 Modify `evaluate()` to return `(fitness, trace)` tuple
- [ ] 1.2 Implement `calculate_nn_dist()` function
- [ ] Write unit tests for trace logging
- [ ] Write unit tests for NN-Dist calculator

### Phase 2: Reward System

- [ ] 2.1 Implement `RewardNormalizer` class
- [ ] 2.1 Implement composite reward calculation
- [ ] 2.2 Implement `AlphaScheduler` class
- [ ] Write unit tests for reward normalization
- [ ] Write unit tests for alpha decay

### Phase 3: Adaptive Engine

- [ ] 3.1 Implement `DiscountedThompsonSampler` class
- [ ] 3.2 Implement `select_arm()` method
- [ ] 3.3 Implement `update()` method
- [ ] Write unit tests for bandit selection
- [ ] Write unit tests for discounted updates

### Phase 4: Integration

- [ ] 4.1 Modify main LLaMEA loop
- [ ] Add logging for bandit decisions
- [ ] Add logging for reward components
- [ ] Integration testing with BBOB benchmarks
- [ ] Performance comparison with standard LLaMEA

---

## References

1. van Stein, N., Bäck, T. (2024). LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics. _IEEE TEVC_
2. van Stein, N., et al. (2025). Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery. _EvoStar_
3. Qi, Y., Guo, Y., Zhu, J. (2025). Discounted Thompson Sampling for Non-Stationary Bandits
4. Hansen, N., et al. (2009). BBOB Benchmark Suite

---

## Hyperparameter Summary

| Parameter       | Symbol | Default | Range       | Description                    |
| --------------- | ------ | ------- | ----------- | ------------------------------ |
| Discount Factor | γ      | 0.9     | [0.8, 0.99] | How fast to forget old rewards |
| Initial Alpha   | α₀     | 0.5     | [0.3, 0.7]  | Initial diversity weight       |
| Variance Scale  | σ      | 1.0     | [0.5, 2.0]  | Thompson sampling variance     |
| Budget          | B      | 10000   | -           | Evaluations per algorithm      |
