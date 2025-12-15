# MADA-LLAMEA v2 Pipeline: Comprehensive Technical Guide

> **Multi-Adaptive Diverse Algorithm with LLM Evolutionary Algorithm (3-Operator Version)**

This document provides a complete technical explanation of the MADA-LLAMEA v2 pipeline, detailing every calculation, decision point, and data flow. This version introduces a **three-operator system** (mutation, crossover, refine) competing via Discounted Thompson Sampling for adaptive operator selection.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Initialization Phase](#2-initialization-phase)
3. [Discounted Thompson Sampling (D-TS)](#3-discounted-thompson-sampling-d-ts)
4. [Three-Operator System](#4-three-operator-system)
5. [Algorithm Evaluation with Trace Collection](#5-algorithm-evaluation-with-trace-collection)
6. [Behavioral Diversity Measurement](#6-behavioral-diversity-measurement)
7. [Composite Reward Calculation](#7-composite-reward-calculation)
8. [Reward Normalization](#8-reward-normalization)
9. [Bandit Update Mechanism](#9-bandit-update-mechanism)
10. [Population Management and Selection](#10-population-management-and-selection)
11. [Complete Example Walkthrough](#11-complete-example-walkthrough)
12. [Hyperparameters and Configuration](#12-hyperparameters-and-configuration)

---

## 1. System Overview

MADA-LLAMEA v2 is an advanced LLM-driven evolutionary algorithm system that autonomously discovers high-performing optimization algorithms through an adaptive multi-operator framework. The system combines:

- **Three Competing Operators**: Mutation, Crossover, and Refine
- **Discounted Thompson Sampling**: Adaptive operator selection based on performance
- **Behavioral Diversity**: Trace-based novelty measurement
- **Composite Rewards**: Balancing exploitation (fitness) and exploration (diversity)

### High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        MADA-LLAMEA v2 ARCHITECTURE                             │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌──────────────┐      ┌─────────────────────────────────┐                    │
│  │ Initialize   │      │   D-TS Bandit (3 Arms)          │                    │
│  │ Population   │──┐   │  • Mutation (Redesign)          │                    │
│  │ (μ parents)  │  │   │  • Crossover (Combine)          │                    │
│  └──────────────┘  │   │  • Refine (Optimize)            │                    │
│         │          │   └─────────────────────────────────┘                    │
│         │          │                    │                                      │
│         │          │                    ▼                                      │
│         │          │   ┌─────────────────────────────────┐                    │
│         │          │   │  Operator Selection             │                    │
│         │          │   │  θᵢ ~ N(μ̂ᵢ, τᵢ)                 │                    │
│         │          │   │  selected = argmax(θᵢ)          │                    │
│         │          │   └─────────────────────────────────┘                    │
│         │          │                    │                                      │
│         │          │                    ▼                                      │
│         │          │   ┌─────────────────────────────────┐                    │
│         │          └──▶│  LLM Prompt Generation          │                    │
│         │              │  • Mutation: Redesign parent    │                    │
│         │              │  • Crossover: Combine 2 parents │                    │
│         │              │  • Refine: Detailed feedback    │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  LLM (Gemini/GPT/etc.)          │                    │
│         │              │  Generates Python Code          │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  BBOB Evaluation + Trace        │                    │
│         │              │  • 24 functions × 3 instances   │                    │
│         │              │  • 3 repetitions × 10K evals    │                    │
│         │              │  • Collect best-so-far trace    │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  Behavioral Diversity (NN-Dist) │                    │
│         │              │  min{d(trace_new, trace_hist)}  │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  Composite Reward               │                    │
│         │              │  R = Δfitness + α × NN-Dist     │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  Reward Normalization           │                    │
│         │              │  R_norm = (R - μ) / σ           │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         │                             ▼                                        │
│         │              ┌─────────────────────────────────┐                    │
│         │              │  D-TS Update (all arms)         │                    │
│         │              │  Nᵢ ← γ·Nᵢ + δ(i,selected)      │                    │
│         │              │  μ̂ᵢ ← Σrewards / Nᵢ              │                    │
│         │              └─────────────────────────────────┘                    │
│         │                             │                                        │
│         └─────────────────────────────┴──────────────────┐                    │
│                                                            ▼                   │
│                              ┌─────────────────────────────────┐              │
│                              │  (μ+λ) Selection                │              │
│                              │  Top μ by fitness + diversity   │              │
│                              └─────────────────────────────────┘              │
│                                                                                │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component                | Purpose                                | Implementation                           |
| ------------------------ | -------------------------------------- | ---------------------------------------- |
| **D-TS Bandit**          | Adaptive 3-arm operator selection      | `DiscountedThompsonSampler`              |
| **MADAOperatorV2**       | Three-operator framework coordinator   | `MADAOperatorV2` class                   |
| **TraceCollector**       | Captures optimization trajectories     | `mada_components.TraceCollector`         |
| **NN-Dist Calculator**   | Behavioral diversity measurement       | `calculate_nn_dist()`                    |
| **Composite Reward**     | Fitness + diversity combination        | `compute_composite_reward()`             |
| **RewardNormalizer**     | Running z-score standardization        | `RewardNormalizer` (Welford's algorithm) |
| **AlphaScheduler**       | Time-varying diversity weight          | `AlphaScheduler` (4 schedules)           |
| **Behavioral Selection** | Diverse parent selection for crossover | `select_behavioral_parents()`            |

### Key Innovation: Three Operators

MADA-LLAMEA v2 introduces a third operator that competes alongside mutation and crossover:

1. **Mutation** (`_mutate`): Generate a NEW and DIFFERENT algorithm
   - Prompt: Simple redesign instruction
   - Goal: Exploration through radical change
2. **Crossover** (`_crossover`): Combine patterns from TWO high-scoring solutions
   - Prompt: Implicit combination of both parents
   - Goal: Exploitation through pattern recombination
3. **Refine** (`_refine`): Redesign and refine (similar to mutation)
   - Prompt: Simple redesign instruction
   - Goal: Alternative improvement strategy (competes with mutation)

---

## 2. Initialization Phase

The initialization phase bootstraps the evolutionary process by creating an initial population of diverse algorithms and establishing baseline statistics for the MADA framework.

### Step 2.1: Configuration and Hyperparameters

```python
# Core Evolutionary Parameters
args = {
    'n_parents': 4,              # μ = population size
    'n_offspring': 16,           # λ = offspring per generation
    'budget': 100,               # Total API calls (LLM queries)
    'eval_budget': 10000,        # Function evaluations per algorithm
    'elitism': True,             # (μ+λ) vs (μ,λ) selection
}

# D-TS Bandit Parameters
bandit_params = {
    'discount': 0.9,             # γ = forgetting factor
    'tau_max': 1.0,              # Maximum posterior uncertainty
    'reward_variance': 1.0,      # Expected variance of normalized rewards
}

# MADA Diversity Parameters
mada_params = {
    'alpha_start': 0.5,          # Initial diversity weight
    'alpha_end': 0.0,            # Final diversity weight
    'alpha_schedule': 'linear',  # Decay: linear/cosine/exponential/constant
    'reward_clamp': 1.0,         # Numerical stability bounds
}

# Behavioral Selection
behavioral_params = {
    'behavioral_selection': True, # Use trace-based parent selection
}
```

**Key Design Decisions:**

- **Budget**: Total LLM API calls = initialization (μ) + generations × offspring (λ)
  - `generations = (budget - n_parents) // n_offspring`
- **Eval Budget**: 10,000 evaluations per algorithm on BBOB benchmark
  - Total BBOB runs per algorithm: 24 functions × 3 instances × 3 repetitions = 216 runs
- **Elitism**: (μ+λ) selection preserves best solutions across generations

### Step 2.2: Initialize MADA Framework Components

```python
# Global trace statistics for NN-Dist normalization
history_traces = []                    # All evaluated algorithm traces
global_min_fitness = float('inf')      # Global minimum (best) fitness seen
global_max_fitness = float('-inf')     # Global maximum (worst) fitness seen

# Reward normalization (Welford's online algorithm)
reward_normalizer = RewardNormalizer(
    warmup_period=5  # Collect 5 rewards before normalizing
)

# Alpha scheduler (diversity weight decay)
alpha_scheduler = AlphaScheduler(
    alpha_start=args.alpha_start,
    alpha_end=args.alpha_end,
    t_max=max(generations - 1, 1),
    schedule=args.alpha_schedule
)

# Three-operator MADA framework
mada_operator = MADAOperatorV2(
    algorithm_manager=algorithm_manager,
    discount=args.discount,
    tau_max=args.tau_max,
    reward_variance=args.reward_variance,
    reward_clamp=args.reward_clamp,
    use_behavioral_selection=args.behavioral_selection,
    enable_mutation=True,
    enable_crossover=True,
    enable_refine=True,
)
```

**Component Initialization Details:**

1. **RewardNormalizer**: Implements Welford's online variance algorithm for numerically stable running statistics

   - During warmup (n < 5): Returns raw rewards
   - After warmup: Returns z-scores R_norm = (R - μ) / σ

2. **AlphaScheduler**: Implements time-varying diversity weight

   - Linear: α(t) = α_start + (α_end - α_start) × (t / T)
   - Cosine: α(t) = α_end + 0.5(α_start - α_end)(1 + cos(πt/T))
   - Exponential: α(t) = α_start × exp(-3t/T)
   - Constant: α(t) = α_start

3. **DiscountedThompsonSampler** (inside MADAOperatorV2):
   - Initializes 3 arms: {mutation, crossover, refine}
   - Each arm: N=1.0, μ̂=0.0, τ=τ_max (uniform prior)

### Step 2.3: Generate Initial Population

The system generates μ diverse seed algorithms through direct LLM prompting without operator selection:

````python
population = []
seen_hashes = set()  # Prevent duplicate algorithms
api_calls = 0

for i in range(n_parents):
    if api_calls >= args.budget:
        break

    print(f"Initializing Parent {i+1}/{n_parents} (API call {api_calls+1})")

    # Retry loop (up to 3 attempts for valid code)
    retries = 0
    while retries <= 3:
        # 1. Query LLM with initialization prompt
    message = algorithm_manager.fetch_algorithm()

        # 2. Extract code and class name
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

        # 3. Check for duplicates (hash-based)
        code_hash = hashlib.sha256(algorithm_code.encode()).hexdigest()
        if code_hash in seen_hashes:
            retries += 1
            continue

        # 4. Create solution object
        solution = MADASolution(
            code=algorithm_code,
            name=algorithm_name,
            generation=0,
        )
        solution.operator = "init"

        # 5. Evaluate on BBOB with trace collection
        print(f"  Evaluating {algorithm_name}...")
    aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
            algorithm_code, algorithm_name, args.eval_budget
        )

        # 6. Handle errors or store successful solution
        if error:
            solution.fitness = 0.0
            solution.error = error
            print(f"  Error: {error}")
            retries += 1
            continue
        else:
            solution.fitness = float(np.mean(aucs))
            solution.fitness_std = float(np.std(aucs))
            solution.aucs = aucs
            solution.detailed_aucs = detailed_aucs  # Per function-group performance
    solution.trace = trace
            seen_hashes.add(code_hash)
            print(f"  Fitness: {solution.fitness:.4f}")

            # 7. Update global trace statistics
    if trace:
        global_min_fitness = min(global_min_fitness, min(trace))
        global_max_fitness = max(global_max_fitness, max(trace))
        history_traces.append(trace)

        # 8. Add to population and log
    population.append(solution)
        explogger.log_code(api_calls, algorithm_name, algorithm_code)
        explogger.log_aucs(api_calls, aucs if aucs else [0])
        api_calls += 1
        break

# 9. Apply diversity-preserving selection
population = selection(population, len(population), elitism=True)
best_ever = population[0]
print(f"Initialization complete. Best: {best_ever.fitness:.4f}")
````

**Initialization Details:**

- **Detailed AUCs**: Performance breakdown by BBOB function groups

  - [0]: Separable "f1-f5"
  - [1]: Low/moderate conditioning (f6-f9)
  - [2]: High conditioning, unimodal (f10-f14)
  - [3]: Multi-modal, adequate structure (f15-f19)
  - [4]: Multi-modal, weak structure (f20-f24)

- **Trace**: Best-so-far fitness at each of 10,000 evaluations

  - Used for behavioral diversity (NN-Dist) calculation
  - Padded with final value if algorithm terminates early

- **Hash-based Deduplication**: Prevents wasting API calls on identical code

---

## 3. Discounted Thompson Sampling (D-TS)

### 3.1 Mathematical Foundation

Discounted Thompson Sampling (D-TS) is a **multi-armed bandit** algorithm that dynamically learns which genetic operator produces the most valuable offspring. It extends classical Thompson Sampling with exponential discounting to handle non-stationary reward distributions.

**Core Principle**: Balance exploitation (use operators that worked well) with exploration (try uncertain operators to gather information).

### 3.2 Arm State Representation

Each operator arm \( i \in \{\text{mutation}, \text{crossover}, \text{refine}\} \) maintains the following statistics:

```python
@dataclass
class ArmStatistics:
    name: str                          # Operator identifier
    N: float = 1.0                     # Effective sample count (discounted)
    mu_tilde: float = 0.0              # Discounted sum of rewards: Σ(γ^k × r_k)
    mu_hat: float = 0.0                # Posterior mean estimate: μ̂ = μ̃ / N
    tau: float = 1.0                   # Posterior standard deviation: τ = σ/√N
    pulls: int = 0                     # Total times selected (cumulative)
    total_reward: float = 0.0          # Cumulative reward (undiscounted)
    total_fitness_reward: float = 0.0  # Fitness component (for analysis)
    total_diversity_reward: float = 0.0 # Diversity component (for analysis)
```

**Mathematical Notation:**

- \( N_i(t) \): Effective sample count for arm \( i \) at time \( t \)
- \( \tilde{\mu}\_i(t) \): Discounted sum of rewards
- \( \hat{\mu}\_i(t) \): Estimated mean reward
- \( \tau_i(t) \): Posterior uncertainty (standard deviation)
- \( \gamma \): Discount factor (default: 0.9)
- \( \sigma \): Reward variance (default: 1.0 for normalized rewards)

### 3.3 Operator Selection via Thompson Sampling

At each offspring generation, the system samples from each operator's posterior and selects the one with the highest sampled value.

**Algorithm:**

```python
def select_arm(self) -> Tuple[str, float, Dict]:
    """
    Select an operator using Thompson Sampling.

    Returns:
        (selected_operator, sampled_theta, arm_snapshot)
    """
    samples = {}
    snapshot = {}

    for arm_name, stats in self.arm_stats.items():
        # Sample from Gaussian posterior: θᵢ ~ N(μ̂ᵢ, τᵢ)
        theta = np.random.normal(stats.mu_hat, stats.tau)
        samples[arm_name] = theta

        # Log current state
        snapshot[arm_name] = {
            'mu_hat': stats.mu_hat,
            'tau': stats.tau,
            'N': stats.N,
            'theta_sampled': theta,
            'pulls': stats.pulls,
        }

    # Select arm with maximum sampled value
    selected_arm = max(samples, key=samples.get)
    selected_theta = samples[selected_arm]

    self.selection_history.append(selected_arm)
    self.total_rounds += 1

    return selected_arm, selected_theta, snapshot
```

**Mathematical Formulation:**

For each arm \( i \), sample:

\[
\theta_i \sim \mathcal{N}(\hat{\mu}\_i, \tau_i^2)
\]

Select:

\[
i^\* = \arg\max\_{i \in \{\text{mut}, \text{xo}, \text{ref}\}} \theta_i
\]

**Example Scenario:**

```
Generation 5, Prior to Selection:

Arm States:
  mutation:  μ̂ = 0.15, τ = 0.42, N = 5.6, pulls = 18
  crossover: μ̂ = 0.28, τ = 0.35, N = 8.2, pulls = 22
  refine:    μ̂ = 0.19, τ = 0.51, N = 3.8, pulls = 12

Thompson Sampling (one draw):
  mutation:  θ ~ N(0.15, 0.42) → θ = 0.32
  crossover: θ ~ N(0.28, 0.35) → θ = 0.41  ← SELECTED
  refine:    θ ~ N(0.19, 0.51) → θ = 0.09

Selected: crossover (highest sampled value)
```

**Interpretation:**

- Crossover has highest mean reward (0.28) and moderate uncertainty (0.35)
- Refine has high uncertainty (0.51), allowing occasional exploration
- Mutation has lower mean but contributes diversity through exploration

### 3.4 Posterior Update with Discounting

After observing reward \( r \) from selected operator, update **all** arms:

**Algorithm:**

```python
def update(self, arm_name: str, reward: float) -> None:
    """
    Update posteriors with exponential discounting.

    Args:
        arm_name: Selected operator
        reward: Observed normalized reward
    """
    sigma = np.sqrt(self.reward_variance)  # σ = 1.0 for normalized rewards

    # Step 1: Apply discount to ALL arms (exponential forgetting)
    for stats in self.arm_stats.values():
        stats.N = self.discount * stats.N              # N ← γ · N
        stats.mu_tilde = self.discount * stats.mu_tilde # μ̃ ← γ · μ̃

        # Update uncertainty after discounting
        if stats.N > 0:
            stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
        else:
            stats.tau = self.tau_max

    # Step 2: Update selected arm with new observation
    if arm_name in self.arm_stats:
    stats = self.arm_stats[arm_name]
        stats.N += 1.0                       # N ← N + 1
        stats.mu_tilde += reward             # μ̃ ← μ̃ + r
        stats.pulls += 1                     # Increment pull count
        stats.total_reward += reward

        # Recompute posterior mean
        if stats.N > 0:
            stats.mu_hat = stats.mu_tilde / stats.N
        else:
            stats.mu_hat = self.prior_mean

        # Recompute posterior uncertainty
        if stats.N > 0:
            stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
        else:
            stats.tau = self.tau_max

    self.reward_history.append((arm_name, reward))
```

**Mathematical Formulation:**

For all arms \( i \):

\[
\begin{align}
N*i(t+1) &= \gamma \cdot N_i(t) + \mathbb{1}[i = i^*] \\
\tilde{\mu}\_i(t+1) &= \gamma \cdot \tilde{\mu}\_i(t) + \mathbb{1}[i = i^*] \cdot r \\
\hat{\mu}\_i(t+1) &= \frac{\tilde{\mu}\_i(t+1)}{N_i(t+1)} \\
\tau_i(t+1) &= \min\left(\frac{\sigma}{\sqrt{N_i(t+1)}}, \tau\*{\max}\right)
\end{align}
\]

Where:

- \( \mathbb{1}[i = i^*] \): Indicator function (1 if arm \( i \) was selected, else 0)
- \( \gamma \): Discount factor (typically 0.9)
- \( \sigma \): Standard deviation of rewards (1.0 for normalized)
- \( \tau\_{\max} \): Maximum uncertainty cap (default: 1.0)

### 3.5 Discounting Dynamics

**Why Discounting?**

1. **Non-Stationary Environments**: Operator effectiveness changes over generations as the population evolves
2. **Recency Bias**: Recent performance is more relevant than ancient history
3. **Exploration Maintenance**: Prevents premature convergence to a single operator

**Discount Factor Effect** (\( \gamma = 0.9 \)):

| Time Steps Ago | Effective Weight | Interpretation |
| -------------- | ---------------- | -------------- |
| 0 (current)    | 1.000            | Full weight    |
| 1              | 0.900            | 90% weight     |
| 5              | 0.590            | ~59% weight    |
| 10             | 0.349            | ~35% weight    |
| 20             | 0.122            | ~12% weight    |
| 50             | 0.005            | <1% weight     |

**Update Example:**

```
Before Update (crossover selected, r = +0.42):
  mutation:  N = 5.6,  μ̃ = 0.84,  μ̂ = 0.15,  τ = 0.42
  crossover: N = 8.2,  μ̃ = 2.30,  μ̂ = 0.28,  τ = 0.35
  refine:    N = 3.8,  μ̃ = 0.72,  μ̂ = 0.19,  τ = 0.51

After Discount (γ = 0.9):
  mutation:  N = 5.04, μ̃ = 0.756, μ̂ = 0.15,  τ = 0.45
  crossover: N = 7.38, μ̃ = 2.070, μ̂ = 0.28,  τ = 0.37
  refine:    N = 3.42, μ̃ = 0.648, μ̂ = 0.19,  τ = 0.54

After Observation (crossover receives r = +0.42):
  mutation:  N = 5.04, μ̃ = 0.756, μ̂ = 0.15,  τ = 0.45
  crossover: N = 8.38, μ̃ = 2.490, μ̂ = 0.30,  τ = 0.35  ← Updated
  refine:    N = 3.42, μ̃ = 0.648, μ̂ = 0.19,  τ = 0.54

Changes:
  - Crossover: μ̂ increased from 0.28 → 0.30 (positive reward)
  - Crossover: τ decreased from 0.37 → 0.35 (less uncertainty)
  - Other arms: μ̂ unchanged, but τ slightly increased (more uncertain relative to selected arm)
```

### 3.6 Exploration-Exploitation Trade-off

**Uncertainty-Driven Exploration:**

- High \( \tau \): Wide posterior → Frequent sampling of high values → More exploration
- Low \( \tau \): Narrow posterior → Rare extreme samples → More exploitation

**Adaptive Selection Probabilities:**

Selection probability \( P(i) \) can be approximated via Monte Carlo (1000 samples):

```python
def get_arm_probabilities(self, n_samples: int = 1000) -> Dict[str, float]:
    """Estimate P(arm selected) via Monte Carlo."""
    counts = {arm: 0 for arm in self.arms}

    for _ in range(n_samples):
        samples = {
            arm: np.random.normal(stats.mu_hat, stats.tau)
            for arm, stats in self.arm_stats.items()
        }
        selected = max(samples, key=samples.get)
        counts[selected] += 1

    return {arm: count / n_samples for arm, count in counts.items()}
```

**Example Probabilities:**

```
Generation 1 (uniform priors):
  P(mutation)  = 33%
  P(crossover) = 34%
  P(refine)    = 33%

Generation 5 (learning):
  P(mutation)  = 18%
  P(crossover) = 56%  ← Emerging preference
  P(refine)    = 26%

Generation 10 (exploitation):
  P(mutation)  = 12%
  P(crossover) = 71%  ← Strong preference
  P(refine)    = 17%
```

---

## 4. Three-Operator System

MADA-LLAMEA v2 introduces a **third operator** (refine) alongside mutation and crossover, each serving a distinct role in the evolutionary search process. All three operators compete via D-TS to learn which is most effective at different stages of evolution.

### 4.1 Operator Design Philosophy

| Operator      | Strategy              | LLM Instruction Style                  | Goal                                               |
| ------------- | --------------------- | -------------------------------------- | -------------------------------------------------- |
| **Mutation**  | Redesign from scratch | "Improve it" (minimal feedback)        | **Exploration**: Generate radical alternatives     |
| **Crossover** | Combine two parents   | "Combine patterns from both solutions" | **Exploitation**: Recombine successful patterns    |
| **Refine**    | Redesign and improve  | "Improve it" (minimal feedback)        | **Alternative Exploration**: Compete with mutation |

### 4.2 Mutation Operator

**Purpose**: Generate behaviorally novel algorithms through redesign.

**Parent Selection**: Random single parent (focal parent or random choice from population).

**Prompt Template**:

```python
EOH_MUTATION_PROMPT = """Last algorithm {name} scored AOCC {score:.2f}. {feedback}Improve it. Format:
# Name: <classname>
# Code: <code>"""
```

**Feedback** (minimal, error-focused):

```python
feedback = ""
if parent.error:
    feedback = f"The algorithm got an error: {parent.error}. "
```

**Implementation**:

````python
def _mutate(self, parent: MADASolution) -> MADASolution:
    """Apply mutation - baseline style redesign."""
    self.child_counter += 1

    # Set conversation context (baseline style)
    self._last_algorithm = f"# Name: {parent.name}\n# Code:\n```python\n{parent.code}\n```"

    # Build feedback
    feedback = ""
    if parent.error:
        feedback = f"The algorithm got an error: {parent.error}. "

    prompt = EOH_MUTATION_PROMPT.format(
        name=parent.name,
        score=parent.fitness,
        std=parent.fitness_std,
        feedback=feedback
    )

    try:
        message = self._call_llm(prompt, parent)
        code = self._extract_code(message)
        class_name = self._extract_class_name(code, f"{parent.name}Mut{self.child_counter}")

        solution = MADASolution(
            code=code,
            name=class_name,
            generation=parent.generation + 1,
            parent_ids=[parent.id]
        )
        solution.operator = "mutation"
        return solution
    except Exception as e:
        # Return failed solution (will be penalized with R = -1.0)
        ...
````

**Example Prompt**:

```
Last algorithm AdaptiveDE scored AOCC 0.52. Improve it. Format:
# Name: <classname>
# Code: <code>
```

**Characteristics:**

- **Concise**: Minimal prompt to encourage creativity
- **Error-aware**: Highlights failures to avoid repeat mistakes
- **No detailed feedback**: Prevents overfitting to specific weaknesses

### 4.3 Refine Operator

**Purpose**: Alternative improvement strategy (similar to mutation, competes via D-TS).

**Parent Selection**: Random single parent (same as mutation).

**Prompt Template**:

```python
EOH_REFINE_PROMPT = """Last algorithm {name} scored AOCC {score:.2f}. {feedback}Improve it. Format:
# Name: <classname>
# Code: <code>"""
```

**Feedback** (minimal, error-focused):

```python
feedback = ""
if parent.error:
    feedback = f"The algorithm got an error: {parent.error}. "
```

**Implementation**:

````python
def _refine(self, parent: MADASolution) -> MADASolution:
    """Apply refine - baseline style redesign."""
    self.child_counter += 1

    # Set conversation context (baseline style)
    self._last_algorithm = f"# Name: {parent.name}\n# Code:\n```python\n{parent.code}\n```"

    # Build feedback (baseline style - error if present)
    feedback = ""
    if parent.error:
        feedback = f"The algorithm got an error: {parent.error}. "

    prompt = EOH_REFINE_PROMPT.format(
        name=parent.name,
        score=parent.fitness,
        std=parent.fitness_std,
        feedback=feedback
    )

    try:
        message = self._call_llm(prompt, parent)
        code = self._extract_code(message)
        class_name = self._extract_class_name(code, f"{parent.name}Ref{self.child_counter}")

        solution = MADASolution(
            code=code,
            name=class_name,
            generation=parent.generation + 1,
            parent_ids=[parent.id]
        )
        solution.operator = "refine"
        return solution
    except Exception as e:
        # Return failed solution
        ...
````

**Example Prompt**:

```
Last algorithm AdaptiveDE scored AOCC 0.52. Improve it. Format:
# Name: <classname>
# Code: <code>
```

**Example with Error**:

```
Last algorithm BrokenDE scored AOCC 0.00. The algorithm got an error: Budget 10000 exceeded at eval 10000. Improve it. Format:
# Name: <classname>
# Code: <code>
```

**Characteristics:**

- **Minimal feedback**: Same as mutation (simple, concise)
- **Error-aware**: Highlights failures to avoid repetition
- **Alternative exploration**: Competes with mutation via D-TS to discover which works better

### 4.4 Crossover Operator

**Purpose**: Combine successful patterns from two high-performing parents.

**Parent Selection**: Behavioral diversity-based selection:

1. **Parent A (Exploitation)**: Highest fitness in population
2. **Parent B (Exploration)**: Maximum trace distance from Parent A

**Prompt Template**:

````python
EOH_CROSSOVER_IMPLICIT_PROMPT = """Two parent solutions (BBOB, 5D, bounds [-5,5]):
Parent A (score {score_a:.2f}):
```python
{code_a}
````

Parent B (score {score_b:.2f}):

```python
{code_b}
```

Create a better offspring by combining patterns from both. Same interface: **init**(self, budget), **call**(self, func).
Reply only with:

# Name: <classname>

# Code: <code>

"""

````

**Implementation**:

```python
def _crossover(self, parent_a: MADASolution, parent_b: MADASolution) -> MADASolution:
    """Apply crossover with behavioral diversity selection."""
    self.child_counter += 1

    # Crossover gets fresh context (both parents, no conversation history)
    self._last_algorithm = ""

    prompt = EOH_CROSSOVER_IMPLICIT_PROMPT.format(
        score_a=parent_a.fitness,
        code_a=parent_a.code,
        score_b=parent_b.fitness,
        code_b=parent_b.code,
    )

    try:
        message = self._call_llm(prompt)
        code = self._extract_code(message)
        class_name = self._extract_class_name(code, f"Hybrid{self.child_counter}")

        solution = MADASolution(
            code=code,
            name=class_name,
            generation=max(parent_a.generation, parent_b.generation) + 1,
            parent_ids=[parent_a.id, parent_b.id]
        )
        solution.operator = "crossover"
        return solution
    except Exception as e:
        # Return failed solution
        ...
````

**Behavioral Parent Selection**:

```python
def _select_diverse_parent(self, sorted_parents, parent_a):
    """Select Parent B with maximum trace distance from Parent A."""
    if len(sorted_parents) < 2:
        return parent_a

    if self.use_behavioral_selection:
        # Use trace-based diversity
        parent_a_sel, parent_b = select_behavioral_parents(
            sorted_parents,
            enabled=True,
            trace_attr='trace',
            fitness_attr='fitness'
        )

        # Log trace distance
        if hasattr(parent_a, 'trace') and hasattr(parent_b, 'trace'):
            if parent_a.trace and parent_b.trace:
                dist = calculate_trace_distance(parent_a.trace, parent_b.trace)
                self._last_selection_info['trace_distance'] = dist

        return parent_b

    # Fallback: Code-based diversity (hash comparison)
    for candidate in sorted_parents[1:]:
        if (candidate.code or "") != (parent_a.code or ""):
            return candidate
    return sorted_parents[1]
```

**Trace Distance Calculation**:

For parents \( A \) and \( B \) with traces \( T_A = [t_A^1, \ldots, t_A^B] \) and \( T_B = [t_B^1, \ldots, t_B^B] \):

\[
d(A, B) = \frac{1}{\sqrt{B}} \sqrt{\sum\_{i=1}^{B} \left( \frac{t_A^i - \min}{\max - \min} - \frac{t_B^i - \min}{\max - \min} \right)^2}
\]

Where:

- \( B = 10{,}000 \): Evaluation budget (trace length)
- \( [\min, \max] \): Global trace normalization bounds

**Example Parent Selection**:

```
Population (sorted by fitness):
  1. AdaptiveDE (fitness: 0.65, trace: [1.0, 0.8, 0.6, ...])  ← Parent A
  2. HybridPSO (fitness: 0.62, trace: [1.0, 0.82, 0.64, ...])
  3. CMAESVar (fitness: 0.58, trace: [1.0, 0.5, 0.3, ...])
  4. RandomDE (fitness: 0.51, trace: [1.0, 0.9, 0.85, ...])

Trace Distances from Parent A:
  d(AdaptiveDE, HybridPSO) = 0.03  (very similar behavior)
  d(AdaptiveDE, CMAESVar)  = 0.31  (different behavior) ← SELECTED as Parent B
  d(AdaptiveDE, RandomDE)  = 0.15  (moderate difference)

Selected Parents:
  Parent A: AdaptiveDE (best fitness)
  Parent B: CMAESVar (most diverse behavior)
```

**Characteristics:**

- **Implicit combination**: LLM decides how to merge patterns
- **No detailed feedback**: Focuses on code structure, not performance details
- **Behavioral diversity**: Ensures Parent B provides complementary search strategies

### 4.5 Operator Comparison

| Aspect                   | Mutation              | Refine                | Crossover               |
| ------------------------ | --------------------- | --------------------- | ----------------------- |
| **Parents**              | 1                     | 1                     | 2                       |
| **Feedback**             | Minimal (errors only) | Minimal (errors only) | None (code only)        |
| **Novelty**              | High                  | High                  | Medium                  |
| **Risk**                 | High (may fail)       | High (may fail)       | Medium                  |
| **Prompt Length**        | Short (~50 tokens)    | Short (~50 tokens)    | Long (~500-2000 tokens) |
| **LLM Temperature**      | 0.8                   | 0.8                   | 0.8                     |
| **Typical Success Rate** | 60-70%                | 60-70%                | 70-80%                  |
| **Best When**            | D-TS decides          | D-TS decides          | D-TS decides            |

### 4.6 Why Two Similar Operators (Mutation + Refine)?

**Design Rationale**: Having both mutation and refine with the **same minimal prompt** allows D-TS to learn which LLM-stochasticity patterns are more effective:

1. **Multiple Draws from Same Distribution**: Each call to the LLM with the same prompt produces different outputs (temperature = 0.8)
2. **Exploration via Randomness**: Two operators with identical prompts effectively double the sampling from creative LLM generations
3. **Data-Driven Discovery**: D-TS automatically discovers if one consistently produces better results (perhaps due to conversation context or sampling order)
4. **Increased Diversity**: More total offspring per generation without increasing prompt complexity

**Alternative Interpretation**: You can view mutation and refine as:

- **Mutation**: "First attempt" at improving a parent
- **Refine**: "Second attempt" at improving a parent (different random seed from LLM)

### 4.7 Operator Dynamics Over Generations

**Hypothetical Evolution** (learned through D-TS):

```
Generation 1-3 (Early Exploration):
  - Mutation: 35%
  - Refine: 30%
  - Crossover: 35%

Generation 4-7 (Learned Preferences):
  - Mutation: 25% (if mutation historically better)
  - Refine: 20%  (if refine historically worse)
  - Crossover: 55% (if crossover consistently strong)

Generation 8+ (Exploitation):
  - Mutation: 15%
  - Refine: 10%
  - Crossover: 75% (dominant operator)
```

**Actual selection is data-driven via D-TS** based on observed composite rewards, not pre-programmed.

**Key Insight**: The system might discover that mutation and refine have similar performance (since they use identical prompts), or it might find that one consistently outperforms the other due to conversation history effects or LLM sampling dynamics.

---

## 4.8 Complete Prompt Specifications (Baseline LLAMEA Style)

This section provides the **exact prompts** used in MADA-LLAMEA v2, following the **baseline LLAMEA prompt structure** for reproducibility.

### 4.8.1 Role Prompt

Used at the beginning of every LLM call to establish the role and constraints:

```python
ROLE_PROMPT = """You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."""
```

**Purpose**:

- Establishes expert persona
- Sets expectations for optimization algorithm design
- Prepended to all task prompts

### 4.8.2 Task Prompt (Baseline LLAMEA)

The core task description used in all prompts:

```python
TASK_PROMPT = """The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
Give an excellent and novel heuristic algorithm to solve this task.
"""
```

### 4.8.3 Example Prompt

Example code structure shown to the LLM:

````python
EXAMPLE_PROMPT = """An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
        return self.f_opt, self.x_opt
```
"""
````

### 4.8.4 Output Format Prompt

Specifies the required response format (NOTE: **Description field is required**):

```python
OUTPUT_FORMAT_PROMPT = """Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
<code>
"""
```

**Key Change from Previous Version**: The **Description field** is now **required** in all LLM responses, enabling richer population context.

### 4.8.5 Initialization Prompt (Composite)

Used during population initialization (combines all components):

```python
INIT_PROMPT = TASK_PROMPT + EXAMPLE_PROMPT + OUTPUT_FORMAT_PROMPT
```

**Key Elements**:

- **Task specification**: BBOB suite, 24 functions, 5D
- **Interface requirements**: `__init__(self, budget)` and `__call__(self, func)`
- **Constraints**: Budget limit (10,000 evals), bounds [-5, 5]
- **Example**: RandomSearch demonstrates the expected structure
- **Output format**: Description + Code (structured response)

### 4.8.6 Mutation Prompt (MADA v2)

Used when D-TS selects mutation. The instruction is **“Generate a NEW and DIFFERENT algorithm.”** Context (population, best, last algorithm) is supplied via conversation history; the prompt itself is minimal:

```python
MUTATION_INSTRUCTION = "Generate a NEW and DIFFERENT algorithm."

prompt = f"{MUTATION_INSTRUCTION} Give the response in the format:\n# Description: <short-description>\n# Code: <code>"
```

**Design Philosophy**:

- **Explicit novelty**: Pushes the LLM to produce a genuinely different algorithm.
- **Simple surface prompt**: All contextual grounding (population, best-so-far, last algorithm) comes from the conversation history built in `_call_llm`.
- **Consistent output shape**: Requires `# Description` and `# Code` fields for downstream parsing.

### 4.8.7 Refine Prompt (Baseline LLAMEA Style)

Used when D-TS selects refine operator. **Identical to mutation** - the only difference is which operator the bandit selects:

**Successful Parent**:

```python
prompt = f"The last proposed algorithm {parent.name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {parent.fitness:.2f}, and a standard deviation of {parent.fitness_std:.2f}. Either refine or redesign to improve the algorithm. Give the response in the format:\n# Description: <short-description>\n# Code: <code>"
```

**Failed Parent** (with error):

```python
prompt = f"The last proposed algorithm {parent.name} got an error: {parent.error}. Either refine or redesign to improve the algorithm. Give the response in the format:\n# Description: <short-description>\n# Code: <code>"
```

**Example** (same as mutation):

```
The last proposed algorithm AdaptiveDE_v4 got an average Area over the convergence curve (AOCC, 1.0 is the best) of 0.55, and a standard deviation of 0.12. Either refine or redesign to improve the algorithm. Give the response in the format:
# Description: <short-description>
# Code: <code>
```

**Design Philosophy**:

- **Same prompt as mutation**: The prompt is identical, but D-TS tracks them separately
- **Different random seed**: LLM generates different output due to temperature=0.8
- **Competes with mutation**: D-TS learns which operator produces better offspring in different contexts

### 4.8.8 Crossover Prompt (Baseline LLAMEA Style)

Used when D-TS selects crossover operator. **Does NOT include best algorithm** - only population context and parent codes:

````python
prompt = f"""Two parent solutions have been selected:
Parent A: {parent_a.name} (AOCC: {parent_a.fitness:.2f})
```python
{parent_a.code}
````

Parent B: {parent_b.name} (AOCC: {parent_b.fitness:.2f})

```python
{parent_b.code}
```

Create a better offspring by combining patterns from both parent algorithms. Give the response in the format:

# Description: <short-description>

# Code: <code>"""

```

**Example Instantiated Prompt**:

```

Two parent solutions have been selected:
Parent A: EnhancedAdaptiveDE (AOCC: 0.62)

```python
class EnhancedAdaptiveDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = 50

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        # ... (full implementation)
```

Parent B: AdaptiveDE_v4 (AOCC: 0.55)

```python
class AdaptiveDE_v4:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.adaptive_F = True

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        # ... (full implementation)
```

Create a better offspring by combining patterns from both parent algorithms. Give the response in the format:

# Description: <short-description>

# Code: <code>

````

**Design Philosophy**:
- **No best algorithm**: Crossover does NOT include the best algorithm in conversation (unlike mutation/refine)
- **Population context only**: Only includes population summary (names, descriptions, scores)
- **Full parent codes**: Both parent implementations provided in entirety
- **Behavioral diversity**: Parent B selected for maximum trace distance from Parent A
- **No conversation history**: Fresh conversation (no `last_algorithm` assistant response)

### 4.8.9 Conversation Structure (Baseline LLAMEA)

MADA-LLAMEA v2 uses a **TRUE baseline LLAMEA conversation structure** to maintain context:

```python
def _call_llm(self, prompt: str, include_best: bool = True) -> str:
    """Call LLM with TRUE baseline LLAMEA conversation structure."""
    messages = [
        {"role": "system", "content": ROLE_PROMPT},
        {"role": "user", "content": INIT_PROMPT},
    ]

    # 1. Add population context (current population summary)
    if self.population:
        pop_summary = "\n".join([
            f"  - {ind.name}: {ind.description} (fitness={ind.fitness:.4f})"
            for ind in self.population[:5]
        ])
        messages.append({"role": "user", "content": f"Current population:\n{pop_summary}"})

    # 2. Add best algorithm so far (skip for crossover when include_best=False)
    if include_best and self.best_ever and self.best_ever.fitness > 0:
        best_context = f"""The best so far proposed algorithm got an average AOCC of {self.best_ever.fitness:.2f} and the code was as follows:
{self.best_ever.code}"""
        messages.append({"role": "user", "content": best_context})

    # 3. Add last algorithm as assistant response (for mutation/refine only)
    if self._last_algorithm:
        messages.append({"role": "assistant", "content": self._last_algorithm})

    # 4. Add current prompt (mutation/refine/crossover request)
    messages.append({"role": "user", "content": prompt})

    # ... LLM call ...
````

**Context Components by Operator**:

| Component           | Mutation | Refine | Crossover |
| ------------------- | -------- | ------ | --------- |
| System Prompt       | ✅       | ✅     | ✅        |
| Init Prompt         | ✅       | ✅     | ✅        |
| Population Summary  | ✅       | ✅     | ✅        |
| Best Algorithm Code | ✅       | ✅     | ❌        |
| Last Algorithm      | ✅       | ✅     | ❌        |
| Current Prompt      | ✅       | ✅     | ✅        |

**Why Crossover Excludes Best Algorithm**:

- Crossover focuses on **combining two parents**, not improving the global best
- Including the best algorithm could bias the LLM toward copying it instead of recombining parent patterns
- Keeps the prompt focused on the structural combination task

**Temperature**: 0.8 (balanced between creativity and coherence)

### 4.8.10 Prompt Comparison (Baseline LLAMEA Style)

| Aspect                   | Mutation                    | Refine                      | Crossover                     |
| ------------------------ | --------------------------- | --------------------------- | ----------------------------- |
| **Prompt Message**       | ~60 tokens                  | ~60 tokens                  | ~50-200 tokens (just parents) |
| **Population Context**   | ✅ Top 5 (name+desc+score)  | ✅ Top 5 (name+desc+score)  | ✅ Top 5 (name+desc+score)    |
| **Best Algorithm Code**  | ✅ Full code                | ✅ Full code                | ❌ Excluded                   |
| **Last Algorithm**       | ✅ Previous response        | ✅ Previous response        | ❌ Fresh conversation         |
| **Parent Context**       | Name + score + std          | Name + score + std          | Full code (both parents)      |
| **Feedback**             | Errors only                 | Errors only                 | None                          |
| **Instruction**          | "Refine or redesign"        | "Refine or redesign"        | "Combine patterns"            |
| **Total Context Tokens** | ~2000-4000 (with best code) | ~2000-4000 (with best code) | ~500-2000 (no best code)      |
| **LLM Creativity**       | High (minimal constraints)  | High (minimal constraints)  | High (implicit merge)         |
| **Expected Output**      | Novel redesign              | Alternative redesign        | Hybrid strategy               |

**Key Differences from Previous Version**:

- **Description field required**: All responses must include `# Description: <text>`
- **Population context**: Now includes descriptions, not just names
- **Best algorithm included**: Mutation/Refine see the best algorithm's full code
- **Crossover isolated**: Crossover focuses only on parents, not influenced by global best

### 4.8.11 Response Parsing (Baseline LLAMEA Format)

All prompts expect the **baseline LLAMEA response format** with Description field:

````
# Description: Short one-line description of the algorithm's main idea
# Code:
```python
class AlgorithmClassName:
    def __init__(self, budget=10000):
        # ...

    def __call__(self, func):
        # ...
```
````

**Extraction Logic**:

````python
def _extract_description(self, message: str) -> str:
    """Extract description from LLM response (baseline LLAMEA style)."""
    pattern = r"#\s*Description:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No description provided"

def _extract_code(self, message: str) -> str:
    """Extract code from LLM response."""
    pattern = r"```(?:python)?\n(.*?)\n```"
    match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    raise NoCodeException("No code block found")

def _extract_class_name(self, code: str, default: str) -> str:
    """Extract class name from code."""
    for line in code.splitlines():
        if line.strip().startswith("class "):
            match = re.search(r"class\s+(\w+)", line)
            if match:
                return match.group(1)
    return default
````

**Error Handling**:

- No code block found → NoCodeException → offspring marked as failed
- No class definition → Use default name (e.g., "Hybrid42")
- Syntax errors → Caught during evaluation phase → fitness = 0.0, error logged

---

## 5. Algorithm Evaluation with Trace Collection

Evaluation serves two purposes in MADA-LLAMEA:

1. **Performance Assessment**: AOCC (Area Over Convergence Curve) on BBOB
2. **Behavioral Characterization**: Optimization trajectory (trace) for diversity measurement

### 5.1 TraceCollector: Behavioral Signature Capture

The `TraceCollector` wraps each BBOB problem to record the **best-so-far fitness** at every function evaluation. This trace becomes the algorithm's behavioral signature.

**Implementation**:

```python
class TraceCollector:
    """
    Wrapper around IOH problem to collect best-so-far trace during optimization.
    """

    def __init__(self, problem, budget: int):
        """
        Args:
            problem: IOH problem instance (callable with .bounds attribute)
            budget: Maximum number of evaluations (10,000 for BBOB)
        """
        self.problem = problem
        self.budget = budget
        self.trace: List[float] = []           # Best-so-far at each eval
        self.best_so_far = float('inf')        # Current best fitness (minimization)
        self.eval_count = 0                    # Current evaluation count

        # Copy bounds from problem for compatibility
        self.bounds = problem.bounds

    def __call__(self, x) -> float:
        """
        Evaluate solution x and update trace.

        Args:
            x: Solution vector (numpy array, dim=5)

        Returns:
            float: Fitness value f(x)

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

        # Record best-so-far at this step
        self.trace.append(self.best_so_far)

        return f

    def get_trace(self) -> List[float]:
        """
        Get the trace vector, padded to budget length.

        Returns:
            list: Best-so-far at each evaluation [f₀, f₁, ..., f_B]
        """
        trace = self.trace.copy()

        # Pad with final value if algorithm terminated early
        if len(trace) < self.budget:
            final_value = trace[-1] if trace else float('inf')
            trace.extend([final_value] * (self.budget - len(trace)))

        return trace[:self.budget]
```

**Trace Properties:**

- **Length**: Always exactly `budget` (10,000), padded if needed
- **Monotonicity**: Non-increasing (best-so-far never worsens)
- **Scale**: Raw fitness values (not normalized during collection)
- **Uniqueness**: Each algorithm produces a unique trajectory

**Example Trace** (first 10 steps):

```
Evaluation:    0      1      2      3      4      5      6      7      8      9
Fitness:     532.1  532.1  412.8  412.8  398.2  398.2  301.5  287.3  287.3  245.6
              └─┘    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘    └─┘
             init   bad    better  bad   better  bad    better better  bad   better
```

### 5.2 BBOB Benchmark Evaluation

Each algorithm is evaluated on the **BBOB (Black-Box Optimization Benchmarking)** suite:

**Benchmark Structure:**

```
24 functions × 3 instances × 3 repetitions = 216 runs per algorithm
└─────────┘   └─────────┘   └──────────┘
   Diversity    Robustness   Statistical
                             Reliability
```

**Evaluation Loop**:

```python
def evaluate_algorithm_with_trace(algorithm_code, algorithm_name, eval_budget):
    """
    Evaluate algorithm on BBOB and return optimization trace.

    Returns:
        tuple: (aucs, detailed_aucs, aggregated_trace, error)
    """
    try:
        # Step 1: Execute algorithm code in safe globals
        safe_globals = {
            "lb": -5.0, "ub": 5.0, "bounds": (-5.0, 5.0),
            "learning_rate": 0.5, "local_search_prob": 0.1,
            "stagnation_multiplier": 1.0, "initial_pop": None,
            "pop_size": 50, "F": 0.5, "CR": 0.9,
            "archive_size_multiplier": 1.0, "eps": 1e-12,
        }

        # Safe np.random.choice wrapper (prevents replace=False errors)
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

        # Step 2: Setup IOH logger for AUC calculation
        l2 = aoc_logger(eval_budget, upper=1e2, triggers=[logger.trigger.ALWAYS])

        aucs = []
        detail_aucs = []
        detailed_aucs = [0, 0, 0, 0, 0]  # Per-group AUCs
        all_traces = []

        # Step 3: Run on all BBOB functions
        for fid in np.arange(1, 25):           # 24 functions
            for iid in [1, 2, 3]:              # 3 instances
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)

                for rep in range(3):           # 3 repetitions
                    np.random.seed(rep)

                    # Create trace collector
                    trace_collector = TraceCollector(problem, eval_budget)

                    try:
                        # Instantiate algorithm
                        algorithm = algorithm_class(eval_budget)
                        lb, ub = problem.bounds.lb, problem.bounds.ub

                        # Inject common attributes (for robustness)
                        for attr, val in [("lb", lb), ("ub", ub), ("bounds", (lb, ub)),
                                         ("pop", []), ("archive_x", []), ("archive_f", []),
                                         ("evals", 0), ("F", 0.5), ("CR", 0.9),
                                         ("archive_size_multiplier", 1.0), ("eps", 1e-12)]:
                            if not hasattr(algorithm, attr):
                                setattr(algorithm, attr, val)
                        if not hasattr(algorithm, "fitness"):
                            algorithm.fitness = lambda x: trace_collector(x)

                        # Run optimization
                        algorithm(trace_collector)

                    except OverBudgetException:
                        pass  # Expected termination
                    except Exception as e:
                        return [], [0, 0, 0, 0, 0], [], str(e)

                    # Compute AUC (Area Over Convergence Curve)
                    auc = correct_aoc(problem, l2, eval_budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)

                    # Collect trace
                    run_trace = trace_collector.get_trace()
                    all_traces.append(run_trace)

                    l2.reset(problem)
                    problem.reset()

            # Aggregate per-group AUCs (5 groups)
            if fid == 5:   # Separable
                detailed_aucs[0] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 9: # Low/moderate conditioning
                detailed_aucs[1] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 14: # High conditioning, unimodal
                detailed_aucs[2] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 19: # Multi-modal, adequate structure
                detailed_aucs[3] = np.mean(detail_aucs); detail_aucs = []
            elif fid == 24: # Multi-modal, weak structure
                detailed_aucs[4] = np.mean(detail_aucs); detail_aucs = []

        # Step 4: Aggregate traces across all 216 runs
        aggregated_trace = aggregate_traces(all_traces, eval_budget)

        return aucs, detailed_aucs, aggregated_trace, ""

    except KeyboardInterrupt:
        return [], [0, 0, 0, 0, 0], [], "KeyboardInterrupt"
    except Exception as e:
        return [], [0, 0, 0, 0, 0], [], str(e)
    finally:
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice
```

**Evaluation Metrics:**

1. **AUC (AOCC)**: Area Over Convergence Curve

   - Measures cumulative performance over the optimization run
   - Higher is better (normalized to [0, 1])
   - Computed per run: 216 AUC values → mean AUC as fitness

2. **Detailed AUCs**: Per function-group performance

   - 5 values, one per BBOB group
   - Used by **refine operator** for targeted feedback

3. **Aggregated Trace**: Representative trajectory
   - Median of 216 traces (robust to outliers)
   - Length: 10,000 (one per evaluation)
   - Used for **NN-Dist** diversity calculation

### 5.3 Trace Aggregation

**Problem**: 216 traces per algorithm → need single representative trace.

**Solution**: Element-wise median (robust to outliers).

```python
def aggregate_traces(traces: List[List[float]], budget: int) -> List[float]:
    """
    Aggregate multiple traces into a single representative trace.
    Uses element-wise median for robustness to outliers.

    Args:
        traces: List of trace lists from different runs (216 traces)
        budget: Expected trace length (10,000)

    Returns:
        list: Aggregated trace of length budget
    """
    if not traces:
        return [0.0] * budget

    # Ensure all traces are same length (pad if needed)
    padded = []
    for t in traces:
        if len(t) < budget:
            final_val = t[-1] if t else 0.0
            t = t + [final_val] * (budget - len(t))
        padded.append(t[:budget])

    # Element-wise median (robust to outliers)
    arr = np.array(padded, dtype=float)  # Shape: (216, 10000)
    aggregated = np.median(arr, axis=0).tolist()

    return aggregated
```

**Why Median?**

| Statistic  | Pros                       | Cons                            |
| ---------- | -------------------------- | ------------------------------- |
| **Mean**   | Efficient, differentiable  | Sensitive to outliers (crashes) |
| **Median** | Robust to outliers, stable | Slightly more computation       |
| **Min**    | Best-case performance      | Not representative              |
| **Max**    | Worst-case robustness      | Overly pessimistic              |

**Example Aggregation** (3 traces, 5 steps):

```
Trace 1:  [100.0, 80.0, 60.0, 50.0, 45.0]
Trace 2:  [110.0, 70.0, 55.0, 48.0, 42.0]
Trace 3:  [105.0, 85.0, 65.0, 52.0, 999.9]  ← outlier at end

Aggregated (median):
          [105.0, 80.0, 60.0, 50.0, 45.0]
           ↑      ↑      ↑      ↑      ↑
         median median median median median
         of 3   of 3   of 3   of 3   of 3
         values values values values values
```

**Aggregation protects against**:

- Algorithm crashes (trace ends early → padded with final value)
- Instance-specific outliers (one bad run doesn't dominate)
- Random seed variance (3 repetitions per instance)

---

## 6. Behavioral Diversity Measurement

Behavioral diversity quantifies how **different** an algorithm's search strategy is from previously evaluated algorithms, independent of performance. MADA-LLAMEA uses **NN-Dist** (Nearest-Neighbor Distance) in trace space.

### 6.1 Motivation: Why Behavioral Diversity?

**Problem**: Traditional diversity metrics (e.g., edit distance on code) don't capture algorithmic behavior.

- Two algorithms with different code may behave identically
- Two algorithms with similar code may explore completely different regions

**Solution**: Use optimization trajectories (traces) as behavioral fingerprints.

- **Trace** = sequence of best-so-far fitness at each evaluation
- Captures exploration strategy, convergence speed, stagnation patterns

**Benefits**:

1. **Phenotypic**: Measures what algorithms **do**, not what they **are**
2. **Task-specific**: Same code may behave differently on different problems
3. **Continuous**: Enables smooth diversity gradients (unlike discrete code comparison)

### 6.2 NN-Dist Definition

**Nearest-Neighbor Distance** (NN-Dist) measures how far a new algorithm's trace is from its closest historical trace.

**Mathematical Definition:**

Given:

- New algorithm trace: \( T\_{\text{new}} = [f_1, f_2, \ldots, f_B] \)
- History of \( N \) traces: \( \mathcal{H} = \{T_1, T_2, \ldots, T_N\} \)
- Global normalization bounds: \( [f_{\min}, f_{\max}] \)

Compute:

\[
\text{NN-Dist}(T*{\text{new}}, \mathcal{H}) = \min*{T*i \in \mathcal{H}} d(T*{\text{new}}, T_i)
\]

Where \( d(\cdot, \cdot) \) is the normalized Euclidean distance:

\[
d(T*a, T_b) = \frac{1}{\sqrt{B}} \sqrt{\sum*{t=1}^{B} \left( \frac{f*a^t - f*{\min}}{f*{\max} - f*{\min}} - \frac{f*b^t - f*{\min}}{f*{\max} - f*{\min}} \right)^2}
\]

**Properties**:

- \( \text{NN-Dist} \in [0, 1] \) (approximately, after normalization)
- Higher = more novel behavior
- \( \text{NN-Dist} = 0 \) iff trace is identical to some historical trace
- \( \text{NN-Dist} \approx 1 \) for maximally different behavior

### 6.3 Implementation

**Step 1: Global Normalization**

To make traces comparable across different fitness scales:

```python
# Update global bounds after each evaluation
if trace:
    global_min_fitness = min(global_min_fitness, min(trace))
    global_max_fitness = max(global_max_fitness, max(trace))
```

**Step 2: NN-Dist Calculation**

```python
def calculate_nn_dist(
    new_trace: List[float],
    history_traces: List[List[float]],
    global_min: Optional[float] = None,
    global_max: Optional[float] = None
) -> float:
    """
    Calculate nearest-neighbor distance for behavioral diversity.

    Args:
        new_trace: Convergence trace of new algorithm [f₁, ..., f_B]
        history_traces: Traces of previously evaluated algorithms
        global_min: Minimum fitness seen across all runs (for normalization)
        global_max: Maximum fitness seen across all runs (for normalization)

    Returns:
        float: Nearest-neighbor distance in [0, ~1] range
    """
    # Edge case: first algorithm has maximum novelty
    if not history_traces:
        return 1.0

    new_arr = np.array(new_trace, dtype=float)

    # Determine normalization range
    if global_min is not None and global_max is not None:
        norm_min, norm_max = global_min, global_max
    else:
        # Auto-compute from new trace (fallback)
        norm_min, norm_max = np.min(new_arr), np.max(new_arr)

    # Normalize new trace to [0, 1]
    range_val = norm_max - norm_min
    if range_val > 1e-10:
        new_normalized = (new_arr - norm_min) / range_val
    else:
        new_normalized = np.zeros_like(new_arr)  # Flat trace

    min_distance = float('inf')

    # Find nearest neighbor in history
    for hist_trace in history_traces:
        hist_arr = np.array(hist_trace, dtype=float)

        # Normalize history trace with same bounds
        if range_val > 1e-10:
            hist_normalized = (hist_arr - norm_min) / range_val
        else:
            hist_normalized = np.zeros_like(hist_arr)

        # Handle length mismatch (use shorter length)
        min_len = min(len(new_normalized), len(hist_normalized))
        new_trimmed = new_normalized[:min_len]
        hist_trimmed = hist_normalized[:min_len]

        # Euclidean distance
        distance = np.sqrt(np.sum((new_trimmed - hist_trimmed) ** 2))

        # Normalize by sqrt(budget) for scale-invariance
        # This approximately maps distances to [0, 1] range
        if min_len > 0:
            distance = distance / np.sqrt(min_len)
        else:
            distance = 0.0

        min_distance = min(min_distance, distance)

    return min_distance if min_distance != float('inf') else 1.0
```

### 6.4 Normalization Details

**Why Normalize?**

1. **Cross-Scale Comparison**: Different BBOB functions have vastly different fitness scales

   - f1 (Sphere): Range ~[0, 1000]
   - f10 (Rosenbrock): Range ~[0, 10000]
   - Without normalization, distances would be dominated by scale

2. **Fairness**: All algorithms evaluated on same benchmark suite should contribute equally

**Normalization Formula**:

\[
\tilde{f}^t = \frac{f^t - f*{\min}}{f*{\max} - f\_{\min}}
\]

Where:

- \( f^t \): Raw fitness at evaluation \( t \)
- \( f\_{\min} \): Global minimum (best) fitness seen so far
- \( f\_{\max} \): Global maximum (worst) fitness seen so far

**Length Normalization**:

\[
d*{\text{normalized}} = \frac{d*{\text{raw}}}{\sqrt{B}}
\]

Where \( B = 10{,}000 \) (budget).

**Rationale**: Euclidean distance grows with dimensionality. Dividing by \( \sqrt{B} \) ensures:

- Distance is approximately bounded in [0, 1]
- Comparable across different budget sizes (if needed)

### 6.5 Example Calculation

**Scenario**: Evaluating a new algorithm with history size = 3.

**Step 1: Traces** (first 5 evals shown; actual length = 10,000):

```
New algorithm:   [1000.0, 800.0, 600.0, 400.0, 200.0, ...]
History trace 1: [1000.0, 850.0, 700.0, 550.0, 400.0, ...]
History trace 2: [1000.0, 500.0, 300.0, 200.0, 100.0, ...]
History trace 3: [ 800.0, 700.0, 600.0, 500.0, 400.0, ...]
```

**Step 2: Global Bounds**:

```
global_min = 0.0    (best fitness ever seen)
global_max = 1000.0 (worst fitness ever seen)
```

**Step 3: Normalize New Trace**:

```
Normalized new: [1.0, 0.8, 0.6, 0.4, 0.2, ...]
                 ↓    ↓    ↓    ↓    ↓
            (1000-0) (800-0) (600-0) (400-0) (200-0)
             -----    -----   -----   -----   -----
              1000     1000    1000    1000    1000
```

**Step 4: Normalize History Traces**:

```
Normalized hist 1: [1.0, 0.85, 0.70, 0.55, 0.40, ...]
Normalized hist 2: [1.0, 0.50, 0.30, 0.20, 0.10, ...]
Normalized hist 3: [0.8, 0.70, 0.60, 0.50, 0.40, ...]
```

**Step 5: Compute Distances** (simplified for 5 steps; actual: 10,000):

```
d(new, hist1) = sqrt((1.0-1.0)² + (0.8-0.85)² + (0.6-0.70)² + (0.4-0.55)² + (0.2-0.40)²)
              = sqrt(0 + 0.0025 + 0.01 + 0.0225 + 0.04)
              = sqrt(0.075) = 0.274
              → 0.274 / sqrt(5) = 0.122

d(new, hist2) = sqrt((1.0-1.0)² + (0.8-0.50)² + (0.6-0.30)² + (0.4-0.20)² + (0.2-0.10)²)
              = sqrt(0 + 0.09 + 0.09 + 0.04 + 0.01)
              = sqrt(0.23) = 0.480
              → 0.480 / sqrt(5) = 0.214

d(new, hist3) = sqrt((1.0-0.8)² + (0.8-0.70)² + (0.6-0.60)² + (0.4-0.50)² + (0.2-0.40)²)
              = sqrt(0.04 + 0.01 + 0 + 0.01 + 0.04)
              = sqrt(0.10) = 0.316
              → 0.316 / sqrt(5) = 0.141
```

**Step 6: Take Minimum**:

```
NN-Dist = min(0.122, 0.214, 0.141) = 0.122
```

**Interpretation**:

- New algorithm is most similar to **hist1** (distance 0.122)
- New algorithm shows moderate novelty (12.2% of maximum possible distance)
- If distance were > 0.5, it would be considered highly novel

### 6.6 Behavioral Diversity in Context

**NN-Dist vs. Fitness**:

| Scenario | Fitness | NN-Dist | Interpretation                      |
| -------- | ------- | ------- | ----------------------------------- |
| 1        | High    | High    | **Excellent**: Novel + effective    |
| 2        | High    | Low     | Good: Effective but similar         |
| 3        | Low     | High    | Useful: Poor but explores new space |
| 4        | Low     | Low     | Bad: Poor and redundant             |

**MADA balances both** via composite reward:
\[
R = \Delta\text{fitness} + \alpha \times \text{NN-Dist}
\]

**Alpha Decay** ensures:

- Early: High \( \alpha \) → Explore diverse behaviors (Scenario 3 valued)
- Late: Low \( \alpha \) → Exploit best solutions (Scenario 2 preferred)

---

## 7. Composite Reward Calculation

The composite reward \( R \) combines **exploitation** (fitness improvement) with **exploration** (behavioral diversity) to guide the three-operator bandit system.

### 7.1 Mathematical Formulation

**Composite Reward**:

\[
R*t = \underbrace{\Delta F}*{\text{Exploitation}} + \underbrace{\alpha*t \times \text{NN-Dist}}*{\text{Exploration}}
\]

Where:

- \( \Delta F = F*{\text{child}} - F*{\text{parent}} \): Fitness improvement
- \( \alpha_t \in [0, 1] \): Time-varying diversity weight
- \( \text{NN-Dist} \in [0, 1] \): Behavioral novelty score

**Design Rationale**:

1. **Fitness Delta** (\( \Delta F \)):

   - Positive when child outperforms parent → reward operator
   - Negative when child underperforms → penalize operator
   - Direct exploitation signal

2. **Diversity Bonus** (\( \alpha_t \times \text{NN-Dist} \)):

   - Rewards novel behaviors even if fitness is worse
   - Prevents premature convergence to local optima
   - Ensures exploration of search space

3. **Time-Varying Alpha** (\( \alpha_t \)):
   - Early generations: High \( \alpha \) → prioritize exploration
   - Late generations: Low \( \alpha \) → prioritize exploitation
   - Smooth transition from exploration to exploitation

### 7.2 Implementation

```python
def compute_composite_reward(
    child_fitness: float,
    parent_fitness: float,
    nn_dist: float,
    alpha: float,
    error: bool = False,
    clamp: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute the MADA composite reward.

    Args:
        child_fitness: Fitness of offspring algorithm (higher is better)
        parent_fitness: Fitness of parent (or best parent for crossover)
        nn_dist: Nearest-neighbor distance (behavioral diversity) in [0, 1]
        alpha: Diversity weight from AlphaScheduler
        error: Whether the offspring had an error during evaluation
        clamp: Maximum absolute reward value for numerical stability

    Returns:
        Tuple of (composite_reward, fitness_delta, diversity_bonus)
    """
    # Heavy penalty for errors (failed code, crashes, budget violations)
    if error:
        return -clamp, 0.0, 0.0

    # Exploitation signal: fitness improvement
    fitness_delta = child_fitness - parent_fitness

    # Exploration signal: behavioral diversity bonus
    diversity_bonus = alpha * nn_dist

    # Composite reward
    raw_reward = fitness_delta + diversity_bonus

    # Clamp for numerical stability (prevents extreme outliers)
    reward = max(-clamp, min(clamp, raw_reward))

    # Small positive signal for ties (prevents bandit stagnation at zero)
    if abs(reward) < 1e-6:
        reward = 0.01

    return reward, fitness_delta, diversity_bonus
```

**Error Handling**:

- Syntax errors, runtime exceptions, budget violations → \( R = -1.0 \)
- Ensures D-TS learns to avoid operators that produce broken code

**Clamping**:

- Prevents extreme rewards from dominating bandit statistics
- Default: \( R \in [-1.0, 1.0] \)
- Normalized rewards are then \( R\_{\text{norm}} \in [-3, +3] \) (approx. 3σ range)

### 7.3 Alpha Scheduling

Alpha \( \alpha(t) \) decays from \( \alpha*{\text{start}} \) to \( \alpha*{\text{end}} \) over \( T \) generations.

**Supported Schedules**:

1. **Linear Decay** (default):
   \[
   \alpha(t) = \alpha*{\text{start}} + (\alpha*{\text{end}} - \alpha\_{\text{start}}) \times \frac{t}{T}
   \]

2. **Cosine Annealing** (smoother):
   \[
   \alpha(t) = \alpha*{\text{end}} + \frac{1}{2}(\alpha*{\text{start}} - \alpha\_{\text{end}}) \times \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
   \]

3. **Exponential Decay**:
   \[
   \alpha(t) = \max\left(\alpha*{\text{start}} \times e^{-3t/T}, \alpha*{\text{end}}\right)
   \]

4. **Constant** (ablation study):
   \[
   \alpha(t) = \alpha\_{\text{start}} \quad \forall t
   \]

**Implementation**:

```python
class AlphaScheduler:
    """Scheduler for the diversity weight parameter α."""

    def __init__(
        self,
        alpha_start: float = 0.5,
        alpha_end: float = 0.0,
        t_max: int = 100,
        schedule: str = 'linear'
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.t_max = max(1, t_max)  # Avoid division by zero
        self.schedule = schedule

    def get_alpha(self, t: int) -> float:
        """Get the diversity weight for generation t (0-indexed)."""
        if self.schedule == 'constant':
            return self.alpha_start

        # Clamp t to [0, t_max]
        t = max(0, min(t, self.t_max))
        progress = t / self.t_max

        if self.schedule == 'linear':
            alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress

        elif self.schedule == 'exponential':
            decay_rate = 3.0
            alpha = self.alpha_start * np.exp(-decay_rate * progress)
            alpha = max(alpha, self.alpha_end)

        elif self.schedule == 'cosine':
            alpha = self.alpha_end + 0.5 * (self.alpha_start - self.alpha_end) * (1 + np.cos(np.pi * progress))

        else:
            alpha = self.alpha_start

        return alpha
```

**Schedule Comparison** (α_start = 0.5, α_end = 0.0, T = 10):

| Generation | Linear | Cosine | Exponential |
| ---------- | ------ | ------ | ----------- |
| 0          | 0.500  | 0.500  | 0.500       |
| 1          | 0.450  | 0.476  | 0.371       |
| 2          | 0.400  | 0.405  | 0.275       |
| 3          | 0.350  | 0.309  | 0.204       |
| 4          | 0.300  | 0.206  | 0.151       |
| 5          | 0.250  | 0.125  | 0.112       |
| 6          | 0.200  | 0.069  | 0.083       |
| 7          | 0.150  | 0.034  | 0.062       |
| 8          | 0.100  | 0.012  | 0.046       |
| 9          | 0.050  | 0.002  | 0.034       |
| 10         | 0.000  | 0.000  | 0.025       |

**Visualization**:

```
α
│
0.5│╲
   │ ╲                    Linear (steady)
   │  ╲___               Cosine (slow start, fast end)
   │      ╲___╲          Exponential (fast start, slow end)
   │          ╲___╲╲___
0.0├──────────────────────────> Generation
   0   2   4   6   8  10
```

**Choosing a Schedule**:

- **Linear**: General-purpose, predictable
- **Cosine**: Maintain exploration longer, then rapidly shift to exploitation
- **Exponential**: Quick shift to exploitation (for fast-converging problems)
- **Constant**: Ablation study (no decay, constant exploration)

### 7.4 Reward Examples

**Example 1: Successful Improvement** (Early Generation)

```
Generation: 2
Alpha: 0.55 (linear schedule, α_start=0.7, α_end=0.1, T=5)

Child fitness:  0.52 (higher is better)
Parent fitness: 0.45
NN-Dist:        0.08 (moderately novel)

Calculation:
  fitness_delta   = 0.52 - 0.45 = +0.070
  diversity_bonus = 0.55 × 0.08 = +0.044

Raw Reward = +0.070 + 0.044 = +0.114

Interpretation:
  - Child improved fitness by 7%
  - Moderate diversity contribution (4.4%)
  - Positive reward → operator encouraged
```

**Example 2: Novel but Worse** (Early Generation)

```
Generation: 2
Alpha: 0.55

Child fitness:  0.42 (worse than parent)
Parent fitness: 0.45
NN-Dist:        0.35 (highly novel)

Calculation:
  fitness_delta   = 0.42 - 0.45 = -0.030
  diversity_bonus = 0.55 × 0.35 = +0.193

Raw Reward = -0.030 + 0.193 = +0.163

Interpretation:
  - Child lost 3% fitness
  - High diversity bonus compensates (19.3%)
  - Positive reward → exploration valued
  - This would be negative in late generations (α ≈ 0)
```

**Example 3: Successful Improvement** (Late Generation)

```
Generation: 9
Alpha: 0.10 (low diversity weight)

Child fitness:  0.68
Parent fitness: 0.65
NN-Dist:        0.05 (low novelty, refining existing strategy)

Calculation:
  fitness_delta   = 0.68 - 0.65 = +0.030
  diversity_bonus = 0.10 × 0.05 = +0.005

Raw Reward = +0.030 + 0.005 = +0.035

Interpretation:
  - Small fitness improvement (3%)
  - Diversity barely matters (α low)
  - Exploitation-focused reward
```

**Example 4: Error Case**

```
Child evaluation: Runtime error ("Budget 10000 exceeded at eval 10000")
NN-Dist: 1.0 (default for failed runs)

Calculation:
  error = True → reward = -1.0 (clamped penalty)

Interpretation:
  - Heavy penalty discourages operators that produce broken code
  - D-TS learns to avoid error-prone strategies
```

### 7.5 Reward Decomposition Tracking

The system tracks **component contributions** for analysis:

```python
# After computing reward, log decomposition
mada_operator.bandit.update_with_decomposition(
    arm_name=operator,
    normalized_reward=reward_normalized,
    fitness_component=fitness_delta,
    diversity_component=diversity_bonus
)
```

**Per-Arm Statistics**:

```
Operator: crossover (after 50 generations)
  Total pulls: 82
  Total reward: +15.3
  Avg reward: +0.187

  Fitness contribution: +12.8  (83.7% of reward)
  Diversity contribution: +2.5 (16.3% of reward)

Interpretation:
  - Crossover primarily rewards fitness improvement
  - Some diversity contribution (early generations)
```

This decomposition helps diagnose operator behavior:

- **High fitness, low diversity**: Exploitative operator
- **Low fitness, high diversity**: Exploratory operator
- **Balanced**: Multi-purpose operator

---

## 8. Reward Normalization

Raw composite rewards vary widely in scale due to:

1. Different fitness improvement magnitudes
2. Time-varying alpha schedules
3. Stochastic evaluation noise

**Problem**: D-TS assumes rewards follow a stable Gaussian distribution. Raw rewards violate this assumption.

**Solution**: Apply **running z-score normalization** to stabilize the reward distribution.

### 8.1 Z-Score Normalization

**Formula**:

\[
R*{\text{normalized}} = \frac{R*{\text{raw}} - \mu_R}{\sigma_R}
\]

Where:

- \( \mu_R \): Running mean of rewards
- \( \sigma_R \): Running standard deviation of rewards

**Properties**:

- \( \mathbb{E}[R_{\text{normalized}}] \approx 0 \)
- \( \text{Var}(R\_{\text{normalized}}) \approx 1 \)
- Outliers are compressed (|z-score| > 3 is rare)

### 8.2 Welford's Online Algorithm

Computing running statistics **without storing all rewards** using Welford's numerically stable algorithm.

**Algorithm**:

```python
class RewardNormalizer:
    """
    Running standardization for bandit rewards using Welford's algorithm.
    """

    def __init__(self, warmup_period: int = 5):
        """
        Args:
            warmup_period: Number of rewards to collect before normalizing.
                          During warmup, raw rewards are returned.
        """
        self.warmup_period = warmup_period
        self.n = 0                      # Number of rewards seen
        self.mean = 0.0                 # Running mean (μ)
        self.M2 = 0.0                   # Sum of squared differences (for variance)
        self.rewards_history: List[float] = []  # For logging/analysis

    def update(self, reward: float) -> None:
        """
        Update running statistics with new reward (Welford's algorithm).

        Welford's Algorithm (numerically stable):
          1. n = n + 1
          2. delta = reward - mean
          3. mean = mean + delta / n
          4. delta2 = reward - mean  (new delta after updating mean)
          5. M2 = M2 + delta * delta2

        Variance = M2 / (n - 1)  (sample variance)
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
            reward: Raw composite reward

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
            return 0.0  # Avoid division by zero (all rewards identical)

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
        """Reset all statistics (for new experiment)."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.rewards_history = []
```

**Mathematical Correctness**:

Welford's algorithm computes:

\[
\begin{align}
\mu*n &= \mu*{n-1} + \frac{r*n - \mu*{n-1}}{n} \\
M2*n &= M2*{n-1} + (r*n - \mu*{n-1})(r_n - \mu_n) \\
\sigma^2_n &= \frac{M2_n}{n-1}
\end{align}
\]

Where:

- \( \mu_n \): Mean after \( n \) observations
- \( M2_n \): Sum of squared differences from mean
- \( \sigma^2_n \): Sample variance

**Advantages over Naive Approach**:

| Approach          | Memory | Numerical Stability                             |
| ----------------- | ------ | ----------------------------------------------- |
| Naive (store all) | O(n)   | Poor (catastrophic cancellation)                |
| Welford's         | O(1)   | Excellent (avoids subtraction of large numbers) |

### 8.3 Warmup Period

**Why Warmup?**

With few samples (\( n < 5 \)), statistics are unstable:

- Mean: Large variance from true mean
- Std: Sample std is biased for small \( n \)

**Solution**: Return raw rewards during warmup period (default: 5 samples).

**Effect**:

```
Reward 1: R = +0.12  →  Normalized = +0.12 (warmup)
Reward 2: R = -0.05  →  Normalized = -0.05 (warmup)
Reward 3: R = +0.20  →  Normalized = +0.20 (warmup)
Reward 4: R = +0.08  →  Normalized = +0.08 (warmup)
Reward 5: R = -0.02  →  Normalized = -0.02 (warmup)

After warmup: μ = 0.066, σ = 0.101

Reward 6: R = +0.15  →  Normalized = (0.15 - 0.066) / 0.101 = +0.83
Reward 7: R = -0.10  →  Normalized = (-0.10 - 0.066) / 0.101 = -1.64
```

### 8.4 Complete Example

**Scenario**: First 10 rewards in a MADA run.

**Step-by-Step**:

| Step | Raw Reward | n   | Mean (μ) | M2    | Std (σ) | Normalized                           |
| ---- | ---------- | --- | -------- | ----- | ------- | ------------------------------------ |
| 1    | +0.100     | 1   | 0.100    | 0.000 | 0.000   | +0.100 (warmup)                      |
| 2    | -0.200     | 2   | -0.050   | 0.045 | 0.212   | -0.200 (warmup)                      |
| 3    | +0.300     | 3   | +0.067   | 0.153 | 0.277   | +0.300 (warmup)                      |
| 4    | +0.100     | 4   | +0.075   | 0.154 | 0.227   | +0.100 (warmup)                      |
| 5    | -0.100     | 5   | +0.040   | 0.228 | 0.239   | -0.100 (warmup)                      |
| 6    | +0.200     | 6   | +0.067   | 0.293 | 0.242   | **(+0.200 - 0.067) / 0.242 = +0.55** |
| 7    | +0.150     | 7   | +0.079   | 0.342 | 0.239   | **(+0.150 - 0.079) / 0.239 = +0.30** |
| 8    | -0.050     | 8   | +0.062   | 0.362 | 0.227   | **(-0.050 - 0.062) / 0.227 = -0.49** |
| 9    | +0.080     | 9   | +0.064   | 0.365 | 0.214   | **(+0.080 - 0.064) / 0.214 = +0.07** |
| 10   | +0.220     | 10  | +0.080   | 0.484 | 0.232   | **(+0.220 - 0.080) / 0.232 = +0.60** |

**Observations**:

1. **Warmup Phase** (steps 1-5):

   - Normalized = Raw (no transformation)
   - Building initial statistics

2. **Active Normalization** (steps 6+):

   - High rewards → positive z-scores
   - Low rewards → negative z-scores
   - Extreme outliers compressed

3. **Statistical Convergence**:
   - Mean stabilizes around +0.08
   - Std stabilizes around 0.23
   - Distribution becomes Gaussian(0, 1) in normalized space

### 8.5 Impact on D-TS

**Before Normalization** (raw rewards):

```
Operator: mutation
  Rewards: [+0.5, +0.02, +0.15, -0.3, +0.8]
  Range: [-0.3, +0.8] = 1.1
  μ̂ = 0.234, τ = 0.447

Operator: crossover
  Rewards: [-0.05, -0.01, +0.03, -0.02, +0.01]
  Range: [-0.05, +0.03] = 0.08
  μ̂ = -0.008, τ = 0.447

Problem: Mutation has high variance → artificially high τ
         Crossover has low variance → doesn't reflect true uncertainty
```

**After Normalization** (z-scores):

```
Global stats: μ_global = 0.113, σ_global = 0.286

Operator: mutation (normalized)
  Rewards: [+1.35, -0.33, +0.13, -1.44, +2.40]
  μ̂ = 0.422, τ = 0.447

Operator: crossover (normalized)
  Rewards: [-0.57, -0.43, -0.29, -0.46, -0.36]
  μ̂ = -0.422, τ = 0.447

Benefit: Both operators now on same scale
         τ reflects sample size, not reward variance
         Fair comparison for Thompson Sampling
```

### 8.6 Edge Cases

**Case 1: All Rewards Identical**

```
Rewards: [0.5, 0.5, 0.5, 0.5, 0.5]
  μ = 0.5, σ = 0.0

New reward: 0.5
  Normalized = (0.5 - 0.5) / 0.0 → Division by zero!

Solution: Return 0.0 if σ < 1e-10
```

**Case 2: Only Errors** (all rewards = -1.0)

```
Rewards: [-1.0, -1.0, -1.0, -1.0, -1.0]
  μ = -1.0, σ = 0.0

New reward: -1.0
  Normalized = 0.0 (handled by std < 1e-10 check)

Impact: All operators look equally bad → random selection
```

**Case 3: Extreme Outlier**

```
Rewards: [0.1, 0.12, 0.09, 0.11, 0.10, 0.13, 5.0]
           \_______stable range_______/    └─outlier

After 6 rewards: μ = 0.108, σ = 0.014
Outlier normalized: (5.0 - 0.108) / 0.014 = +349.4 (!)

After 7 rewards: μ = 0.807, σ = 1.83
  Future rewards normalized: ~[-0.5, +2.5] range

Solution: Clamping raw rewards prevents this (clamp=1.0)
```

---

## 9. Bandit Update Mechanism

After evaluating an offspring and computing its normalized reward, the D-TS bandit updates **all three operators** to refine their posteriors.

### 9.1 Update Flow

**High-Level Process**:

```python
# 1. Offspring generation & evaluation
child, selection_info = mada_operator.generate_offspring(parents)
operator = selection_info['operator']  # 'mutation', 'crossover', or 'refine'

aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
    child.code, child.name, eval_budget
)

# 2. Compute composite reward
reward, fitness_delta, diversity_bonus = compute_composite_reward(
    child_fitness=child.fitness,
    parent_fitness=parent_fitness,
    nn_dist=nn_dist,
    alpha=current_alpha,
    error=bool(error),
    clamp=args.reward_clamp
)

# 3. Normalize reward
normalized_reward = reward_normalizer.normalize(reward)

# 4. Update D-TS bandit (all arms)
mada_operator.bandit.update_with_decomposition(
    arm_name=operator,
    normalized_reward=normalized_reward,
    fitness_component=fitness_delta,
    diversity_component=diversity_bonus
)
```

### 9.2 Update Implementation

**Standard D-TS Update** (as defined in Section 3.4):

```python
def update(self, arm_name: str, reward: float) -> None:
    """
    Update posteriors with exponential discounting.

    This implements the core D-TS update from Section 3.4.
    """
    sigma = np.sqrt(self.reward_variance)  # σ = 1.0 for normalized rewards

    # Step 1: Apply discount to ALL arms (exponential forgetting)
    for stats in self.arm_stats.values():
        stats.N = self.discount * stats.N              # N ← γ · N
        stats.mu_tilde = self.discount * stats.mu_tilde # μ̃ ← γ · μ̃

        # Update uncertainty after discounting
        if stats.N > 0:
            stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
        else:
            stats.tau = self.tau_max

    # Step 2: Update selected arm with new observation
    if arm_name in self.arm_stats:
        stats = self.arm_stats[arm_name]
        stats.N += 1.0                       # N ← N + 1
        stats.mu_tilde += reward             # μ̃ ← μ̃ + r
        stats.pulls += 1                     # Increment pull count
        stats.total_reward += reward

        # Recompute posterior mean
        if stats.N > 0:
            stats.mu_hat = stats.mu_tilde / stats.N
        else:
            stats.mu_hat = self.prior_mean

        # Recompute posterior uncertainty
        if stats.N > 0:
            stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
        else:
            stats.tau = self.tau_max

    self.reward_history.append((arm_name, reward))
```

**Extended Update with Decomposition Tracking**:

```python
def update_with_decomposition(
    self,
    arm_name: str,
    normalized_reward: float,
    fitness_component: float,
    diversity_component: float
) -> None:
    """
    Update bandit with reward and track component contributions.

    This extends the standard D-TS update with MADA-specific analytics.

    Args:
        arm_name: Selected operator ('mutation', 'crossover', 'refine')
        normalized_reward: Z-score normalized composite reward
        fitness_component: Raw fitness improvement (Δfitness)
        diversity_component: Raw diversity bonus (α × NN-Dist)
    """
    # 1. Standard D-TS update (modifies all arms)
    self.update(arm_name, normalized_reward)

    # 2. Track component contributions (for analysis & diagnostics)
    if arm_name in self.arm_stats:
        self.arm_stats[arm_name].total_fitness_reward += fitness_component
        self.arm_stats[arm_name].total_diversity_reward += diversity_component
```

**Component Tracking Purpose**:

Decomposition tracking enables post-hoc analysis of operator behavior:

- Which operators exploit fitness vs. explore diversity?
- Does operator preference shift over time?
- Are reward components aligned with operator design?

### 9.3 Update Example

**Scenario**: Generation 5, crossover selected, positive reward.

**Before Update**:

```
Arm States (before offspring evaluation):
  mutation:  N = 8.2,  μ̃ = 1.23,  μ̂ = 0.15,  τ = 0.35,  pulls = 18
  crossover: N = 12.4, μ̃ = 3.47,  μ̂ = 0.28,  τ = 0.28,  pulls = 28
  refine:    N = 5.6,  μ̃ = 0.84,  μ̂ = 0.15,  τ = 0.42,  pulls = 15

Selection: crossover (θ sampled = 0.41)
```

**Offspring Evaluation**:

```
Child: HybridDE_v7
  Fitness: 0.67 (parent: 0.62)
  NN-Dist: 0.12
  Alpha: 0.30 (generation 5)

Composite Reward:
  Δfitness = 0.67 - 0.62 = +0.050
  Diversity = 0.30 × 0.12 = +0.036
  Raw reward = +0.050 + 0.036 = +0.086

Normalization (n=61, μ=0.042, σ=0.187):
  Normalized = (0.086 - 0.042) / 0.187 = +0.235
```

**Update Process**:

**Step 1: Discount ALL arms** (γ = 0.9):

```
mutation:  N = 0.9 × 8.2  = 7.38,  μ̃ = 0.9 × 1.23 = 1.107,  μ̂ = 1.107 / 7.38 = 0.15
crossover: N = 0.9 × 12.4 = 11.16, μ̃ = 0.9 × 3.47 = 3.123,  μ̂ = 3.123 / 11.16 = 0.28
refine:    N = 0.9 × 5.6  = 5.04,  μ̃ = 0.9 × 0.84 = 0.756,  μ̂ = 0.756 / 5.04 = 0.15
```

**Step 2: Update crossover** (selected arm):

```
crossover: N = 11.16 + 1.0 = 12.16
           μ̃ = 3.123 + 0.235 = 3.358
           μ̂ = 3.358 / 12.16 = 0.276
           pulls = 28 + 1 = 29
```

**Step 3: Recompute uncertainties** (τ = σ / √N):

```
mutation:  τ = 1.0 / √7.38  = 0.368
crossover: τ = 1.0 / √12.16 = 0.286  ← Decreased (more confident)
refine:    τ = 1.0 / √5.04  = 0.445
```

**Step 4: Track components** (crossover only):

```
crossover: total_fitness_reward  += 0.050  (now: 1.84)
           total_diversity_reward += 0.036  (now: 0.52)
```

**After Update**:

```
Arm States (after update):
  mutation:  N = 7.38,  μ̃ = 1.107, μ̂ = 0.15,  τ = 0.368, pulls = 18
  crossover: N = 12.16, μ̃ = 3.358, μ̂ = 0.276, τ = 0.286, pulls = 29  ← Updated
  refine:    N = 5.04,  μ̃ = 0.756, μ̂ = 0.15,  τ = 0.445, pulls = 15

Changes:
  - Crossover: μ̂ increased slightly (0.28 → 0.276, due to discounting effect)
  - Crossover: τ decreased (0.28 → 0.286, more confident but discounting also applies)
  - Mutation/Refine: μ̂ unchanged, τ slightly increased (relative uncertainty grows)
```

**Impact on Next Selection**:

```
Next offspring generation (Monte Carlo probabilities):
  P(mutation)  ≈ 15%  (low μ̂, moderate τ → occasional exploration)
  P(crossover) ≈ 62%  (high μ̂, low τ → strong preference)
  P(refine)    ≈ 23%  (moderate μ̂, high τ → exploratory sampling)
```

### 9.4 Cumulative Effects Over Generations

**Generation Progression Example** (simplified, 10 generations):

| Gen | Operator  | Reward | mutation (μ̂, τ) | crossover (μ̂, τ) | refine (μ̂, τ) | Selection Probs |
| --- | --------- | ------ | --------------- | ---------------- | ------------- | --------------- |
| 1   | mutation  | +0.12  | (0.12, 1.00)    | (0.00, 1.00)     | (0.00, 1.00)  | 35%, 32%, 33%   |
| 2   | crossover | +0.35  | (0.11, 0.94)    | (0.35, 0.71)     | (0.00, 0.94)  | 22%, 51%, 27%   |
| 3   | refine    | -0.15  | (0.10, 0.89)    | (0.31, 0.67)     | (-0.15, 0.82) | 24%, 52%, 24%   |
| 4   | crossover | +0.28  | (0.09, 0.84)    | (0.30, 0.58)     | (-0.14, 0.77) | 21%, 57%, 22%   |
| 5   | crossover | +0.24  | (0.08, 0.79)    | (0.29, 0.52)     | (-0.12, 0.73) | 19%, 60%, 21%   |
| ... | ...       | ...    | ...             | ...              | ...           | ...             |
| 10  | crossover | +0.18  | (0.05, 0.64)    | (0.25, 0.35)     | (-0.05, 0.58) | 14%, 68%, 18%   |

**Observations**:

1. **Crossover Dominance**: High positive rewards → increasing μ̂ → higher selection probability
2. **Refine Struggle**: Early negative reward → suppressed but not eliminated (exploration via τ)
3. **Mutation Stability**: Moderate rewards → stable niche role
4. **Uncertainty Decay**: All τ decrease over time (more data → more confidence)
5. **Non-Stationarity**: Discounting allows re-exploration if conditions change

### 9.5 Snapshot Logging

After each generation, the system logs a complete bandit snapshot for post-hoc analysis:

```python
def log_bandit_snapshot(explogger, generation, bandit_state):
    """Log bandit state snapshot to bandit_snapshots.jsonl."""
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "bandit_snapshots.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"generation": generation, **bandit_state}, default=str) + "\n")
```

**Snapshot Content**:

```json
{
  "generation": 5,
  "arm_stats": {
    "mutation": {
      "mu_hat": 0.15,
      "tau": 0.368,
      "N": 7.38,
      "pulls": 18,
      "total_reward": 2.7,
      "avg_reward": 0.15,
      "total_fitness_reward": 2.42,
      "total_diversity_reward": 0.28
    },
    "crossover": {
      /* ... */
    },
    "refine": {
      /* ... */
    }
  },
  "selection_probs": {
    "mutation": 0.15,
    "crossover": 0.62,
    "refine": 0.23
  },
  "total_rounds": 61,
  "alpha": 0.3,
  "reward_stats": {
    "n": 61,
    "mean": 0.042,
    "std": 0.187,
    "variance": 0.035
  },
  "history_size": 61
}
```

**Analysis Uses**:

- Plot μ̂ evolution over time
- Visualize selection probability dynamics
- Diagnose operator balance issues
- Correlate alpha decay with operator preferences

---

## 10. Population Management and Selection

MADA-LLAMEA v2 uses **(μ+λ) evolutionary selection** with **diversity preservation** to maintain a balance between fitness and code diversity.

### 10.1 Selection Strategies

**Two Modes**:

1. **(μ+λ) Selection** (elitism=True, default):

   - Combine parents and offspring: \( P*{\text{combined}} = P*{\text{parents}} \cup P\_{\text{offspring}} \)
   - Select best μ individuals
   - Guarantees non-decreasing best fitness
   - Slower population turnover

2. **(μ,λ) Selection** (elitism=False):
   - Select only from offspring: \( P\_{\text{new}} = \text{best}(\lambda \text{ offspring}) \)
   - Parents are discarded each generation
   - Faster exploration but risky (can lose best solution)

**MADA-LLAMEA v2 Default**: (μ+λ) with μ=4, λ=16.

### 10.2 Selection Implementation

**Algorithm**: Fitness-first with code-based diversity preservation.

```python
def selection(population, n_parents, elitism=True):
    """
    Select best individuals from population with diversity preservation.

    Strategy:
      1. Always select the best individual (highest fitness)
      2. Greedily add diverse individuals (different code)
      3. Fill remaining slots with next-best fitness

    Args:
        population: List of MADASolution objects
        n_parents: Number of individuals to select (μ)
        elitism: If True, preserves best fitness (always includes global best)

    Returns:
        List of selected individuals (length = n_parents)
    """
    # Sort by fitness (descending: higher is better)
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)

    if not sorted_pop:
        return []

    # Step 1: Always include the best individual (elitism)
    selected = [sorted_pop[0]]
    if n_parents == 1:
        return selected

    best_code = sorted_pop[0].code or ""

    # Step 2: Add one diverse individual (different code from best)
    for ind in sorted_pop[1:]:
        if (ind.code or "") != best_code:
            selected.append(ind)
            break

    # Ensure at least 2 individuals if population allows
    if len(selected) < 2 and len(sorted_pop) > 1:
        selected.append(sorted_pop[1])

    # Step 3: Add more diverse individuals (different codes)
    seen_codes = {s.code or "" for s in selected}
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        if (ind.code or "") not in seen_codes:
            seen_codes.add(ind.code or "")
            selected.append(ind)

    # Step 4: Fill remaining slots with next-best fitness (even if duplicate code)
    for ind in sorted_pop:
        if len(selected) >= n_parents:
            break
        if ind not in selected:
            selected.append(ind)

    return selected
```

**Selection Priorities**:

1. **Fitness** (primary): Sort population by fitness first
2. **Code Diversity** (secondary): Prefer individuals with different code
3. **Completeness** (tertiary): Fill to exactly μ individuals

### 10.3 Diversity Preservation Rationale

**Problem**: Pure fitness-based selection leads to:

- Premature convergence (all individuals become nearly identical)
- Loss of exploration potential
- Redundant LLM queries (crossover of identical parents)

**Solution**: Code-based diversity filter.

**Example** (μ=4, population after generation 5):

```
Combined Population (parents + offspring = 20 individuals):

Rank | Name                  | Fitness | Code Hash | Selected?
-----|-----------------------|---------|-----------|----------
1    | AdaptiveDE_v7         | 0.68    | abc123    | ✓ (best)
2    | AdaptiveDE_v7         | 0.68    | abc123    | ✗ (duplicate)
3    | HybridPSO_v3          | 0.65    | def456    | ✓ (diverse)
4    | EnhancedDE_v2         | 0.62    | ghi789    | ✓ (diverse)
5    | AdaptiveDE_v6         | 0.61    | abc122    | ✓ (diverse enough)
6    | MutatedDE_v1          | 0.58    | jkl012    | ✗ (μ=4 reached)
...

Selected Population (next generation parents):
  1. AdaptiveDE_v7  (0.68)
  2. HybridPSO_v3   (0.65)
  3. EnhancedDE_v2  (0.62)
  4. AdaptiveDE_v6  (0.61)
```

**Code Hash**: Uses SHA-256 hash of algorithm code for exact duplicate detection.

### 10.4 Selection in Evolutionary Context

**Generation Flow**:

```python
for gen in range(1, generations + 1):
    # 1. Start with current parents
    parents = population  # Size: μ

    # 2. Generate offspring using MADA operators
    offspring = []
    for i in range(n_offspring):
        child, info = mada_operator.generate_offspring(parents)
        # Evaluate child...
        offspring.append(child)

    # 3. Selection (depends on elitism)
    if args.elitism:
        # (μ+λ): Combine and select best μ
        combined = parents + offspring  # Size: μ + λ
        population = selection(combined, n_parents, elitism=True)
    else:
        # (μ,λ): Select only from offspring
        population = selection(offspring, n_parents, elitism=False)

    # 4. Next generation uses selected population as parents
```

**Population Dynamics** (μ=4, λ=8, elitism=True):

| Generation | Parents (In)   | Offspring | Combined | Selected (Out) |
| ---------- | -------------- | --------- | -------- | -------------- |
| 1          | 4 (init)       | 8         | 12       | 4 (best of 12) |
| 2          | 4 (from gen 1) | 8         | 12       | 4 (best of 12) |
| 3          | 4 (from gen 2) | 8         | 12       | 4 (best of 12) |
| ...        | ...            | ...       | ...      | ...            |

**Without Elitism** (μ=4, λ=8, elitism=False):

| Generation | Parents (In)   | Offspring | Selected (Out) |
| ---------- | -------------- | --------- | -------------- |
| 1          | 4 (init)       | 8         | 4 (best of 8)  |
| 2          | 4 (from gen 1) | 8         | 4 (best of 8)  |
| 3          | 4 (from gen 2) | 8         | 4 (best of 8)  |
| ...        | ...            | ...       | ...            |

### 10.5 Diversity Preservation vs. Behavioral Selection

MADA-LLAMEA v2 uses **two levels of diversity**:

1. **Population Diversity** (this section):

   - Ensures multiple different algorithms in population
   - Based on **code hash** (exact duplicate detection)
   - Operates during **selection** (survival)

2. **Behavioral Parent Selection** (Section 4.4):
   - Selects diverse parents for **crossover**
   - Based on **trace distance** (behavioral diversity)
   - Operates during **offspring generation**

**Interaction**:

```
Generation N:
  Population: [Alg_A, Alg_B, Alg_C, Alg_D]  ← Code-diverse (from selection)
                 ↓
  Crossover Parent Selection:
    Parent A: Alg_A (best fitness)
    Parent B: Alg_C (most behaviorally different from A, even if code is different from A already)
                 ↓
  Offspring: Hybrid_AC
                 ↓
Generation N+1:
  Population: [Alg_A, Hybrid_AC, Alg_B, Alg_D]  ← Code-diverse (selection removed low-fitness Alg_C)
```

### 10.6 Selection Example

**Scenario**: Generation 5, μ=4, λ=8, elitism=True.

**Parents** (start of generation):

```
1. AdaptiveDE_v5  (fitness: 0.65, code_hash: aaa111)
2. HybridPSO_v2   (fitness: 0.62, code_hash: bbb222)
3. EnhancedDE_v1  (fitness: 0.58, code_hash: ccc333)
4. MutatedCMA_v3  (fitness: 0.55, code_hash: ddd444)
```

**Offspring** (after evaluation):

```
5. AdaptiveDE_v6     (fitness: 0.68, code_hash: aaa112, operator: refine)
6. HybridPSO_v3      (fitness: 0.63, code_hash: bbb223, operator: refine)
7. AdaptiveDE_v5     (fitness: 0.65, code_hash: aaa111, operator: mutation) ← DUPLICATE
8. FailedAlgorithm   (fitness: 0.00, code_hash: eee555, operator: crossover, error: True)
9. NovelDE_v1        (fitness: 0.52, code_hash: fff666, operator: mutation)
10. EnhancedDE_v2    (fitness: 0.60, code_hash: ccc334, operator: crossover)
11. AdaptiveDE_v7    (fitness: 0.70, code_hash: aaa113, operator: refine) ← NEW BEST
12. HybridCMA_v1     (fitness: 0.54, code_hash: ggg777, operator: crossover)
```

**Combined** (sorted by fitness):

```
Rank | Name             | Fitness | Code Hash | Operator  | Origin
-----|------------------|---------|-----------|-----------|--------
1    | AdaptiveDE_v7    | 0.70    | aaa113    | refine    | Offspring
2    | AdaptiveDE_v6    | 0.68    | aaa112    | refine    | Offspring
3    | AdaptiveDE_v5    | 0.65    | aaa111    | -         | Parent
4    | AdaptiveDE_v5    | 0.65    | aaa111    | mutation  | Offspring (DUPLICATE)
5    | HybridPSO_v3     | 0.63    | bbb223    | refine    | Offspring
6    | HybridPSO_v2     | 0.62    | bbb222    | -         | Parent
7    | EnhancedDE_v2    | 0.60    | ccc334    | crossover | Offspring
8    | EnhancedDE_v1    | 0.58    | ccc333    | -         | Parent
9    | MutatedCMA_v3    | 0.55    | ddd444    | -         | Parent
10   | HybridCMA_v1     | 0.54    | ggg777    | crossover | Offspring
11   | NovelDE_v1       | 0.52    | fff666    | mutation  | Offspring
12   | FailedAlgorithm  | 0.00    | eee555    | crossover | Offspring (ERROR)
```

**Selection Process** (μ=4):

```
Step 1: Select best (rank 1)
  → AdaptiveDE_v7 (0.70, aaa113)

Step 2: Select first diverse individual (different code from best)
  → HybridPSO_v3 (0.63, bbb223)  [rank 5, different from aaa113]

Step 3: Add more diverse individuals
  → EnhancedDE_v2 (0.60, ccc334)  [rank 7, different from {aaa113, bbb223}]
  → MutatedCMA_v3 (0.55, ddd444)  [rank 9, different from {aaa113, bbb223, ccc334}]

Selected (μ=4 reached):
  1. AdaptiveDE_v7  (0.70)
  2. HybridPSO_v3   (0.63)
  3. EnhancedDE_v2  (0.60)
  4. MutatedCMA_v3  (0.55)
```

**Observations**:

- **Best solution promoted**: AdaptiveDE_v7 (offspring via refine)
- **Duplicates eliminated**: AdaptiveDE_v5 (rank 4, duplicate of rank 3)
- **Diversity valued**: MutatedCMA_v3 (rank 9) selected over higher-fitness AdaptiveDE_v6 (rank 2) to maintain code diversity
- **Errors naturally eliminated**: FailedAlgorithm (rank 12) has fitness 0.00, never selected
- **Old parents can survive**: MutatedCMA_v3 from parents survives due to diversity

### 10.7 Selection Pressure Analysis

**With Elitism** (μ+λ):

- **Selection Pressure**: \( \frac{\mu}{\mu + \lambda} = \frac{4}{12} = 33\% \) survival rate
- **Pros**: Best solution guaranteed to survive, stable convergence
- **Cons**: Slower exploration, harder for new individuals to replace old ones

**Without Elitism** (μ,λ):

- **Selection Pressure**: \( \frac{\mu}{\lambda} = \frac{4}{8} = 50\% \) survival rate (of offspring only)
- **Pros**: Faster turnover, more exploration
- **Cons**: Best solution can be lost, unstable fitness

**MADA-LLAMEA v2 Recommendation**: Use elitism for stable, risk-averse search (default).

---

## 11. Complete Example Walkthrough

This section demonstrates a complete offspring generation cycle in Generation 5 of a MADA-LLAMEA v2 run.

### 11.1 Initial State (Generation 5)

**Configuration**:

```
Hyperparameters:
  μ = 4 (parents)
  λ = 8 (offspring per generation)
  α_start = 0.7, α_end = 0.1, schedule = linear
  γ = 0.9 (discount factor)
  τ_max = 1.0

Run Statistics:
  API calls so far: 36
  Generations completed: 4
  History traces: 32
  Best fitness: 0.62 (EnhancedAdaptiveDE)
```

**Population** (4 parents):

```
1. EnhancedAdaptiveDE  (fitness: 0.62, generation: 3, operator: refine)
2. HybridPSO_v2        (fitness: 0.58, generation: 2, operator: crossover)
3. AdaptiveDE_v4       (fitness: 0.55, generation: 1, operator: mutation)
4. MutatedCMAES        (fitness: 0.51, generation: 2, operator: mutation)
```

**D-TS Bandit State**:

```
Operator        μ̂      τ     N    Pulls  Avg Reward  Fitness Contrib  Div Contrib
---------------------------------------------------------------------------------
mutation     +0.12  0.42  5.6    14      +0.12       +1.68           +0.21
crossover    +0.28  0.35  8.2    18      +0.28       +4.28           +0.76
refine       +0.15  0.51  3.8     9      +0.15       +1.35           +0.27
```

**Alpha Schedule**:

```
Generation 5 (t=4, t_max=9):
  progress = 4/9 = 0.444
  α(4) = 0.7 + (0.1 - 0.7) × 0.444 = 0.433
```

**Reward Normalizer**:

```
n = 36, μ = 0.042, σ = 0.187
```

### 11.2 Step-by-Step Execution

**Offspring 3/8 (API Call 39)**

---

**STEP 1: Operator Selection via D-TS**

```
Thompson Sampling (sample from posteriors):

mutation:  μ̂ = 0.12, τ = 0.42
  → θ ~ N(0.12, 0.42) = np.random.normal(0.12, 0.42) = 0.28

crossover: μ̂ = 0.28, τ = 0.35
  → θ ~ N(0.28, 0.35) = np.random.normal(0.28, 0.35) = 0.51  ← MAX

refine:    μ̂ = 0.15, τ = 0.51
  → θ ~ N(0.15, 0.51) = np.random.normal(0.15, 0.51) = 0.09

Selected Operator: crossover (θ = 0.51)
```

**Console Output**:

```
Offspring 3/8 (API call 39)
  D-TS: crossover (θ=0.510)
```

---

**STEP 2: Crossover Parent Selection**

```
Behavioral Parent Selection (enabled):

Sorted population by fitness (descending):
  1. EnhancedAdaptiveDE (0.62)
  2. HybridPSO_v2 (0.58)
  3. AdaptiveDE_v4 (0.55)
  4. MutatedCMAES (0.51)

Parent A: EnhancedAdaptiveDE (best fitness)

Trace distances from Parent A:
  d(EnhancedAdaptiveDE, HybridPSO_v2)  = 0.031  (similar behavior)
  d(EnhancedAdaptiveDE, AdaptiveDE_v4) = 0.124  (different behavior) ← MAX
  d(EnhancedAdaptiveDE, MutatedCMAES)  = 0.088  (moderate difference)

Parent B: AdaptiveDE_v4 (maximum trace distance = 0.124)
```

---

**STEP 3: LLM Prompt Generation**

````
Prompt (EOH_CROSSOVER_IMPLICIT_PROMPT):

"Two parent solutions (BBOB, 5D, bounds [-5,5]):
Parent A (score 0.62):
```python
class EnhancedAdaptiveDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.F = 0.5
        self.CR = 0.9
        # ... (full code)
````

Parent B (score 0.55):

```python
class AdaptiveDE_v4:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        # ... (full code)
```

Create a better offspring by combining patterns from both. Same interface: **init**(self, budget), **call**(self, func).
Reply only with:

# Name: <classname>

# Code: <code>

"

```

**LLM Call**:

```

Model: gemini-2.0-flash
Temperature: 0.8
API Call ID: 39

```

---

**STEP 4: LLM Response & Code Extraction**

```

LLM Response:

"# Name: SynergisticAdaptiveDE

# Code:

```python
class SynergisticAdaptiveDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = 50

    def __call__(self, func):
        # Combined strategy from both parents:
        # - Parent A's adaptive parameter control
        # - Parent B's population diversity mechanism
        # ... (implementation)
```

"

Extracted:
class_name = "SynergisticAdaptiveDE"
code = <extracted from code block>

```

**Console Output**:

```

Evaluating SynergisticAdaptiveDE...

```

---

**STEP 5: BBOB Evaluation with Trace Collection**

```

Evaluation Loop:
24 functions × 3 instances × 3 repetitions = 216 runs
Budget per run: 10,000 evaluations

Selected Run Example (f1, instance 1, rep 0):
Problem: Sphere function (f1)
Trace: [532.1, 532.1, 412.8, 398.2, ..., 12.5] (length 10,000)

AUC Calculation:
216 AUC values: [0.58, 0.62, 0.55, ..., 0.49]
Mean AUC: 0.57

Detailed AUCs (per function group):
Separable (f1-f5): 0.68
Low/moderate conditioning (f6-f9): 0.59
High conditioning (f10-f14): 0.52
Multi-modal adequate (f15-f19): 0.48
Multi-modal weak (f20-f24): 0.45

Trace Aggregation:
216 traces → element-wise median → aggregated_trace (length 10,000)

```

**Console Output**:

```

Fitness: 0.5700

```

---

**STEP 6: Behavioral Diversity (NN-Dist)**

```

Global Normalization:
global_min = 0.0 (best fitness ever seen)
global_max = 1250.0 (worst fitness ever seen)

Normalize New Trace:
new_trace_normalized = (aggregated_trace - 0.0) / 1250.0

Compute Distances to History (32 traces):
d(new, hist_0) = 0.082
d(new, hist_1) = 0.095
d(new, hist_2) = 0.067 ← MIN
...
d(new, hist_31) = 0.143

NN-Dist = min(distances) = 0.067

```

**Console Output**:

```

NN-Dist: 0.0670

```

---

**STEP 7: Composite Reward Calculation**

```

Parent Fitness:
parent_a_fitness = 0.62 (EnhancedAdaptiveDE)
parent_b_fitness = 0.55 (AdaptiveDE_v4)
parent_fitness_for_reward = max(0.62, 0.55) = 0.62

Child Fitness:
child_fitness = 0.57

Fitness Delta:
Δfitness = 0.57 - 0.62 = -0.050 (worse than best parent)

Diversity Bonus:
α(gen 5) = 0.433
diversity_bonus = 0.433 × 0.067 = +0.029

Composite Reward:
raw_reward = -0.050 + 0.029 = -0.021

Clamping (clamp=1.0):
reward = max(-1.0, min(1.0, -0.021)) = -0.021

```

**Console Output**:

```

Reward: raw=-0.0210, norm=?
(Δfit=-0.0500, div=+0.0290)

```

---

**STEP 8: Reward Normalization**

```

Reward Normalizer State (before):
n = 36, μ = 0.042, σ = 0.187

Update with New Reward (Welford's):
n = 37
delta = -0.021 - 0.042 = -0.063
μ = 0.042 + (-0.063 / 37) = 0.040
delta2 = -0.021 - 0.040 = -0.061
M2 = M2_old + (-0.063) × (-0.061) = M2_old + 0.00384
σ = sqrt(M2 / 36) = 0.188

Z-Score Normalization:
normalized_reward = (-0.021 - 0.040) / 0.188 = -0.324

```

**Console Output**:

```

Reward: raw=-0.0210, norm=-0.3240
(Δfit=-0.0500, div=+0.0290)

```

---

**STEP 9: D-TS Bandit Update**

```

Before Update:
mutation: N = 5.6, μ̃ = 0.672, μ̂ = 0.12
crossover: N = 8.2, μ̃ = 2.296, μ̂ = 0.28
refine: N = 3.8, μ̃ = 0.570, μ̂ = 0.15

Step 1: Apply Discount (γ = 0.9) to ALL arms:
mutation: N = 0.9 × 5.6 = 5.04, μ̃ = 0.9 × 0.672 = 0.605
crossover: N = 0.9 × 8.2 = 7.38, μ̃ = 0.9 × 2.296 = 2.066
refine: N = 0.9 × 3.8 = 3.42, μ̃ = 0.9 × 0.570 = 0.513

Step 2: Update Crossover (selected arm, reward = -0.324):
N = 7.38 + 1.0 = 8.38
μ̃ = 2.066 + (-0.324) = 1.742
μ̂ = 1.742 / 8.38 = 0.208 (decreased from 0.28!)
pulls = 18 + 1 = 19

Step 3: Recompute Uncertainties (τ = 1/√N):
mutation: τ = 1/√5.04 = 0.445
crossover: τ = 1/√8.38 = 0.345
refine: τ = 1/√3.42 = 0.541

Step 4: Track Components (crossover only):
total_fitness_reward += -0.050 (now: 4.23)
total_diversity_reward += +0.029 (now: 0.79)

After Update:
mutation: N = 5.04, μ̃ = 0.605, μ̂ = 0.12, τ = 0.445
crossover: N = 8.38, μ̃ = 1.742, μ̂ = 0.208, τ = 0.345 ← Updated
refine: N = 3.42, μ̃ = 0.513, μ̂ = 0.15, τ = 0.541

```

**Impact**: Crossover received negative reward → μ̂ decreased (0.28 → 0.208), but still leading operator.

---

**STEP 10: State Update**

```

Add Trace to History:
history_traces.append(aggregated_trace)
history_size = 33

Update Global Bounds:
global_min = 0.0 (unchanged)
global_max = 1250.0 (unchanged)

Best Ever:
Still EnhancedAdaptiveDE (0.62) (offspring 0.57 did not improve)

````

---

**STEP 11: Logging**

**Offspring Record** (`mada_offspring.jsonl`):

```json
{
  "attempt": 39,
  "operator": "crossover",
  "fitness": 0.5700,
  "parent_fitness": 0.62,
  "nn_dist": 0.0670,
  "alpha": 0.433,
  "raw_reward": -0.0210,
  "normalized_reward": -0.3240,
  "fitness_delta": -0.0500,
  "diversity_bonus": 0.0290,
  "theta_sampled": 0.510,
  "parent_ids": ["mada_000012", "mada_000008"],
  "generation": 5,
  "error": ""
}
````

**Code File** (`code/try-39-SynergisticAdaptiveDE.py`):

```python
class SynergisticAdaptiveDE:
    # ... (full code saved)
```

**AUC File** (`try-39-aucs.txt`):

```
# len=216 mean=0.5700 std=0.048
0.58
0.62
0.55
...
```

---

### 11.3 Summary

**Key Observations**:

1. **Operator Selection**: Crossover won D-TS sampling despite negative trend (exploration via uncertainty τ)
2. **Parent Selection**: Behavioral diversity (trace distance) selected diverse parents (0.124 apart)
3. **Fitness**: Offspring worse than best parent (-5%) but still reasonable (0.57)
4. **Diversity**: Moderate novelty (NN-Dist = 0.067), contributing +0.029 to reward
5. **Net Reward**: Negative (-0.021 raw, -0.324 normalized) due to fitness loss
6. **Bandit Impact**: Crossover posterior μ̂ decreased (0.28 → 0.208), reducing future selection probability
7. **Alpha Effect**: α=0.433 partially compensated for fitness loss (without diversity, reward would be -0.050)

**Interpretation**:

- Crossover attempted to combine two diverse parents
- Result was behaviorally novel but lower fitness
- D-TS learned from failure: crossover less attractive in future
- Diversity bonus (α=0.433) softened the penalty, allowing continued exploration

---

## 12. Hyperparameters and Configuration

### 12.1 Command-Line Interface

```bash
python main-thesis-mada-v2.py \
  --evolutionary-mode \
  --elitism \
  --budget 100 \
  --n-parents 4 \
  --n-offspring 16 \
  --alpha-start 0.5 \
  --alpha-end 0.0 \
  --alpha-schedule linear \
  --discount 0.9 \
  --tau-max 1.0 \
  --behavioral-selection \
  --model gemini-2.0-flash \
  --experiment-name mada-v2-experiment
```

### 12.2 Hyperparameter Reference

**Core Evolutionary Parameters**:

| Parameter       | Default | Range          | Description                               |
| --------------- | ------- | -------------- | ----------------------------------------- |
| `--budget`      | 100     | [10, 500]      | Total API calls (LLM queries)             |
| `--eval-budget` | 10000   | [1000, 100000] | Function evaluations per algorithm (BBOB) |
| `--n-parents`   | 4       | [2, 20]        | Population size (μ)                       |
| `--n-offspring` | 16      | [4, 50]        | Offspring per generation (λ)              |
| `--elitism`     | False   | {True, False}  | Enable (μ+λ) selection                    |
| `--generations` | None    | [1, 100]       | Fixed generations (auto if None)          |

**D-TS Bandit Parameters**:

| Parameter           | Default | Range      | Description                               |
| ------------------- | ------- | ---------- | ----------------------------------------- |
| `--discount`        | 0.9     | [0.5, 1.0] | Forgetting factor γ (1.0 = no forgetting) |
| `--tau-max`         | 1.0     | [0.5, 5.0] | Maximum posterior uncertainty             |
| `--reward-variance` | 1.0     | [0.5, 2.0] | Expected variance of normalized rewards   |
| `--reward-clamp`    | 1.0     | [0.5, 5.0] | Clamp raw rewards to [-clamp, +clamp]     |

**MADA Diversity Parameters**:

| Parameter          | Default | Range                                   | Description                |
| ------------------ | ------- | --------------------------------------- | -------------------------- |
| `--alpha-start`    | 0.5     | [0.0, 1.0]                              | Initial diversity weight α |
| `--alpha-end`      | 0.0     | [0.0, 1.0]                              | Final diversity weight α   |
| `--alpha-schedule` | linear  | {linear, cosine, exponential, constant} | Decay schedule             |

**Behavioral Selection**:

| Parameter                   | Default | Range         | Description                                      |
| --------------------------- | ------- | ------------- | ------------------------------------------------ |
| `--behavioral-selection`    | True    | {True, False} | Use trace-based parent selection for crossover   |
| `--no-behavioral-selection` | -       | -             | Disable behavioral selection (use fitness-based) |

**Operator Control**:

| Parameter             | Default | Range         | Description                               |
| --------------------- | ------- | ------------- | ----------------------------------------- |
| `--disable-mutation`  | False   | {True, False} | Disable mutation operator (2-arm bandit)  |
| `--disable-crossover` | False   | {True, False} | Disable crossover operator (2-arm bandit) |

**Model Configuration**:

| Parameter           | Default            | Range        | Description                                        |
| ------------------- | ------------------ | ------------ | -------------------------------------------------- |
| `--model`           | gemini-2.0-flash   | string       | LLM model identifier                               |
| `--base-url`        | None               | URL          | Custom API base URL (for OpenAI-compatible APIs)   |
| `--api-key`         | None               | string       | API key (or use OPENAI_API_KEY / AIML_KEY env var) |
| `--max-tokens`      | None               | [1000, 8000] | Maximum tokens per LLM response                    |
| `--experiment-name` | mada-v2-experiment | string       | Experiment suffix for logging directory            |

### 12.3 Recommended Configurations

**Fast Exploration** (limited budget):

```bash
--budget 50 \
--n-parents 4 \
--n-offspring 8 \
--alpha-start 0.7 \
--alpha-end 0.1 \
--alpha-schedule linear \
--discount 0.85 \
--elitism
```

**Deep Exploitation** (large budget):

```bash
--budget 200 \
--n-parents 8 \
--n-offspring 16 \
--alpha-start 0.5 \
--alpha-end 0.0 \
--alpha-schedule cosine \
--discount 0.9 \
--elitism
```

**Ablation: No Diversity**:

```bash
--alpha-start 0.0 \
--alpha-end 0.0 \
--alpha-schedule constant \
--no-behavioral-selection
```

**Ablation: Only Crossover**:

```bash
--disable-mutation \
--enable-crossover \
--behavioral-selection
```

### 12.4 Hyperparameter Sensitivity

**Critical Parameters** (high impact on performance):

1. **Alpha Start/End**: Controls exploration-exploitation balance

   - High α → More exploration, slower convergence
   - Low α → More exploitation, risk of premature convergence

2. **Discount (γ)**: Controls bandit adaptability

   - γ = 1.0: No forgetting, slow adaptation
   - γ = 0.8: Fast forgetting, rapid adaptation (but noisy)
   - **Recommended**: 0.9 (balanced)

3. **Behavioral Selection**: Major impact on crossover quality
   - Enabled: Diverse parents → novel offspring
   - Disabled: Fitness-based parents → exploitative offspring

**Robust Parameters** (low impact):

1. **Tau Max**: Usually 1.0 works well (exploration maintained by discounting)
2. **Reward Variance**: Should match normalized reward variance (~1.0)
3. **Reward Clamp**: Numerical stability, rarely hit with BBOB

---

## 13. Output Files and Logging

### 13.1 Directory Structure

```
exp-12-12_112343-gemini-2.0-flash-mada-v2-experiment-evolutionary-elitism/
├── code/
│   ├── try-1-RandomSearch.py
│   ├── try-2-AdaptiveDE.py
│   ├── try-3-HybridPSO.py
│   └── ... (all 100 generated algorithms)
│
├── ioh/
│   └── (IOH logger data, if enabled)
│
├── mada_offspring.jsonl         # Per-offspring metadata
├── bandit_snapshots.jsonl       # Per-generation bandit states
├── BEST_ALGORITHM.py            # Best algorithm with metadata
├── conversationlog.txt          # Full LLM conversation history
├── try-1-aucs.txt               # AUC values for algorithm 1
├── try-2-aucs.txt               # AUC values for algorithm 2
└── ... (AUC files for all algorithms)
```

### 13.2 Key Output Files

**`mada_offspring.jsonl`** (JSON Lines format):

```json
{"attempt": 1, "operator": "init", "fitness": 0.45, "parent_fitness": 0.0, "nn_dist": 1.0, "alpha": 0.7, "raw_reward": 0.0, "normalized_reward": 0.0, "fitness_delta": 0.0, "diversity_bonus": 0.0, "theta_sampled": 0.0, "parent_ids": [], "generation": 0, "error": ""}
{"attempt": 5, "operator": "mutation", "fitness": 0.52, "parent_fitness": 0.48, "nn_dist": 0.12, "alpha": 0.67, "raw_reward": 0.12, "normalized_reward": 0.64, "fitness_delta": 0.04, "diversity_bonus": 0.08, "theta_sampled": 0.32, "parent_ids": ["mada_000003"], "generation": 1, "error": ""}
{"attempt": 12, "operator": "crossover", "fitness": 0.58, "parent_fitness": 0.52, "nn_dist": 0.09, "alpha": 0.58, "raw_reward": 0.11, "normalized_reward": 0.51, "fitness_delta": 0.06, "diversity_bonus": 0.05, "theta_sampled": 0.41, "parent_ids": ["mada_000005", "mada_000007"], "generation": 2, "error": ""}
...
```

**Fields**:

- `attempt`: API call number (1 to budget)
- `operator`: Operator used (init, mutation, crossover, refine)
- `fitness`: Child fitness (mean AOCC)
- `parent_fitness`: Best parent fitness
- `nn_dist`: Nearest-neighbor distance (behavioral diversity)
- `alpha`: Current diversity weight
- `raw_reward`: Raw composite reward
- `normalized_reward`: Z-score normalized reward
- `fitness_delta`: Exploitation component
- `diversity_bonus`: Exploration component
- `theta_sampled`: D-TS sampled value
- `parent_ids`: Parent solution IDs
- `generation`: Generation number
- `error`: Error message (empty if successful)

---

**`bandit_snapshots.jsonl`** (JSON Lines format):

```json
{
  "generation": 5,
  "arm_stats": {
    "mutation": {"mu_hat": 0.12, "tau": 0.42, "N": 5.6, "pulls": 14, "total_reward": 1.68, "avg_reward": 0.12, "total_fitness_reward": 1.42, "total_diversity_reward": 0.26},
    "crossover": {"mu_hat": 0.28, "tau": 0.35, "N": 8.2, "pulls": 18, "total_reward": 5.04, "avg_reward": 0.28, "total_fitness_reward": 4.28, "total_diversity_reward": 0.76},
    "refine": {"mu_hat": 0.15, "tau": 0.51, "N": 3.8, "pulls": 9, "total_reward": 1.35, "avg_reward": 0.15, "total_fitness_reward": 1.08, "total_diversity_reward": 0.27}
  },
  "selection_probs": {"mutation": 0.18, "crossover": 0.59, "refine": 0.23},
  "total_rounds": 41,
  "alpha": 0.43,
  "reward_stats": {"n": 41, "mean": 0.042, "std": 0.187, "variance": 0.035},
  "history_size": 41
}
...
```

---

**`BEST_ALGORITHM.py`**:

```python
# Best Algorithm: EnhancedAdaptiveDE_v7
# Fitness: 0.6812
# Generation: 8
# Operator: refine

import numpy as np

class EnhancedAdaptiveDE_v7:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = 50

    def __call__(self, func):
        # Implementation...
        pass
```

---

**`conversationlog.txt`**:

````
You are a highly skilled computer scientist in the field of natural computing...

The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite...

# Name: RandomSearch
# Code:
```python
class RandomSearch:
    ...
````

Last algorithm RandomSearch scored AOCC 0.45. Improve it. Format:

# Name: <classname>

# Code: <code>

# Name: AdaptiveDE

# Code:

```python
class AdaptiveDE:
    ...
```

... (full conversation history)

```

---

**`try-N-aucs.txt`**:

```

# len=216 mean=0.5700 std=0.048

0.58
0.62
0.55
0.59
... (216 AUC values from 24 functions × 3 instances × 3 repetitions)

```

---

## 14. Pipeline Summary

### 14.1 Component Overview

| Step | Component | Input | Output | Purpose |
|------|-----------|-------|--------|---------|
| 1 | **D-TS Bandit** | Arm posteriors (μ̂, τ) | Selected operator (mutation/crossover/refine) | Adaptive operator selection |
| 2 | **Operator Logic** | Parents, operator type | LLM prompt | Generate task-specific prompts |
| 3 | **LLM** | Prompt | Python code | Generate algorithm candidate |
| 4 | **BBOB Evaluation** | Algorithm code | 216 AUC values, trace | Measure performance & behavior |
| 5 | **NN-Dist** | New trace vs. history | Diversity score [0, 1] | Quantify behavioral novelty |
| 6 | **Composite Reward** | Δfitness, NN-Dist, α | Raw reward | Balance exploitation & exploration |
| 7 | **Normalizer** | Raw reward | Z-score | Stabilize reward distribution |
| 8 | **D-TS Update** | Normalized reward | Updated posteriors | Adapt operator preferences |
| 9 | **Selection** | Population + offspring | Next generation (μ parents) | Preserve best + diversity |

### 14.2 Mathematical Notation Summary

| Symbol | Definition | Range |
|--------|------------|-------|
| \( \mu \) | Population size (parents) | [2, 20] |
| \( \lambda \) | Offspring per generation | [4, 50] |
| \( \gamma \) | D-TS discount factor | [0, 1] |
| \( \tau_i \) | Posterior uncertainty for arm \( i \) | [0, τ_max] |
| \( \hat{\mu}_i \) | Estimated mean reward for arm \( i \) | ℝ |
| \( \alpha_t \) | Diversity weight at generation \( t \) | [0, 1] |
| \( \text{NN-Dist} \) | Nearest-neighbor distance | [0, ~1] |
| \( \Delta F \) | Fitness improvement | ℝ |
| \( R \) | Composite reward | ℝ |
| \( R_{\text{norm}} \) | Normalized reward (z-score) | ℝ |

### 14.3 Key Equations

1. **Thompson Sampling**: \( \theta_i \sim \mathcal{N}(\hat{\mu}_i, \tau_i^2), \quad i^* = \arg\max_i \theta_i \)

2. **D-TS Update**: \( N_i \leftarrow \gamma N_i + \mathbb{1}[i = i^*], \quad \hat{\mu}_i \leftarrow \frac{\tilde{\mu}_i}{N_i}, \quad \tau_i \leftarrow \min\left(\frac{\sigma}{\sqrt{N_i}}, \tau_{\max}\right) \)

3. **Behavioral Diversity**: \( \text{NN-Dist} = \min_{T_j \in \mathcal{H}} \frac{1}{\sqrt{B}} \sqrt{\sum_{t=1}^{B} \left( \frac{f_{\text{new}}^t - f_{\min}}{f_{\max} - f_{\min}} - \frac{f_j^t - f_{\min}}{f_{\max} - f_{\min}} \right)^2} \)

4. **Composite Reward**: \( R = \Delta F + \alpha \times \text{NN-Dist} \)

5. **Reward Normalization**: \( R_{\text{norm}} = \frac{R - \mu_R}{\sigma_R} \)

6. **Alpha Scheduling (Linear)**: \( \alpha(t) = \alpha_{\text{start}} + (\alpha_{\text{end}} - \alpha_{\text{start}}) \times \frac{t}{T} \)

---

## 15. Conclusion

MADA-LLAMEA v2 represents a sophisticated evolutionary framework for automated algorithm design, integrating:

- **Three-Operator System**: Mutation, crossover, and refine compete via adaptive multi-armed bandits
- **Behavioral Diversity**: Trace-based novelty measurement ensures exploration of the algorithm space
- **Discounted Thompson Sampling**: Non-stationary bandit algorithm adapts operator preferences over time
- **Composite Rewards**: Balance exploitation (fitness) and exploration (diversity) with time-varying weights
- **Robust Implementation**: Production-ready codebase with comprehensive logging and error handling

The system autonomously discovers high-performing optimization algorithms through a principled balance of exploration and exploitation, guided by both performance signals and behavioral novelty.

**For Thesis Inclusion**: This document provides complete technical specifications suitable for methodology chapters, supplementary materials, and reproducibility sections.

```
