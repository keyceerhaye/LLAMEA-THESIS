"""
Evolution of Heuristics with Discounted Thompson Sampling (EoH-DTS)

This version extends main-thesis-eoh.py with Discounted Thompson Sampling (D-TS)
to adaptively balance between mutation (exploration) and crossover (exploitation).

Mathematical Foundation (from MADA 5.0 and D-TS paper):
- Uses Gaussian posteriors for each arm (mutation/crossover)
- Incorporates discount factor γ ∈ (0,1) to adapt to non-stationary rewards
- Posterior variance is bounded by τ_max to prevent over/under-exploration

DS-TS Algorithm:
1. Sample θ(i) ~ N(μ̂_t(i), τ_t(i)²) for each arm i
2. Select arm with maximum sampled value
3. Update discounted statistics:
   - N_{t+1}(i) = γ·N_t(i) + I{selected arm = i}
   - μ̃_{t+1}(i) = γ·μ̃_t(i) + reward·I{selected arm = i}
   - μ̂_{t+1}(i) = μ̃_{t+1}(i) / N_{t+1}(i)
   - τ_{t+1}(i) = min(1/√N_{t+1}(i), τ_max)

Arms:
- Mutation (Exploration): Generate novel algorithms, may discover new strategies
- Crossover (Exploitation): Combine proven solutions, preserves good genetic material

References:
- MADA 5.0: Holistic Semantic Refinement Architecture
- Discounted Thompson Sampling (Raj & Kalyani, 2017)
- FunSearch (Google DeepMind)
- Evolution of Heuristics (EoH)

Usage:
    python main-thesis-eoh-dts.py --evolutionary-mode --n-parents 4 --n-offspring 8 --budget 50 --elitism
"""

import os
import sys
import hashlib
from pathlib import Path
import numpy as np
from ioh import get_problem, logger
import re
import argparse
import random
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
LLAMEA_SRC = PROJECT_ROOT / "llamea-experimentation" / "src"
if LLAMEA_SRC.exists() and str(LLAMEA_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMEA_SRC))

# Use existing managers from managers.py (same as main-thesis.py)
from managers import AlgorithmManager, ExperimentLogger
from utils import OverBudgetException, aoc_logger, correct_aoc
from llamea.utils import NoCodeException

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    dotenv_loaded = load_dotenv()
    print(f"DEBUG: .env file loaded: {dotenv_loaded}")
except ImportError:
    print("DEBUG: python-dotenv not installed, using system environment variables only")
    pass


# ==============================================================================
# DISCOUNTED THOMPSON SAMPLING (D-TS) IMPLEMENTATION
# ==============================================================================

@dataclass
class ArmStatistics:
    """Statistics for a single bandit arm."""
    name: str
    N: float = 1.0          # Discounted effective sample size (initialized to 1 for stability)
    mu_tilde: float = 0.0   # Discounted cumulative reward
    mu_hat: float = 0.0     # Posterior mean estimate
    tau: float = 1.0        # Posterior standard deviation
    pulls: int = 0          # Total number of pulls (for logging)
    total_reward: float = 0.0  # Total undiscounted reward (for logging)


class DiscountedThompsonSampler:
    """
    Discounted Thompson Sampling with Gaussian Posteriors.
    
    Implements the DS-TS algorithm from MADA 5.0:
    - Handles non-stationary reward distributions via discounting
    - Uses Gaussian posteriors with variance truncation
    - Balances exploration vs exploitation adaptively
    
    Mathematical Details:
    - Prior: N(0, τ_max²) for each arm
    - Likelihood: Gaussian with known variance σ² (reward_variance)
    - Posterior updates use discount factor γ to forget old observations
    
    Args:
        arms: List of arm names (e.g., ['mutation', 'crossover'])
        discount: Discount factor γ ∈ (0, 1), default 0.9
                  Lower values = faster forgetting, more reactive to recent rewards
        tau_max: Maximum posterior std dev, default 1.0
                 Prevents over-exploration when sample size is small
        reward_variance: Assumed variance of rewards, default 0.25
                        Affects how quickly posterior concentrates
        prior_mean: Initial mean for all arms, default 0.0
    """
    
    def __init__(
        self,
        arms: List[str],
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 0.25,
        prior_mean: float = 0.0,
    ):
        self.arms = arms
        self.discount = discount
        self.tau_max = tau_max
        self.reward_variance = reward_variance
        self.prior_mean = prior_mean
        
        # Initialize arm statistics
        self.arm_stats: Dict[str, ArmStatistics] = {}
        for arm in arms:
            self.arm_stats[arm] = ArmStatistics(
                name=arm,
                N=1.0,  # Start with pseudo-count of 1 for numerical stability
                mu_tilde=prior_mean,
                mu_hat=prior_mean,
                tau=tau_max,
            )
        
        self.total_rounds = 0
        self.selection_history: List[str] = []
        self.reward_history: List[Tuple[str, float]] = []
    
    def select_arm(self) -> Tuple[str, float, Dict]:
        """
        Select an arm using Thompson Sampling.
        
        Returns:
            Tuple of (selected_arm_name, sampled_theta, snapshot_dict)
        
        Algorithm:
            1. For each arm i, sample θ_i ~ N(μ̂_i, τ_i²)
            2. Select arm with maximum θ_i
        """
        samples = {}
        snapshot = {}
        
        for arm_name, stats in self.arm_stats.items():
            # Sample from posterior: θ ~ N(μ̂, τ²)
            theta = np.random.normal(stats.mu_hat, stats.tau)
            samples[arm_name] = theta
            
            # Create snapshot for logging
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
    
    def update(self, arm_name: str, reward: float) -> None:
        """
        Update the posterior for the selected arm after observing reward.
        
        Args:
            arm_name: Name of the arm that was pulled
            reward: Observed reward (fitness improvement)
        
        Algorithm (DS-TS with Gaussian posteriors):
            1. Decay all arms: N_i ← γ·N_i, μ̃_i ← γ·μ̃_i
            2. For selected arm: N_i ← N_i + 1, μ̃_i ← μ̃_i + reward
            3. Update posterior mean: μ̂_i = μ̃_i / N_i
            4. Update posterior std: τ_i = min(1/√N_i, τ_max)
        """
        # Step 1: Apply discount to ALL arms (passage of time)
        for stats in self.arm_stats.values():
            stats.N = self.discount * stats.N
            stats.mu_tilde = self.discount * stats.mu_tilde
        
        # Step 2: Update selected arm with new observation
        if arm_name in self.arm_stats:
            stats = self.arm_stats[arm_name]
            stats.N += 1.0
            stats.mu_tilde += reward
            stats.pulls += 1
            stats.total_reward += reward
            
            # Step 3: Update posterior mean
            # μ̂ = μ̃ / N (discounted average)
            if stats.N > 0:
                stats.mu_hat = stats.mu_tilde / stats.N
            else:
                stats.mu_hat = self.prior_mean
            
            # Step 4: Update posterior std with truncation
            # τ = min(1/√N, τ_max)
            if stats.N > 0:
                stats.tau = min(1.0 / np.sqrt(stats.N), self.tau_max)
            else:
                stats.tau = self.tau_max
        
        self.reward_history.append((arm_name, reward))
    
    def get_state_snapshot(self) -> Dict:
        """Get current state of all arms for logging."""
        return {
            arm_name: {
                'mu_hat': stats.mu_hat,
                'tau': stats.tau,
                'N': stats.N,
                'pulls': stats.pulls,
                'total_reward': stats.total_reward,
                'avg_reward': stats.total_reward / stats.pulls if stats.pulls > 0 else 0.0,
            }
            for arm_name, stats in self.arm_stats.items()
        }
    
    def get_arm_probabilities(self, n_samples: int = 1000) -> Dict[str, float]:
        """
        Estimate selection probabilities via Monte Carlo simulation.
        
        Args:
            n_samples: Number of samples for estimation
            
        Returns:
            Dictionary mapping arm names to selection probabilities
        """
        counts = {arm: 0 for arm in self.arms}
        
        for _ in range(n_samples):
            samples = {}
            for arm_name, stats in self.arm_stats.items():
                samples[arm_name] = np.random.normal(stats.mu_hat, stats.tau)
            selected = max(samples, key=samples.get)
            counts[selected] += 1
        
        return {arm: count / n_samples for arm, count in counts.items()}


# ==============================================================================
# EoH PROMPT TEMPLATES
# ==============================================================================

EOH_SYSTEM_PROMPT = """You are a highly skilled computer scientist specializing in evolutionary algorithm design.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc. since these are generally not well performing.
Focus on well-established techniques like Differential Evolution, CMA-ES, Evolution Strategies, or novel hybrid approaches."""

EOH_INIT_PROMPT = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
An example of such code (a simple random search), is as follows:
```python
import numpy as np

class RandomSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        if self.dim is None:
            self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        for i in range(self.budget):
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
            
        return self.f_opt, self.x_opt
```
Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <classname>
# Code: <code>
"""

# Strategy A: The "Merge & Refine" Prompt (Explicit Crossover)
EOH_CROSSOVER_EXPLICIT_PROMPT = """You are an expert Algorithm Designer. Your task is to create a new, more efficient algorithm by combining the strengths of two existing parent algorithms.

Parent Algorithm A (Score: {score_a:.4f}):
```python
{code_a}
```

Parent Algorithm B (Score: {score_b:.4f}):
```python
{code_b}
```

Task: Write a new Python optimizer class that combines the best aspects of both Parent A and Parent B. 
- Identify the strengths of each parent (e.g., search strategy, adaptation mechanism, population management)
- Merge these strengths into a cohesive new algorithm
- The new algorithm should have a unique class name
- Ensure the code is syntactically correct and follows the same interface (__init__(self, budget) and __call__(self, func))

Give the response in the format:
# Name: <classname>
# Code: <code>
"""

# Strategy B: The "Contextual Continuation" Prompt (Implicit Crossover - FunSearch style)
EOH_CROSSOVER_IMPLICIT_PROMPT = """I am designing metaheuristic algorithms to solve black-box optimization problems (BBOB benchmark, 5D, bounds [-5, 5]).

Here are two high-scoring solutions from my evolutionary search:

Solution 1 (Score: {score_a:.4f}):
```python
{code_a}
```

Solution 2 (Score: {score_b:.4f}):
```python
{code_b}
```

Based on the logic and strategies shown in both solutions above, generate a Solution 3 that is likely to achieve an even higher score. The new solution should:
- Learn from the successful patterns in both parents
- Combine or improve upon their search strategies
- Have a unique class name
- Follow the same interface: __init__(self, budget) and __call__(self, func)

Give the response in the format:
# Name: <classname>
# Code: <code>
"""

# Mutation prompt - generate a novel algorithm inspired by parent
EOH_MUTATION_PROMPT = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.

For reference, here is a previous algorithm that achieved a score of {score:.4f}:
```python
{code}
```

{feedback}

Your task: Design a NEW and DIFFERENT algorithm that can achieve an even higher score. Do not just copy the reference - create something novel using different techniques such as:
- Differential Evolution variants (DE/rand, DE/best, SHADE, L-SHADE)
- Evolution Strategies (CMA-ES, self-adaptive ES)
- Particle Swarm Optimization variants
- Hybrid approaches combining multiple strategies
- Novel adaptive mechanisms
- Different selection/mutation/recombination operators

Give an excellent and novel heuristic algorithm to solve this task and also give it a name. Give the response in the format:
# Name: <classname>
# Code: <code>
"""


# ==============================================================================
# SOLUTION CLASS
# ==============================================================================

class EoHSolution:
    """Simple solution container for EoH."""
    
    _id_counter = 0
    
    def __init__(self, code, name, description="", generation=0, parent_ids=None):
        EoHSolution._id_counter += 1
        self.id = f"eoh_{EoHSolution._id_counter:06d}"
        self.code = code
        self.name = name
        self.description = description or name
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.fitness = 0.0
        self.aucs = []
        self.detailed_aucs = [0, 0, 0, 0, 0]
        self.error = ""
        self.operator = ""  # 'init', 'mutation', 'crossover'


# ==============================================================================
# EOH OPERATOR WITH DISCOUNTED THOMPSON SAMPLING
# ==============================================================================

class EoHOperatorDTS:
    """
    Evolution of Heuristics Operator with Discounted Thompson Sampling.
    
    Uses D-TS to adaptively balance between:
    - Mutation (Exploration): Generate novel algorithms
    - Crossover (Exploitation): Combine existing good solutions
    
    The bandit learns which operator is more effective in the current
    evolutionary phase and adjusts selection probabilities accordingly.
    """
    
    def __init__(
        self,
        algorithm_manager,
        crossover_style: str = "implicit",
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 0.25,
        reward_clamp: float = 1.0,
    ):
        """
        Args:
            algorithm_manager: Manager for LLM API calls
            crossover_style: "implicit" (FunSearch style) or "explicit" (merge & refine)
            discount: D-TS discount factor γ ∈ (0, 1)
                      Lower = faster adaptation, higher = more stable
            tau_max: Maximum posterior std dev (controls exploration)
            reward_variance: Assumed reward variance for posterior updates
            reward_clamp: Clamp rewards to [-clamp, +clamp] for stability
        """
        self.algorithm_manager = algorithm_manager
        self.crossover_style = crossover_style
        self.child_counter = 0
        self.reward_clamp = reward_clamp
        
        # Initialize D-TS bandit with two arms
        self.bandit = DiscountedThompsonSampler(
            arms=['mutation', 'crossover'],
            discount=discount,
            tau_max=tau_max,
            reward_variance=reward_variance,
            prior_mean=0.0,  # Neutral prior
        )
        
        # Track statistics for logging
        self._mutation_count = 0
        self._crossover_count = 0
        self._last_selection_info = {}
    
    def reset_generation_counts(self):
        """Reset offspring counts at the start of each generation."""
        self._mutation_count = 0
        self._crossover_count = 0
    
    def get_operator_stats(self) -> Dict:
        """Get current generation's operator usage statistics."""
        return {
            'mutation_count': self._mutation_count,
            'crossover_count': self._crossover_count,
            'total': self._mutation_count + self._crossover_count,
        }
    
    def get_bandit_state(self) -> Dict:
        """Get current bandit state for logging."""
        state = self.bandit.get_state_snapshot()
        probs = self.bandit.get_arm_probabilities()
        return {
            'arm_stats': state,
            'selection_probs': probs,
            'total_rounds': self.bandit.total_rounds,
        }
    
    def generate_offspring(
        self,
        parents: List[EoHSolution],
        focal_parent: Optional[EoHSolution] = None,
        population_summary: str = "",
    ) -> Tuple[EoHSolution, Dict]:
        """
        Generate a new offspring using D-TS selected operator.
        
        Args:
            parents: List of parent solutions
            focal_parent: Primary parent for mutation
            population_summary: String describing current population
        
        Returns:
            Tuple of (offspring_solution, selection_info)
        """
        if not parents:
            raise ValueError("EoHOperatorDTS requires at least one parent")
        
        # Need at least 2 parents for crossover
        if len(parents) < 2:
            # Force mutation if only one parent
            operator = 'mutation'
            theta = 0.0
            snapshot = {}
        else:
            # Use D-TS to select operator
            operator, theta, snapshot = self.bandit.select_arm()
        
        # Store selection info for logging
        self._last_selection_info = {
            'operator': operator,
            'theta_sampled': theta,
            'snapshot': snapshot,
        }
        
        # Execute selected operator
        if operator == 'mutation':
            self._mutation_count += 1
            parent = focal_parent or random.choice(parents)
            child = self._mutate(parent)
        else:
            self._crossover_count += 1
            sorted_parents = sorted(parents, key=lambda p: p.fitness, reverse=True)
            parent_a = sorted_parents[0]
            parent_b = self._select_diverse_parent(sorted_parents, parent_a)
            child = self._crossover(parent_a, parent_b)
        
        return child, self._last_selection_info
    
    def update_bandit(
        self,
        operator: str,
        child_fitness: float,
        parent_fitness: float,
        error: bool = False,
    ) -> float:
        """
        Update the bandit with observed reward after evaluation.
        
        Args:
            operator: Which operator was used ('mutation' or 'crossover')
            child_fitness: Fitness of the generated offspring
            parent_fitness: Best parent fitness for comparison
            error: Whether the offspring had an error
        
        Returns:
            The computed reward value
        """
        if error:
            # Penalize errors
            reward = -self.reward_clamp
        else:
            # Reward = fitness improvement (delta)
            raw_reward = child_fitness - parent_fitness
            # Clamp for stability
            reward = max(-self.reward_clamp, min(self.reward_clamp, raw_reward))
            
            # Give small positive signal for ties (prevents stuck bandits)
            if reward == 0.0:
                reward = 0.01
        
        self.bandit.update(operator, reward)
        return reward
    
    def _mutate(self, parent: EoHSolution) -> EoHSolution:
        """Apply mutation: Generate novel algorithm inspired by parent."""
        self.child_counter += 1
        
        # Build feedback based on parent's detailed performance
        feedback_parts = []
        if parent.detailed_aucs and any(parent.detailed_aucs):
            feedback_parts.append("Performance breakdown:")
            feedback_parts.append(f"  - Separable functions: {parent.detailed_aucs[0]:.4f}")
            feedback_parts.append(f"  - Low/moderate conditioning: {parent.detailed_aucs[1]:.4f}")
            feedback_parts.append(f"  - High conditioning & unimodal: {parent.detailed_aucs[2]:.4f}")
            feedback_parts.append(f"  - Multimodal (adequate structure): {parent.detailed_aucs[3]:.4f}")
            feedback_parts.append(f"  - Multimodal (weak structure): {parent.detailed_aucs[4]:.4f}")
        if parent.error:
            feedback_parts.append(f"Previous error: {parent.error}")
        
        feedback = "\n".join(feedback_parts) if feedback_parts else "No specific feedback available."
        
        prompt = EOH_MUTATION_PROMPT.format(
            score=parent.fitness,
            code=parent.code,
            feedback=feedback
        )
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            class_name = self._extract_class_name(code, default=f"{parent.name}Mut{self.child_counter}")
            description = self._extract_name(message) or class_name
            
            solution = EoHSolution(
                code=code,
                name=class_name,
                description=description,
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "mutation"
            return solution
            
        except Exception as e:
            solution = EoHSolution(
                code=parent.code,
                name=f"{parent.name}MutFail{self.child_counter}",
                description=f"Mutation failed: {e}",
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "mutation_failed"
            solution.error = str(e)
            return solution
    
    def _crossover(self, parent_a: EoHSolution, parent_b: EoHSolution) -> EoHSolution:
        """Apply crossover: Combine two parent algorithms."""
        self.child_counter += 1
        
        if self.crossover_style == "explicit":
            prompt = EOH_CROSSOVER_EXPLICIT_PROMPT.format(
                score_a=parent_a.fitness,
                code_a=parent_a.code,
                score_b=parent_b.fitness,
                code_b=parent_b.code
            )
        else:
            prompt = EOH_CROSSOVER_IMPLICIT_PROMPT.format(
                score_a=parent_a.fitness,
                code_a=parent_a.code,
                score_b=parent_b.fitness,
                code_b=parent_b.code
            )
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            class_name = self._extract_class_name(code, default=f"Hybrid{self.child_counter}")
            description = self._extract_name(message) or class_name
            
            solution = EoHSolution(
                code=code,
                name=class_name,
                description=description,
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=[parent_a.id, parent_b.id]
            )
            solution.operator = "crossover"
            return solution
            
        except Exception as e:
            solution = EoHSolution(
                code=parent_a.code,
                name=f"{parent_a.name}XoFail{self.child_counter}",
                description=f"Crossover failed: {e}",
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=[parent_a.id, parent_b.id]
            )
            solution.operator = "crossover_failed"
            solution.error = str(e)
            return solution
    
    def _select_diverse_parent(self, sorted_parents: List[EoHSolution], parent_a: EoHSolution) -> EoHSolution:
        """Select a second parent that is code-diverse from parent_a."""
        for candidate in sorted_parents[1:]:
            if (candidate.code or "") != (parent_a.code or ""):
                return candidate
        return sorted_parents[1] if len(sorted_parents) > 1 else parent_a
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt using AlgorithmManager's client."""
        messages = [
            {"role": "system", "content": EOH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Log using AlgorithmManager's logger
        self.algorithm_manager.logger.log_conversation(f"\n[EoH-DTS Prompt]\n{prompt}\n")
        
        call_kwargs = {
            "model": self.algorithm_manager.ai_model,
            "messages": messages,
            "temperature": 0.8,
        }
        if self.algorithm_manager.max_tokens:
            call_kwargs["max_tokens"] = self.algorithm_manager.max_tokens
        
        response = self.algorithm_manager.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        
        self.algorithm_manager.logger.log_conversation(f"\n[EoH-DTS Response]\n{message}\n")
        
        return message
    
    def _extract_code(self, message: str) -> str:
        """Extract Python code from LLM response."""
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        raise NoCodeException("No code block found in response")
    
    def _extract_class_name(self, code: str, default: str) -> str:
        """Extract class name from code."""
        for line in code.splitlines():
            if line.strip().startswith("class "):
                match = re.search(r"class\s+(\w+)", line)
                if match:
                    return match.group(1)
        return default
    
    def _extract_name(self, message: str) -> str:
        """Extract algorithm name from message."""
        pattern = r"#\s*Name:\s*(\w+)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""


# ==============================================================================
# EOH-DTS SPECIFIC LOGGING HELPERS
# ==============================================================================

def log_eoh_offspring(explogger, attempt, record):
    """Log EoH offspring metadata to JSONL file."""
    # Ensure directory exists (in case it was deleted during run)
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "eoh_offspring.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": attempt, **record}, default=str) + "\n")


def log_bandit_snapshot(explogger, generation, bandit_state):
    """Log D-TS bandit state snapshot to JSONL file."""
    # Ensure directory exists (in case it was deleted during run)
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "bandit_snapshots.jsonl")
    record = {"generation": generation, **bandit_state}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ==============================================================================
# EVALUATION
# ==============================================================================

EXPECTED_AUC_LEN = 24 * 3 * 3


def _hash_aucs(aucs):
    try:
        arr = np.asarray(aucs, dtype=float)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return ""


def _summarize_aucs(aucs):
    try:
        arr = np.asarray(aucs, dtype=float)
        return len(arr), float(np.mean(arr)), float(np.std(arr))
    except Exception:
        return 0, 0.0, 0.0


def evaluate_algorithm(algorithm_code, algorithm_name, eval_budget):
    """Evaluates a single algorithm on BBOB benchmark suite."""
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
            return [], [0, 0, 0, 0, 0], f"Class {algorithm_name} not found"
        
        algorithm_class = exec_globals[algorithm_name]
        l2 = aoc_logger(eval_budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        aucs = []
        detail_aucs = []
        detailed_aucs = [0, 0, 0, 0, 0]
        
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)
                
                for rep in range(3):
                    np.random.seed(rep)
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
                            algorithm.fitness = lambda x: problem(x)
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    except Exception as e:
                        return [], [0, 0, 0, 0, 0], str(e)
                    
                    auc = correct_aoc(problem, l2, eval_budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()
            
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
        
        return aucs, detailed_aucs, ""

    except KeyboardInterrupt:
        return [], [0, 0, 0, 0, 0], "KeyboardInterrupt"
    except Exception as e:
        return [], [0, 0, 0, 0, 0], str(e)
    finally:
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice


# ==============================================================================
# SELECTION
# ==============================================================================

def selection(population, n_parents, elitism=True, minimization=False):
    """Select the best individuals from the population."""
    reverse = not minimization
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)
    
    if not sorted_pop:
        return []
    
    selected = [sorted_pop[0]]
    if n_parents == 1:
        return selected
    
    best_code = sorted_pop[0].code or ""
    
    # Find diverse second parent
    for ind in sorted_pop[1:]:
        if (ind.code or "") != best_code:
            selected.append(ind)
            break
    
    if len(selected) < 2 and len(sorted_pop) > 1:
        selected.append(sorted_pop[1])
    
    # Fill remaining slots
    seen_codes = {s.code or "" for s in selected}
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        if (ind.code or "") not in seen_codes:
            seen_codes.add(ind.code or "")
            selected.append(ind)
    
    # Fallback
    for ind in sorted_pop:
        if len(selected) >= n_parents:
            break
        if ind not in selected:
            selected.append(ind)
    
    return selected


# ==============================================================================
# MAIN EVOLUTIONARY LOOP
# ==============================================================================

def run_eoh_dts_evolutionary_mode(
    algorithm_manager,
    eoh_operator,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
):
    """
    Population-based evolutionary mode with D-TS adaptive operator selection.
    """
    population = []
    best_ever = None
    api_calls = 0
    generation = 0
    last_auc_hash = ""
    
    # Phase 1: Initialize parent population
    print(f"\n{'='*60}")
    print(f"INITIALIZATION: Generating {n_parents} parent algorithms")
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
                # Use AlgorithmManager's fetch_algorithm() method
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
                aucs, detailed_aucs, error = evaluate_algorithm(
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
                    auc_len, auc_mean_val, auc_std_val = _summarize_aucs(aucs)
                    auc_hash = _hash_aucs(aucs)
                    solution.fitness = float(np.mean(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    seen_hashes.add(code_hash)
                    print(f"  Fitness: {solution.fitness:.4f}")
                    last_auc_hash = auc_hash

                population.append(solution)
                explogger.log_code(api_calls, algorithm_name, algorithm_code)
                explogger.log_aucs(api_calls, aucs if aucs else [0], metadata={
                    "len": auc_len if aucs else 0,
                    "mean": auc_mean_val if aucs else 0.0,
                    "std": auc_std_val if aucs else 0.0,
                })
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
    
    # Phase 2: Evolutionary loop with D-TS
    for gen in range(1, generations + 1):
        if api_calls >= args.budget:
            print(f"\nBudget exhausted at generation {gen}")
            break
        
        generation = gen
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{generations}")
        print(f"API Calls: {api_calls}/{args.budget}")
        print(f"Best so far: {best_ever.fitness:.4f} ({best_ever.name})")
        
        # Show current bandit state
        bandit_state = eoh_operator.get_bandit_state()
        probs = bandit_state['selection_probs']
        print(f"D-TS Selection Probabilities: mutation={probs['mutation']:.1%}, crossover={probs['crossover']:.1%}")
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
                child, selection_info = eoh_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                )
                
                operator = selection_info['operator']
                print(f"  D-TS selected: {operator} (θ={selection_info['theta_sampled']:.3f})")
                print(f"  Evaluating {child.name}...")
                
                aucs, detailed_aucs, error = evaluate_algorithm(
                    child.code, child.name, args.eval_budget
                )

                if error:
                    child.fitness = 0.0
                    child.error = error
                    print(f"  Error: {error}")
                else:
                    auc_len, auc_mean_val, auc_std_val = _summarize_aucs(aucs)
                    child.fitness = float(np.mean(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    print(f"  Fitness: {child.fitness:.4f}")
                    
                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                        best_ever = child

                # Update D-TS bandit with reward
                reward = eoh_operator.update_bandit(
                    operator=operator,
                    child_fitness=child.fitness,
                    parent_fitness=best_parent_fitness,
                    error=bool(error),
                )
                print(f"  Reward: {reward:+.4f}")

                # Log offspring using helper function
                offspring_record = {
                    'operator': child.operator,
                    'fitness': child.fitness,
                    'parent_fitness': best_parent_fitness,
                    'reward': reward,
                    'theta_sampled': selection_info['theta_sampled'],
                    'parent_ids': child.parent_ids,
                    'generation': generation,
                    'error': child.error,
                }
                log_eoh_offspring(explogger, api_calls, offspring_record)
                
                offspring.append(child)
                explogger.log_code(api_calls, child.name, child.code)
                explogger.log_aucs(api_calls, aucs if aucs else [0], metadata={
                    "mean": auc_mean_val if aucs else 0.0,
                })
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

        # Log bandit state using helper function
        bandit_state = eoh_operator.get_bandit_state()
        log_bandit_snapshot(explogger, generation, bandit_state)
        
        stats = eoh_operator.get_operator_stats()
        print(f"Operator usage: {stats['mutation_count']} mutations, {stats['crossover_count']} crossovers")
        
        # Show updated arm statistics
        arm_stats = bandit_state['arm_stats']
        print(f"Bandit state after gen {generation}:")
        for arm, s in arm_stats.items():
            print(f"  {arm}: μ̂={s['mu_hat']:.3f}, τ={s['tau']:.3f}, N={s['N']:.2f}, pulls={s['pulls']}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("EOH-DTS EVOLUTIONARY OPTIMIZATION COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations completed: {generation}")
    print(f"Best algorithm: {best_ever.name}")
    print(f"Best fitness: {best_ever.fitness:.4f}")
    print(f"Best generation: {best_ever.generation}")
    
    # Final bandit statistics
    final_state = eoh_operator.get_bandit_state()
    print(f"\nFinal D-TS Bandit Statistics:")
    for arm, s in final_state['arm_stats'].items():
        print(f"  {arm}: pulls={s['pulls']}, avg_reward={s['avg_reward']:.4f}, μ̂={s['mu_hat']:.3f}")
    
    # Save best algorithm
    with open(f"{explogger.dirname}/BEST_ALGORITHM.py", "w") as f:
        f.write(f"# Best Algorithm: {best_ever.name}\n")
        f.write(f"# Fitness: {best_ever.fitness:.4f}\n")
        f.write(f"# Generation: {best_ever.generation}\n")
        f.write(f"# Operator: {best_ever.operator}\n\n")
        f.write(best_ever.code)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EoH-DTS: Evolution of Heuristics with Discounted Thompson Sampling',
        epilog='''
EoH with Discounted Thompson Sampling (D-TS) for adaptive operator selection.

The D-TS bandit learns which operator (mutation vs crossover) is more effective
and adjusts selection probabilities accordingly. This enables:
- Automatic exploration/exploitation balance
- Adaptation to different evolutionary phases
- No manual tuning of mutation rate needed

D-TS Parameters:
- discount (γ): How fast to forget old observations (lower = faster adaptation)
- tau_max: Maximum exploration (higher = more random early on)
- reward_clamp: Stability bound for rewards

Examples:
  # Basic run with D-TS
  python main-thesis-eoh-dts.py --evolutionary-mode --n-parents 4 --n-offspring 8 --budget 50 --elitism
  
  # Custom D-TS parameters
  python main-thesis-eoh-dts.py --evolutionary-mode --discount 0.95 --tau-max 2.0 --budget 100
  
  # Fast adaptation (more reactive to recent performance)
  python main-thesis-eoh-dts.py --evolutionary-mode --discount 0.8 --budget 100
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # API Configuration
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--base-url', type=str, help='Custom base URL for OpenAI-compatible APIs')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens for API responses')
    
    # Model Configuration  
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='AI model to use')
    parser.add_argument('--experiment-name', type=str, default='eoh-dts-experiment', help='Experiment name')
    
    # Experimental Configuration
    parser.add_argument('--budget', type=int, default=100, help='Total API calls budget')
    parser.add_argument('--eval-budget', type=int, default=10000, help='Evaluation budget per algorithm')
    parser.add_argument('--elitism', action='store_true', help='Enable elitism (μ+λ) strategy')
    
    # Population Configuration
    parser.add_argument('--n-parents', type=int, default=4, help='Number of parents')
    parser.add_argument('--n-offspring', type=int, default=16, help='Number of offspring per generation')
    parser.add_argument('--generations', type=int, default=None, help='Number of generations')
    parser.add_argument('--evolutionary-mode', action='store_true', help='Enable evolutionary mode')
    
    # EoH Configuration
    parser.add_argument('--crossover-style', type=str, default='implicit',
                       choices=['implicit', 'explicit'], help='Crossover prompt style')
    
    # D-TS Configuration
    parser.add_argument('--discount', type=float, default=0.9,
                       help='D-TS discount factor γ (default: 0.9). Lower = faster adaptation.')
    parser.add_argument('--tau-max', type=float, default=1.0,
                       help='D-TS maximum posterior std dev τ_max (default: 1.0)')
    parser.add_argument('--reward-variance', type=float, default=0.25,
                       help='D-TS assumed reward variance (default: 0.25)')
    parser.add_argument('--reward-clamp', type=float, default=1.0,
                       help='Clamp rewards to [-clamp, +clamp] (default: 1.0)')
    
    args = parser.parse_args()
    
    # Debug
    print("DEBUG: Environment variables found:")
    print(f"  OPENAI_API_KEY: {'***' if os.getenv('OPENAI_API_KEY') else 'Not found'}")
    print(f"  AIML_KEY: {'***' if os.getenv('AIML_KEY') else 'Not found'}")
    print(f"  BASE_URL: {os.getenv('BASE_URL') or 'Not found'}")
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AIML_KEY")
    if not api_key:
        raise ValueError("API key is required.")
    
    base_url = args.base_url or os.getenv("BASE_URL")
    
    # Setup
    ai_model = args.model
    experiment_suffix = f"{ai_model}-{args.experiment_name}"
    if args.evolutionary_mode:
        experiment_suffix += "-evolutionary"
    if args.elitism:
        experiment_suffix += "-elitism"
    
    # Use existing managers from managers.py (same as main-thesis.py)
    explogger = ExperimentLogger(experiment_suffix)
    algorithm_manager = AlgorithmManager(
        api_key, 
        explogger, 
        ai_model,
        elitism=args.elitism,
        detailed_feedback=False,
        base_url=base_url,
        max_tokens=args.max_tokens
    )

    if args.evolutionary_mode:
        # Create EoH-DTS operator
        eoh_operator = EoHOperatorDTS(
            algorithm_manager,
            crossover_style=args.crossover_style,
            discount=args.discount,
            tau_max=args.tau_max,
            reward_variance=args.reward_variance,
            reward_clamp=args.reward_clamp,
        )
        
        # Calculate generations
        if args.generations is not None:
            generations = args.generations
        else:
            generations = max(1, (args.budget - args.n_parents) // args.n_offspring)
        
        print(f"\nStarting EoH-DTS (Evolution of Heuristics with Discounted Thompson Sampling):")
        print(f"  Mode: EVOLUTIONARY with D-TS Adaptive Operator Selection")
        print(f"  Model: {ai_model}")
        print(f"  Base URL: {base_url or 'Default OpenAI'}")
        print(f"  API Budget: {args.budget}")
        print(f"  Evaluation Budget: {args.eval_budget}")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
        print(f"  Generations: {generations}")
        print(f"  Strategy: {'(μ+λ) Elitism' if args.elitism else '(μ,λ) Comma'}")
        print(f"  Crossover Style: {args.crossover_style}")
        print(f"  D-TS Parameters:")
        print(f"    Discount (γ): {args.discount}")
        print(f"    τ_max: {args.tau_max}")
        print(f"    Reward variance: {args.reward_variance}")
        print(f"    Reward clamp: ±{args.reward_clamp}")
        print("-" * 60)
        
        run_eoh_dts_evolutionary_mode(
            algorithm_manager,
            eoh_operator,
            explogger,
            args,
            generations,
            args.n_parents,
            args.n_offspring,
        )
    else:
        print("ERROR: EoH-DTS requires --evolutionary-mode flag")
        return
    
    print(f"\nExperiment completed. Results saved in: {explogger.dirname}")


if __name__ == "__main__":
    main()


