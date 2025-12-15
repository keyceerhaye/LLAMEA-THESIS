"""
MADA-LLAMEA v2: Multi-Adaptive Diverse Algorithm with LLM Evolutionary Algorithm

This extends main-thesis-mada.py with a third operator: REFINE.

Key Enhancements in v2:
1. Three Operators: mutation, crossover, refine
2. Refine Operator: Mutation-style but with detailed benchmark feedback
   - Focuses on redesigning and refining the parent algorithm
   - Includes per-function-group performance breakdown
3. All three operators compete via Discounted Thompson Sampling

Operator Descriptions:
- Mutation: Generate a NEW and DIFFERENT algorithm
- Crossover: Combine patterns from two high-scoring solutions
- Refine: Redesign and refine the strategy to improve specific weaknesses

Usage:
    python main-thesis-mada-v2.py --evolutionary-mode --elitism --budget 50
    python main-thesis-mada-v2.py --evolutionary-mode --elitism --alpha-start 0.7 --alpha-schedule cosine --budget 100
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

# Use existing managers
from managers import AlgorithmManager, ExperimentLogger
from utils import OverBudgetException, aoc_logger, correct_aoc
from llamea.utils import NoCodeException

# Import MADA components
from mada_components import (
    TraceCollector,
    aggregate_traces,
    calculate_nn_dist,
    calculate_trace_distance,
    RewardNormalizer,
    AlphaScheduler,
    compute_composite_reward,
    select_behavioral_parents,
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# DISCOUNTED THOMPSON SAMPLING (Extended for 3 arms)
# ==============================================================================

@dataclass
class ArmStatistics:
    """Statistics for a single bandit arm."""
    name: str
    N: float = 1.0
    mu_tilde: float = 0.0
    mu_hat: float = 0.0
    tau: float = 1.0
    pulls: int = 0
    total_reward: float = 0.0
    # MADA additions
    total_fitness_reward: float = 0.0
    total_diversity_reward: float = 0.0


class DiscountedThompsonSampler:
    """
    Discounted Thompson Sampling with Gaussian Posteriors.
    Extended for MADA v2 with 3 operators and reward component tracking.
    """
    
    def __init__(
        self,
        arms: List[str],
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 1.0,  # Normalized rewards have variance ~1
        prior_mean: float = 0.0,
    ):
        self.arms = arms
        self.discount = discount
        self.tau_max = tau_max
        self.reward_variance = reward_variance
        self.prior_mean = prior_mean
        
        self.arm_stats: Dict[str, ArmStatistics] = {}
        for arm in arms:
            self.arm_stats[arm] = ArmStatistics(
                name=arm,
                N=0.0,  # Paper uses N=0 for proper D-TS initialization
                mu_tilde=0.0,  # Also 0 per paper
                mu_hat=prior_mean,  # Prior mean when N=0
                tau=tau_max,  # Maximum exploration initially
            )
        
        self.total_rounds = 0
        self.selection_history: List[str] = []
        self.reward_history: List[Tuple[str, float]] = []
    
    def select_arm(self) -> Tuple[str, float, Dict]:
        """Select an arm using Thompson Sampling."""
        samples = {}
        snapshot = {}
        
        for arm_name, stats in self.arm_stats.items():
            theta = np.random.normal(stats.mu_hat, stats.tau)
            samples[arm_name] = theta
            snapshot[arm_name] = {
                'mu_hat': stats.mu_hat,
                'tau': stats.tau,
                'N': stats.N,
                'theta_sampled': theta,
                'pulls': stats.pulls,
            }
        
        selected_arm = max(samples, key=samples.get)
        selected_theta = samples[selected_arm]
        
        self.selection_history.append(selected_arm)
        self.total_rounds += 1
        
        return selected_arm, selected_theta, snapshot
    
    def update(self, arm_name: str, reward: float) -> None:
        """Update the posterior for the selected arm."""
        # Precompute sigma for tau calculation
        sigma = np.sqrt(self.reward_variance)
        
        # Apply discount to ALL arms and update their tau
        for stats in self.arm_stats.values():
            stats.N = self.discount * stats.N
            stats.mu_tilde = self.discount * stats.mu_tilde
            # Update tau for all arms after discounting
            # tau = sigma / sqrt(N) is the posterior std dev
            if stats.N > 0:
                stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
            else:
                stats.tau = self.tau_max
        
        # Update selected arm
        if arm_name in self.arm_stats:
            stats = self.arm_stats[arm_name]
            stats.N += 1.0
            stats.mu_tilde += reward
            stats.pulls += 1
            stats.total_reward += reward
            
            if stats.N > 0:
                stats.mu_hat = stats.mu_tilde / stats.N
            else:
                stats.mu_hat = self.prior_mean
            
            # Update tau for selected arm after adding new observation
            if stats.N > 0:
                stats.tau = min(sigma / np.sqrt(stats.N), self.tau_max)
            else:
                stats.tau = self.tau_max
        
        self.reward_history.append((arm_name, reward))
    
    def update_with_decomposition(
        self,
        arm_name: str,
        normalized_reward: float,
        fitness_component: float,
        diversity_component: float
    ) -> None:
        """Update bandit with reward and track component contributions."""
        self.update(arm_name, normalized_reward)
        
        if arm_name in self.arm_stats:
            self.arm_stats[arm_name].total_fitness_reward += fitness_component
            self.arm_stats[arm_name].total_diversity_reward += diversity_component
    
    def get_state_snapshot(self) -> Dict:
        """Get current state of all arms."""
        return {
            arm_name: {
                'mu_hat': stats.mu_hat,
                'tau': stats.tau,
                'N': stats.N,
                'pulls': stats.pulls,
                'total_reward': stats.total_reward,
                'avg_reward': stats.total_reward / stats.pulls if stats.pulls > 0 else 0.0,
                'total_fitness_reward': stats.total_fitness_reward,
                'total_diversity_reward': stats.total_diversity_reward,
            }
            for arm_name, stats in self.arm_stats.items()
        }
    
    def get_arm_probabilities(self, n_samples: int = 1000) -> Dict[str, float]:
        """Estimate selection probabilities via Monte Carlo."""
        counts = {arm: 0 for arm in self.arms}
        
        for _ in range(n_samples):
            samples = {}
            for arm_name, stats in self.arm_stats.items():
                samples[arm_name] = np.random.normal(stats.mu_hat, stats.tau)
            selected = max(samples, key=samples.get)
            counts[selected] += 1
        
        return {arm: count / n_samples for arm, count in counts.items()}


# ==============================================================================
# BASELINE LLAMEA PROMPT TEMPLATES (v2 with 3 operators)
# ==============================================================================

# Role/System Prompt
ROLE_PROMPT = """You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."""

# Task Prompt
TASK_PROMPT = """The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code to minimize the function value. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.
Give an excellent and novel heuristic algorithm to solve this task.
"""

# Example Prompt
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

# Output Format Prompt
OUTPUT_FORMAT_PROMPT = """Provide the Python code and a one-line description with the main idea (without enters). Give the response in the format:
# Description: <short-description>
# Code:
<code>
"""

# Mutation Instruction - Emphasize novelty and exploration
MUTATION_INSTRUCTION = """Generate a COMPLETELY NEW metaheuristic algorithm that is fundamentally DIFFERENT from all existing algorithms in the population.

Your goal is to BROADEN the search space by exploring novel algorithmic paradigms, mechanisms, and strategies that have NOT been tried yet.

Requirements:
- Use a DIFFERENT optimization mechanism (e.g., if current algorithms use DE, try CMA-ES, Bayesian, surrogate-assisted, etc.)
- Do NOT simply modify parameters or add minor variations
- Create a genuinely novel approach with distinct characteristics"""

# Initialization Prompt (combines task + example + format)
INIT_PROMPT = TASK_PROMPT + EXAMPLE_PROMPT + OUTPUT_FORMAT_PROMPT


# ==============================================================================
# SOLUTION CLASS
# ==============================================================================

class MADASolution:
    """Solution container for MADA-LLAMEA with trace support."""
    
    _id_counter = 0
    
    def __init__(self, code, name, description="", generation=0, parent_ids=None):
        MADASolution._id_counter += 1
        self.id = f"mada_{MADASolution._id_counter:06d}"
        self.code = code
        self.name = name
        self.description = description or name
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.fitness = 0.0
        self.fitness_std = 0.0  # Standard deviation of AUCs
        self.aucs = []
        self.detailed_aucs = [0, 0, 0, 0, 0]
        self.trace = []  # MADA: Optimization trace for diversity
        self.error = ""
        self.operator = ""
        self.feedback = ""  # Baseline LLAMEA style feedback
    
    def get_summary(self):
        """Returns formatted summary for population context (baseline LLAMEA style)."""
        return f"{self.name}: {self.description} (Score: {self.fitness:.4f})"


# ==============================================================================
# MADA OPERATOR v2 (with 3 operators)
# ==============================================================================

class MADAOperatorV2:
    """
    MADA Operator v2 with Discounted Thompson Sampling and 3 operators.
    
    Operators:
    - mutation: Generate a NEW and DIFFERENT algorithm
    - crossover: Combine patterns from two solutions
    - refine: Redesign and refine to address specific weaknesses
    """
    
    def __init__(
        self,
        algorithm_manager,
        crossover_style: str = "implicit",
        discount: float = 0.9,
        tau_max: float = 1.0,
        reward_variance: float = 1.0,  # Normalized rewards have variance ~1
        reward_clamp: float = 1.0,
        use_behavioral_selection: bool = True,
        detailed_feedback: bool = False,
        enable_mutation: bool = True,
        enable_crossover: bool = True,
        enable_refine: bool = True,
        warmup_refine: int = 0,  # Number of generations to force refine
    ):
        self.algorithm_manager = algorithm_manager
        self.crossover_style = crossover_style
        self.child_counter = 0
        self.reward_clamp = reward_clamp
        self.use_behavioral_selection = use_behavioral_selection
        self.detailed_feedback = detailed_feedback
        self.enable_mutation = enable_mutation
        self.enable_crossover = enable_crossover
        self.enable_refine = enable_refine
        self.warmup_refine = warmup_refine  # Force refine for first N generations
        self.current_generation = 0  # Track current generation for warmup
        self.population = []  # Track current population for baseline-style prompts
        self.best_ever = None  # Track best algorithm for baseline-style prompts
        
        # Build arm set based on enabled operators
        arms = []
        if self.enable_mutation:
            arms.append('mutation')
        if self.enable_crossover:
            arms.append('crossover')
        if self.enable_refine:
            arms.append('refine')
        if not arms:
            raise ValueError("At least one operator must be enabled")
        
        # v2: bandit over enabled arms
        self.bandit = DiscountedThompsonSampler(
            arms=arms,
            discount=discount,
            tau_max=tau_max,
            reward_variance=reward_variance,
        )
        
        self._mutation_count = 0
        self._crossover_count = 0
        self._refine_count = 0
        self._last_selection_info = {}
        self._last_algorithm = ""  # Track last algorithm response (baseline style)
    
    def reset_generation_counts(self):
        self._mutation_count = 0
        self._crossover_count = 0
        self._refine_count = 0
    
    def get_operator_stats(self) -> Dict:
        return {
            'mutation_count': self._mutation_count,
            'crossover_count': self._crossover_count,
            'refine_count': self._refine_count,
            'total': self._mutation_count + self._crossover_count + self._refine_count,
        }
    
    def get_bandit_state(self) -> Dict:
        state = self.bandit.get_state_snapshot()
        probs = self.bandit.get_arm_probabilities()
        return {
            'arm_stats': state,
            'selection_probs': probs,
            'total_rounds': self.bandit.total_rounds,
        }
    
    def generate_offspring(
        self,
        parents: List[MADASolution],
        focal_parent: Optional[MADASolution] = None,
    ) -> Tuple[MADASolution, Dict]:
        """Generate offspring using D-TS selected operator."""
        if not parents:
            raise ValueError("MADAOperator requires at least one parent")
        
        operator, theta, snapshot = self.bandit.select_arm()
        
        # Warmup: Force refine for first N generations
        warmup_forced = False
        if self.warmup_refine > 0 and self.current_generation <= self.warmup_refine:
            operator = 'refine'
            warmup_forced = True
        
        # If crossover selected but unavailable (disabled or insufficient parents), fallback to refine
        if operator == 'crossover' and (not self.enable_crossover or len(parents) < 2):
            operator = 'refine'
        # If mutation selected but disabled, fallback to refine
        if operator == 'mutation' and not self.enable_mutation:
            operator = 'refine'
        # If refine disabled (edge case), fallback to mutation or crossover if enabled
        if operator == 'refine' and not self.enable_refine:
            if self.enable_mutation:
                operator = 'mutation'
            elif self.enable_crossover and len(parents) >= 2:
                operator = 'crossover'
            else:
                raise ValueError("No enabled operator available for offspring generation")
        
        self._last_selection_info = {
            'operator': operator,
            'theta_sampled': theta,
            'snapshot': snapshot,
            'warmup_forced': warmup_forced,
        }
        
        if operator == 'mutation':
            self._mutation_count += 1
            parent = focal_parent or random.choice(parents)
            child = self._mutate(parent, all_parents=parents)
        elif operator == 'refine':
            self._refine_count += 1
            parent = focal_parent or random.choice(parents)
            child = self._refine(parent)
        else:  # crossover
            self._crossover_count += 1
            sorted_parents = sorted(parents, key=lambda p: p.fitness, reverse=True)
            parent_a = sorted_parents[0]
            parent_b = self._select_diverse_parent(sorted_parents, parent_a)
            child = self._crossover(parent_a, parent_b)
        
        return child, self._last_selection_info
    
    def _mutate(self, parent: MADASolution, all_parents: List[MADASolution] = None) -> MADASolution:
        """Apply mutation - generate a completely NEW algorithm different from all existing ones."""
        self.child_counter += 1
        
        # Set last_algorithm for conversation context
        self._last_algorithm = f"# Name: {parent.name}\n# Description: {parent.description}\n# Code:\n```python\n{parent.code}\n```"
        
        # Build detailed feedback for parent's performance breakdown (if enabled)
        feedback_parts = []
        if self.detailed_feedback and parent.detailed_aucs and any(parent.detailed_aucs):
            feedback_parts.append("Performance breakdown of the reference algorithm:")
            feedback_parts.append(f"  - Separable functions: {parent.detailed_aucs[0]:.4f}")
            feedback_parts.append(f"  - Low/moderate conditioning: {parent.detailed_aucs[1]:.4f}")
            feedback_parts.append(f"  - High conditioning & unimodal: {parent.detailed_aucs[2]:.4f}")
            feedback_parts.append(f"  - Multimodal (adequate structure): {parent.detailed_aucs[3]:.4f}")
            feedback_parts.append(f"  - Multimodal (weak structure): {parent.detailed_aucs[4]:.4f}")
        if parent.error:
            feedback_parts.append(f"Previous error to avoid: {parent.error}")
        
        detailed_feedback = "\n".join(feedback_parts) if feedback_parts else ""
        
        # Build list of existing algorithms to avoid duplicating
        existing_algorithms = []
        if all_parents:
            for p in all_parents[:5]:  # Show top 5 algorithms
                existing_algorithms.append(f"  - {p.name}: {p.description} (AOCC: {p.fitness:.2f})")
        existing_summary = "\n".join(existing_algorithms) if existing_algorithms else "None yet."
        
        # Build mutation prompt with exploration emphasis
        prompt = f"""{MUTATION_INSTRUCTION}

**EXISTING ALGORITHMS (DO NOT duplicate these approaches):**
{existing_summary}

{detailed_feedback}

Generate a completely NEW algorithm with a DIFFERENT mechanism. Give the response in the format:
# Description: <short-description>
# Code: <code>"""
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            description = self._extract_description(message)
            class_name = self._extract_class_name(code, f"{parent.name}Mut{self.child_counter}")
            
            # Update last_algorithm with new response
            self._last_algorithm = message
            
            solution = MADASolution(
                code=code,
                name=class_name,
                description=description,
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "mutation"
            return solution
            
        except Exception as e:
            solution = MADASolution(
                code=parent.code,
                name=f"{parent.name}MutFail{self.child_counter}",
                description=f"Mutation failed: {e}",
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "mutation_failed"
            solution.error = str(e)
            return solution
    
    def _refine(self, parent: MADASolution) -> MADASolution:
        """Apply refine - with detailed feedback to guide improvements."""
        self.child_counter += 1
        
        # Set last_algorithm for conversation context
        self._last_algorithm = f"# Name: {parent.name}\n# Description: {parent.description}\n# Code:\n```python\n{parent.code}\n```"
        
        # Build detailed feedback for parent's performance breakdown (if enabled)
        feedback_parts = []
        if self.detailed_feedback and parent.detailed_aucs and any(parent.detailed_aucs):
            feedback_parts.append("Performance breakdown:")
            feedback_parts.append(f"  - Separable functions: {parent.detailed_aucs[0]:.4f}")
            feedback_parts.append(f"  - Low/moderate conditioning: {parent.detailed_aucs[1]:.4f}")
            feedback_parts.append(f"  - High conditioning & unimodal: {parent.detailed_aucs[2]:.4f}")
            feedback_parts.append(f"  - Multimodal (adequate structure): {parent.detailed_aucs[3]:.4f}")
            feedback_parts.append(f"  - Multimodal (weak structure): {parent.detailed_aucs[4]:.4f}")
            
            # Identify weakest area(s) for targeted improvement
            min_auc = min(parent.detailed_aucs)
            weakest_idx = parent.detailed_aucs.index(min_auc)
            weakness_names = ["Separable functions", "Low/moderate conditioning", 
                            "High conditioning & unimodal", "Multimodal (adequate structure)", 
                            "Multimodal (weak structure)"]
            feedback_parts.append(f"\n**FOCUS AREA**: The algorithm struggles most with {weakness_names[weakest_idx]} (AOCC: {min_auc:.4f}). Consider improving this aspect.")
        
        detailed_feedback = "\n".join(feedback_parts) if feedback_parts else ""
        
        # Build prompt with detailed feedback
        if parent.error:
            prompt = f"""The last proposed algorithm {parent.name} got an error: {parent.error}.

{detailed_feedback}

Either refine or redesign to fix the error and improve the algorithm. Give the response in the format:
# Description: <short-description>
# Code: <code>"""
        else:
            prompt = f"""The last proposed algorithm {parent.name} got an average Area over the convergence curve (AOCC, 1.0 is the best) of {parent.fitness:.2f}, and a standard deviation of {parent.fitness_std:.2f}.

{detailed_feedback}

Either refine or redesign to improve the algorithm, especially focusing on the weak areas identified above. Give the response in the format:
# Description: <short-description>
# Code: <code>"""
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            description = self._extract_description(message)
            class_name = self._extract_class_name(code, f"{parent.name}Ref{self.child_counter}")
            
            # Update last_algorithm with new response
            self._last_algorithm = message
            
            solution = MADASolution(
                code=code,
                name=class_name,
                description=description,
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "refine"
            return solution
            
        except Exception as e:
            solution = MADASolution(
                code=parent.code,
                name=f"{parent.name}RefFail{self.child_counter}",
                description=f"Refine failed: {e}",
                generation=parent.generation + 1,
                parent_ids=[parent.id]
            )
            solution.operator = "refine_failed"
            solution.error = str(e)
            return solution
    
    def _crossover(self, parent_a: MADASolution, parent_b: MADASolution) -> MADASolution:
        """Apply crossover with detailed feedback for both parents."""
        self.child_counter += 1
        
        # Crossover doesn't use last_algorithm (fresh conversation)
        self._last_algorithm = ""
        
        # Build detailed feedback for Parent A (if enabled)
        feedback_a_parts = []
        if self.detailed_feedback and parent_a.detailed_aucs and any(parent_a.detailed_aucs):
            feedback_a_parts.append("Performance breakdown:")
            feedback_a_parts.append(f"  - Separable functions: {parent_a.detailed_aucs[0]:.4f}")
            feedback_a_parts.append(f"  - Low/moderate conditioning: {parent_a.detailed_aucs[1]:.4f}")
            feedback_a_parts.append(f"  - High conditioning & unimodal: {parent_a.detailed_aucs[2]:.4f}")
            feedback_a_parts.append(f"  - Multimodal (adequate structure): {parent_a.detailed_aucs[3]:.4f}")
            feedback_a_parts.append(f"  - Multimodal (weak structure): {parent_a.detailed_aucs[4]:.4f}")
        feedback_a = "\n".join(feedback_a_parts) if feedback_a_parts else ""
        
        # Build detailed feedback for Parent B (if enabled)
        feedback_b_parts = []
        if self.detailed_feedback and parent_b.detailed_aucs and any(parent_b.detailed_aucs):
            feedback_b_parts.append("Performance breakdown:")
            feedback_b_parts.append(f"  - Separable functions: {parent_b.detailed_aucs[0]:.4f}")
            feedback_b_parts.append(f"  - Low/moderate conditioning: {parent_b.detailed_aucs[1]:.4f}")
            feedback_b_parts.append(f"  - High conditioning & unimodal: {parent_b.detailed_aucs[2]:.4f}")
            feedback_b_parts.append(f"  - Multimodal (adequate structure): {parent_b.detailed_aucs[3]:.4f}")
            feedback_b_parts.append(f"  - Multimodal (weak structure): {parent_b.detailed_aucs[4]:.4f}")
        feedback_b = "\n".join(feedback_b_parts) if feedback_b_parts else ""
        
        # Build error analysis section
        error_parts = []
        if parent_a.error or parent_b.error:
            error_parts.append("\n**ERROR ANALYSIS - Avoid these issues in the new solution:**")
            if parent_a.error:
                error_parts.append(f"  - Parent A error: {parent_a.error}")
            if parent_b.error:
                error_parts.append(f"  - Parent B error: {parent_b.error}")
            error_parts.append("  - Ensure proper bounds checking, avoid division by zero, handle edge cases.")
        error_analysis = "\n".join(error_parts) if error_parts else ""
        
        # Crossover prompt with detailed feedback
        prompt = f"""Two parent solutions have been selected:

**Parent A: {parent_a.name}** (AOCC: {parent_a.fitness:.2f})
```python
{parent_a.code}
```
{feedback_a}

**Parent B: {parent_b.name}** (AOCC: {parent_b.fitness:.2f})
```python
{parent_b.code}
```
{feedback_b}
{error_analysis}

Create a better offspring by combining the STRENGTHS from both parent algorithms. Use the performance breakdown to identify which parent is better at which function types, and combine their best strategies.

Give the response in the format:
# Description: <short-description>
# Code: <code>"""
        
        try:
            # Crossover does NOT include best algorithm (include_best=False)
            message = self._call_llm(prompt, include_best=False)
            code = self._extract_code(message)
            description = self._extract_description(message)
            class_name = self._extract_class_name(code, f"Hybrid{self.child_counter}")
            
            solution = MADASolution(
                code=code,
                name=class_name,
                description=description,
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=[parent_a.id, parent_b.id]
            )
            solution.operator = "crossover"
            return solution
            
        except Exception as e:
            solution = MADASolution(
                code=parent_a.code,
                name=f"{parent_a.name}XoFail{self.child_counter}",
                description=f"Crossover failed: {e}",
                generation=max(parent_a.generation, parent_b.generation) + 1,
                parent_ids=[parent_a.id, parent_b.id]
            )
            solution.operator = "crossover_failed"
            solution.error = str(e)
            return solution
    
    def _select_diverse_parent(self, sorted_parents, parent_a):
        """
        Select a diverse parent for crossover.
        """
        if len(sorted_parents) < 2:
            return parent_a
        
        # Use behavioral selection if enabled and traces are available
        if self.use_behavioral_selection:
            parent_a_sel, parent_b = select_behavioral_parents(
                sorted_parents,
                enabled=True,
                trace_attr='trace',
                fitness_attr='fitness'
            )
            
            # Log selection info
            self._last_selection_info['behavioral_selection'] = True
            if hasattr(parent_a, 'trace') and hasattr(parent_b, 'trace'):
                if parent_a.trace and parent_b.trace:
                    dist = calculate_trace_distance(parent_a.trace, parent_b.trace)
                    self._last_selection_info['trace_distance'] = dist
            
            return parent_b
        
        # Fallback: Code-based diversity (original behavior)
        self._last_selection_info['behavioral_selection'] = False
        for candidate in sorted_parents[1:]:
            if (candidate.code or "") != (parent_a.code or ""):
                return candidate
        return sorted_parents[1]
    
    def _call_llm(self, prompt: str, include_best: bool = True) -> str:
        """Call LLM with TRUE baseline LLAMEA conversation structure."""
        # Build conversation history like baseline LLAMEA
        messages = [
            {"role": "system", "content": ROLE_PROMPT},
            {"role": "user", "content": INIT_PROMPT},
        ]
        
        # Add population context (current population summary)
        if self.population:
            pop_summary = "\n".join([
                f"  - {ind.name}: {ind.description} (fitness={ind.fitness:.4f})"
                for ind in self.population[:5]
            ])
            messages.append({"role": "user", "content": f"Current population:\n{pop_summary}"})
        
        # Add best algorithm so far (like baseline LLAMEA) - skip for crossover
        if include_best and self.best_ever and self.best_ever.fitness > 0:
            best_context = f"""The best so far proposed algorithm got an average AOCC of {self.best_ever.fitness:.2f} and the code was as follows:
{self.best_ever.code}"""
            messages.append({"role": "user", "content": best_context})
        
        # Add last algorithm as assistant response (for mutation/refine only)
        if self._last_algorithm:
            messages.append({"role": "assistant", "content": self._last_algorithm})
        
        # Add current prompt (mutation/refine/crossover request)
        messages.append({"role": "user", "content": prompt})
        
        # Log the conversation
        for msg in messages:
            self.algorithm_manager.logger.log_conversation(f"\n[{msg['role']}]\n{msg['content']}")
        
        call_kwargs = {
            "model": self.algorithm_manager.ai_model,
            "messages": messages,
            "temperature": 0.8,
        }
        if self.algorithm_manager.max_tokens:
            call_kwargs["max_tokens"] = self.algorithm_manager.max_tokens
        
        response = self.algorithm_manager.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        
        self.algorithm_manager.logger.log_conversation(f"\n[assistant]\n{message}")
        
        return message
    
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
    
    def _extract_description(self, message: str) -> str:
        """Extract description from LLM response (baseline LLAMEA style)."""
        pattern = r"#\s*Description:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "No description provided"
    
    def _construct_population_summary(self, parents: List[MADASolution]) -> str:
        """Generate population summary for baseline LLAMEA style prompts."""
        return "\n".join([ind.get_summary() for ind in parents])


# ==============================================================================
# EVALUATION WITH TRACE
# ==============================================================================

def evaluate_algorithm_with_trace(algorithm_code, algorithm_name, eval_budget):
    """
    Evaluate algorithm on BBOB and return optimization trace.
    
    Returns:
        tuple: (aucs, detailed_aucs, aggregated_trace, error)
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
        all_traces = []
        
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)
                
                for rep in range(3):
                    np.random.seed(rep)
                    
                    # Create trace collector
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
                        
                        algorithm(trace_collector)
                        
                    except OverBudgetException:
                        pass
                    except Exception as e:
                        return [], [0, 0, 0, 0, 0], [], str(e)
                    
                    auc = correct_aoc(problem, l2, eval_budget)
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    
                    # Collect trace
                    run_trace = trace_collector.get_trace()
                    all_traces.append(run_trace)
                    
                    l2.reset(problem)
                    problem.reset()
            
            # Safely aggregate per-group AUCs (avoid mean of empty slice)
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
        
        # Aggregate traces
        aggregated_trace = aggregate_traces(all_traces, eval_budget)
        
        return aucs, detailed_aucs, aggregated_trace, ""

    except KeyboardInterrupt:
        return [], [0, 0, 0, 0, 0], [], "KeyboardInterrupt"
    except Exception as e:
        return [], [0, 0, 0, 0, 0], [], str(e)
    finally:
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice


# ==============================================================================
# LOGGING HELPERS
# ==============================================================================

def log_mada_offspring(explogger, attempt, record):
    """Log MADA offspring metadata."""
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "mada_offspring.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": attempt, **record}, default=str) + "\n")


def log_bandit_snapshot(explogger, generation, bandit_state):
    """Log bandit state snapshot."""
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "bandit_snapshots.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"generation": generation, **bandit_state}, default=str) + "\n")


# ==============================================================================
# SELECTION
# ==============================================================================

def selection(population, n_parents, elitism=True):
    """
    Select the best individuals from the population.
    
    This matches the baseline LLaMEA selection function exactly:
    - Always deterministic best-first selection
    - The elitism logic (μ+λ vs μ,λ) is handled by the CALLING code,
      which passes either parents+offspring or just offspring.
    
    Args:
        population: List of individuals to select from
        n_parents: Number of individuals to select
        elitism: Parameter kept for API compatibility, but the actual
                 elitism logic happens in the calling code.
                 
    Returns:
        List of top n_parents individuals sorted by fitness (descending).
    """
    if not population:
        return []
    
    # Simple deterministic selection: sort by fitness and take top n
    # This matches baseline LLaMEA exactly
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    return sorted_pop[:n_parents]


# ==============================================================================
# MAIN EVOLUTIONARY LOOP
# ==============================================================================

def run_mada_evolutionary_mode(
    algorithm_manager,
    mada_operator,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
):
    """
    MADA-LLAMEA v2 main evolutionary loop with 3 operators.
    """
    
    # MADA State Initialization
    history_traces = []
    global_min_fitness = float('inf')
    global_max_fitness = float('-inf')
    reward_normalizer = RewardNormalizer(warmup_period=5)
    alpha_scheduler = AlphaScheduler(
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        t_max=max(generations - 1, 1),
        schedule=args.alpha_schedule
    )
    
    population = []
    best_ever = None
    api_calls = 0
    generation = 0
    
    # Phase 1: Initialization
    print(f"\n{'='*60}")
    print(f"MADA-LLAMEA v2 INITIALIZATION: Generating {n_parents} parents")
    print('='*60)
    
    seen_hashes = set()
    for i in range(n_parents):
        if api_calls >= args.budget:
            break
            
        print(f"\nInitializing Parent {i+1}/{n_parents} (API call {api_calls+1})")
        
        try:
            retries = 0
            while retries <= 3:
                message = algorithm_manager.fetch_algorithm()
                
                # Extract code
                pattern = r"```(?:python)?\n(.*?)\n```"
                match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
                if not match:
                    retries += 1
                    continue
                algorithm_code = match.group(1)
                
                # Extract class name
                class_match = re.search(r"class\s+(\w+)", algorithm_code)
                if not class_match:
                    retries += 1
                    continue
                algorithm_name = class_match.group(1)
                
                # Extract description
                desc_pattern = r"#\s*Description:\s*(.+?)(?:\n|$)"
                desc_match = re.search(desc_pattern, message, re.IGNORECASE)
                algorithm_description = desc_match.group(1).strip() if desc_match else algorithm_name

                code_hash = hashlib.sha256(algorithm_code.encode()).hexdigest()
                if code_hash in seen_hashes:
                    retries += 1
                    continue

                solution = MADASolution(
                    code=algorithm_code,
                    name=algorithm_name,
                    description=algorithm_description,
                    generation=generation,
                )
                solution.operator = "init"

                print(f"  Evaluating {algorithm_name}...")
                aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
                    algorithm_code, algorithm_name, args.eval_budget
                )

                if error:
                    solution.fitness = 0.0
                    solution.error = error
                    solution.feedback = f"The algorithm got an error: {error}. "
                    print(f"  Error: {error}")
                    retries += 1
                    continue
                else:
                    solution.fitness = float(np.mean(aucs))
                    solution.fitness_std = float(np.std(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    solution.trace = trace
                    solution.feedback = ""  # No feedback for successful execution
                    seen_hashes.add(code_hash)
                    print(f"  Fitness: {solution.fitness:.4f}")
                    
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
    print(f"\nInitialization complete. Best: {best_ever.fitness:.4f}")
    
    # Phase 2: Evolutionary Loop
    for gen in range(1, generations + 1):
        if api_calls >= args.budget:
            print(f"\nBudget exhausted at generation {gen}")
            break
        
        generation = gen
        mada_operator.current_generation = gen  # Track for warmup
        current_alpha = alpha_scheduler.get_alpha(generation - 1)
        
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{generations}")
        print(f"API Calls: {api_calls}/{args.budget}")
        print(f"Best: {best_ever.fitness:.4f} ({best_ever.name})")
        print(f"MADA α: {current_alpha:.3f}")
        
        bandit_state = mada_operator.get_bandit_state()
        probs = bandit_state['selection_probs']
        # Print only enabled arms
        prob_strs = []
        for arm in mada_operator.bandit.arms:
            prob_strs.append(f"{arm[:3]}={probs[arm]:.1%}")
        print("D-TS: " + ", ".join(prob_strs))
        
        parents = selection(population, n_parents, elitism=args.elitism)
        best_parent_fitness = max((p.fitness for p in parents), default=0.0)
        
        # Set population and best_ever for baseline LLAMEA style prompts
        mada_operator.population = parents
        mada_operator.best_ever = best_ever
        
        mada_operator.reset_generation_counts()
        offspring = []
        
        for i in range(n_offspring):
            if api_calls >= args.budget:
                break
            
            parent = random.choice(parents)
            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")

            try:
                child, selection_info = mada_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                )
                
                operator = selection_info['operator']
                warmup_str = " [WARMUP]" if selection_info.get('warmup_forced', False) else ""
                print(f"  D-TS: {operator} (θ={selection_info['theta_sampled']:.3f}){warmup_str}")
                print(f"  Evaluating {child.name}...")
                
                aucs, detailed_aucs, trace, error = evaluate_algorithm_with_trace(
                    child.code, child.name, args.eval_budget
                )

                if error:
                    child.fitness = 0.0
                    child.error = error
                    child.trace = []
                    # Build feedback (baseline LLAMEA style)
                    child.feedback = f"The algorithm got an error: {error}. "
                    print(f"  Error: {error}")
                else:
                    child.fitness = float(np.mean(aucs))
                    child.fitness_std = float(np.std(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    child.trace = trace
                    # Build feedback (baseline LLAMEA style - empty for successful execution)
                    child.feedback = ""
                    print(f"  Fitness: {child.fitness:.4f}")
                    
                    if trace:
                        global_min_fitness = min(global_min_fitness, min(trace))
                        global_max_fitness = max(global_max_fitness, max(trace))
                    
                    if child.fitness > best_ever.fitness:
                        print(f"  *** NEW BEST! {child.fitness:.4f} ***")
                        best_ever = child

                # MADA: Calculate NN-Dist
                if trace and history_traces:
                    nn_dist = calculate_nn_dist(trace, history_traces, global_min_fitness, global_max_fitness)
                else:
                    nn_dist = 1.0
                
                print(f"  NN-Dist: {nn_dist:.4f}")
                
                # Determine parent fitness from the actual parent(s) of this child
                parent_fitness_for_reward = best_parent_fitness
                if child.parent_ids:
                    parent_lookup = {p.id: p.fitness for p in parents}
                    parent_fitness_values = [
                        parent_lookup.get(pid, None) for pid in child.parent_ids
                    ]
                    parent_fitness_values = [pf for pf in parent_fitness_values if pf is not None]
                    if parent_fitness_values:
                        parent_fitness_for_reward = max(parent_fitness_values)

                # MADA: Composite reward
                reward, fitness_delta, diversity_bonus = compute_composite_reward(
                    child_fitness=child.fitness,
                    parent_fitness=parent_fitness_for_reward,
                    nn_dist=nn_dist,
                    alpha=current_alpha,
                    error=bool(error),
                    clamp=args.reward_clamp
                )
                
                # MADA: Normalize reward
                normalized_reward = reward_normalizer.normalize(reward)
                
                print(f"  Reward: raw={reward:+.4f}, norm={normalized_reward:+.4f}")
                print(f"    (Δfit={fitness_delta:+.4f}, div={diversity_bonus:+.4f})")
                
                # Update bandit with decomposition
                mada_operator.bandit.update_with_decomposition(
                    operator, normalized_reward, fitness_delta, diversity_bonus
                )
                
                # Add trace to history
                if trace:
                    history_traces.append(trace)

                # Log
                offspring_record = {
                    'operator': child.operator,
                    'fitness': child.fitness,
                    'parent_fitness': parent_fitness_for_reward,
                    'nn_dist': nn_dist,
                    'alpha': current_alpha,
                    'raw_reward': reward,
                    'normalized_reward': normalized_reward,
                    'fitness_delta': fitness_delta,
                    'diversity_bonus': diversity_bonus,
                    'theta_sampled': selection_info['theta_sampled'],
                    'parent_ids': child.parent_ids,
                    'generation': generation,
                    'error': child.error,
                }
                log_mada_offspring(explogger, api_calls, offspring_record)
                
                offspring.append(child)
                explogger.log_code(api_calls, child.name, child.code)
                explogger.log_aucs(api_calls, aucs if aucs else [0])
                
                api_calls += 1

            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
        
        # Selection
        if args.elitism:
            combined = parents + offspring
            population = selection(combined, n_parents, elitism=True)
        else:
            population = selection(offspring, n_parents, elitism=False)

        # Log bandit state
        bandit_state = mada_operator.get_bandit_state()
        bandit_state['alpha'] = current_alpha
        bandit_state['reward_stats'] = reward_normalizer.get_stats()
        bandit_state['history_size'] = len(history_traces)
        log_bandit_snapshot(explogger, generation, bandit_state)
        
        stats = mada_operator.get_operator_stats()
        print(f"Operators: {stats['mutation_count']} mut, {stats['crossover_count']} xo, {stats['refine_count']} ref")
    
    # Final Summary
    print(f"\n{'='*60}")
    print("MADA-LLAMEA v2 COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations: {generation}")
    print(f"Best: {best_ever.name} ({best_ever.fitness:.4f})")
    print(f"Traces collected: {len(history_traces)}")
    
    final_state = mada_operator.get_bandit_state()
    print(f"\nFinal Bandit (3 operators):")
    for arm, s in final_state['arm_stats'].items():
        print(f"  {arm}: pulls={s['pulls']}, μ̂={s['mu_hat']:.3f}, fit_r={s['total_fitness_reward']:.3f}, div_r={s['total_diversity_reward']:.3f}")
    
    # Save best
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
        description='MADA-LLAMEA v2: Multi-Adaptive Diverse Algorithm with 3 Operators',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # API Configuration
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--base-url', type=str, help='Custom API base URL')
    parser.add_argument('--max-tokens', type=int, help='Max tokens')
    
    # Model Configuration  
    parser.add_argument('--model', type=str, default='google/gemini-2.5-flash', help='AI model')
    parser.add_argument('--experiment-name', type=str, default='mada-v2-experiment', help='Experiment name')
    
    # Experiment Configuration
    parser.add_argument('--budget', type=int, default=100, help='API budget')
    parser.add_argument('--eval-budget', type=int, default=10000, help='Eval budget per algorithm')
    parser.add_argument('--elitism', action='store_true', help='Enable (μ+λ) elitism')
    
    # Population Configuration
    parser.add_argument('--n-parents', type=int, default=4, help='Number of parents')
    parser.add_argument('--n-offspring', type=int, default=16, help='Offspring per generation')
    parser.add_argument('--generations', type=int, default=None, help='Fixed generations')
    parser.add_argument('--evolutionary-mode', action='store_true', help='Enable evolutionary mode')
    
    # D-TS Configuration
    parser.add_argument('--discount', type=float, default=0.9, help='D-TS discount γ')
    parser.add_argument('--tau-max', type=float, default=1.0, help='D-TS τ_max')
    parser.add_argument('--reward-variance', type=float, default=1.0, help='D-TS reward variance')
    parser.add_argument('--reward-clamp', type=float, default=1.0, help='Reward clamp')
    parser.add_argument('--disable-mutation', action='store_true', help='Disable mutation operator')
    parser.add_argument('--disable-crossover', action='store_true', help='Disable crossover operator')
    parser.add_argument('--warmup-refine', type=int, default=0,
                       help='Force refine operator for first N generations (default: 0, disabled)')
    
    # MADA Configuration
    parser.add_argument('--alpha-start', type=float, default=0.5, help='Initial diversity weight α')
    parser.add_argument('--alpha-end', type=float, default=0.0, help='Final diversity weight α')
    parser.add_argument('--alpha-schedule', type=str, default='linear',
                       choices=['linear', 'exponential', 'cosine', 'constant'],
                       help='Alpha decay schedule')
    
    # Behavioral Selection (Diversity-Based Crossover)
    parser.add_argument('--behavioral-selection', dest='behavioral_selection',
                       action='store_true', default=True,
                       help='Enable trace-based behavioral parent selection for crossover (default: enabled)')
    parser.add_argument('--no-behavioral-selection', dest='behavioral_selection',
                       action='store_false',
                       help='Disable behavioral selection, use fitness-based selection instead')
    
    # Detailed Feedback
    parser.add_argument('--detailed-feedback', action='store_true', default=False,
                       help='Enable detailed per-function-group performance feedback in prompts')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AIML_KEY")
    if not api_key:
        raise ValueError("API key required")
    
    base_url = args.base_url or os.getenv("BASE_URL")
    
    # Setup
    ai_model = args.model
    experiment_suffix = f"{ai_model}-{args.experiment_name}"
    if args.evolutionary_mode:
        experiment_suffix += "-evolutionary"
    if args.elitism:
        experiment_suffix += "-elitism"
    
    explogger = ExperimentLogger(experiment_suffix)
    algorithm_manager = AlgorithmManager(
        api_key, explogger, ai_model,
        elitism=args.elitism,
        detailed_feedback=False,
        base_url=base_url,
        max_tokens=args.max_tokens
    )

    if args.evolutionary_mode:
        mada_operator = MADAOperatorV2(
            algorithm_manager,
            discount=args.discount,
            tau_max=args.tau_max,
            reward_variance=args.reward_variance,
            reward_clamp=args.reward_clamp,
            use_behavioral_selection=args.behavioral_selection,
            detailed_feedback=args.detailed_feedback,
            enable_mutation=not args.disable_mutation,
            enable_crossover=not args.disable_crossover,
            enable_refine=True,
            warmup_refine=args.warmup_refine,
        )
        
        if args.generations is not None:
            generations = args.generations
        else:
            generations = max(1, (args.budget - args.n_parents) // args.n_offspring)
        
        print(f"\nStarting MADA-LLAMEA v2 (3 Operators):")
        print(f"  Model: {ai_model}")
        print(f"  Budget: {args.budget}")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
        print(f"  Generations: {generations}")
        print(f"  D-TS: γ={args.discount}, τ_max={args.tau_max}")
        print(f"  MADA: α={args.alpha_start}→{args.alpha_end} ({args.alpha_schedule})")
        print(f"  Operators: mutation, crossover, refine")
        print(f"  Warmup Refine: {args.warmup_refine} generations" if args.warmup_refine > 0 else "  Warmup Refine: disabled")
        print(f"  Behavioral Selection: {'enabled' if args.behavioral_selection else 'disabled'}")
        print(f"  Detailed Feedback: {'enabled' if args.detailed_feedback else 'disabled'}")
        print("-" * 60)
        
        run_mada_evolutionary_mode(
            algorithm_manager,
            mada_operator,
            explogger,
            args,
            generations,
            args.n_parents,
            args.n_offspring,
        )
    else:
        print("ERROR: MADA-LLAMEA v2 requires --evolutionary-mode flag")
        return
    
    print(f"\nResults saved in: {explogger.dirname}")


if __name__ == "__main__":
    main()


