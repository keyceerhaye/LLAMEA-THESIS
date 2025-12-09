"""
MADA-LLAMEA: Multi-Adaptive Diverse Algorithm with LLM Evolutionary Algorithm

This extends main-thesis-eoh-dts.py with behavioral diversity rewards based on
the MADA5.0 design document. Key enhancements:

1. Trace Logging: Evaluation captures optimization trajectories (best-so-far)
2. NN-Dist: Nearest-neighbor distance measures behavioral diversity
3. Composite Reward: R = ΔFitness + α × NN-Dist
4. Alpha Scheduler: Diversity weight decays from exploration to exploitation
5. Reward Normalization: Running standardization for stable bandit learning

Mathematical Foundation:
- Composite Reward: R_t = (Fit_child - Fit_parent) + α × d(trace_new, trace_history)
- Alpha Decay: α_t = α_start × (1 - t/T_max)
- NN-Dist: d(x,y) = sqrt(sum((x[t] - y[t])²)) / sqrt(B)

References:
- MADA 5.0: Design Document
- van Stein et al. (2025): Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery
- Qi, Guo, Zhu (2025): Discounted Thompson Sampling

Usage:
    python main-thesis-mada.py --evolutionary-mode --elitism --budget 50
    python main-thesis-mada.py --evolutionary-mode --elitism --alpha-start 0.7 --alpha-schedule cosine --budget 100
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
    RewardNormalizer,
    AlphaScheduler,
    compute_composite_reward,
)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# DISCOUNTED THOMPSON SAMPLING (from main-thesis-eoh-dts.py)
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
    Extended for MADA with reward component tracking.
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
        
        self.arm_stats: Dict[str, ArmStatistics] = {}
        for arm in arms:
            self.arm_stats[arm] = ArmStatistics(
                name=arm,
                N=1.0,
                mu_tilde=prior_mean,
                mu_hat=prior_mean,
                tau=tau_max,
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
        # Apply discount to ALL arms
        for stats in self.arm_stats.values():
            stats.N = self.discount * stats.N
            stats.mu_tilde = self.discount * stats.mu_tilde
        
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
            
            if stats.N > 0:
                stats.tau = min(1.0 / np.sqrt(stats.N), self.tau_max)
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
# EoH PROMPT TEMPLATES
# ==============================================================================

EOH_SYSTEM_PROMPT = """You are a highly skilled computer scientist specializing in evolutionary algorithm design.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.
Do not use hyped nature-inspired algorithms such as Harmony Search, Grey Wolf, Firefly, Whale optimizer etc.
Focus on well-established techniques like Differential Evolution, CMA-ES, Evolution Strategies, or novel hybrid approaches."""

EOH_INIT_PROMPT = """
The optimization algorithm should handle a wide range of tasks, evaluated on the BBOB test suite of 24 noiseless functions. 
Write the optimization algorithm in Python code with an `__init__(self, budget)` function and `def __call__(self, func)`.
The func() can only be called as many times as the budget allows. Search space: [-5.0, 5.0], dimensionality: 5.

Example (simple random search):
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

Give an excellent and novel heuristic algorithm. Format:
# Name: <classname>
# Code: <code>
"""

EOH_CROSSOVER_IMPLICIT_PROMPT = """I am designing metaheuristic algorithms for black-box optimization (BBOB benchmark, 5D, bounds [-5, 5]).

Here are two high-scoring solutions:

Solution 1 (Score: {score_a:.4f}):
```python
{code_a}
```

Solution 2 (Score: {score_b:.4f}):
```python
{code_b}
```

Generate a Solution 3 that achieves an even higher score by combining patterns from both.
Follow the same interface: __init__(self, budget) and __call__(self, func)

Format:
# Name: <classname>
# Code: <code>
"""

EOH_MUTATION_PROMPT = """
The optimization algorithm should handle BBOB test suite functions. Write Python code with `__init__(self, budget)` and `def __call__(self, func)`.
Budget: 10000 evaluations. Search space: [-5.0, 5.0], dim: 5.

Reference algorithm (score: {score:.4f}):
```python
{code}
```

{feedback}

Design a NEW and DIFFERENT algorithm for a higher score. Use techniques like:
- Differential Evolution variants (DE/rand, SHADE, L-SHADE)
- Evolution Strategies (CMA-ES, self-adaptive ES)
- Particle Swarm Optimization variants
- Hybrid approaches, novel adaptive mechanisms

Format:
# Name: <classname>
# Code: <code>
"""


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
        self.aucs = []
        self.detailed_aucs = [0, 0, 0, 0, 0]
        self.trace = []  # MADA: Optimization trace for diversity
        self.error = ""
        self.operator = ""


# ==============================================================================
# MADA OPERATOR
# ==============================================================================

class MADAOperator:
    """
    MADA Operator with Discounted Thompson Sampling and behavioral diversity.
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
        self.algorithm_manager = algorithm_manager
        self.crossover_style = crossover_style
        self.child_counter = 0
        self.reward_clamp = reward_clamp
        
        self.bandit = DiscountedThompsonSampler(
            arms=['mutation', 'crossover'],
            discount=discount,
            tau_max=tau_max,
            reward_variance=reward_variance,
        )
        
        self._mutation_count = 0
        self._crossover_count = 0
        self._last_selection_info = {}
    
    def reset_generation_counts(self):
        self._mutation_count = 0
        self._crossover_count = 0
    
    def get_operator_stats(self) -> Dict:
        return {
            'mutation_count': self._mutation_count,
            'crossover_count': self._crossover_count,
            'total': self._mutation_count + self._crossover_count,
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
        
        if len(parents) < 2:
            operator = 'mutation'
            theta = 0.0
            snapshot = {}
        else:
            operator, theta, snapshot = self.bandit.select_arm()
        
        self._last_selection_info = {
            'operator': operator,
            'theta_sampled': theta,
            'snapshot': snapshot,
        }
        
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
    
    def _mutate(self, parent: MADASolution) -> MADASolution:
        """Apply mutation."""
        self.child_counter += 1
        
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
        
        feedback = "\n".join(feedback_parts) if feedback_parts else "No specific feedback."
        
        prompt = EOH_MUTATION_PROMPT.format(
            score=parent.fitness,
            code=parent.code,
            feedback=feedback
        )
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            class_name = self._extract_class_name(code, f"{parent.name}Mut{self.child_counter}")
            
            solution = MADASolution(
                code=code,
                name=class_name,
                description=class_name,
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
    
    def _crossover(self, parent_a: MADASolution, parent_b: MADASolution) -> MADASolution:
        """Apply crossover."""
        self.child_counter += 1
        
        prompt = EOH_CROSSOVER_IMPLICIT_PROMPT.format(
            score_a=parent_a.fitness,
            code_a=parent_a.code,
            score_b=parent_b.fitness,
            code_b=parent_b.code
        )
        
        try:
            message = self._call_llm(prompt)
            code = self._extract_code(message)
            class_name = self._extract_class_name(code, f"Hybrid{self.child_counter}")
            
            solution = MADASolution(
                code=code,
                name=class_name,
                description=class_name,
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
        for candidate in sorted_parents[1:]:
            if (candidate.code or "") != (parent_a.code or ""):
                return candidate
        return sorted_parents[1] if len(sorted_parents) > 1 else parent_a
    
    def _call_llm(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": EOH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        self.algorithm_manager.logger.log_conversation(f"\n[MADA Prompt]\n{prompt}\n")
        
        call_kwargs = {
            "model": self.algorithm_manager.ai_model,
            "messages": messages,
            "temperature": 0.8,
        }
        if self.algorithm_manager.max_tokens:
            call_kwargs["max_tokens"] = self.algorithm_manager.max_tokens
        
        response = self.algorithm_manager.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        
        self.algorithm_manager.logger.log_conversation(f"\n[MADA Response]\n{message}\n")
        
        return message
    
    def _extract_code(self, message: str) -> str:
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        raise NoCodeException("No code block found")
    
    def _extract_class_name(self, code: str, default: str) -> str:
        for line in code.splitlines():
            if line.strip().startswith("class "):
                match = re.search(r"class\s+(\w+)", line)
                if match:
                    return match.group(1)
        return default


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
    """Select best individuals from population."""
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
    
    if not sorted_pop:
        return []
    
    selected = [sorted_pop[0]]
    if n_parents == 1:
        return selected
    
    best_code = sorted_pop[0].code or ""
    
    for ind in sorted_pop[1:]:
        if (ind.code or "") != best_code:
            selected.append(ind)
            break
    
    if len(selected) < 2 and len(sorted_pop) > 1:
        selected.append(sorted_pop[1])
    
    seen_codes = {s.code or "" for s in selected}
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        if (ind.code or "") not in seen_codes:
            seen_codes.add(ind.code or "")
            selected.append(ind)
    
    for ind in sorted_pop:
        if len(selected) >= n_parents:
            break
        if ind not in selected:
            selected.append(ind)
    
    return selected


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
    MADA-LLAMEA main evolutionary loop with behavioral diversity rewards.
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
    print(f"MADA-LLAMEA INITIALIZATION: Generating {n_parents} parents")
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

                solution = MADASolution(
                    code=algorithm_code,
                    name=algorithm_name,
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
                    print(f"  Error: {error}")
                    retries += 1
                    continue
                else:
                    solution.fitness = float(np.mean(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    solution.trace = trace
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
        current_alpha = alpha_scheduler.get_alpha(generation - 1)
        
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{generations}")
        print(f"API Calls: {api_calls}/{args.budget}")
        print(f"Best: {best_ever.fitness:.4f} ({best_ever.name})")
        print(f"MADA α: {current_alpha:.3f}")
        
        bandit_state = mada_operator.get_bandit_state()
        probs = bandit_state['selection_probs']
        print(f"D-TS: mutation={probs['mutation']:.1%}, crossover={probs['crossover']:.1%}")
        print('='*60)
        
        parents = selection(population, n_parents, elitism=args.elitism)
        best_parent_fitness = max((p.fitness for p in parents), default=0.0)
        
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
                print(f"  D-TS: {operator} (θ={selection_info['theta_sampled']:.3f})")
                print(f"  Evaluating {child.name}...")
                
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
                    
                    if trace:
                        global_min_fitness = min(global_min_fitness, min(trace))
                        global_max_fitness = max(global_max_fitness, max(trace))
                    
                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f}")
                        best_ever = child

                # MADA: Calculate NN-Dist
                if trace and history_traces:
                    nn_dist = calculate_nn_dist(trace, history_traces, global_min_fitness, global_max_fitness)
                else:
                    nn_dist = 1.0
                
                print(f"  NN-Dist: {nn_dist:.4f}")
                
                # MADA: Composite reward
                reward, fitness_delta, diversity_bonus = compute_composite_reward(
                    child_fitness=child.fitness,
                    parent_fitness=best_parent_fitness,
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
                    'parent_fitness': best_parent_fitness,
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
        print(f"Operators: {stats['mutation_count']} mut, {stats['crossover_count']} xo")
    
    # Final Summary
    print(f"\n{'='*60}")
    print("MADA-LLAMEA COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations: {generation}")
    print(f"Best: {best_ever.name} ({best_ever.fitness:.4f})")
    print(f"Traces collected: {len(history_traces)}")
    
    final_state = mada_operator.get_bandit_state()
    print(f"\nFinal Bandit:")
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
        description='MADA-LLAMEA: Multi-Adaptive Diverse Algorithm with LLM Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # API Configuration
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--base-url', type=str, help='Custom API base URL')
    parser.add_argument('--max-tokens', type=int, help='Max tokens')
    
    # Model Configuration  
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', help='AI model')
    parser.add_argument('--experiment-name', type=str, default='mada-experiment', help='Experiment name')
    
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
    parser.add_argument('--reward-variance', type=float, default=0.25, help='D-TS reward variance')
    parser.add_argument('--reward-clamp', type=float, default=1.0, help='Reward clamp')
    
    # MADA Configuration
    parser.add_argument('--alpha-start', type=float, default=0.5, help='Initial diversity weight α')
    parser.add_argument('--alpha-end', type=float, default=0.0, help='Final diversity weight α')
    parser.add_argument('--alpha-schedule', type=str, default='linear',
                       choices=['linear', 'exponential', 'cosine', 'constant'],
                       help='Alpha decay schedule')
    
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
        mada_operator = MADAOperator(
            algorithm_manager,
            discount=args.discount,
            tau_max=args.tau_max,
            reward_variance=args.reward_variance,
            reward_clamp=args.reward_clamp,
        )
        
        if args.generations is not None:
            generations = args.generations
        else:
            generations = max(1, (args.budget - args.n_parents) // args.n_offspring)
        
        print(f"\nStarting MADA-LLAMEA:")
        print(f"  Model: {ai_model}")
        print(f"  Budget: {args.budget}")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
        print(f"  Generations: {generations}")
        print(f"  D-TS: γ={args.discount}, τ_max={args.tau_max}")
        print(f"  MADA: α={args.alpha_start}→{args.alpha_end} ({args.alpha_schedule})")
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
        print("ERROR: MADA-LLAMEA requires --evolutionary-mode flag")
        return
    
    print(f"\nResults saved in: {explogger.dirname}")


if __name__ == "__main__":
    main()

