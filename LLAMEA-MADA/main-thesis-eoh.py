"""
Evolution of Heuristics (EoH) - Simplified LLM-driven Genetic Programming

This is a simplified version of main-thesis.py that uses the Evolution of Heuristics (EoH)
approach instead of the complex MADA block-based system.

Key Features:
- Simple LLM-based mutation: Ask the LLM to improve a single algorithm
- Simple LLM-based crossover: Ask the LLM to combine two parent algorithms (FunSearch/LMX style)
- No block parsing, no bandits, no complex dependency management
- Same benchmarking using AUC on BBOB 5D functions

References:
- FunSearch (Google DeepMind): Context window implicit crossover
- LMX (Language Model Crossover): Few-shot prompting for variation
- EoH: Evolution of Heuristics with LLM operators

Usage:
    python main-thesis-eoh.py --evolutionary-mode --n-parents 4 --n-offspring 16 --budget 100
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
# SOLUTION CLASS (Simplified)
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
# EOH OPERATOR (Simple LLM-based mutation and crossover)
# ==============================================================================

class EoHOperator:
    """
    Evolution of Heuristics Operator - Simple LLM-based genetic operators.
    
    Uses two main operators:
    1. Mutation: Ask LLM to improve a single algorithm
    2. Crossover: Ask LLM to combine two parent algorithms
    
    Supports two modes for controlling operator distribution:
    1. Rate-based: Use mutation_rate for probabilistic selection
    2. Count-based: Specify exact counts for mutation and crossover offspring
    """
    
    def __init__(self, algorithm_manager, mutation_rate=0.5, crossover_style="implicit",
                 n_mutation=None, n_crossover=None):
        """
        Args:
            algorithm_manager: Manager for LLM API calls
            mutation_rate: Probability of mutation vs crossover (0.5 = equal chance)
                          Only used if n_mutation and n_crossover are not set.
            crossover_style: "implicit" (FunSearch style) or "explicit" (merge & refine)
            n_mutation: Exact number of offspring from mutation per generation (optional)
            n_crossover: Exact number of offspring from crossover per generation (optional)
        """
        self.algorithm_manager = algorithm_manager
        self.mutation_rate = mutation_rate
        self.crossover_style = crossover_style
        self.child_counter = 0
        
        # Count-based mode
        self.n_mutation = n_mutation
        self.n_crossover = n_crossover
        self.use_count_mode = (n_mutation is not None) or (n_crossover is not None)
        
        # Track offspring counts within a generation (for count-based mode)
        self._mutation_count = 0
        self._crossover_count = 0
    
    def reset_generation_counts(self):
        """Reset offspring counts at the start of each generation."""
        self._mutation_count = 0
        self._crossover_count = 0
    
    def get_operator_stats(self):
        """Get current generation's operator usage statistics."""
        return {
            'mutation_count': self._mutation_count,
            'crossover_count': self._crossover_count,
            'total': self._mutation_count + self._crossover_count
        }
    
    def _decide_operator(self, parents):
        """
        Decide whether to use mutation or crossover.
        
        Returns:
            str: 'mutation' or 'crossover'
        """
        # Need at least 2 parents for crossover
        if len(parents) < 2:
            return 'mutation'
        
        if self.use_count_mode:
            # Count-based mode: follow exact quotas
            n_mut = self.n_mutation if self.n_mutation is not None else 0
            n_xo = self.n_crossover if self.n_crossover is not None else 0
            
            # Check if we've hit the mutation quota
            mutation_done = self._mutation_count >= n_mut
            # Check if we've hit the crossover quota
            crossover_done = self._crossover_count >= n_xo
            
            if mutation_done and crossover_done:
                # Both quotas met, use rate-based fallback
                return 'mutation' if random.random() < self.mutation_rate else 'crossover'
            elif mutation_done:
                return 'crossover'
            elif crossover_done:
                return 'mutation'
            else:
                # Both have quota remaining - use proportional selection
                remaining_mut = n_mut - self._mutation_count
                remaining_xo = n_xo - self._crossover_count
                total_remaining = remaining_mut + remaining_xo
                if total_remaining <= 0:
                    return 'mutation' if random.random() < 0.5 else 'crossover'
                prob_mutation = remaining_mut / total_remaining
                return 'mutation' if random.random() < prob_mutation else 'crossover'
        else:
            # Rate-based mode: probabilistic selection
            return 'mutation' if random.random() < self.mutation_rate else 'crossover'
    
    def generate_offspring(self, parents, focal_parent=None, population_summary=""):
        """
        Generate a new offspring using either mutation or crossover.
        
        Args:
            parents: List of parent solutions
            focal_parent: Primary parent for mutation (if not set, random choice)
            population_summary: String describing current population
        
        Returns:
            EoHSolution: New offspring solution
        """
        if not parents:
            raise ValueError("EoHOperator requires at least one parent")
        
        # Decide operator
        operator = self._decide_operator(parents)
        
        if operator == 'mutation':
            self._mutation_count += 1
            parent = focal_parent or random.choice(parents)
            return self._mutate(parent)
        else:
            self._crossover_count += 1
            # Select two parents (prefer higher fitness parents)
            sorted_parents = sorted(parents, key=lambda p: p.fitness, reverse=True)
            parent_a = sorted_parents[0]
            # Select second parent - prefer diversity
            parent_b = self._select_diverse_parent(sorted_parents, parent_a)
            return self._crossover(parent_a, parent_b)
    
    def _mutate(self, parent):
        """Apply mutation: Ask LLM to improve the parent algorithm."""
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
            # Fallback: return a copy of parent with error
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
    
    def _crossover(self, parent_a, parent_b):
        """Apply crossover: Ask LLM to combine two parent algorithms."""
        self.child_counter += 1
        
        # Choose crossover style
        if self.crossover_style == "explicit":
            prompt = EOH_CROSSOVER_EXPLICIT_PROMPT.format(
                score_a=parent_a.fitness,
                code_a=parent_a.code,
                score_b=parent_b.fitness,
                code_b=parent_b.code
            )
        else:  # implicit (FunSearch style)
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
            # Fallback: return a copy of better parent with error
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
    
    def _select_diverse_parent(self, sorted_parents, parent_a):
        """Select a second parent that is code-diverse from parent_a."""
        for candidate in sorted_parents[1:]:
            if (candidate.code or "") != (parent_a.code or ""):
                return candidate
        # Fallback to second best
        return sorted_parents[1] if len(sorted_parents) > 1 else parent_a
    
    def _call_llm(self, prompt):
        """Call the LLM with the given prompt using AlgorithmManager's client."""
        messages = [
            {"role": "system", "content": EOH_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Log the prompt using AlgorithmManager's logger
        self.algorithm_manager.logger.log_conversation(f"\n[EoH Prompt]\n{prompt}\n")
        
        call_kwargs = {
            "model": self.algorithm_manager.ai_model,
            "messages": messages,
            "temperature": 0.8,
        }
        if self.algorithm_manager.max_tokens:
            call_kwargs["max_tokens"] = self.algorithm_manager.max_tokens
        
        response = self.algorithm_manager.client.chat.completions.create(**call_kwargs)
        message = response.choices[0].message.content
        
        # Log the response
        self.algorithm_manager.logger.log_conversation(f"\n[EoH Response]\n{message}\n")
        
        return message
    
    def _extract_code(self, message):
        """Extract Python code from LLM response."""
        pattern = r"```(?:python)?\n(.*?)\n```"
        match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1)
        raise NoCodeException("No code block found in response")
    
    def _extract_class_name(self, code, default):
        """Extract class name from code."""
        for line in code.splitlines():
            if line.strip().startswith("class "):
                match = re.search(r"class\s+(\w+)", line)
                if match:
                    return match.group(1)
        return default
    
    def _extract_name(self, message):
        """Extract algorithm name from message."""
        pattern = r"#\s*Name:\s*(\w+)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""


# ==============================================================================
# EOH-SPECIFIC LOGGING HELPER
# ==============================================================================

def log_eoh_offspring(explogger, attempt, record):
    """Log EoH offspring metadata to JSONL file."""
    # Ensure directory exists (in case it was deleted during run)
    os.makedirs(explogger.dirname, exist_ok=True)
    log_path = os.path.join(explogger.dirname, "eoh_offspring.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": attempt, **record}, default=str) + "\n")


# ==============================================================================
# EVALUATION (Same as main-thesis.py)
# ==============================================================================

EXPECTED_AUC_LEN = 24 * 3 * 3  # 24 fids × 3 iids × 3 reps


def _hash_aucs(aucs):
    """Fast SHA256 hash for duplicate detection."""
    try:
        arr = np.asarray(aucs, dtype=float)
        return hashlib.sha256(arr.tobytes()).hexdigest()
    except Exception:
        return ""


def _summarize_aucs(aucs):
    """Return length, mean, std for quick logging."""
    try:
        arr = np.asarray(aucs, dtype=float)
        return len(arr), float(np.mean(arr)), float(np.std(arr))
    except Exception:
        return 0, 0.0, 0.0


def evaluate_algorithm(algorithm_code, algorithm_name, eval_budget):
    """Evaluates a single algorithm on BBOB benchmark suite."""
    try:
        safe_globals = {
            "lb": -5.0,
            "ub": 5.0,
            "bounds": (-5.0, 5.0),
            "learning_rate": 0.5,
            "local_search_prob": 0.1,
            "stagnation_multiplier": 1.0,
            "initial_pop": None,
            "pop_size": 50,
            "F": 0.5,
            "CR": 0.9,
            "archive_size_multiplier": 1.0,
            "eps": 1e-12,
        }

        _orig_np_choice = np.random.choice

        def _safe_choice(a, size=None, replace=False, p=None, axis=0, shuffle=True):
            try:
                if not replace and size is not None:
                    n = len(a)
                    if n < size:
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

                        lb = problem.bounds.lb
                        ub = problem.bounds.ub
                        if not hasattr(algorithm, "lb"):
                            algorithm.lb = lb
                        if not hasattr(algorithm, "ub"):
                            algorithm.ub = ub
                        if not hasattr(algorithm, "bounds"):
                            algorithm.bounds = (lb, ub)
                        if not hasattr(algorithm, "pop"):
                            algorithm.pop = []
                        if not hasattr(algorithm, "archive_x"):
                            algorithm.archive_x = []
                        if not hasattr(algorithm, "archive_f"):
                            algorithm.archive_f = []
                        if not hasattr(algorithm, "evals"):
                            algorithm.evals = 0
                        if not hasattr(algorithm, "F"):
                            algorithm.F = 0.5
                        if not hasattr(algorithm, "CR"):
                            algorithm.CR = 0.9
                        if not hasattr(algorithm, "archive_size_multiplier"):
                            algorithm.archive_size_multiplier = 1.0
                        if not hasattr(algorithm, "eps"):
                            algorithm.eps = 1e-12
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
                detailed_aucs[0] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 9:
                detailed_aucs[1] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 14:
                detailed_aucs[2] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 19:
                detailed_aucs[3] = np.mean(detail_aucs)
                detail_aucs = []
            if fid == 24:
                detailed_aucs[4] = np.mean(detail_aucs)
                detail_aucs = []
        
        return aucs, detailed_aucs, ""

    except KeyboardInterrupt:
        return [], [0, 0, 0, 0, 0], "KeyboardInterrupt during evaluation"
    except Exception as e:
        return [], [0, 0, 0, 0, 0], str(e)
    finally:
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice


# ==============================================================================
# SELECTION (Same as main-thesis.py)
# ==============================================================================

def selection(population, n_parents, elitism=True, minimization=False):
    """Select the best individuals from the population."""
    reverse = not minimization
    sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=reverse)

    if not sorted_pop:
        return []

    selected = []
    best = sorted_pop[0]
    selected.append(best)

    if n_parents == 1:
        return selected

    best_fitness = best.fitness
    best_code = best.code or ""

    second = None
    for ind in sorted_pop[1:]:
        code_key = ind.code or ""
        if code_key == best_code:
            continue
        if ind.fitness < best_fitness:
            second = ind
            break

    if second is None:
        for ind in sorted_pop[1:]:
            code_key = ind.code or ""
            if code_key != best_code:
                second = ind
                break

    if second is None and len(sorted_pop) > 1:
        second = sorted_pop[1]

    if second is not None:
        selected.append(second)

    seen_codes = {best_code, second.code or "" if second is not None else ""}
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        code_key = ind.code or ""
        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)
        selected.append(ind)

    if len(selected) < n_parents:
        for ind in sorted_pop:
            if len(selected) >= n_parents:
                break
            if ind not in selected:
                selected.append(ind)

    return selected


# ==============================================================================
# MAIN EVOLUTIONARY LOOP
# ==============================================================================

def run_eoh_evolutionary_mode(
    algorithm_manager,
    eoh_operator,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
):
    """
    Population-based evolutionary mode using EoH methodology.
    Simple LLM-based mutation and crossover without complex block parsing.
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
            print(f"Budget exhausted during initialization at {api_calls} calls")
            break
            
        print(f"\nInitializing Parent {i+1}/{n_parents} (API call {api_calls+1})")
        
        try:
            retries = 0
            max_retries = 3
            while retries <= max_retries:
                # Use AlgorithmManager's fetch_algorithm() method
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

                code_hash = hashlib.sha256((algorithm_code or "").encode("utf-8")).hexdigest()
                if code_hash in seen_hashes:
                    print("  Duplicate code hash detected, retrying initialization...")
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
                        print("  Max retries exceeded; keeping last attempt despite error.")
                    else:
                        continue
                else:
                    auc_len, auc_mean_val, auc_std_val = _summarize_aucs(aucs)
                    auc_hash = _hash_aucs(aucs)
                    if auc_len != EXPECTED_AUC_LEN:
                        print(f"  WARNING: AUC length {auc_len} != expected {EXPECTED_AUC_LEN}")
                    if last_auc_hash and auc_hash == last_auc_hash:
                        print("  WARNING: AUC vector identical to previous attempt")
                    solution.fitness = float(np.mean(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    solution.error = ""
                    seen_hashes.add(code_hash)
                    print(f"  Fitness: {solution.fitness:.4f}")
                    last_auc_hash = auc_hash

                population.append(solution)
                explogger.log_code(api_calls, algorithm_name, algorithm_code)
                explogger.log_aucs(
                    api_calls,
                    aucs if aucs else [0],
                    metadata={
                        "len": auc_len if aucs else 0,
                        "mean": auc_mean_val if aucs else 0.0,
                        "std": auc_std_val if aucs else 0.0,
                        "sha256": auc_hash if aucs else "",
                    },
                )
                api_calls += 1
                break

        except Exception as e:
            print(f"  Initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    if not population:
        print("ERROR: Failed to initialize any individuals!")
        return
    
    population = selection(population, len(population), elitism=True, minimization=False)
    best_ever = population[0]
    
    print(f"\nInitialization complete. Best initial fitness: {best_ever.fitness:.4f}")
    
    # Phase 2: Evolutionary loop
    for gen in range(1, generations + 1):
        if api_calls >= args.budget:
            print(f"\nBudget exhausted at generation {gen}")
            break
        
        generation = gen
        print(f"\n{'='*60}")
        print(f"GENERATION {generation}/{generations}")
        print(f"API Calls: {api_calls}/{args.budget}")
        print(f"Best so far: {best_ever.fitness:.4f} ({best_ever.name})")
        print('='*60)
        
        parents = selection(population, n_parents, elitism=args.elitism, minimization=False)
        pop_summary = "\n".join(
            [f"  - {ind.name}: fitness={ind.fitness:.4f}" for ind in parents[:5]]
        )
        best_parent_fitness = max((p.fitness for p in parents), default=0.0)
        
        # Reset operator counts for this generation
        eoh_operator.reset_generation_counts()
        
        offspring = []
        for i in range(n_offspring):
            if api_calls >= args.budget:
                print(f"Budget exhausted at offspring {i}/{n_offspring}")
                break
            
            parent = random.choice(parents)

            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")
            print(f"  Parent: {parent.name} (fitness: {parent.fitness:.4f})")

            try:
                child = eoh_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                    population_summary=f"Current population:\n{pop_summary}",
                )

                print(f"  Evaluating {child.name} ({child.operator})...")
                aucs, detailed_aucs, error = evaluate_algorithm(
                    child.code, child.name, args.eval_budget
                )

                if error:
                    child.fitness = 0.0
                    child.error = error
                    print(f"  Error: {error}")
                else:
                    auc_len, auc_mean_val, auc_std_val = _summarize_aucs(aucs)
                    auc_hash = _hash_aucs(aucs)
                    if auc_len != EXPECTED_AUC_LEN:
                        print(f"  WARNING: AUC length {auc_len} != expected {EXPECTED_AUC_LEN}")
                    if last_auc_hash and auc_hash == last_auc_hash:
                        print("  WARNING: AUC vector identical to previous attempt")
                    child.fitness = float(np.mean(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    child.error = ""
                    last_auc_hash = auc_hash

                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                        best_ever = child

                # Log offspring using helper function
                op_stats = eoh_operator.get_operator_stats()
                offspring_record = {
                    'operator': child.operator,
                    'fitness': child.fitness,
                    'parent_fitness': best_parent_fitness,
                    'parent_ids': child.parent_ids,
                    'generation': generation,
                    'error': child.error,
                    'gen_mutation_count': op_stats['mutation_count'],
                    'gen_crossover_count': op_stats['crossover_count'],
                }
                log_eoh_offspring(explogger, api_calls, offspring_record)
                
                offspring.append(child)
                explogger.log_code(api_calls, child.name, child.code)
                explogger.log_aucs(
                    api_calls,
                    aucs if aucs else [0],
                    metadata={
                        "len": auc_len if aucs else 0,
                        "mean": auc_mean_val if aucs else 0.0,
                        "std": auc_std_val if aucs else 0.0,
                        "sha256": auc_hash if aucs else "",
                    },
                )
                api_calls += 1

            except Exception as e:
                print(f"  Offspring generation error: {e}")
                import traceback
                traceback.print_exc()
        
        # Selection for next generation
        if args.elitism:
            combined = parents + offspring
            population = selection(combined, n_parents, elitism=True, minimization=False)
            print(f"\nSelection: (μ+λ) - Best {n_parents} from {len(combined)} individuals")
        else:
            population = selection(offspring, n_parents, elitism=False, minimization=False)
            print(f"\nSelection: (μ,λ) - Best {n_parents} from {len(offspring)} offspring")

        print(
            "New population parents:",
            [f"{ind.name} (fitness={ind.fitness:.4f})" for ind in population[:n_parents]],
        )
        
        # Print operator usage statistics for this generation
        stats = eoh_operator.get_operator_stats()
        print(f"Operator usage: {stats['mutation_count']} mutations, {stats['crossover_count']} crossovers")
    
    # Final summary
    print(f"\n{'='*60}")
    print("EOH EVOLUTIONARY OPTIMIZATION COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations completed: {generation}")
    print(f"Best algorithm: {best_ever.name}")
    print(f"Best fitness: {best_ever.fitness:.4f}")
    print(f"Best generation: {best_ever.generation}")
    
    # Save best algorithm
    with open(f"{explogger.dirname}/BEST_ALGORITHM.py", "w") as f:
        f.write(f"# Best Algorithm: {best_ever.name}\n")
        f.write(f"# Fitness: {best_ever.fitness:.4f}\n")
        f.write(f"# Generation: {best_ever.generation}\n")
        f.write(f"# Operator: {best_ever.operator}\n\n")
        f.write(best_ever.code)


# ==============================================================================
# ITERATIVE MODE (Simple 1+1 refinement)
# ==============================================================================

def run_iterative_mode(algorithm_manager, explogger, args):
    """Legacy iterative refinement mode - simple (1+1) strategy."""
    
    best_solution = None
    
    for api_call in range(args.budget):
        print(f"\n{'='*60}")
        print(f"API Call {api_call + 1}/{args.budget}")
        print('='*60)
        
        try:
            if api_call == 0:
                # Generate initial algorithm using AlgorithmManager
                message = algorithm_manager.fetch_algorithm()
            else:
                # Mutate the best solution
                eoh_op = EoHOperator(algorithm_manager, mutation_rate=1.0)  # Always mutate
                child = eoh_op._mutate(best_solution)
                
                # Evaluate
                aucs, detailed_aucs, error = evaluate_algorithm(
                    child.code, child.name, args.eval_budget
                )
                
                if error:
                    child.fitness = 0.0
                    child.error = error
                    print(f"Error: {error}")
                else:
                    auc_len, auc_mean, auc_std = _summarize_aucs(aucs)
                    child.fitness = float(np.mean(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    print(f"Algorithm: {child.name}")
                    print(f"AUC Mean: {child.fitness:.4f} ± {auc_std:.4f}")
                    
                    if best_solution is None or child.fitness > best_solution.fitness:
                        best_solution = child
                        print(f"  🎉 NEW BEST!")
                
                explogger.log_code(api_call, child.name, child.code)
                explogger.log_aucs(api_call, aucs if aucs else [0])
                continue
            
            # Handle first iteration
            pattern = r"```(?:python)?\n(.*?)\n```"
            match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
            if not match:
                print("Error: No code extracted from LLM response")
                continue
            
            algorithm_code = match.group(1)
            class_match = re.search(r"class\s+(\w+)", algorithm_code)
            algorithm_name = class_match.group(1) if class_match else "Unknown"
            
            explogger.log_code(api_call, algorithm_name, algorithm_code)
            
            aucs, detailed_aucs, error = evaluate_algorithm(
                algorithm_code, algorithm_name, args.eval_budget
            )
            
            if error:
                print(f"Error: {error}")
            else:
                auc_len, auc_mean, auc_std = _summarize_aucs(aucs)
                print(f"Algorithm: {algorithm_name}")
                print(f"AUC Mean: {auc_mean:.4f} ± {auc_std:.4f}")
                
                best_solution = EoHSolution(
                    code=algorithm_code,
                    name=algorithm_name,
                    description=algorithm_name,
                    generation=0,
                )
                best_solution.fitness = auc_mean
                best_solution.aucs = aucs
                best_solution.detailed_aucs = detailed_aucs
            
            explogger.log_aucs(api_call, aucs if aucs else [0])
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='EoH - Evolution of Heuristics with LLM-based Genetic Operators',
        epilog='''
Evolution of Heuristics (EoH) - Simplified LLM-driven Genetic Programming

This implementation uses simple LLM prompts for mutation and crossover:
- Mutation: Ask LLM to improve a single algorithm
- Crossover: Ask LLM to combine two parent algorithms (FunSearch/LMX style)

Operator Distribution Modes:
1. RATE-BASED (default): Use --mutation-rate for probabilistic selection
2. COUNT-BASED: Use --n-mutation and --n-crossover for exact counts per generation

Examples:
  # Population-based evolutionary mode (rate-based, 50/50 mutation/crossover)
  python main-thesis-eoh.py --evolutionary-mode --n-parents 4 --n-offspring 16 --budget 100
  
  # Rate-based: 70% mutation, 30% crossover
  python main-thesis-eoh.py --evolutionary-mode --mutation-rate 0.7 --budget 100
  
  # Count-based: exactly 10 mutations and 6 crossovers per generation
  python main-thesis-eoh.py --evolutionary-mode --n-offspring 16 --n-mutation 10 --n-crossover 6 --budget 100
  
  # Count-based: only crossover (no mutation)
  python main-thesis-eoh.py --evolutionary-mode --n-offspring 16 --n-mutation 0 --n-crossover 16 --budget 100
  
  # Count-based: only mutation (no crossover)
  python main-thesis-eoh.py --evolutionary-mode --n-offspring 16 --n-mutation 16 --n-crossover 0 --budget 100
  
  # With explicit crossover style (Merge & Refine)
  python main-thesis-eoh.py --evolutionary-mode --crossover-style explicit --budget 100
  
  # Iterative refinement mode (1+1 strategy)
  python main-thesis-eoh.py --budget 50
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # API Configuration
    parser.add_argument('--api-key', type=str, help='OpenAI API key (or use OPENAI_API_KEY in .env file)')
    parser.add_argument('--base-url', type=str, help='Custom base URL for OpenAI-compatible APIs (or use BASE_URL in .env file)')
    parser.add_argument('--max-tokens', type=int, help='Maximum tokens for API responses')
    
    # Model Configuration  
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', 
                       help='AI model to use (default: gemini-2.0-flash)')
    parser.add_argument('--experiment-name', type=str, default='eoh-experiment',
                       help='Name for the experiment (default: eoh-experiment)')
    
    # Experimental Configuration
    parser.add_argument('--budget', type=int, default=100,
                       help='Total API calls budget (default: 100)')
    parser.add_argument('--eval-budget', type=int, default=10000,
                       help='Evaluation budget per algorithm (default: 10000)')
    parser.add_argument('--elitism', action='store_true',
                       help='Enable elitism (μ+λ) strategy, otherwise uses (μ,λ)')
    
    # Population Configuration
    parser.add_argument('--n-parents', type=int, default=4,
                       help='Number of parent algorithms to maintain (default: 4)')
    parser.add_argument('--n-offspring', type=int, default=16,
                       help='Number of offspring algorithms per generation (default: 16)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of generations (if set, overrides budget)')
    parser.add_argument('--evolutionary-mode', action='store_true',
                       help='Enable population-based evolutionary mode')
    
    # EoH-specific Configuration
    parser.add_argument('--mutation-rate', type=float, default=0.5,
                       help='Probability of mutation vs crossover (default: 0.5). '
                            'Only used if --n-mutation and --n-crossover are not set.')
    parser.add_argument('--crossover-style', type=str, default='implicit',
                       choices=['implicit', 'explicit'],
                       help='Crossover prompt style: implicit (FunSearch) or explicit (Merge & Refine)')
    parser.add_argument('--n-mutation', type=int, default=None,
                       help='Exact number of offspring from mutation per generation. '
                            'If set, enables count-based mode instead of rate-based.')
    parser.add_argument('--n-crossover', type=int, default=None,
                       help='Exact number of offspring from crossover per generation. '
                            'If set, enables count-based mode instead of rate-based.')
    
    args = parser.parse_args()
    
    # Debug: Show environment variables
    print("DEBUG: Environment variables found:")
    print(f"  OPENAI_API_KEY: {'***' if os.getenv('OPENAI_API_KEY') else 'Not found'}")
    print(f"  AIML_KEY: {'***' if os.getenv('AIML_KEY') else 'Not found'}")
    print(f"  BASE_URL: {os.getenv('BASE_URL') or 'Not found'}")
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AIML_KEY")
    if not api_key:
        raise ValueError(
            "API key is required. Options:\n"
            "  1. Use --api-key argument\n" 
            "  2. Set OPENAI_API_KEY environment variable\n"
            "  3. Set AIML_KEY environment variable\n"
            "  4. Create a .env file with OPENAI_API_KEY=your_key"
        )
    
    base_url = args.base_url or os.getenv("BASE_URL")
    
    # Setup experiment
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
        # Create EoH operator
        eoh_operator = EoHOperator(
            algorithm_manager,
            mutation_rate=args.mutation_rate,
            crossover_style=args.crossover_style,
            n_mutation=args.n_mutation,
            n_crossover=args.n_crossover
        )
        
        # Calculate generations
        if args.generations is not None:
            generations = args.generations
            total_budget = args.n_parents + (generations * args.n_offspring)
            print(f"INFO: Fixed generations mode")
            print(f"INFO: Generations: {generations}")
            print(f"INFO: Total API calls needed: {total_budget}")
            if total_budget > args.budget:
                print(f"WARNING: Total calls ({total_budget}) exceed budget ({args.budget})")
        else:
            generations = max(1, (args.budget - args.n_parents) // args.n_offspring)
            total_budget = args.n_parents + (generations * args.n_offspring)
            print(f"INFO: Budget-driven evolutionary mode")
            print(f"INFO: Calculated generations: {generations}")
        
        print(f"\nStarting EoH (Evolution of Heuristics) experiment:")
        print(f"  Mode: EVOLUTIONARY (Population-based)")
        print(f"  Model: {ai_model}")
        print(f"  Base URL: {base_url or 'Default OpenAI'}")
        print(f"  Max Tokens: {args.max_tokens or 'Default'}")
        print(f"  API Budget: {args.budget}")
        print(f"  Evaluation Budget: {args.eval_budget}")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
        print(f"  Generations: {generations}")
        print(f"  Strategy: {'(μ+λ) Elitism' if args.elitism else '(μ,λ) Comma'}")
        print(f"  Crossover Style: {args.crossover_style}")
        
        # Show operator distribution mode
        if args.n_mutation is not None or args.n_crossover is not None:
            n_mut = args.n_mutation if args.n_mutation is not None else 0
            n_xo = args.n_crossover if args.n_crossover is not None else 0
            print(f"  Operator Mode: COUNT-BASED")
            print(f"    Mutation offspring per gen: {n_mut}")
            print(f"    Crossover offspring per gen: {n_xo}")
            if n_mut + n_xo != args.n_offspring:
                print(f"    NOTE: n_mutation + n_crossover ({n_mut + n_xo}) != n_offspring ({args.n_offspring})")
                print(f"          Extra offspring will use rate-based fallback ({args.mutation_rate:.0%} mutation)")
        else:
            print(f"  Operator Mode: RATE-BASED")
            print(f"    Mutation probability: {args.mutation_rate:.0%}")
            print(f"    Crossover probability: {1 - args.mutation_rate:.0%}")
        print("-" * 60)
        
        run_eoh_evolutionary_mode(
            algorithm_manager,
            eoh_operator,
            explogger,
            args,
            generations,
            args.n_parents,
            args.n_offspring,
        )
    else:
        # Legacy iterative mode
        print(f"\nStarting EoH Iterative experiment:")
        print(f"  Mode: ITERATIVE (1+1 refinement)")
        print(f"  Model: {ai_model}")
        print(f"  Base URL: {base_url or 'Default OpenAI'}")
        print(f"  Max Tokens: {args.max_tokens or 'Default'}")
        print(f"  API Budget: {args.budget}")
        print(f"  Evaluation Budget: {args.eval_budget}")
        print("-" * 60)
        
        run_iterative_mode(algorithm_manager, explogger, args)
    
    print(f"\nExperiment completed. Results saved in: {explogger.dirname}")


if __name__ == "__main__":
    main()
