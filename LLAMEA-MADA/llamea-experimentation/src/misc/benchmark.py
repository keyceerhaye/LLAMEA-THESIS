"""
Standardized BBOB Benchmarking Utilities
========================================

This module provides a wrapper-based approach for computing AOCC (Area Over Convergence Curve)
metrics that works with any version of the IOH library.

The wrapper pattern intercepts function calls to track evaluations and compute AOCC without
relying on IOH's attach_logger() method, which has compatibility issues with custom Python loggers.

Usage:
------
    from misc.benchmark import AUCTracker, WrappedProblem, evaluate_algorithm
    
    # Simple usage with evaluate_algorithm helper
    auc = evaluate_algorithm(algorithm_class, budget=10000, dim=5, fid=1, iid=1)
    
    # Manual usage with tracker and wrapper
    tracker = AUCTracker(budget=10000)
    problem = get_problem(fid, iid, dim)
    wrapped = WrappedProblem(problem, tracker)
    
    algorithm = MyAlgorithm(budget, dim)
    algorithm(wrapped)
    
    auc = tracker.get_auc()
"""

import numpy as np
from ioh import get_problem


class OverBudgetException(Exception):
    """Raised when the algorithm exceeds the evaluation budget."""
    pass


class AUCTracker:
    """
    Track Area Under Curve (AOCC) for optimization without IOH logger attachment.
    
    This class computes the AOCC metric by tracking function evaluations and the
    best objective value seen so far. It uses logarithmic scaling by default to
    handle the wide range of objective values in BBOB.
    
    The AOCC formula:
        - Each evaluation contributes: (log(best_y) - log(lower)) / (log(upper) - log(lower))
        - Final AOCC = 1 - (accumulated_aoc + remaining_budget * final_fraction) / budget
    
    Higher AOCC is better:
        - 1.0 = Perfect (found optimum on first evaluation)
        - 0.0 = Never improved from worst case
    
    Attributes:
        budget (int): Maximum number of function evaluations.
        lower (float): Lower bound for clipping objective values (default: 1e-8).
        upper (float): Upper bound for clipping objective values (default: 1e2).
        scale_log (bool): Whether to use log10 scaling (default: True).
    """
    
    def __init__(self, budget, lower=1e-8, upper=1e2, scale_log=True):
        """
        Initialize the AUC tracker.
        
        Args:
            budget (int): Maximum number of function evaluations.
            lower (float): Lower bound for clipping (default: 1e-8).
            upper (float): Upper bound for clipping (default: 1e2).
            scale_log (bool): Use log10 scaling (default: True).
        """
        self.budget = budget
        self.lower = lower
        self.upper = upper
        self.scale_log = scale_log
        self.transform = (lambda x: np.log10(x)) if scale_log else (lambda x: x)
        self.reset()
    
    def reset(self):
        """Reset the tracker for a new run."""
        self.aoc = 0.0
        self.evaluations = 0
        self.best_y = np.inf
    
    def update(self, y_value):
        """
        Update AUC with a new function evaluation.
        
        Args:
            y_value (float): The objective value from the function evaluation.
            
        Raises:
            OverBudgetException: If evaluations exceed the budget.
        """
        self.evaluations += 1
        
        # Update best value seen
        if y_value < self.best_y:
            self.best_y = y_value
        
        # Check budget constraints
        if self.evaluations > self.budget:
            raise OverBudgetException()
        
        # Don't accumulate on the last evaluation (matches original behavior)
        if self.evaluations == self.budget:
            return
        
        # Accumulate AOC contribution
        y_clipped = np.clip(self.best_y, self.lower, self.upper)
        self.aoc += (self.transform(y_clipped) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )
    
    def get_auc(self):
        """
        Get the normalized AUC (AOCC) value.
        
        This corrects for early stopping by assuming the final best value
        would be maintained for any remaining evaluations.
        
        Returns:
            float: AOCC value between 0 and 1 (higher is better).
        """
        y_clipped = np.clip(self.best_y, self.lower, self.upper)
        fraction = (self.transform(y_clipped) - self.transform(self.lower)) / (
            self.transform(self.upper) - self.transform(self.lower)
        )
        aoc = (self.aoc + np.clip(self.budget - self.evaluations, 0, self.budget) * fraction) / self.budget
        return 1 - aoc


class WrappedProblem:
    """
    Wrapper around IOH problem to track evaluations for AUC calculation.
    
    This wrapper intercepts function calls and forwards them to both the
    actual IOH problem and the AUC tracker. It exposes the problem's bounds
    and metadata so algorithms can access them.
    
    Attributes:
        problem: The underlying IOH problem.
        tracker: The AUCTracker instance.
        bounds: Problem bounds (exposes lb and ub arrays).
        meta_data: Problem metadata.
    """
    
    def __init__(self, problem, tracker):
        """
        Initialize the wrapper.
        
        Args:
            problem: An IOH problem instance.
            tracker: An AUCTracker instance.
        """
        self.problem = problem
        self.tracker = tracker
        self.bounds = problem.bounds
        self.meta_data = problem.meta_data
    
    def __call__(self, x):
        """
        Evaluate the objective function and track the result.
        
        Args:
            x: Input vector to evaluate.
            
        Returns:
            float: The objective value.
        """
        y = self.problem(x)
        self.tracker.update(y)
        return y
    
    def reset(self):
        """Reset both the problem and tracker."""
        self.problem.reset()
        self.tracker.reset()


def evaluate_algorithm(algorithm_class, budget=10000, dim=5, fid=1, iid=1, 
                       reps=1, seed=None, upper=1e2, lower=1e-8):
    """
    Evaluate an algorithm on a single BBOB problem.
    
    Args:
        algorithm_class: The algorithm class to evaluate.
        budget (int): Function evaluation budget.
        dim (int): Problem dimension.
        fid (int): BBOB function ID (1-24).
        iid (int): Instance ID.
        reps (int): Number of repetitions.
        seed (int): Random seed (None for random).
        upper (float): Upper bound for AOCC calculation.
        lower (float): Lower bound for AOCC calculation.
        
    Returns:
        float or list: Single AUC value if reps=1, else list of AUC values.
    """
    tracker = AUCTracker(budget, lower=lower, upper=upper)
    problem = get_problem(fid, iid, dim)
    
    aucs = []
    for rep in range(reps):
        if seed is not None:
            np.random.seed(seed + rep)
        
        tracker.reset()
        wrapped = WrappedProblem(problem, tracker)
        
        try:
            alg = algorithm_class(budget, dim)
            alg(wrapped)
        except OverBudgetException:
            pass
        except Exception as e:
            # Algorithm error - return 0 AUC
            aucs.append(0.0)
            problem.reset()
            continue
        
        aucs.append(tracker.get_auc())
        problem.reset()
    
    return aucs[0] if reps == 1 else aucs


def evaluate_bbob_suite(algorithm_class, budget=10000, dim=5, 
                        fids=range(1, 25), iids=[1, 2, 3], reps=3,
                        upper=1e2, lower=1e-8, verbose=False):
    """
    Evaluate an algorithm on the full BBOB suite or a subset.
    
    Args:
        algorithm_class: The algorithm class to evaluate.
        budget (int): Function evaluation budget per run.
        dim (int): Problem dimension.
        fids (iterable): Function IDs to evaluate (default: 1-24).
        iids (iterable): Instance IDs to evaluate (default: [1, 2, 3]).
        reps (int): Repetitions per (function, instance) pair.
        upper (float): Upper bound for AOCC calculation.
        lower (float): Lower bound for AOCC calculation.
        verbose (bool): Print progress.
        
    Returns:
        dict: Results containing:
            - 'mean_auc': Mean AOCC across all runs
            - 'std_auc': Standard deviation
            - 'all_aucs': List of all AUC values
            - 'function_aucs': Dict mapping fid to list of AUCs
            - 'group_aucs': Dict with BBOB function group scores
    """
    tracker = AUCTracker(budget, lower=lower, upper=upper)
    
    all_aucs = []
    function_aucs = {}
    group_aucs = {
        'separable': [],      # f1-f5
        'moderate': [],       # f6-f9
        'ill_conditioned': [],  # f10-f14
        'multimodal': [],     # f15-f19
        'weakly_structured': []  # f20-f24
    }
    
    for fid in fids:
        problem = get_problem(fid, iids[0], dim)
        fid_aucs = []
        
        for iid in iids:
            problem = get_problem(fid, iid, dim)
            
            for rep in range(reps):
                np.random.seed(rep)
                tracker.reset()
                wrapped = WrappedProblem(problem, tracker)
                
                try:
                    alg = algorithm_class(budget, dim)
                    alg(wrapped)
                except OverBudgetException:
                    pass
                except Exception as e:
                    fid_aucs.append(0.0)
                    problem.reset()
                    continue
                
                auc = tracker.get_auc()
                fid_aucs.append(auc)
                all_aucs.append(auc)
                problem.reset()
        
        function_aucs[fid] = fid_aucs
        
        # Assign to function groups
        if fid <= 5:
            group_aucs['separable'].extend(fid_aucs)
        elif fid <= 9:
            group_aucs['moderate'].extend(fid_aucs)
        elif fid <= 14:
            group_aucs['ill_conditioned'].extend(fid_aucs)
        elif fid <= 19:
            group_aucs['multimodal'].extend(fid_aucs)
        else:
            group_aucs['weakly_structured'].extend(fid_aucs)
        
        if verbose:
            print(f"  f{fid}: {np.mean(fid_aucs):.4f} ± {np.std(fid_aucs):.4f}")
    
    # Compute group means
    group_means = {k: np.mean(v) if v else 0.0 for k, v in group_aucs.items()}
    
    return {
        'mean_auc': np.mean(all_aucs) if all_aucs else 0.0,
        'std_auc': np.std(all_aucs) if all_aucs else 0.0,
        'all_aucs': all_aucs,
        'function_aucs': function_aucs,
        'group_aucs': group_aucs,
        'group_means': group_means
    }

