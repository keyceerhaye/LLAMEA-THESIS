#!/usr/bin/env python3
"""
MADA 2.0 Benchmark Script
=========================
Implements the adaptive outer-loop controller with:
- Discounted Thompson Sampling to select between Mutation and Crossover
- Strict 1+1 elitism (offspring replaces parent only if better)
- Dual mutation prompts (Explore vs Simplify)
- Pure crossover stitching vs MADA innovation (50/50)

Usage:
  python main-thesis.py --budget 50 --mada2
  python main-thesis.py --budget 100 --mada2 --outer-gamma 0.9 --outer-tau 3.0

Legacy modes (MADA 1.0 or iterative) are still available via flags.
"""

import os
import sys
import argparse
import random
import re
import traceback

import numpy as np
from ioh import get_problem

# ---------------------------------------------------------------------------
# Path setup: ensure src/ is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from llamea import Gemini_LLM, LLaMEA, OpenAI_LLM, Solution
from llamea.utils import NoCodeException

# Standardized benchmarking utilities (wrapper-based, compatible with all IOH versions)
from misc.benchmark import AUCTracker, WrappedProblem, OverBudgetException

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def evaluate_bbob(solution, explogger=None):
    """
    Evaluate a Solution on the BBOB benchmark suite.

    Sets solution.fitness to the mean AOCC and attaches detailed metadata.
    """
    code = solution.code or ""
    name = solution.name or "UnknownAlgorithm"

    try:
        exec(code, globals())
    except Exception as exc:
        solution.set_scores(0.0, f"Code execution error: {exc}", str(exc))
        return solution

    if name not in globals():
        solution.set_scores(0.0, f"Class {name} not found after exec", "ClassNotFound")
        return solution

    algorithm_class = globals()[name]
    eval_budget = int(os.getenv("EVAL_BUDGET", "10000"))
    tracker = AUCTracker(eval_budget, upper=1e2)

    aucs = []
    detail_aucs = []
    detailed_aucs = [0.0, 0.0, 0.0, 0.0, 0.0]

    dim = 5  # BBOB dimension
    try:
        for fid in range(1, 25):
            for iid in [1, 2, 3]:
                problem = get_problem(fid, iid, dim)
                for rep in range(3):
                    np.random.seed(rep)
                    tracker.reset()
                    wrapped_problem = WrappedProblem(problem, tracker)
                    try:
                        # Pass both budget and dim to algorithm
                        alg = algorithm_class(eval_budget, dim)
                        alg(wrapped_problem)
                    except OverBudgetException:
                        pass
                    except Exception as e:
                        # Log but continue on algorithm errors
                        pass
                    auc = tracker.get_auc()
                    aucs.append(auc)
                    detail_aucs.append(auc)
                    problem.reset()

            # Track detailed AUCs by function groups
            if fid == 5:
                detailed_aucs[0] = float(np.mean(detail_aucs))
                detail_aucs = []
            elif fid == 9:
                detailed_aucs[1] = float(np.mean(detail_aucs))
                detail_aucs = []
            elif fid == 14:
                detailed_aucs[2] = float(np.mean(detail_aucs))
                detail_aucs = []
            elif fid == 19:
                detailed_aucs[3] = float(np.mean(detail_aucs))
                detail_aucs = []
            elif fid == 24:
                detailed_aucs[4] = float(np.mean(detail_aucs))
                detail_aucs = []

    except Exception as exc:
        solution.set_scores(0.0, f"Evaluation error: {exc}", str(exc))
        return solution

    score = float(np.mean(aucs))
    std = float(np.std(aucs))
    feedback = (
        f"Algorithm {name} achieved AOCC={score:.4f} ± {std:.4f} on BBOB. "
        f"Group scores: {[f'{v:.3f}' for v in detailed_aucs]}"
    )
    solution.set_scores(score, feedback)
    solution.add_metadata("aucs", aucs)
    solution.add_metadata("detailed_aucs", detailed_aucs)
    return solution


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="MADA 2.0 Benchmark - Adaptive Outer Loop with DTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # MADA 2.0 mode (recommended)
  python main-thesis.py --mada2 --budget 50

  # Tune outer-loop bandit
  python main-thesis.py --mada2 --budget 100 --outer-gamma 0.95 --outer-tau 2.0

  # Legacy population mode (MADA 1.0 block-level bandits)
  python main-thesis.py --evolutionary-mode --n-parents 4 --n-offspring 8 --budget 100

  # Simple iterative refinement (no bandits)
  python main-thesis.py --budget 50
""",
    )

    # API
    parser.add_argument("--api-key", type=str, help="API key (or set GEMINI_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--base-url", type=str, help="Custom base URL for OpenAI-compatible APIs")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash", help="LLM model name")

    # Experiment
    parser.add_argument("--experiment-name", type=str, default="mada2-benchmark")
    parser.add_argument("--budget", type=int, default=50, help="Total LLM calls budget")
    parser.add_argument("--eval-budget", type=int, default=10000, help="Function evaluations per algorithm")

    # MADA 2.0 mode
    parser.add_argument("--mada2", action="store_true", help="Enable MADA 2.0 adaptive outer loop with DTS")
    parser.add_argument("--outer-gamma", type=float, default=0.9, help="Discount factor γ for outer-loop DTS")
    parser.add_argument("--outer-tau", type=float, default=3.0, help="Variance clamp τ_max for outer-loop DTS")
    
    # Population configuration (works for both MADA 2.0 and legacy modes)
    parser.add_argument("--n-parents", type=int, default=1, help="Number of parents (μ). Use 1 for strict 1+1")
    parser.add_argument("--n-offspring", type=int, default=1, help="Number of offspring per generation (λ)")
    parser.add_argument("--elitism", action="store_true", help="Enable (μ+λ) elitism, otherwise uses (μ,λ)")

    # Legacy modes
    parser.add_argument("--evolutionary-mode", action="store_true", help="Legacy MADA 1.0 population mode (block-level bandits)")

    return parser


def resolve_llm(args):
    """Instantiate the appropriate LLM based on model name and API key."""
    # Check for AIML key first (uses OpenAI-compatible endpoint)
    aiml_key = os.getenv("AIML_KEY")
    if aiml_key:
        # AIML uses OpenAI-compatible API
        base_url = args.base_url or os.getenv("AIML_BASE_URL", "https://api.aimlapi.com/v1")
        return OpenAI_LLM(aiml_key, model=args.model, base_url=base_url)
    
    api_key = (
        args.api_key
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "API key required. Set --api-key, GEMINI_API_KEY, OPENAI_API_KEY, or AIML_KEY."
        )

    model = args.model.lower()
    if "gemini" in model:
        return Gemini_LLM(api_key, model=args.model)
    else:
        return OpenAI_LLM(api_key, model=args.model, base_url=args.base_url)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    # Set eval budget as env var so evaluate_bbob can read it
    os.environ["EVAL_BUDGET"] = str(args.eval_budget)

    llm = resolve_llm(args)

    print("=" * 60)
    print("MADA 2.0 Benchmark")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Budget: {args.budget}")
    print(f"  Eval Budget: {args.eval_budget}")
    
    if args.mada2:
        mode_str = "MADA 2.0"
        if args.n_parents == 1 and args.n_offspring == 1:
            mode_str += " (strict 1+1)"
        else:
            mode_str += f" (μ={args.n_parents}, λ={args.n_offspring})"
            mode_str += f" {'elitism' if args.elitism else 'comma'}"
        print(f"  Mode: {mode_str}")
        print(f"  Outer γ: {args.outer_gamma}")
        print(f"  Outer τ: {args.outer_tau}")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
        print(f"  Elitism: {args.elitism}")
    elif args.evolutionary_mode:
        print(f"  Mode: Legacy MADA 1.0 (block-level bandits)")
        print(f"  Parents (μ): {args.n_parents}")
        print(f"  Offspring (λ): {args.n_offspring}")
    else:
        print(f"  Mode: Simple iterative refinement")
    print("-" * 60)

    if args.mada2:
        # ----------------------------------------------------------------
        # MADA 2.0: Adaptive outer loop with DTS-guided mutation/crossover
        # Supports both strict 1+1 (n_parents=1, n_offspring=1) and
        # population-based evolution (μ+λ or μ,λ)
        # ----------------------------------------------------------------
        optimizer = LLaMEA(
            f=evaluate_bbob,
            llm=llm,
            n_parents=args.n_parents,
            n_offspring=args.n_offspring,
            budget=args.budget,
            experiment_name=args.experiment_name,
            elitism=args.elitism,
            adaptive_outer_loop=True,
            outer_loop_gamma=args.outer_gamma,
            outer_loop_tau=args.outer_tau,
        )
        best = optimizer.run()

        print("\n" + "=" * 60)
        print("MADA 2.0 RUN COMPLETE")
        print("=" * 60)
        print(f"Best algorithm: {best.name}")
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Generation: {best.generation}")
        print(f"Total evaluations: {len(optimizer.run_history)}")

        # Bandit summary
        bandit_summary = optimizer.get_bandit_summary()
        
        if "outer_loop" in bandit_summary:
            print("\nOuter-loop bandit state (Mutation vs Crossover):")
            for arm, stats in bandit_summary["outer_loop"].items():
                print(f"  {arm}: count={stats['count']:.2f}, mean={stats['mean']:.4f}")
        
        if "block_bandits" in bandit_summary:
            print("\nBlock-level bandit state (for MADA Innovation):")
            for block, arms in bandit_summary["block_bandits"].items():
                print(f"  {block}:")
                for arm, stats in arms.items():
                    print(f"    {arm}: count={stats['count']:.2f}, mean={stats['mean']:.4f}")

    elif args.evolutionary_mode:
        # ----------------------------------------------------------------
        # Legacy MADA 1.0: Population-based with block-level bandits
        # ----------------------------------------------------------------
        optimizer = LLaMEA(
            f=evaluate_bbob,
            llm=llm,
            n_parents=args.n_parents,
            n_offspring=args.n_offspring,
            budget=args.budget,
            experiment_name=args.experiment_name,
            elitism=args.elitism,
            adaptive_outer_loop=False,
        )
        best = optimizer.run()

        print("\n" + "=" * 60)
        print("LEGACY EVOLUTIONARY RUN COMPLETE")
        print("=" * 60)
        print(f"Best algorithm: {best.name}")
        print(f"Best fitness: {best.fitness:.4f}")

    else:
        # ----------------------------------------------------------------
        # Simple iterative refinement (no bandits)
        # ----------------------------------------------------------------
        optimizer = LLaMEA(
            f=evaluate_bbob,
            llm=llm,
            n_parents=1,
            n_offspring=1,
            budget=args.budget,
            experiment_name=args.experiment_name,
            elitism=True,
            adaptive_outer_loop=False,
        )
        best = optimizer.run()

        print("\n" + "=" * 60)
        print("ITERATIVE REFINEMENT COMPLETE")
        print("=" * 60)
        print(f"Best algorithm: {best.name}")
        print(f"Best fitness: {best.fitness:.4f}")

    # Save best algorithm
    if optimizer.logger:
        out_path = f"{optimizer.logger.dirname}/BEST_ALGORITHM.py"
        with open(out_path, "w") as f:
            f.write(f"# Best Algorithm: {best.name}\n")
            f.write(f"# Fitness: {best.fitness:.4f}\n")
            f.write(f"# Generation: {best.generation}\n\n")
            f.write(best.code or "")
        print(f"\nResults saved to: {optimizer.logger.dirname}")


if __name__ == "__main__":
    main()
