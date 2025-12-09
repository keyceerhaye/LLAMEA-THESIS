import os
import sys
import hashlib
from pathlib import Path
import numpy as np
from ioh import get_problem, logger
import re
import argparse
import random

PROJECT_ROOT = Path(__file__).resolve().parent
LLAMEA_SRC = PROJECT_ROOT / "llamea-experimentation" / "src"
if LLAMEA_SRC.exists() and str(LLAMEA_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMEA_SRC))

from managers import AlgorithmManager, ExperimentLogger
from utils import OverBudgetException, aoc_logger, correct_aoc
from llamea.solution import Solution
from llamea.mada import MADAOperator, BlockParser
from llamea.utils import NoCodeException

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    dotenv_loaded = load_dotenv()
    print(f"DEBUG: .env file loaded: {dotenv_loaded}")
except ImportError:
    # If python-dotenv is not installed, just continue without it
    print("DEBUG: python-dotenv not installed, using system environment variables only")
    pass


def make_solution(code, class_name, description, generation, parent_ids=None):
    """Instantiate a MADA-aware Solution with the bookkeeping fields we rely on."""
    solution = Solution(
        code=code,
        name=class_name,
        description=description,
        generation=generation,
        parent_ids=parent_ids or [],
    )
    solution.aucs = []
    solution.detailed_aucs = [0, 0, 0, 0, 0]
    solution.error = ""
    return solution


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
    """
    Evaluates a single algorithm on BBOB benchmark suite.
    
    Args:
        algorithm_code: Python code string
        algorithm_name: Name of the algorithm class
        eval_budget: Evaluation budget per run
        
    Returns:
        tuple: (aucs, detailed_aucs, error_message)
    """
    try:
        # Provide defensive defaults so LLM-generated code that references
        # common missing symbols (seen in logs) does not crash with NameError.
        # Defaults to keep early generations from crashing on missing params.
        safe_globals = {
            "lb": -5.0,
            "ub": 5.0,
            "bounds": (-5.0, 5.0),
            "learning_rate": 0.5,
            "local_search_prob": 0.1,
            "stagnation_multiplier": 1.0,
            "initial_pop": None,
            "pop_size": 50,  # default to avoid missing-pop size errors in DE-style code
            # Common DE params that LLM code omits.
            "F": 0.5,
            "CR": 0.9,
            "archive_size_multiplier": 1.0,
            # Small epsilon to avoid divide-by-zero.
            "eps": 1e-12,
        }

        # Patch numpy.random.choice to fall back to replace=True when the
        # requested sample is larger than the population (prevents early-gen
        # crashes on tiny populations).
        _orig_np_choice = np.random.choice

        def _safe_choice(a, size=None, replace=False, p=None, axis=0, shuffle=True):
            try:
                if not replace and size is not None:
                    n = len(a)
                    if n < size:
                        replace = True
            except Exception:
                pass
            # RandomState.choice does not support axis/shuffle; ignore them.
            return _orig_np_choice(a, size=size, replace=replace, p=p)

        np.random.choice = _safe_choice

        # Execute the algorithm code in a dedicated globals dict so we can
        # retrieve the generated class reliably.
        exec_globals = {**globals(), **safe_globals}
        exec(algorithm_code, exec_globals)
        
        # Get the algorithm class
        if algorithm_name not in exec_globals:
            return [], [0, 0, 0, 0, 0], f"Class {algorithm_name} not found"
        
        algorithm_class = exec_globals[algorithm_name]
        
        # Setup logging (match original LLaMEA: wide clamp, log scale)
        l2 = aoc_logger(eval_budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        aucs = []
        detail_aucs = []
        detailed_aucs = [0, 0, 0, 0, 0]
        
        # Run on all BBOB functions
        for fid in np.arange(1, 25):
            for iid in [1, 2, 3]:
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)
                
                for rep in range(3):
                    np.random.seed(rep)
                    
                    try:
                        algorithm = algorithm_class(eval_budget)

                        # Inject common attributes/helpers expected by LLM code
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
            
            # Track detailed AUCs by function groups
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
        # Gracefully handle manual interrupts during evaluation so the main
        # loop can record this as an error instead of crashing with a traceback.
        return [], [0, 0, 0, 0, 0], "KeyboardInterrupt during evaluation"

    except Exception as e:
        return [], [0, 0, 0, 0, 0], str(e)
    finally:
        # Restore numpy choice to avoid side effects outside evaluation.
        if "_orig_np_choice" in locals():
            np.random.choice = _orig_np_choice


def selection(population, n_parents, elitism=True, minimization=False):
    """
    Select the best individuals from the population, preferring diversity in code.

    Args:
        population: List of Individual objects
        n_parents: Number of parents to select
        elitism: Whether to use elitism (μ+λ) or comma strategy (μ,λ)
        minimization: Whether we're minimizing (False = maximizing)

    Returns:
        List of selected individuals
    """
    reverse = not minimization  # For maximization, sort in reverse
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

    # Prefer a second parent that is both code-distinct and strictly worse in fitness.
    second = None
    for ind in sorted_pop[1:]:
        code_key = ind.code or ""
        if code_key == best_code:
            continue
        if ind.fitness < best_fitness:
            second = ind
            break

    # If all candidates with different code have the same fitness, allow equal fitness but distinct code.
    if second is None:
        for ind in sorted_pop[1:]:
            code_key = ind.code or ""
            if code_key != best_code:
                second = ind
                break

    # As a last resort (e.g., only clones in the population), fall back to the next best individual.
    if second is None and len(sorted_pop) > 1:
        second = sorted_pop[1]

    if second is not None:
        selected.append(second)

    # If more parents are requested, fill the remaining slots with the next best
    # individuals, avoiding exact code duplicates when possible.
    seen_codes = {best_code, second.code or "" if second is not None else ""}
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        code_key = ind.code or ""
        if code_key in seen_codes:
            continue
        seen_codes.add(code_key)
        selected.append(ind)

    # Final safety fallback in case we still don't have enough parents.
    if len(selected) < n_parents:
        for ind in sorted_pop:
            if len(selected) >= n_parents:
                break
            if ind not in selected:
                selected.append(ind)

    return selected


def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='LLaMEA Thesis - Population-based Evolutionary Algorithm with LLM',
        epilog='''
Environment Variables:
  The program will automatically load environment variables from a .env file if present.
  Supported variables: OPENAI_API_KEY, BASE_URL, AIML_KEY
  
Examples:
  # Simple iterative mode (legacy, no populations)
  python main-thesis.py --api-key your_key --budget 50
  
  # Population-based evolutionary mode (NEW!)
  python main-thesis.py --evolutionary-mode --n-parents 4 --n-offspring 16 --budget 100
  
  # With elitism and detailed feedback
  python main-thesis.py --evolutionary-mode --elitism --detailed-feedback --budget 100
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
    parser.add_argument('--experiment-name', type=str, default='thesis-experiment',
                       help='Name for the experiment (default: thesis-experiment)')
    
    # Experimental Configuration
    parser.add_argument('--budget', type=int, default=100,
                       help='Total API calls budget (default: 100)')
    parser.add_argument('--eval-budget', type=int, default=10000,
                       help='Evaluation budget per algorithm (default: 10000)')
    parser.add_argument('--elitism', action='store_true',
                       help='Enable elitism (μ+λ) strategy, otherwise uses (μ,λ)')
    parser.add_argument('--detailed-feedback', action='store_true', 
                       help='Enable detailed feedback about algorithm performance')
    
    # Population Configuration
    parser.add_argument('--n-parents', type=int, default=4,
                       help='Number of parent algorithms to maintain (default: 4)')
    parser.add_argument('--n-offspring', type=int, default=16,
                       help='Number of offspring algorithms per generation (default: 16)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of generations (if set, overrides budget)')
    parser.add_argument('--evolutionary-mode', action='store_true',
                       help='Enable population-based evolutionary mode (recommended!)')
    
    # MADA 4.0 Configuration (DS-TS Bandit & Strategy Weights)
    parser.add_argument('--mada-strategy-weights', type=str, default='0.4,0.4,0.2',
                       help='Strategy weights for innovation,recombination,legacy (default: 0.4,0.4,0.2)')
    parser.add_argument('--mada-discount', type=float, default=0.97,
                       help='DS-TS discount factor gamma (default: 0.97)')
    parser.add_argument('--mada-tau-max', type=float, default=3.0,
                       help='DS-TS maximum posterior std dev tau_max (default: 3.0)')
    parser.add_argument('--mada-reward-variance', type=float, default=0.25,
                       help='DS-TS reward variance for posterior updates (default: 0.25)')
    parser.add_argument('--mada-placeholder-threshold', type=float, default=0.5,
                       help='Placeholder ratio threshold for warnings (default: 0.5)')
    parser.add_argument('--mada-semantic-linter', action='store_true',
                       help='Enable semantic linter for helper conflict resolution')
    parser.add_argument('--mada-reward-clamp', type=float, default=1.0,
                       help='Clamp reward to [-clamp, +clamp] for stability (default: 1.0)')
    parser.add_argument('--mada-log-bandits', action='store_true',
                       help='Log bandit state snapshots every generation')
    
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
    
    # Get base_url from argument or environment
    base_url = args.base_url or os.getenv("BASE_URL")
    
    # Setup experiment
    ai_model = args.model
    experiment_suffix = f"{ai_model}-{args.experiment_name}"
    if args.evolutionary_mode:
        experiment_suffix += "-evolutionary"
    if args.elitism:
        experiment_suffix += "-elitism"
    
    explogger = ExperimentLogger(experiment_suffix)
    algorithm_manager = AlgorithmManager(
        api_key, 
        explogger, 
        ai_model, 
        args.elitism, 
        args.detailed_feedback,
        base_url=base_url,
        max_tokens=args.max_tokens
    )

    # Determine mode and calculate generations
    if args.evolutionary_mode:
        # Parse MADA 4.0 strategy weights
        strategy_weights = None
        if args.mada_strategy_weights:
            try:
                weights = [float(w.strip()) for w in args.mada_strategy_weights.split(',')]
                if len(weights) == 3:
                    strategy_weights = {
                        'innovation': weights[0],
                        'recombination': weights[1],
                        'legacy': weights[2]
                    }
            except ValueError:
                print(f"WARNING: Invalid strategy weights '{args.mada_strategy_weights}', using defaults")
        
        mada_parser = BlockParser()
        mada_operator = MADAOperator(
            algorithm_manager,
            parser=mada_parser,
            discount=args.mada_discount,
            tau_max=args.mada_tau_max,
            reward_variance=args.mada_reward_variance,
            rng_seed=None,
            strategy_weights=strategy_weights,
            enable_semantic_linter=args.mada_semantic_linter,
            placeholder_threshold=args.mada_placeholder_threshold,
        )
        # Store MADA config for logging
        mada_config = {
            'strategy_weights': strategy_weights or {'innovation': 0.4, 'recombination': 0.4, 'legacy': 0.2},
            'discount': args.mada_discount,
            'tau_max': args.mada_tau_max,
            'reward_variance': args.mada_reward_variance,
            'placeholder_threshold': args.mada_placeholder_threshold,
            'semantic_linter': args.mada_semantic_linter,
            'reward_clamp': args.mada_reward_clamp,
            'log_bandits': args.mada_log_bandits,
        }
        # Population-based evolutionary mode
        if args.generations is not None:
            generations = args.generations
            total_budget = args.n_parents + (generations * args.n_offspring)
            print(f"INFO: Fixed generations mode")
            print(f"INFO: Generations: {generations}")
            print(f"INFO: Total API calls needed: {total_budget}")
            if total_budget > args.budget:
                print(f"WARNING: Total calls ({total_budget}) exceed budget ({args.budget})")
                print(f"WARNING: Will stop at budget limit")
        else:
            # Budget-driven calculation
            generations = max(1, (args.budget - args.n_parents) // args.n_offspring)
            total_budget = args.n_parents + (generations * args.n_offspring)
            print(f"INFO: Budget-driven evolutionary mode")
            print(f"INFO: Calculated generations: {generations}")
            print(f"INFO: Total API calls: {total_budget}")
        
        print(f"\nStarting Population-Based LLaMEA Thesis experiment:")
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
        print(f"  Detailed Feedback: {args.detailed_feedback}")
        print(f"  MADA 4.0 Config:")
        print(f"    Strategy Weights: {mada_config['strategy_weights']}")
        print(f"    DS-TS Discount (γ): {mada_config['discount']}")
        print(f"    DS-TS τ_max: {mada_config['tau_max']}")
        print(f"    Reward Clamp: ±{mada_config['reward_clamp']}")
        print(f"    Semantic Linter: {mada_config['semantic_linter']}")
        print(f"    Log Bandits: {mada_config['log_bandits']}")
        print("-" * 60)
        
        # Run evolutionary mode
        run_evolutionary_mode(
            algorithm_manager,
            explogger,
            args,
            generations,
            args.n_parents,
            args.n_offspring,
            mada_operator,
            mada_config,
        )
    else:
        # Legacy iterative refinement mode (original behavior)
        print(f"\nStarting Iterative LLaMEA Thesis experiment:")
        print(f"  Mode: ITERATIVE (1+1 refinement)")
        print(f"  Model: {ai_model}")
        print(f"  Base URL: {base_url or 'Default OpenAI'}")
        print(f"  Max Tokens: {args.max_tokens or 'Default'}")
        print(f"  API Budget: {args.budget}")
        print(f"  Evaluation Budget: {args.eval_budget}")
        print(f"  Elitism: {args.elitism}")
        print(f"  Detailed Feedback: {args.detailed_feedback}")
        print("-" * 60)
        
        # Run legacy mode
        run_iterative_mode(algorithm_manager, explogger, args)
    
    print(f"\nExperiment completed. Results saved in: {explogger.dirname}")

def run_iterative_mode(algorithm_manager, explogger, args):
    """
    Legacy iterative refinement mode (original main-thesis.py behavior).
    Simple (1+1) strategy: generate one algorithm, refine it iteratively.
    """
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0, 0, 0, 0, 0]
    algorithm_name = algorithm_name_long = ""
    
    last_auc_hash = ""
    for openai_try in range(args.budget):
        print(f"\n{'='*60}")
        print(f"API Call {openai_try + 1}/{args.budget}")
        print('='*60)
        
        if openai_try == 0:
            message = algorithm_manager.fetch_algorithm()
        else:
            message = algorithm_manager.refine_algorithm(auc_mean, auc_std, algorithm_name_long, detailed_aucs)

        try:
            new_algorithm = algorithm_manager.extract_algorithm_code(message)
            algorithm_name = re.findall("class\\s*(\\w*)\\:", new_algorithm, re.IGNORECASE)[0]
            algorithm_name_long = algorithm_manager.extract_algorithm_name(message)
            if algorithm_name_long == "":
                algorithm_name_long = algorithm_name

            explogger.log_code(openai_try, algorithm_name, new_algorithm)

            # Evaluate
            aucs, detailed_aucs, error = evaluate_algorithm(new_algorithm, algorithm_name, args.eval_budget)
            
            if error:
                auc_mean = 0.0
                auc_std = 0.0
                algorithm_manager.last_error = error
                print(f"Error: {error}")
            else:
                # Sanity checks before logging
                auc_len, auc_mean, auc_std = _summarize_aucs(aucs)
                auc_hash = _hash_aucs(aucs)
                if auc_len != EXPECTED_AUC_LEN:
                    print(f"WARNING: AUC length {auc_len} != expected {EXPECTED_AUC_LEN}")
                if last_auc_hash and auc_hash == last_auc_hash:
                    print("WARNING: AUC vector identical to previous attempt; possible reuse/caching.")
                explogger.log_aucs(
                    openai_try,
                    aucs,
                    metadata={"len": auc_len, "mean": auc_mean, "std": auc_std, "sha256": auc_hash},
                )
                last_auc_hash = auc_hash
                print(f"Algorithm: {algorithm_name_long}")
                print(f"AUC Mean: {auc_mean:.4f} ± {auc_std:.4f}")
                algorithm_manager.last_error = ""
                algorithm_manager.last_algorithm = message
            
        except NoCodeException:
            auc_mean = 0.0
            auc_std = 0.0
            print("Error: No code extracted from LLM response")
        except Exception as e:
            auc_mean = 0.0
            auc_std = 0.0
            algorithm_manager.last_error = repr(e)
            print(f"Error: {algorithm_manager.last_error}")
            import traceback
            traceback.print_exc()


def run_evolutionary_mode(
    algorithm_manager,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
    mada_operator,
    mada_config=None,
):
    """
    Population-based evolutionary mode using LLAMEA methodology.
    Implements (μ+λ) or (μ,λ) evolutionary strategy with LLM-generated algorithms.
    
    MADA 4.0 features:
    - DS-TS bandit-guided block selection (innovation/recombination/legacy)
    - Reward clamping for stability
    - Structured logging of offspring lineage and bandit states
    """
    mada_config = mada_config or {}
    reward_clamp = mada_config.get('reward_clamp', 1.0)
    log_bandits = mada_config.get('log_bandits', False)
    
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
                message = algorithm_manager.fetch_algorithm()
                algorithm_code = algorithm_manager.extract_algorithm_code(message)
                algorithm_name = re.findall("class\\s*(\\w*)\\:", algorithm_code, re.IGNORECASE)[0]
                algorithm_name_long = algorithm_manager.extract_algorithm_name(message)
                if algorithm_name_long == "":
                    algorithm_name_long = algorithm_name

                code_hash = hashlib.sha256((algorithm_code or "").encode("utf-8")).hexdigest()
                if code_hash in seen_hashes:
                    print("  Duplicate code hash detected, retrying initialization...")
                    retries += 1
                    continue

                solution = make_solution(
                    code=algorithm_code,
                    class_name=algorithm_name,
                    description=algorithm_name_long,
                    generation=generation,
                )

                print(f"  Evaluating {algorithm_name}...")
                aucs = []
                detailed_aucs = [0, 0, 0, 0, 0]
                aucs, detailed_aucs, error = evaluate_algorithm(
                    algorithm_code, algorithm_name, args.eval_budget
                )

                auc_len = 0
                auc_mean_val = 0.0
                auc_std_val = 0.0
                auc_hash = ""
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
                        print("  WARNING: AUC vector identical to previous attempt; possible reuse/caching.")
                    solution.fitness = float(np.mean(aucs))
                    solution.aucs = aucs
                    solution.detailed_aucs = detailed_aucs
                    solution.error = ""
                    seen_hashes.add(code_hash)
                    print(f"  Fitness: {solution.fitness:.4f}")
                    try:
                        mada_operator.ensure_blocks(solution)
                    except Exception as parser_err:
                        print(f"  Parser warning: {parser_err}")

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
                last_auc_hash = auc_hash if aucs else last_auc_hash
                api_calls += 1
                break

        except Exception as e:
            print(f"  Initialization error: {e}")
            import traceback
            traceback.print_exc()
    
    if not population:
        print("ERROR: Failed to initialize any individuals!")
        return
    
    # Sort initial population
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
        
        # Select parents for reproduction
        parents = selection(population, n_parents, elitism=args.elitism, minimization=False)
        pop_summary = "\n".join(
            [f"  - {ind.name}: fitness={ind.fitness:.4f}" for ind in parents[:5]]
        )
        best_parent_fitness = max((p.fitness for p in parents), default=0.0)
        
        # Generate offspring
        offspring = []
        for i in range(n_offspring):
            if api_calls >= args.budget:
                print(f"Budget exhausted at offspring {i}/{n_offspring}")
                break
            
            parent = random.choice(parents)

            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")
            print(f"  Parent: {parent.name} (fitness: {parent.fitness:.4f})")

            try:
                proposal = mada_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                    population_summary=f"Current population:\n{pop_summary}",
                )
                child = make_solution(
                    code=proposal.code,
                    class_name=proposal.class_name,
                    description=proposal.description,
                    generation=generation,
                    parent_ids=proposal.parent_ids,
                )
                child.operator = proposal.strategy
                child.set_lineage(proposal.lineage)

                print(f"  Evaluating {child.name} ({proposal.strategy})...")
                aucs = []
                detailed_aucs = [0, 0, 0, 0, 0]
                aucs, detailed_aucs, error = evaluate_algorithm(
                    proposal.code, proposal.class_name, args.eval_budget
                )

                raw_reward = -1.0  # Default for error cases
                auc_len = 0
                auc_mean_val = 0.0
                auc_std_val = 0.0
                auc_hash = ""
                if error:
                    child.fitness = 0.0
                    child.error = error
                    reward = -reward_clamp  # Clamp negative reward
                    print(f"  Error: {error}")
                else:
                    auc_len, auc_mean_val, auc_std_val = _summarize_aucs(aucs)
                    auc_hash = _hash_aucs(aucs)
                    if auc_len != EXPECTED_AUC_LEN:
                        print(f"  WARNING: AUC length {auc_len} != expected {EXPECTED_AUC_LEN}")
                    if last_auc_hash and auc_hash == last_auc_hash:
                        print("  WARNING: AUC vector identical to previous attempt; possible reuse/caching.")
                    child.fitness = float(np.mean(aucs))
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    child.error = ""
                    # MADA 4.0: Compute and clamp reward (Section 5.3)
                    raw_reward = child.fitness - best_parent_fitness
                    reward = max(-reward_clamp, min(reward_clamp, raw_reward))
                    # MADA 5.0: give a small positive signal for valid innovation ties
                    if proposal.strategy == "innovation" and reward == 0.0:
                        epsilon_reward = min(reward_clamp, 0.01)
                        reward = epsilon_reward
                        raw_reward = max(raw_reward, epsilon_reward)
                    try:
                        mada_operator.ensure_blocks(child)
                    except Exception as parser_err:
                        print(f"  Parser warning: {parser_err}")

                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                        best_ever = child

                # Update DS-TS bandits with richer reward signals:
                # - per-block reward uses raw_reward (unclamped) to give bandits more gradient.
                # - constraint flag propagates strong penalty on errors.
                per_block_rewards = {}
                lineage = child.get_lineage() or {}
                for decision in lineage.get("decisions", []):
                    blk = decision.get("block")
                    if blk:
                        per_block_rewards[blk] = raw_reward
                reward_detail = {
                    "per_block_rewards": per_block_rewards,
                    "constraint_violation": bool(child.error),
                    "constraint_penalty": -abs(reward_clamp),
                }
                # Update DS-TS bandits with clamped reward (for stability) plus detail.
                mada_operator.update_bandits(lineage, reward, reward_detail=reward_detail)
                
                # MADA 4.0: Structured offspring logging
                lineage = child.get_lineage() or {}
                offspring_record = {
                    'strategy': proposal.strategy,
                    'decisions': lineage.get('decisions', []),
                    'reward': reward,
                    'raw_reward': raw_reward,
                    'fitness': child.fitness,
                    'parent_fitness': best_parent_fitness,
                    'parent_ids': proposal.parent_ids,
                    'generation': generation,
                }
                explogger.log_mada_offspring(api_calls, offspring_record)
                
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
                last_auc_hash = auc_hash if aucs else last_auc_hash
                api_calls += 1

            except Exception as e:
                print(f"  Offspring generation error: {e}")
                import traceback
                traceback.print_exc()
        
        # Selection for next generation
        if args.elitism:
            # (μ+λ) strategy: select from parents + offspring
            combined = parents + offspring
            population = selection(combined, n_parents, elitism=True, minimization=False)
            print(f"\nSelection: (μ+λ) - Best {n_parents} from {len(combined)} individuals")
        else:
            # (μ,λ) strategy: select only from offspring
            population = selection(offspring, n_parents, elitism=False, minimization=False)
            print(f"\nSelection: (μ,λ) - Best {n_parents} from {len(offspring)} offspring")

        # Log both fitness and algorithm names for the new parent population
        print(
            "New population parents:",
            [f"{ind.name} (fitness={ind.fitness:.4f})" for ind in population[:n_parents]],
        )
        
        # MADA 4.0: Log bandit state snapshots if enabled
        if log_bandits:
            try:
                bandit_states = {}
                for block_name, sampler in mada_operator.bandits.items():
                    bandit_states[block_name] = sampler.get_state_snapshot(block_name)
                explogger.log_bandit_snapshot(generation, bandit_states)
            except Exception as bandit_err:
                print(f"  Bandit logging warning: {bandit_err}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("EVOLUTIONARY OPTIMIZATION COMPLETED")
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
        f.write(f"# Generation: {best_ever.generation}\n\n")
        f.write(best_ever.code)


if __name__ == "__main__":
    main()
