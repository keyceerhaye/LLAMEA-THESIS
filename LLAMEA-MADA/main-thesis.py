import os
import sys
from pathlib import Path
import numpy as np
from ioh import get_problem, logger
import re
import argparse
import random
import traceback
from collections import Counter, deque
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent
LLAMEA_SRC = PROJECT_ROOT / "llamea-experimentation" / "src"
if LLAMEA_SRC.exists() and str(LLAMEA_SRC) not in sys.path:
    sys.path.insert(0, str(LLAMEA_SRC))

from managers import (
    AlgorithmManager,
    ExperimentLogger,
    LLMTransientError,
    InvalidAlgorithmError,
)
from utils import OverBudgetException, aoc_logger, correct_aoc
from llamea.solution import Solution
from llamea.mada import MADAOperator, BlockParser
from llamea.mada.ds_ts import DiscountedThompsonSampler
from llamea.utils import NoCodeException

NEGATIVE_REWARD_FLOOR = -0.3
SYNTAX_ERROR_REWARD = -0.05
POSITIVE_REWARD_CEILING = 0.3
REWARD_BASELINE_DECAY = 0.9
TRACEBACK_TAIL_LINES = 8
STAGNATION_WINDOW = 4
STAGNATION_THRESHOLD = 0.01
STAGNATION_RECOVERY_GENERATIONS = 3

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


def _family_key(individual):
    """Best-effort identifier for an algorithm 'family' based on lineage metadata."""

    lineage = None
    getter = getattr(individual, "get_lineage", None)
    if callable(getter):
        lineage = getter()
    elif hasattr(individual, "metadata"):
        lineage = individual.metadata.get("mada_lineage")

    if isinstance(lineage, dict):
        family_id = lineage.get("family_id")
        if family_id:
            return f"family:{family_id}"
        parents = lineage.get("parents")
        if parents:
            return "parents:" + "-".join(sorted(parents))
        strategy = lineage.get("strategy")
        if strategy:
            return f"strategy:{strategy}"

    operator = getattr(individual, "operator", None)
    if operator:
        return f"operator:{operator}"
    description = getattr(individual, "description", None)
    if description:
        return f"description:{description}"
    return getattr(individual, "name", "") or getattr(individual, "id", "")


def _fitness_close(a, b, rel_tol=1e-4):
    """Return True if two fitness values are effectively equal."""

    scale = max(1.0, abs(a), abs(b))
    return abs(a - b) <= rel_tol * scale


class RewardBaselineTracker:
    """Maintains a smooth global baseline for bandit rewards."""

    def __init__(self, decay: float = REWARD_BASELINE_DECAY):
        self.decay = decay
        self.value = None

    def observe(self, observation):
        if observation is None:
            return self.current()
        if self.value is None:
            self.value = observation
        else:
            self.value = (self.decay * self.value) + ((1 - self.decay) * observation)
        return self.value

    def current(self, fallback=0.0):
        return self.value if self.value is not None else fallback


class StrategyBaselineManager:
    """Tracks independent EWMA baselines for each strategy arm."""

    def __init__(self, strategies, decay: float = REWARD_BASELINE_DECAY):
        self.decay = decay
        self.trackers = {name: RewardBaselineTracker(decay=decay) for name in strategies}

    def current(self, strategy: str, fallback: float = 0.0) -> float:
        tracker = self.trackers.setdefault(strategy, RewardBaselineTracker(self.decay))
        return tracker.current(fallback)

    def observe(self, strategy: str, value: float):
        tracker = self.trackers.setdefault(strategy, RewardBaselineTracker(self.decay))
        tracker.observe(value)


def _format_exception_with_traceback(label: str, exc: BaseException) -> str:
    """Return a short message plus the tail of the traceback."""

    tb = traceback.format_exc()
    if not tb:
        return f"{label} ({exc.__class__.__name__}): {exc}"
    tail = "\n".join(tb.strip().splitlines()[-TRACEBACK_TAIL_LINES:])
    return f"{label} ({exc.__class__.__name__}): {exc}\n{tail}"


def _classify_failure(error_text: str, invalid_code: bool = False) -> str:
    """Return 'syntax' or 'runtime' to differentiate reward shaping."""

    if invalid_code:
        return "syntax"
    if not error_text:
        return "success"
    lowered = error_text.lower()
    syntax_markers = (
        "syntaxerror",
        "invalid syntax",
        "undefined identifier",
        "undefined_identifiers",
        "no code extracted",
        "invalidalgorithmerror",
        "indentationerror",
    )
    if any(marker in lowered for marker in syntax_markers):
        return "syntax"
    return "runtime"


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
        # Execute the algorithm code
        exec(algorithm_code, globals())
        
        # Get the algorithm class
        if algorithm_name not in globals():
            return [], [0, 0, 0, 0, 0], f"Class {algorithm_name} not found"
        
        algorithm_class = globals()[algorithm_name]
        
        # Setup logging
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
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    except Exception as e:
                        error_msg = _format_exception_with_traceback(
                            "Evaluation runtime failure", e
                        )
                        return [], [0, 0, 0, 0, 0], error_msg
                    
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
        error_msg = _format_exception_with_traceback("Evaluation error", e)
        return [], [0, 0, 0, 0, 0], error_msg


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
    best_family = _family_key(best)

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
        if _fitness_close(second.fitness, best_fitness) and _family_key(second) == best_family:
            alternate = next(
                (
                    ind
                    for ind in sorted_pop[1:]
                    if ind is not second
                    and _fitness_close(ind.fitness, best_fitness)
                    and _family_key(ind) != best_family
                ),
                None,
            )
            if alternate is not None:
                second = alternate

    if second is not None:
        selected.append(second)

    # If more parents are requested, fill the remaining slots with the next best
    # individuals, avoiding exact code duplicates when possible.
    seen_codes = {best_code}
    if second is not None:
        seen_codes.add(second.code or "")
    seen_families = {best_family}
    if second is not None:
        seen_families.add(_family_key(second))
    deferred = []
    for ind in sorted_pop[2:]:
        if len(selected) >= n_parents:
            break
        code_key = ind.code or ""
        if code_key in seen_codes:
            continue
        family = _family_key(ind)
        if _fitness_close(ind.fitness, best_fitness) and family in seen_families:
            deferred.append(ind)
            continue
        seen_codes.add(code_key)
        seen_families.add(family)
        selected.append(ind)

    if len(selected) < n_parents:
        for ind in deferred:
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
    parser.add_argument('--max-tokens', type=int, default=16384, help='Maximum tokens for API responses')
    
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
    parser.add_argument(
        '--syntax-error-reward',
        type=float,
        default=SYNTAX_ERROR_REWARD,
        help='Reward assigned to syntax/validation failures (default: -0.05)',
    )
    parser.add_argument(
        '--negative-reward-floor',
        type=float,
        default=NEGATIVE_REWARD_FLOOR,
        help='Minimum reward cap for runtime failures (default: -0.3)',
    )
    parser.add_argument(
        '--positive-reward-ceiling',
        type=float,
        default=POSITIVE_REWARD_CEILING,
        help='Maximum reward cap for successes (default: 0.3)',
    )
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
    parser.add_argument('--allow-refresh', action='store_true',
                       help='Allow parent refresh after extended stagnation')
    
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
    
    syntax_error_reward = args.syntax_error_reward
    negative_reward_floor = args.negative_reward_floor
    positive_reward_ceiling = args.positive_reward_ceiling

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
        mada_parser = BlockParser()
        mada_operator = MADAOperator(
            algorithm_manager,
            parser=mada_parser,
            rng_seed=None,
        )
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
            syntax_error_reward,
            negative_reward_floor,
            positive_reward_ceiling,
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

            algorithm_manager.validate_identifier_usage(new_algorithm, allow_tutoring=True, soft_undefined=True)
            explogger.log_code(openai_try, algorithm_name, new_algorithm)

            # Evaluate
            aucs, detailed_aucs, error = evaluate_algorithm(new_algorithm, algorithm_name, args.eval_budget)
            
            if error:
                auc_mean = 0.0
                auc_std = 0.0
                algorithm_manager.last_error = error
                algorithm_manager.last_traceback_tail = error
                print(f"Error: {error}")
            else:
                auc_mean = np.mean(aucs)
                auc_std = np.std(aucs)
                explogger.log_aucs(openai_try, aucs)
                print(f"Algorithm: {algorithm_name_long}")
                print(f"AUC Mean: {auc_mean:.4f} ± {auc_std:.4f}")
                algorithm_manager.last_error = ""
                algorithm_manager.last_algorithm = message
                algorithm_manager.last_traceback_tail = ""
            
        except NoCodeException:
            auc_mean = 0.0
            auc_std = 0.0
            print("Error: No code extracted from LLM response")
            algorithm_manager.last_traceback_tail = ""
        except InvalidAlgorithmError as exc:
            auc_mean = 0.0
            auc_std = 0.0
            algorithm_manager.last_error = str(exc)
            algorithm_manager.last_traceback_tail = algorithm_manager.last_error
            print(f"Error: {algorithm_manager.last_error}")
            explogger.log_conversation(f"[INVALID ALG] {algorithm_manager.last_error}")
        except Exception as e:
            auc_mean = 0.0
            auc_std = 0.0
            algorithm_manager.last_error = repr(e)
            print(f"Error: {algorithm_manager.last_error}")
            import traceback
            tb = traceback.format_exc()
            algorithm_manager.last_traceback_tail = tb or repr(e)
            traceback.print_exc()


def run_evolutionary_mode(
    algorithm_manager,
    explogger,
    args,
    generations,
    n_parents,
    n_offspring,
    mada_operator,
    syntax_error_reward,
    negative_reward_floor,
    positive_reward_ceiling,
):
    """
    Population-based evolutionary mode using LLAMEA methodology.
    Implements (μ+λ) or (μ,λ) evolutionary strategy with LLM-generated algorithms.
    """
    population = []
    best_ever = None
    api_calls = 0
    generation = 0
    reward_baseline = RewardBaselineTracker()
    strategy_baselines = StrategyBaselineManager(["innovation", "recombination", "legacy"])
    outer_controller = DiscountedThompsonSampler(
        ["mada", "legacy"], discount=0.97, tau_max=3.0, reward_variance=0.25
    )
    outer_choice_counter = Counter()
    refresh_events: List[Dict[str, object]] = []
    stagnation_stretch = 0
    LEGACY_SYNTAX_THRESHOLD = 3
    LEGACY_NUDGE_DURATION = 4
    legacy_syntax_streak = 0
    legacy_nudge_cooldown = 0

    def _attempt_parent_refresh(current_generation: int) -> bool:
        nonlocal api_calls, population, best_ever
        if api_calls >= args.budget:
            return False
        print(
            f"\n[REFRESH] Stagnation exceeded threshold; requesting new parent "
            f"(API call {api_calls + 1})."
        )
        try:
            message = algorithm_manager.fetch_algorithm()
            algorithm_code = algorithm_manager.extract_algorithm_code(message)
            algorithm_manager.validate_identifier_usage(algorithm_code, allow_tutoring=True, soft_undefined=True)
            algorithm_name = re.findall("class\\s*(\\w*)\\:", algorithm_code, re.IGNORECASE)[0]
            algorithm_name_long = algorithm_manager.extract_algorithm_name(message) or algorithm_name
        except (NoCodeException, InvalidAlgorithmError) as exc:
            print(f"[REFRESH] Failed to obtain valid algorithm: {exc}")
            explogger.log_conversation(f"[REFRESH FAILURE] {exc}")
            return False

        refresh_solution = make_solution(
            code=algorithm_code,
            class_name=algorithm_name,
            description=algorithm_name_long,
            generation=current_generation,
        )
        aucs, detailed_aucs, error = evaluate_algorithm(
            algorithm_code, algorithm_name, args.eval_budget
        )
        if error:
            print(f"[REFRESH] Evaluation error: {error}")
            explogger.log_conversation(f"[REFRESH FAILURE:{algorithm_name}] {error}")
            algorithm_manager.last_error = error
            algorithm_manager.last_traceback_tail = error
            return False

        refresh_solution.fitness = float(np.mean(aucs))
        refresh_solution.aucs = aucs
        refresh_solution.detailed_aucs = detailed_aucs
        refresh_solution.error = ""
        try:
            mada_operator.ensure_blocks(refresh_solution)
        except Exception as parser_err:
            print(f"[REFRESH] Parser warning: {parser_err}")

        # Replace worst parent
        if population:
            worst_parent = min(population, key=lambda sol: sol.fitness)
            population.remove(worst_parent)
        population.append(refresh_solution)
        reward_baseline.observe(refresh_solution.fitness)
        if refresh_solution.fitness > 0:
            strategy_baselines.observe("legacy", refresh_solution.fitness)
        if refresh_solution.fitness > best_ever.fitness:
            best_ever = refresh_solution
        explogger.log_code(api_calls, refresh_solution.name, refresh_solution.code)
        explogger.log_aucs(api_calls, aucs if aucs else [0])
        refresh_events.append(
            {
                "generation": current_generation,
                "name": refresh_solution.name,
                "fitness": refresh_solution.fitness,
            }
        )
        api_calls += 1
        print(
            f"[REFRESH] Replaced weakest parent with {refresh_solution.name} "
            f"(fitness={refresh_solution.fitness:.4f})."
        )
        explogger.log_conversation(
            f"[REFRESH] Generation {current_generation}: introduced {refresh_solution.name} "
            f"(fitness {refresh_solution.fitness:.4f})."
        )
        return True
    
    # Phase 1: Initialize parent population
    print(f"\n{'='*60}")
    print(f"INITIALIZATION: Generating {n_parents} parent algorithms")
    print('='*60)
    
    for i in range(n_parents):
        if api_calls >= args.budget:
            print(f"Budget exhausted during initialization at {api_calls} calls")
            break
            
        print(f"\nInitializing Parent {i+1}/{n_parents} (API call {api_calls+1})")
        
        try:
            message = algorithm_manager.fetch_algorithm()
            algorithm_code = algorithm_manager.extract_algorithm_code(message)
            algorithm_name = re.findall("class\\s*(\\w*)\\:", algorithm_code, re.IGNORECASE)[0]
            algorithm_name_long = algorithm_manager.extract_algorithm_name(message)
            if algorithm_name_long == "":
                algorithm_name_long = algorithm_name

            solution = make_solution(
                code=algorithm_code,
                class_name=algorithm_name,
                description=algorithm_name_long,
                generation=generation,
            )

            try:
                algorithm_manager.validate_identifier_usage(algorithm_code, allow_tutoring=True, soft_undefined=True)
            except InvalidAlgorithmError as exc:
                solution.fitness = 0.0
                solution.error = str(exc)
                algorithm_manager.last_error = solution.error
                algorithm_manager.last_traceback_tail = solution.error
                print(f"  Error: {solution.error}")
                explogger.log_conversation(f"[INVALID:{algorithm_name}] {solution.error}")
                population.append(solution)
                explogger.log_code(api_calls, algorithm_name, algorithm_code)
                explogger.log_aucs(api_calls, [0])
                api_calls += 1
                continue

            print(f"  Evaluating {algorithm_name}...")
            aucs = []
            detailed_aucs = [0, 0, 0, 0, 0]
            aucs, detailed_aucs, error = evaluate_algorithm(
                algorithm_code, algorithm_name, args.eval_budget
            )

            if error:
                solution.fitness = 0.0
                solution.error = error
                print(f"  Error: {error}")
            else:
                solution.fitness = float(np.mean(aucs))
                solution.aucs = aucs
                solution.detailed_aucs = detailed_aucs
                solution.error = ""
                print(f"  Fitness: {solution.fitness:.4f}")
                try:
                    mada_operator.ensure_blocks(solution)
                except Exception as parser_err:
                    print(f"  Parser warning: {parser_err}")
                if solution.fitness > 0:
                    reward_baseline.observe(solution.fitness)

            population.append(solution)
            explogger.log_code(api_calls, algorithm_name, algorithm_code)
            explogger.log_aucs(api_calls, aucs if aucs else [0])
            api_calls += 1

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
    
    best_history = deque(maxlen=STAGNATION_WINDOW)
    stagnation_generations_remaining = 0
    
    # Phase 2: Evolutionary loop
    for gen in range(1, generations + 1):
        if api_calls >= args.budget:
            print(f"\nBudget exhausted at generation {gen}")
            break
        
        generation = gen
        mada_operator.step_generation()
        best_history.append(best_ever.fitness if best_ever else 0.0)
        stagnation_trigger = False
        if len(best_history) == best_history.maxlen:
            earlier = best_history[0]
            latest = best_history[-1]
            if earlier > 0:
                improvement = (latest - earlier) / max(1e-8, abs(earlier))
            else:
                improvement = latest - earlier
            if improvement < STAGNATION_THRESHOLD:
                stagnation_trigger = True
        if stagnation_trigger:
            stagnation_stretch += 1
        else:
            stagnation_stretch = 0
        if stagnation_trigger and stagnation_generations_remaining <= 0:
            stagnation_generations_remaining = STAGNATION_RECOVERY_GENERATIONS
            mada_operator.boost_exploration(duration=STAGNATION_RECOVERY_GENERATIONS)
            stagnation_msg = (
                f"[STAGNATION] Generation {generation}: best fitness "
                f"{best_ever.fitness:.4f} stalled; boosting exploration."
            )
            print(stagnation_msg)
            explogger.log_conversation(stagnation_msg)
        stagnation_active = stagnation_generations_remaining > 0

        if (
            args.allow_refresh
            and stagnation_stretch >= 10
            and _attempt_parent_refresh(generation)
        ):
            stagnation_stretch = 0

        stagnation_recovered = False
        legacy_injected = not stagnation_active
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
        if best_parent_fitness > 0:
            reward_baseline.observe(best_parent_fitness)
        
        # Generate offspring
        offspring = []
        generation_strategy_counter = Counter()
        generation_failure_counts = Counter()
        controller_nudges: List[Dict[str, object]] = []
        for i in range(n_offspring):
            if api_calls >= args.budget:
                print(f"Budget exhausted at offspring {i}/{n_offspring}")
                break
            
            if legacy_nudge_cooldown > 0:
                legacy_nudge_cooldown -= 1

            parent = random.choice(parents)

            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")
            print(f"  Parent: {parent.name} (fitness: {parent.fitness:.4f})")

            proposal = None
            force_strategy = None
            outer_choice, _, _ = outer_controller.select_arm("outer_loop")
            effective_outer_choice = outer_choice
            legacy_suppressed = False
            if (
                legacy_nudge_cooldown > 0
                and outer_choice == "legacy"
                and not stagnation_active
            ):
                legacy_suppressed = True
                effective_outer_choice = "mada"
            if effective_outer_choice == "legacy":
                force_strategy = "legacy"
            if stagnation_active and not legacy_injected:
                force_strategy = "legacy"
                effective_outer_choice = "legacy"
                legacy_suppressed = False
            if legacy_suppressed:
                suppression_msg = (
                    "[OUTER] Suppressed a legacy request due to consecutive syntax penalties."
                )
                print(suppression_msg)
                explogger.log_conversation(suppression_msg)
                controller_nudges.append(
                    {
                        "generation": generation,
                        "event": "legacy_suppressed",
                        "offspring": i + 1,
                    }
                )
            try:
                proposal = mada_operator.generate_offspring(
                    parents=parents,
                    focal_parent=parent,
                    population_summary=f"Current population:\n{pop_summary}",
                    force_strategy=force_strategy,
                )
                if force_strategy == "legacy":
                    legacy_injected = True
                child = make_solution(
                    code=proposal.code,
                    class_name=proposal.class_name,
                    description=proposal.description,
                    generation=generation,
                    parent_ids=proposal.parent_ids,
                )
                child.operator = proposal.strategy
                child.set_lineage(proposal.lineage)
                generation_strategy_counter[child.operator] += 1

                print(f"  Evaluating {child.name} ({proposal.strategy})...")
                aucs: List[float] = []
                detailed_aucs = [0, 0, 0, 0, 0]
                invalid_code_error = None
                strategy_label = child.operator
                strategy_baseline_value = strategy_baselines.current(
                    strategy_label, best_parent_fitness
                )
                global_baseline_value = reward_baseline.current(best_parent_fitness)
                tutoring_mode = all(
                    cooldown > 0 for cooldown in mada_operator.strategy_cooldowns.values()
                )
                tutoring_mode_local = tutoring_mode
                tutoring_issues = []
                success = False
                try:
                    # Heuristic auto-repair for tuple destructuring before validation
                    proposal_code = proposal.code
                    repaired_code, fixes = algorithm_manager.coerce_tuple_loops(proposal_code)
                    if fixes:
                        explogger.log_conversation(
                            f"[AUTO-FIX] Applied {fixes} tuple loop repair(s) to {child.name}"
                        )
                        proposal_code = repaired_code

                    tutoring_issues = algorithm_manager.validate_identifier_usage(
                        proposal_code, allow_tutoring=tutoring_mode_local, soft_undefined=True
                    ) or []
                except InvalidAlgorithmError as exc:
                    invalid_code_error = str(exc)
                    # If we applied any repair and the failure is undefined identifiers,
                    # soften to tutoring and proceed to evaluation for telemetry.
                    if fixes and "Undefined identifiers detected" in invalid_code_error:
                        tutoring_mode_local = True
                        try:
                            tutoring_issues = (
                                algorithm_manager.validate_identifier_usage(
                                    proposal_code, allow_tutoring=True, soft_undefined=True
                                )
                                or []
                            )
                            invalid_code_error = None
                        except InvalidAlgorithmError:
                            tutoring_mode_local = tutoring_mode
                            invalid_code_error = str(exc)

                failure_kind = "success"
                if invalid_code_error:
                    child.fitness = 0.0
                    child.error = invalid_code_error
                    reward = syntax_error_reward
                    failure_kind = "syntax"
                    strategy_baselines.observe(strategy_label, best_parent_fitness)
                    algorithm_manager.last_error = child.error
                    algorithm_manager.last_traceback_tail = child.error
                    print(f"  Error: {child.error}")
                    explogger.log_conversation(f"[INVALID:{child.name}] {child.error}")
                elif tutoring_mode_local and tutoring_issues:
                    # Separate fatal vs soft tutoring issues
                    fatal_issues = [
                        issue
                        for issue in tutoring_issues
                        if issue.startswith("forbidden_archive") or issue.startswith("undefined_identifiers")
                    ]
                    if fatal_issues:
                        issue_text = "; ".join(fatal_issues)
                        child.fitness = 0.0
                        child.error = f"TUTORING_FATAL: {issue_text}"
                        reward = syntax_error_reward
                        failure_kind = "syntax"
                        strategy_baselines.observe(strategy_label, best_parent_fitness)
                        algorithm_manager.last_error = child.error
                        algorithm_manager.last_traceback_tail = child.error
                        print(f"  Tutoring fatal: {issue_text}")
                        explogger.log_conversation(f"[INVALID-TUTORING:{child.name}] {issue_text}")
                    else:
                        issue_text = "; ".join(tutoring_issues)
                        print(f"  Tutoring mode: allowing evaluation despite issues: {issue_text}")
                        explogger.log_conversation(f"[TUTORING:{child.name}] {issue_text}")
                        child.error = f"TUTORING_WARN: {issue_text}"
                else:
                    aucs, detailed_aucs, error = evaluate_algorithm(
                        proposal_code, proposal.class_name, args.eval_budget
                    )

                    success = not bool(error)
                    if success:
                        child.fitness = float(np.mean(aucs))
                        child.aucs = aucs
                        child.detailed_aucs = detailed_aucs
                        child.error = ""
                        reward = child.fitness - strategy_baseline_value
                        strategy_baselines.observe(strategy_label, child.fitness)
                        reward_baseline.observe(child.fitness)
                        try:
                            mada_operator.ensure_blocks(child)
                        except Exception as parser_err:
                            print(f"  Parser warning: {parser_err}")
                        if child.fitness > best_ever.fitness:
                            if stagnation_generations_remaining > 0:
                                stagnation_generations_remaining = 0
                                stagnation_recovered = True
                                recovery_msg = (
                                    f"[STAGNATION] Cleared via {child.name} "
                                    f"({child.fitness:.4f}). Restoring default weights."
                                )
                                print(recovery_msg)
                                explogger.log_conversation(recovery_msg)
                                mada_operator.reset_strategy_weights()
                            print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                            best_ever = child
                        algorithm_manager.last_error = ""
                        algorithm_manager.last_traceback_tail = ""
                    else:
                        child.fitness = 0.0
                        child.error = error
                        failure_kind = _classify_failure(child.error)
                        if failure_kind == "syntax":
                            reward = syntax_error_reward
                        else:
                            reward = negative_reward_floor
                        strategy_baselines.observe(strategy_label, best_parent_fitness)
                        algorithm_manager.last_error = child.error
                        algorithm_manager.last_traceback_tail = child.error
                        print(f"  Error: {error}")
                        explogger.log_conversation(f"[FAILURE:{child.name}] {error}")

                if failure_kind != "success":
                    generation_failure_counts[failure_kind] += 1
                else:
                    generation_failure_counts["success"] += 1

                if strategy_label == "legacy":
                    if failure_kind == "syntax":
                        legacy_syntax_streak += 1
                    elif success:
                        legacy_syntax_streak = 0
                    else:
                        legacy_syntax_streak = 0
                elif success:
                    legacy_syntax_streak = 0

                if (
                    legacy_syntax_streak >= LEGACY_SYNTAX_THRESHOLD
                    and legacy_nudge_cooldown == 0
                ):
                    legacy_nudge_cooldown = LEGACY_NUDGE_DURATION
                    legacy_syntax_streak = 0
                    nudge_msg = (
                        f"[OUTER] Legacy accrued {LEGACY_SYNTAX_THRESHOLD} consecutive syntax penalties. "
                        f"Down-weighting legacy for the next {LEGACY_NUDGE_DURATION} offspring."
                    )
                    print(nudge_msg)
                    explogger.log_conversation(nudge_msg)
                    controller_nudges.append(
                        {
                            "generation": generation,
                            "event": "legacy_nudge",
                            "duration": LEGACY_NUDGE_DURATION,
                        }
                    )
                    outer_controller.update(
                        "outer_loop", "legacy", negative_reward_floor
                    )

                reward = max(
                    min(reward, positive_reward_ceiling),
                    negative_reward_floor,
                )
                mada_operator.record_strategy_outcome(
                    child.operator, success, child.error if not success else ""
                )
                mada_operator.update_bandits(child.get_lineage(), reward)
                outer_controller.update("outer_loop", effective_outer_choice, reward)
                outer_choice_counter[effective_outer_choice] += 1
                explogger.log_reward_snapshot(
                    api_calls,
                    child.name,
                    child.operator,
                    child.fitness,
                    global_baseline_value,
                    reward,
                    strategy_baseline_value,
                    effective_outer_choice,
                    failure_kind,
                    child.error,
                )
                offspring.append(child)
                explogger.log_code(api_calls, child.name, child.code)
                explogger.log_aucs(api_calls, aucs if aucs else [0])
                api_calls += 1

            except Exception as e:
                formatted = _format_exception_with_traceback(
                    "Offspring generation error", e
                )
                algorithm_manager.last_error = formatted
                algorithm_manager.last_traceback_tail = formatted
                if isinstance(e, LLMTransientError):
                    print(f"  LLM request failed after retries: {formatted}")
                    explogger.log_conversation(f"[LLM TRANSIENT] {formatted}")
                    if force_strategy == "legacy":
                        legacy_injected = False
                    continue
                failed_strategy = force_strategy or getattr(proposal, "strategy", "innovation")
                mada_operator.record_strategy_outcome(failed_strategy, False, formatted)
                print(f"  Offspring generation error: {formatted}")
                explogger.log_conversation(f"[OFFSPRING ERROR] {formatted}")
                outer_controller.update("outer_loop", effective_outer_choice, negative_reward_floor)
                if force_strategy == "legacy":
                    legacy_injected = False
        
        if stagnation_active and not legacy_injected:
            warning_msg = (
                f"[STAGNATION] Generation {generation}: legacy fallback could not be "
                "scheduled before hitting the budget."
            )
            print(warning_msg)
            explogger.log_conversation(warning_msg)
        
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
        print(f"Outer loop choices this generation: {dict(outer_choice_counter)}")
        print(
            "Failure counts - syntax: "
            f"{generation_failure_counts.get('syntax', 0)}, runtime: "
            f"{generation_failure_counts.get('runtime', 0)}"
        )
        placeholder_snapshot = []
        for ind in population[:n_parents]:
            entry = {"name": ind.name}
            try:
                parsed = mada_operator.ensure_blocks(ind)
                coverage = getattr(parsed, "coverage", {}) or {}
                entry.update(
                    placeholder_ratio=coverage.get("placeholder_ratio"),
                    real_blocks=coverage.get("real_blocks"),
                    total_blocks=coverage.get("total_blocks"),
                )
            except Exception as parser_err:
                entry["error"] = repr(parser_err)
            placeholder_snapshot.append(entry)
        monitoring_payload = {
            "schema_version": "1.1",
            "generation": generation,
            "strategy_counts": dict(generation_strategy_counter),
            "placeholder_snapshot": placeholder_snapshot,
            "bandits": mada_operator.export_bandit_snapshot(),
            "strategy_cooldowns": dict(mada_operator.strategy_cooldowns),
            "strategy_failure_streaks": dict(mada_operator.strategy_failure_counts),
        }
        monitoring_payload["syntax_penalties"] = generation_failure_counts.get("syntax", 0)
        monitoring_payload["runtime_failures"] = generation_failure_counts.get("runtime", 0)
        monitoring_payload["strategy_baselines_snapshot"] = {
            name: strategy_baselines.current(name)
            for name in strategy_baselines.trackers.keys()
        }
        if getattr(mada_operator, "cooldown_events", None):
            monitoring_payload["cooldown_events"] = list(mada_operator.cooldown_events)
        monitoring_payload["outer_controller"] = outer_controller.get_state_snapshot("outer_loop")
        monitoring_payload["outer_choice_counts"] = dict(outer_choice_counter)
        if controller_nudges:
            monitoring_payload["controller_nudges"] = list(controller_nudges)
        if refresh_events:
            monitoring_payload["parent_refresh"] = list(refresh_events)
            monitoring_payload["refresh_events"] = list(refresh_events)
        explogger.log_monitoring_snapshot(monitoring_payload)
        if getattr(mada_operator, "cooldown_events", None) is not None:
            mada_operator.cooldown_events.clear()
        outer_choice_counter.clear()
        refresh_events.clear()
        
        if not stagnation_recovered and stagnation_generations_remaining > 0:
            stagnation_generations_remaining -= 1
            if stagnation_generations_remaining == 0:
                decay_msg = "[STAGNATION] Boost window elapsed; reverting strategy weights."
                print(decay_msg)
                explogger.log_conversation(decay_msg)
                mada_operator.reset_strategy_weights()
    
    # Final summary
    print(f"\n{'='*60}")
    print("EVOLUTIONARY OPTIMIZATION COMPLETED")
    print('='*60)
    print(f"Total API calls: {api_calls}")
    print(f"Generations completed: {generation}")
    print(f"Best algorithm: {best_ever.name}")
    print(f"Best fitness: {best_ever.fitness:.4f}")
    print(f"Best generation: {best_ever.generation}")
    
    # Save best algorithm with explicit UTF-8 encoding for Windows safety
    best_path = Path(explogger.dirname) / "BEST_ALGORITHM.py"
    best_contents = (
        f"# Best Algorithm: {best_ever.name}\n"
        f"# Fitness: {best_ever.fitness:.4f}\n"
        f"# Generation: {best_ever.generation}\n\n"
        f"{best_ever.code}"
    )
    best_path.write_text(best_contents, encoding="utf-8")


if __name__ == "__main__":
    main()
