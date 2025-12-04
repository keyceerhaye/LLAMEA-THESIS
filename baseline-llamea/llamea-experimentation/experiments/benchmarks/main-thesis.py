from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import os
import numpy as np
from ioh import get_problem, logger
import re
import argparse
import random
from managers import AlgorithmManager, ExperimentLogger
from utils import OverBudgetException, aoc_logger, correct_aoc
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


class Individual:
    """Represents an individual algorithm in the population"""
    def __init__(self, code="", name="", description="", fitness=0.0, generation=0):
        self.code = code
        self.name = name
        self.description = description
        self.fitness = fitness
        self.generation = generation
        self.aucs = []
        self.detailed_aucs = [0, 0, 0, 0, 0]
        self.error = ""
    
    def __repr__(self):
        return f"Individual({self.name}, fitness={self.fitness:.4f}, gen={self.generation})"



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
        
    except Exception as e:
        return [], [0, 0, 0, 0, 0], str(e)


def selection(population, n_parents, elitism=True, minimization=False):
    """
    Select the best individuals from the population.
    
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
    return sorted_pop[:n_parents]


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
            algorithm_manager, explogger, args, 
            generations, args.n_parents, args.n_offspring
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

            explogger.log_code(openai_try, algorithm_name, new_algorithm)

            # Evaluate
            aucs, detailed_aucs, error = evaluate_algorithm(new_algorithm, algorithm_name, args.eval_budget)
            
            if error:
                auc_mean = 0.0
                auc_std = 0.0
                algorithm_manager.last_error = error
                print(f"Error: {error}")
            else:
                auc_mean = np.mean(aucs)
                auc_std = np.std(aucs)
                explogger.log_aucs(openai_try, aucs)
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


def run_evolutionary_mode(algorithm_manager, explogger, args, generations, n_parents, n_offspring):
    """
    Population-based evolutionary mode using LLAMEA methodology.
    Implements (μ+λ) or (μ,λ) evolutionary strategy with LLM-generated algorithms.
    """
    population = []
    best_ever = None
    api_calls = 0
    generation = 0
    
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
            
            # Create individual
            individual = Individual(
                code=algorithm_code,
                name=algorithm_name,
                description=algorithm_name_long,
                generation=generation
            )
            
            # Evaluate
            print(f"  Evaluating {algorithm_name}...")
            aucs, detailed_aucs, error = evaluate_algorithm(algorithm_code, algorithm_name, args.eval_budget)
            
            if error:
                individual.fitness = 0.0
                individual.error = error
                print(f"  Error: {error}")
            else:
                individual.fitness = np.mean(aucs)
                individual.aucs = aucs
                individual.detailed_aucs = detailed_aucs
                print(f"  Fitness: {individual.fitness:.4f}")
            
            population.append(individual)
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
        
        # Generate offspring
        offspring = []
        for i in range(n_offspring):
            if api_calls >= args.budget:
                print(f"Budget exhausted at offspring {i}/{n_offspring}")
                break
            
            # Select a random parent
            parent = random.choice(parents)
            
            print(f"\nOffspring {i+1}/{n_offspring} (API call {api_calls+1})")
            print(f"  Parent: {parent.name} (fitness: {parent.fitness:.4f})")
            
            try:
                # Build population summary for LLM context
                pop_summary = "\n".join([
                    f"  - {ind.name}: fitness={ind.fitness:.4f}"
                    for ind in parents[:5]  # Show top 5
                ])
                
                # Create refinement prompt with population context
                algorithm_manager.tried_algorithms = f"Current population:\n{pop_summary}"
                algorithm_manager.last_algorithm = f"# Name: {parent.description}\n# Code:\n```python\n{parent.code}\n```"
                
                message = algorithm_manager.refine_algorithm(
                    parent.fitness,
                    np.std(parent.aucs) if parent.aucs else 0.0,
                    parent.description,
                    parent.detailed_aucs
                )
                
                algorithm_code = algorithm_manager.extract_algorithm_code(message)
                algorithm_name = re.findall("class\\s*(\\w*)\\:", algorithm_code, re.IGNORECASE)[0]
                algorithm_name_long = algorithm_manager.extract_algorithm_name(message)
                if algorithm_name_long == "":
                    algorithm_name_long = algorithm_name
                
                # Create offspring individual
                child = Individual(
                    code=algorithm_code,
                    name=algorithm_name,
                    description=algorithm_name_long,
                    generation=generation
                )
                
                # Evaluate
                print(f"  Evaluating {algorithm_name}...")
                aucs, detailed_aucs, error = evaluate_algorithm(algorithm_code, algorithm_name, args.eval_budget)
                
                if error:
                    child.fitness = 0.0
                    child.error = error
                    print(f"  Error: {error}")
                else:
                    child.fitness = np.mean(aucs)
                    child.aucs = aucs
                    child.detailed_aucs = detailed_aucs
                    print(f"  Fitness: {child.fitness:.4f}")
                    
                    if child.fitness > best_ever.fitness:
                        print(f"  🎉 NEW BEST! {child.fitness:.4f} > {best_ever.fitness:.4f}")
                        best_ever = child
                
                offspring.append(child)
                explogger.log_code(api_calls, algorithm_name, algorithm_code)
                explogger.log_aucs(api_calls, aucs if aucs else [0])
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
        
        print(f"New population fitness: {[f'{ind.fitness:.4f}' for ind in population[:3]]}")
    
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
