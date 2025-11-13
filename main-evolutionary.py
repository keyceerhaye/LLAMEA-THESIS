#!/usr/bin/env python3
"""
LLaMEA Evolutionary Thesis Script
Using the full LLaMEA framework with configurable parents and offspring
"""

import os
import argparse
import numpy as np
import time
import openai
from ioh import get_problem, logger
from llamea import LLaMEA
from llamea.llm import OpenAI_LLM
from utils import OverBudgetException, aoc_logger, correct_aoc

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    dotenv_loaded = load_dotenv()
    print(f"DEBUG: .env file loaded: {dotenv_loaded}")
except ImportError:
    print("DEBUG: python-dotenv not installed, using system environment variables only")


def evaluate_single_solution(solution, explogger=None, eval_budget=10000):
    """
    Evaluates a single solution on the BBOB test suite.
    
    Args:
        solution: The algorithm solution object
        explogger: Optional experiment logger
        eval_budget: Evaluation budget per algorithm
        
    Returns:
        tuple: (feedback_string, fitness_score, error_message)
    """
    try:
        # Extract algorithm from solution
        algorithm_code = solution.code
        algorithm_name = solution.name
        
        # If algorithm name is empty, try to extract from code
        if not algorithm_name:
            import re
            class_match = re.findall(r"class\s*(\w+)\s*:", algorithm_code, re.IGNORECASE)
            if class_match:
                algorithm_name = class_match[0]
            else:
                algorithm_name = f"UnknownAlgorithm_{solution.generation}"
        
        # Execute the algorithm code
        exec(algorithm_code, globals())
        
        # Get the algorithm class using the extracted algorithm name
        try:
            algorithm_class = globals()[algorithm_name]
        except KeyError:
            return f"Algorithm class {algorithm_name} not found after execution", 0.0, f"Class {algorithm_name} not in globals"
        
        # Evaluation budget per algorithm
        budget = eval_budget
        
        # Setup logging
        l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
        aucs = []
        
        # Run on all BBOB functions (matching main.py implementation)
        for fid in np.arange(1, 25):  # All 24 BBOB functions
            for iid in [1, 2, 3]:  # 3 instances
                problem = get_problem(fid, iid, 5)
                problem.attach_logger(l2)
                
                for rep in range(3):  # 3 repetitions
                    np.random.seed(rep)
                    
                    try:
                        algorithm = algorithm_class(budget)
                        algorithm(problem)
                    except OverBudgetException:
                        pass
                    except Exception as e:
                        return f"Algorithm execution error: {str(e)}", 0.0, str(e)
                    
                    auc = correct_aoc(problem, l2, budget)
                    aucs.append(auc)
                    l2.reset(problem)
                    problem.reset()
        
        # Calculate final fitness
        auc_mean = np.mean(aucs)
        auc_std = np.std(aucs)
        
        # Log AUCs if explogger provided (LLaMEA handles code logging separately)
        if explogger:
            # Use solution ID for unique AUC file naming  
            solution_id = solution.id[:8]  # Use first 8 characters of UUID for short unique ID
            explogger.log_aucs(solution_id, aucs)
        
        # Provide feedback to LLM
        feedback = f"Algorithm {algorithm_name} achieved AUC mean: {auc_mean:.4f} ± {auc_std:.4f}"
        if auc_mean > 0.5:
            feedback += " - Good performance!"
        elif auc_mean > 0.3:
            feedback += " - Moderate performance, try to improve."
        else:
            feedback += " - Low performance, significant improvements needed."
        
        return feedback, auc_mean, ""
        
    except Exception as e:
        return f"Evaluation error: {str(e)}", 0.0, str(e)


def bbob_population_evaluation_function(population, parents=None, explogger=None, eval_budget=10000):
    """
    Population evaluation function for BBOB test suite that LLaMEA expects.
    
    Args:
        population: List of solution objects to evaluate
        parents: Optional parent population (not modified in our case)
        explogger: Optional experiment logger
        eval_budget: Evaluation budget per algorithm
        
    Returns:
        tuple: (evaluated_population, parents)
    """
    for solution in population:
        try:
            feedback, fitness, error = evaluate_single_solution(solution, explogger, eval_budget)
            solution.set_scores(fitness, feedback, error)
        except Exception as e:
            solution.set_scores(0.0, f"Evaluation failed: {str(e)}", str(e))
    
    return population, parents


def main():
    """Main function with evolutionary LLaMEA"""
    parser = argparse.ArgumentParser(
        description='LLaMEA Evolutionary Thesis - True evolutionary optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # API Configuration
    parser.add_argument('--api-key', type=str, help='OpenAI API key')
    parser.add_argument('--base-url', type=str, help='Custom base URL for OpenAI-compatible APIs')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Maximum tokens for API responses (default: 8192)')
    
    # Model Configuration
    parser.add_argument('--model', type=str, default='gemini-2.0-flash', 
                       help='AI model to use (default: gemini-2.0-flash)')
    parser.add_argument('--experiment-name', type=str, default='evolutionary-experiment',
                       help='Name for the experiment (default: evolutionary-experiment)')
    
    # Evolutionary Configuration
    parser.add_argument('--n-parents', type=int, default=4,
                       help='Number of parent algorithms (default: 4)')
    parser.add_argument('--n-offspring', type=int, default=16,
                       help='Number of offspring per generation (default: 16)')
    parser.add_argument('--api-budget', type=int, default=100,
                       help='Total API calls budget (default: 100)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Fixed number of generations (overrides budget-based calculation)')
    parser.add_argument('--elitism', action='store_true', default=True,
                       help='Enable elitism (μ+λ) strategy (default: True)')
    parser.add_argument('--eval-budget', type=int, default=10000,
                       help='Evaluation budget per algorithm (default: 5000)')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("AIML_KEY")
    if not api_key:
        raise ValueError("API key is required")
    
    # Get base URL
    base_url = args.base_url or os.getenv("BASE_URL")
    
    # Setup LLM - need to handle base_url specially for OpenAI_LLM
    llm = OpenAI_LLM(api_key=api_key, model=args.model)
    
    # If custom base_url is needed, update the client after initialization
    if base_url:
        llm.base_url = base_url
        llm._client_kwargs["base_url"] = base_url
        llm.client = openai.OpenAI(**llm._client_kwargs)
    
    # Handle max_tokens by modifying the client behavior
    if args.max_tokens:
        original_query = llm.query
        def query_with_max_tokens(session_messages, max_retries=5, default_delay=10):
            # Store original query method
            attempt = 0
            while True:
                try:
                    response = llm.client.chat.completions.create(
                        model=llm.model,
                        messages=session_messages,
                        temperature=llm.temperature,
                        max_tokens=args.max_tokens
                    )
                    return response.choices[0].message.content
                except openai.RateLimitError as err:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    retry_after = None
                    if getattr(err, "response", None) is not None:
                        retry_after = err.response.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after else default_delay * attempt
                    time.sleep(wait)
                except (openai.APITimeoutError, openai.APIConnectionError, openai.APIError) as err:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    time.sleep(default_delay * attempt)
        
        llm.query = query_with_max_tokens
    
    # Calculate generations based on budget
    if args.generations is not None:
        # User specified fixed generations
        generations = args.generations
        estimated_api_calls = args.n_parents + (generations * args.n_offspring)
        total_budget = estimated_api_calls
        print(f"INFO: Using fixed generations: {generations}")
        print(f"INFO: Estimated API calls needed: {estimated_api_calls}")
        if estimated_api_calls > args.api_budget:
            print(f"WARNING: Estimated API calls ({estimated_api_calls}) exceed budget ({args.api_budget})")
    else:
        # Calculate generations based on API budget
        # Initial parents + generations * offspring <= api_budget
        # generations <= (api_budget - n_parents) / n_offspring
        max_generations = max(1, (args.api_budget - args.n_parents) // args.n_offspring)
        generations = max_generations
        actual_api_calls = args.n_parents + (generations * args.n_offspring)
        # LLaMEA's budget is the total number of individuals to generate
        total_budget = actual_api_calls
        print(f"INFO: Budget-based calculation:")
        print(f"INFO: API Budget: {args.api_budget}")
        print(f"INFO: Parents: {args.n_parents} (initial API calls)")
        print(f"INFO: Offspring per generation: {args.n_offspring}")
        print(f"INFO: Calculated generations: {generations}")
        print(f"INFO: Total API calls will be: {actual_api_calls}")
        print(f"INFO: LLaMEA budget (total individuals): {total_budget}")

    # Define task prompt
    task_prompt = """
The optimization algorithm should handle a wide range of tasks, which is evaluated on the BBOB test suite of 24 noiseless functions. Your task is to write the optimization algorithm in Python code. The code should contain an `__init__(self, budget)` function and the function `def __call__(self, func)`, which should optimize the black box function `func` using `self.budget` function evaluations.
The func() can only be called as many times as the budget allows, not more. Each of the optimization functions has a search space between -5.0 (lower bound) and 5.0 (upper bound). The dimensionality is set to 5.

Give an excellent and novel heuristic algorithm to solve this task and also give it a name.
"""
    
    # Create evaluation function wrapper with eval_budget
    def evaluation_wrapper(population, parents=None, explogger=None):
        return bbob_population_evaluation_function(population, parents, explogger, args.eval_budget)
    
    # Create LLaMEA instance
    llamea = LLaMEA(
        f=evaluation_wrapper,
        llm=llm,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        task_prompt=task_prompt,
        experiment_name=args.experiment_name,
        elitism=args.elitism,
        budget=total_budget,  # Use total number of individuals to generate
        minimization=False,  # We want to maximize AUC scores
        max_workers=2,  # Limit parallelism for stability
        log=True,
        evaluate_population=True,  # Enable population-based evaluation
    )
    
    print(f"\nStarting LLaMEA Evolutionary experiment:")
    print(f"  Model: {args.model}")
    print(f"  Base URL: {base_url or 'Default OpenAI'}")
    print(f"  Max Tokens: {args.max_tokens or 'Default'}")
    print(f"  Parents: {args.n_parents}")
    print(f"  Offspring: {args.n_offspring}")
    print(f"  Generations: {generations}")
    print(f"  Elitism: {args.elitism}")
    print(f"  API Budget: {args.api_budget}")
    print(f"  Evaluation Budget per Algorithm: {args.eval_budget}")
    print("-" * 60)
    
    # Run evolutionary optimization
    best_solution = llamea.run()
    
    print(f"\nEvolutionary optimization completed!")
    print(f"Best Algorithm: {best_solution.name}")
    print(f"Best Fitness: {best_solution.fitness:.4f}")
    print(f"Results saved in: {llamea.logger.dirname}")


if __name__ == "__main__":
    main()
