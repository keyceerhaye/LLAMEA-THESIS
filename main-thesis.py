import os
import numpy as np
from ioh import get_problem, logger
import re
import argparse
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



def main():
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(
        description='LLaMEA Thesis - Enhanced main program with API customization',
        epilog='''
Environment Variables:
  The program will automatically load environment variables from a .env file if present.
  Supported variables: OPENAI_API_KEY, BASE_URL, AIML_KEY
  
Examples:
  python main-thesis.py --api-key your_key
  python main-thesis.py --base-url https://api.custom-endpoint.com/v1
  python main-thesis.py --max-tokens 4096 --elitism
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
                       help='OpenAI API calls budget (default: 100)')
    parser.add_argument('--eval-budget', type=int, default=10000,
                       help='Evaluation budget per algorithm (default: 10000)')
    parser.add_argument('--elitism', action='store_true',
                       help='Enable elitism strategy')
    parser.add_argument('--detailed-feedback', action='store_true', 
                       help='Enable detailed feedback about algorithm performance')
    
    # Population Configuration (for future evolutionary enhancement)
    parser.add_argument('--n-parents', type=int, default=4,
                       help='Number of parent algorithms to maintain (default: 1)')
    parser.add_argument('--n-offspring', type=int, default=16,
                       help='Number of offspring algorithms per generation (default: 1)')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of generations (if set, overrides budget)')
    parser.add_argument('--evolutionary-mode', action='store_true',
                       help='Enable true evolutionary mode with populations')
    
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

    # Initialize metrics tracking (same as main.py)
    auc_mean = 0
    auc_std = 0
    detailed_aucs = [0,0,0,0,0]
    openai_budget = args.budget
    algorithm_name = algorithm_name_long = ""

    # Calculate effective generations for population-based runs (future use)
    if args.generations is not None:
        effective_generations = args.generations
        estimated_calls = args.n_parents + (effective_generations * args.n_offspring)
        print(f"INFO: If using evolutionary mode:")
        print(f"INFO: Fixed generations: {effective_generations}")
        print(f"INFO: Estimated API calls: {estimated_calls}")
        if estimated_calls > openai_budget:
            print(f"WARNING: Estimated calls exceed budget!")
    else:
        # Budget-driven calculation
        max_gens = max(1, (openai_budget - args.n_parents) // args.n_offspring)
        print(f"INFO: Budget-driven evolutionary potential:")
        print(f"INFO: Max generations with {args.n_parents} parents, {args.n_offspring} offspring: {max_gens}")

    print(f"\nStarting LLaMEA Thesis experiment:")
    print(f"  Model: {ai_model}")
    print(f"  Base URL: {base_url or 'Default OpenAI'}")
    print(f"  Max Tokens: {args.max_tokens or 'Default'}")
    print(f"  API Budget: {openai_budget}")
    print(f"  Evaluation Budget: {args.eval_budget}")
    print(f"  Elitism: {args.elitism}")
    print(f"  Detailed Feedback: {args.detailed_feedback}")
    print(f"  Population Config: {args.n_parents} parents, {args.n_offspring} offspring")
    print("-" * 50)

    # Main optimization loop (identical to main.py)
    for openai_try in np.arange(openai_budget):
        print(f"\nAPI Call {openai_try + 1}/{openai_budget}")
        
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

            exec(new_algorithm, globals())
            algorithm_manager.last_algorithm = message

            explogger.log_code(openai_try, algorithm_name, new_algorithm)

            budget = args.eval_budget
            
            # Setup logging (same as main.py) 
            l2 = aoc_logger(budget, upper=1e2, triggers=[logger.trigger.ALWAYS])
            aucs = []
            detail_aucs = []
            
            # Benchmark evaluation loop (identical to main.py)
            for fid in np.arange(1,25):
                for iid in [1, 2, 3]:
                    problem = get_problem(fid, iid, 5)
                    problem.attach_logger(l2)

                    for rep in range(3):
                        np.random.seed(rep)
                        
                        try:
                            algorithm = globals()[algorithm_name](budget)
                            algorithm(problem)
                        except OverBudgetException:
                            pass
                        
                        auc = correct_aoc(problem, l2, budget)
                        aucs.append(auc)
                        detail_aucs.append(auc)
                        l2.reset(problem)
                        problem.reset()
                        
                # Track detailed AUCs by function groups (same as main.py)
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

            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs)
            explogger.log_aucs(openai_try, aucs)

            print(f"Algorithm: {algorithm_name_long}")
            print(f"AUC Mean: {auc_mean:.4f} ± {auc_std:.4f}")
            algorithm_manager.last_error = ""
            
        except NoCodeException:
            auc_mean = 0.0
            auc_std = 0.0
            print("Error: No code extracted from LLM response")
        except Exception as e:
            auc_mean = 0.0
            auc_std = 0.0
            algorithm_manager.last_error = repr(e)
            print(f"Error: {algorithm_manager.last_error}")
            print(f"Algorithm name: {algorithm_name}")
            print(f"Generated algorithm excerpt: {new_algorithm[:200] if 'new_algorithm' in locals() else 'N/A'}...")
            import traceback
            traceback.print_exc()

    print(f"\nExperiment completed. Results saved in: {explogger.dirname}")


if __name__ == "__main__":
    main()
