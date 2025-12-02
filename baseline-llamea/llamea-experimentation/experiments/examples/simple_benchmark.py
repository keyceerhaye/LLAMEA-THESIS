#!/usr/bin/env python3
"""
Simple LLaMEA Benchmark Example
===============================

A minimal example showing how to use the LLaMEA framework to evolve
optimization algorithms. This example uses the Dummy LLM for testing
without requiring API keys.
"""

import os
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_PATH):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from llamea import LLaMEA, Dummy_LLM, OpenAI_LLM, Gemini_LLM


def simple_evaluation(solution, explogger=None):
    """
    A simple evaluation function that tests generated algorithms
    on basic optimization problems.
    """
    code = solution.code
    algorithm_name = solution.name
    
    try:
        # Execute the generated algorithm code
        exec_globals = {"np": np, "numpy": np}
        exec(code, exec_globals)
        
        if algorithm_name not in exec_globals:
            raise ValueError(f"Algorithm class '{algorithm_name}' not found")
        
        # Define simple test functions
        def sphere(x):
            """Sphere function: simple unimodal function"""
            return np.sum(x**2)
        
        def rosenbrock(x):
            """Rosenbrock function: harder multimodal function"""
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        # Create a simple function wrapper
        class TestFunction:
            def __init__(self, func, dim):
                self.func = func
                self.dim = dim
                self.bounds = type('Bounds', (), {
                    'lb': np.full(dim, -5.0),
                    'ub': np.full(dim, 5.0)
                })()
                self.call_count = 0
            
            def __call__(self, x):
                self.call_count += 1
                return self.func(x)
        
        # Test the algorithm on multiple problems
        scores = []
        test_functions = [sphere, rosenbrock]
        dimensions = [2, 5]
        
        for dim in dimensions:
            budget = 500 * dim
            
            for test_func in test_functions:
                for rep in range(3):  # Multiple runs for robustness
                    np.random.seed(rep)
                    
                    # Create function instance
                    func = TestFunction(test_func, dim)
                    
                    try:
                        # Initialize and run the algorithm
                        algorithm = exec_globals[algorithm_name](budget=budget, dim=dim)
                        result = algorithm(func)
                        
                        # Extract best value found
                        if isinstance(result, tuple):
                            best_value = result[0]
                        else:
                            best_value = result
                        
                        # Convert to a score (higher is better)
                        # Use negative log to convert minimization to maximization
                        score = max(0, 1.0 - np.log10(max(best_value + 1e-10, 1e-10)) / 10)
                        scores.append(score)
                        
                    except Exception as e:
                        # Algorithm failed, assign zero score
                        scores.append(0.0)
                        print(f"Algorithm execution failed: {e}")
        
        # Calculate final fitness
        if scores:
            fitness = np.mean(scores)
            std_dev = np.std(scores)
            feedback = f"Algorithm {algorithm_name} achieved fitness {fitness:.4f} ± {std_dev:.4f} across test problems"
        else:
            fitness = 0.0
            feedback = f"Algorithm {algorithm_name} failed on all test problems"
        
        solution.set_scores(fitness, feedback)
        print(f"  → {algorithm_name}: fitness = {fitness:.4f}")
        
    except Exception as e:
        # Handle compilation or execution errors
        error_msg = f"Evaluation failed: {str(e)}"
        solution.set_scores(0.0, error_msg, str(e))
        print(f"  → Error evaluating {algorithm_name}: {error_msg}")
    
    return solution


def main():
    """Run a simple LLaMEA benchmark."""
    print("🚀 Starting Simple LLaMEA Benchmark")
    print("-" * 50)
    
    # Set up LLM (using Dummy LLM for this example)
    # To use a real LLM, uncomment one of the following:
    # llm = OpenAI_LLM(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
    # llm = Gemini_LLM(api_key=os.getenv("GEMINI_API_KEY"), model="gemini-1.5-flash")
    llm = Dummy_LLM(model="dummy-test")
    
    # Define the task prompt
    task_prompt = """
Your task is to create an optimization algorithm that can minimize black-box functions.
The algorithm should be implemented as a Python class with:
- __init__(self, budget, dim): Initialize with evaluation budget and problem dimension
- __call__(self, func): Optimize the function and return (best_value, best_point)

The function has bounds from -5.0 to 5.0 in each dimension.
Focus on creating robust algorithms that work well across different problem types.
"""
    
    # Create and configure LLaMEA
    llamea = LLaMEA(
        f=simple_evaluation,          # Our evaluation function
        llm=llm,                      # Language model to use
        n_parents=3,                  # Small population for quick testing
        n_offspring=3,                # Generate 3 offspring per generation
        task_prompt=task_prompt,      # Problem description
        experiment_name="simple_test", # Experiment identifier
        elitism=True,                 # Keep best solutions
        HPO=False,                    # No hyperparameter optimization
        budget=10,                    # Run for 10 generations
        minimization=False,           # We're maximizing fitness scores
        max_workers=2,                # Limit parallelism
        log=True,                     # Enable logging
    )
    
    print(f"Configuration:")
    print(f"  → LLM: {llm.model}")
    print(f"  → Population: {llamea.n_parents} parents, {llamea.n_offspring} offspring")
    print(f"  → Budget: {llamea.budget} generations")
    print(f"  → Elitism: {llamea.elitism}")
    print("-" * 50)
    
    # Run the evolution
    print("🧬 Starting evolution...")
    best_solution = llamea.run()
    
    # Display results
    print("-" * 50)
    print("✅ Evolution completed!")
    print(f"🏆 Best algorithm: {best_solution.name}")
    print(f"📊 Best fitness: {best_solution.fitness:.6f}")
    print(f"💬 Feedback: {best_solution.feedback}")
    print("-" * 50)
    print("🧬 Best algorithm code:")
    print(best_solution.code)
    print("-" * 50)
    
    return best_solution


if __name__ == "__main__":
    try:
        best_solution = main()
        print("🎉 Benchmark completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
