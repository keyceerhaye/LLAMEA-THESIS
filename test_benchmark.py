#!/usr/bin/env python3
"""
Test script for LLaMEA benchmarking tools.

This script tests the basic functionality of the benchmarking framework
without requiring external dependencies or API keys.
"""

import os
import sys
import traceback

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from llamea import LLaMEA, Dummy_LLM
        print("  ✅ LLaMEA core imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import LLaMEA core: {e}")
        return False
    
    try:
        from misc import OverBudgetException, aoc_logger, correct_aoc
        print("  ✅ Misc utilities imported successfully")
    except ImportError as e:
        print(f"  ⚠️  Misc utilities not available: {e}")
        print("     This is okay if IOH is not installed")
    
    try:
        import numpy as np
        print("  ✅ NumPy imported successfully")
    except ImportError as e:
        print(f"  ❌ Failed to import NumPy: {e}")
        return False
    
    return True


def test_dummy_llm():
    """Test the Dummy LLM functionality."""
    print("\n🤖 Testing Dummy LLM...")
    
    try:
        from llamea import Dummy_LLM
        
        llm = Dummy_LLM(model="test-model")
        
        # Test basic query
        messages = [{"role": "user", "content": "Generate an algorithm"}]
        response = llm.query(messages)
        
        if response and "RandomSearch" in response:
            print("  ✅ Dummy LLM generates responses correctly")
            return True
        else:
            print("  ❌ Dummy LLM response unexpected")
            return False
            
    except Exception as e:
        print(f"  ❌ Dummy LLM test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation_function():
    """Test the evaluation function."""
    print("\n📊 Testing evaluation function...")
    
    try:
        from llamea import Solution
        import numpy as np
        
        # Create a test solution with a simple algorithm
        test_code = """
import numpy as np

class TestAlgorithm:
    def __init__(self, budget=1000, dim=5):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
    
    def __call__(self, func):
        for i in range(min(self.budget, 100)):  # Limit for testing
            x = np.random.uniform(func.bounds.lb, func.bounds.ub)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x
        return self.f_opt, self.x_opt
"""
        
        solution = Solution(name="TestAlgorithm", code=test_code)
        
        # Test evaluation
        def simple_eval(solution, explogger=None):
            code = solution.code
            algorithm_name = solution.name
            
            try:
                exec_globals = {"np": np, "numpy": np}
                exec(code, exec_globals)
                
                if algorithm_name in exec_globals:
                    # Simple test
                    def sphere(x):
                        return np.sum(x**2)
                    
                    class TestFunc:
                        def __init__(self):
                            self.bounds = type('Bounds', (), {
                                'lb': np.array([-5.0, -5.0]),
                                'ub': np.array([5.0, 5.0])
                            })()
                        
                        def __call__(self, x):
                            return sphere(x)
                    
                    func = TestFunc()
                    algorithm = exec_globals[algorithm_name](budget=100, dim=2)
                    result = algorithm(func)
                    
                    fitness = 1.0 if isinstance(result, tuple) else 0.5
                    solution.set_scores(fitness, f"Test evaluation completed")
                    return solution
                else:
                    solution.set_scores(0.0, "Algorithm not found")
                    return solution
                    
            except Exception as e:
                solution.set_scores(0.0, f"Evaluation failed: {e}")
                return solution
        
        evaluated_solution = simple_eval(solution)
        
        if evaluated_solution.fitness > 0:
            print(f"  ✅ Evaluation function works (fitness: {evaluated_solution.fitness})")
            return True
        else:
            print(f"  ❌ Evaluation failed: {evaluated_solution.feedback}")
            return False
            
    except Exception as e:
        print(f"  ❌ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def test_llamea_basic():
    """Test basic LLaMEA functionality."""
    print("\n🧬 Testing LLaMEA basic functionality...")
    
    try:
        from llamea import LLaMEA, Dummy_LLM, Solution
        import numpy as np
        
        # Simple evaluation function
        def eval_func(solution, explogger=None):
            # Always return a random fitness for testing
            fitness = np.random.rand()
            solution.set_scores(fitness, f"Test fitness: {fitness:.3f}")
            return solution
        
        llm = Dummy_LLM(model="test")
        
        # Create minimal LLaMEA instance
        llamea = LLaMEA(
            f=eval_func,
            llm=llm,
            n_parents=2,
            n_offspring=2,
            budget=3,  # Very small for testing
            task_prompt="Create a simple optimization algorithm",
            experiment_name="test",
            log=False,  # Disable logging for testing
        )
        
        print(f"  ✅ LLaMEA instance created successfully")
        print(f"     Population: {llamea.n_parents} parents, {llamea.n_offspring} offspring")
        print(f"     Budget: {llamea.budget} generations")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LLaMEA basic test failed: {e}")
        traceback.print_exc()
        return False


def test_benchmark_tool():
    """Test the benchmark tool."""
    print("\n🔧 Testing benchmark tool...")
    
    try:
        from llamea_benchmark import LLaMEABenchmark
        
        # Create benchmark with minimal settings
        benchmark = LLaMEABenchmark(
            llm_provider="dummy",
            experiment_name="test",
            budget=2,  # Very small for testing
            n_parents=2,
            n_offspring=2,
            evaluation_strategy="simple",
            verbose=False,
        )
        
        print("  ✅ Benchmark tool initialized successfully")
        print(f"     LLM Provider: {benchmark.llm_provider}")
        print(f"     Evaluation: {benchmark.evaluation_strategy}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Benchmark tool test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 LLaMEA Benchmark Test Suite")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_import),
        ("Dummy LLM Test", test_dummy_llm),
        ("Evaluation Function Test", test_evaluation_function),
        ("LLaMEA Basic Test", test_llamea_basic),
        ("Benchmark Tool Test", test_benchmark_tool),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ❌ {test_name} FAILED")
        except Exception as e:
            print(f"  ❌ {test_name} CRASHED: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The benchmark tool is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
