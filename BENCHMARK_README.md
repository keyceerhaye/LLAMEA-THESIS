# LLaMEA Benchmarking Tools

This directory contains comprehensive benchmarking tools for the LLaMEA (Large Language Model Evolutionary Algorithm) framework. These tools allow you to evolve optimization algorithms using different LLM providers and evaluation strategies.

## 🚀 Quick Start

### 1. Simple Example (No API Keys Required)

```bash
# Run a simple benchmark with dummy LLM
python examples/simple_benchmark.py
```

This runs a minimal example that demonstrates the LLaMEA framework without requiring any API keys.

### 2. Full Benchmark Tool

```bash
# Basic usage with dummy LLM
python llamea_benchmark.py --llm dummy --budget 20 --experiment test

# With OpenAI GPT-4 (requires OPENAI_API_KEY environment variable)
python llamea_benchmark.py --llm openai --model gpt-4-turbo --budget 50 --evaluation bbob

# With Google Gemini (requires GEMINI_API_KEY environment variable)
python llamea_benchmark.py --llm gemini --budget 50 --evaluation simple
```

## 🔧 Installation Requirements

### Core Requirements

```bash
pip install numpy
```

### For Environment File Support (Recommended)

```bash
pip install python-dotenv
```

### For BBOB Benchmarks (Optional)

```bash
pip install ioh
```

### For LLM Providers

```bash
# For OpenAI
pip install openai

# For Google Gemini
pip install google-generativeai
```

## 📊 Available Evaluation Strategies

### 1. Simple Evaluation (`--evaluation simple`)

- Tests algorithms on basic functions (sphere, Rosenbrock)
- Fast execution, good for development and testing
- No external dependencies

### 2. BBOB Evaluation (`--evaluation bbob`)

- Full BBOB (Black-Box Optimization Benchmarking) test suite
- Industry-standard benchmark for optimization algorithms
- Requires `ioh` package
- More comprehensive but slower

### 3. Custom Evaluation (`--evaluation custom`)

- Placeholder for custom evaluation functions
- Can be extended by modifying the `_evaluate_custom` method

## 🤖 Supported LLM Providers

### 1. Dummy LLM (`--llm dummy`)

- No API key required
- Returns fixed example algorithms
- Perfect for testing and development
- Fast execution

### 2. OpenAI (`--llm openai`)

- Requires OpenAI API key
- Set `OPENAI_API_KEY` environment variable
- Supports models: gpt-4-turbo, gpt-4o, gpt-3.5-turbo, etc.
- High-quality algorithm generation

### 3. Google Gemini (`--llm gemini`)

- Requires Google AI API key
- Set `GEMINI_API_KEY` environment variable
- Supports models: gemini-2.0-flash, gemini-1.5-flash, etc.
- Competitive performance with OpenAI

## 🎛️ Configuration Options

### Command Line Arguments

```bash
# LLM Configuration
--llm {openai,gemini,dummy}     # LLM provider
--model MODEL_NAME              # Specific model name
--api-key API_KEY              # API key (or use env variables)
--base-url BASE_URL            # Custom API base URL (or use BASE_URL env var)
--max-tokens N                 # Maximum tokens for responses (default: 8192)

# Experiment Configuration
--experiment NAME              # Experiment name
--budget N                     # Number of generations
--parents N                    # Population size (parents)
--offspring N                  # Offspring per generation

# Strategy Configuration
--evaluation {bbob,simple,custom}  # Evaluation strategy
--no-elitism                   # Disable elitist selection
--hpo                          # Enable hyperparameter optimization

# Output Configuration
--quiet                        # Reduce verbosity
```

### Environment Variables

You can set up environment variables in a `.env` file (copy from `env_example.txt`):

```bash
# API Keys
OPENAI_API_KEY=your-openai-api-key
AIML_KEY=your-custom-api-key          # Alternative for custom endpoints
GEMINI_API_KEY=your-gemini-api-key

# Custom API Configuration
BASE_URL=https://api.your-endpoint.com/v1  # For custom OpenAI-compatible APIs

# Or export them directly:
export OPENAI_API_KEY="your-openai-api-key"
export AIML_KEY="your-custom-api-key"
export BASE_URL="https://api.your-endpoint.com/v1"
export GEMINI_API_KEY="your-gemini-api-key"

# Run benchmark
python llamea_benchmark.py --llm openai --budget 50
```

## 📝 Example Usage Scenarios

### Development and Testing

```bash
# Quick test with dummy LLM
python llamea_benchmark.py --llm dummy --budget 5 --parents 2 --offspring 2

# Test with simple evaluation
python llamea_benchmark.py --llm dummy --evaluation simple --budget 10
```

### Research Experiments

```bash
# Full BBOB benchmark with OpenAI
python llamea_benchmark.py --llm openai --evaluation bbob --budget 100 --experiment research_v1

# Using custom API endpoint with max tokens
python llamea_benchmark.py --llm openai --base-url https://api.your-endpoint.com/v1 --max-tokens 8192 --budget 50

# Compare different LLMs
python llamea_benchmark.py --llm openai --budget 50 --experiment openai_test
python llamea_benchmark.py --llm gemini --budget 50 --experiment gemini_test
```

### Hyperparameter Optimization

```bash
# Enable HPO mode
python llamea_benchmark.py --llm openai --hpo --budget 30 --experiment hpo_test
```

### Large Scale Benchmarks

```bash
# Large population, long run
python llamea_benchmark.py --llm openai --budget 200 --parents 10 --offspring 10 --experiment large_scale
```

## 📊 Output and Results

### Console Output

The tool provides real-time feedback including:

- Algorithm evaluation progress
- Fitness scores and statistics
- Best solution information
- Error messages and debugging info

### Saved Results

Results are automatically saved to files:

- `llamea_result_{experiment}_{timestamp}.py` - Contains the best algorithm code
- Log files in experiment directories (when logging is enabled)

### Example Output Structure

```
🚀 Starting LLaMEA Benchmark: test_experiment
  → LLM Provider: openai
  → Model: gpt-4-turbo
  → Evaluation: simple
  → Budget: 50 generations
  → Population: 5 parents, 5 offspring
------------------------------------------------------------
Generation 1, best so far: 0.0234
  → RandomSearchV1: Score = 0.0234 ± 0.0123
Generation 2, best so far: 0.0456
  → AdaptiveGradientSearch: Score = 0.0456 ± 0.0089
...
------------------------------------------------------------
✅ Benchmark completed in 123.45 seconds
🏆 Best solution: OptimizedEvolutionStrategy
📊 Best fitness: 0.8234
```

## 🔬 Framework Architecture

### Core Components

1. **LLaMEABenchmark Class**: Main orchestrator
2. **Evaluation Functions**: Test generated algorithms
3. **LLM Interfaces**: Connect to different language models
4. **Configuration System**: Flexible parameter management

### Evaluation Flow

1. LLM generates optimization algorithm code
2. Code is executed in controlled environment
3. Algorithm is tested on benchmark problems
4. Performance metrics are calculated
5. Feedback is provided to LLM for next iteration

### Algorithm Requirements

Generated algorithms must implement:

```python
class YourAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Optimize func using self.budget evaluations
        # Return (best_value, best_point) or just best_value
        pass
```

## 🛠️ Extending the Framework

### Adding Custom Evaluation

```python
def custom_evaluation(solution, explogger=None):
    # Your custom evaluation logic
    code = solution.code
    algorithm_name = solution.name

    # Execute and test the algorithm
    # ...

    # Set fitness and feedback
    solution.set_scores(fitness, feedback)
    return solution

# Use in benchmark
benchmark = LLaMEABenchmark(evaluation_strategy="custom")
benchmark._evaluate_custom = custom_evaluation
```

### Adding New LLM Provider

Follow the pattern in `llamea/llm.py` to implement new LLM interfaces.

## ⚠️ Important Notes

### Rate Limits

- Be aware of API rate limits when using commercial LLMs
- The tool implements basic retry logic but may need adjustment for your use case

### Computational Resources

- BBOB evaluations can be computationally intensive
- Adjust population sizes and budgets based on available resources
- Consider using `--quiet` mode for automated runs

### Reproducibility

- Random seeds are set for evaluation consistency
- LLM responses may vary between runs due to temperature settings
- Save configuration and results for reproducible experiments

## 🐛 Troubleshooting

### Common Issues

1. **"IOH not available" warning**: Install `pip install ioh` for BBOB benchmarks
2. **API key errors**: Set environment variables or pass `--api-key`
3. **Timeout errors**: Reduce population size or increase timeout values
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

```bash
# Run with dummy LLM for debugging
python llamea_benchmark.py --llm dummy --budget 5 --quiet
```

## 📚 References

- [LLaMEA Paper](https://arxiv.org/abs/your-paper-link)
- [BBOB Benchmark Suite](https://github.com/IOHprofiler/IOHexperimenter)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google AI Documentation](https://ai.google.dev/docs)

## 🤝 Contributing

To contribute improvements or new features:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the same terms as the LLaMEA framework.
