# LLaMEA Experimentation Setup

This folder contains all essential files for running LLAMEA (Large Language Model Evolutionary Algorithm) experiments, including benchmarking and IOH experimenter integration.

## 📁 Folder Structure

```
llamea-experimentation/
├── llamea/                 # Core LLAMEA framework
│   ├── __init__.py
│   ├── llamea.py          # Main evolutionary engine
│   ├── llm.py             # LLM provider abstraction
│   ├── solution.py        # Solution data structure
│   ├── utils.py           # Utility functions
│   └── loggers.py         # Experiment logging
├── examples/              # Example scripts
│   ├── minimum_example.py
│   ├── simple_benchmark.py
│   ├── black-box-optimization.py
│   ├── black-box-opt-with-HPO.py
│   └── automl_example.py
├── benchmarks/            # Benchmarking scripts
│   ├── main.py           # Legacy simple script
│   ├── main-thesis.py    # Enhanced thesis script with CLI
│   ├── main-evolutionary.py  # Production framework script
│   ├── managers.py       # Custom manager classes
│   └── utils.py          # BBOB-specific utilities
├── misc/                  # Utilities and tools
│   ├── iohrun.py         # IOH experimenter runner
│   ├── utils.py          # General utilities
│   ├── plot_aucs.py      # Visualization tools
│   └── ...               # Other helper scripts
├── docs/                  # Documentation (optional)
├── requirements.txt       # Python dependencies
├── benchmark_config.yaml  # Benchmark configurations
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your_openai_key_here
# GEMINI_API_KEY=your_gemini_key_here
```

### 3. Run Your First Experiment

#### Option A: Simple Example (No API Keys Required)

```bash
python examples/minimum_example.py
```

#### Option B: With Dummy LLM (Testing)

```bash
python benchmarks/main-thesis.py --llm dummy --budget 5
```

#### Option C: Full Benchmark with OpenAI

```bash
python benchmarks/main-evolutionary.py --llm openai --model gpt-4-turbo --budget 50
```

## 📊 Available Scripts

### Examples (`examples/`)

1. **minimum_example.py** - Simplest possible LLAMEA usage
2. **simple_benchmark.py** - Basic benchmarking setup
3. **black-box-optimization.py** - Black-box optimization example
4. **black-box-opt-with-HPO.py** - With hyperparameter optimization
5. **automl_example.py** - AutoML use case

### Benchmarks (`benchmarks/`)

1. **main.py** - Legacy (1+1) evolution script
   - Simple, hardcoded configuration
   - Good for understanding basics

2. **main-thesis.py** - Enhanced CLI script
   - Single-algorithm (1+1) evolution
   - Full command-line interface
   - Environment variable support
   ```bash
   python benchmarks/main-thesis.py --llm openai --budget 20 --detailed-feedback
   ```

3. **main-evolutionary.py** - Production framework ⭐ RECOMMENDED
   - True population-based evolution (μ+λ)
   - All LLaMEA framework features
   - Niching, HPO, diff mode support
   ```bash
   python benchmarks/main-evolutionary.py --llm openai --n-parents 5 --n-offspring 5 --budget 100
   ```

### IOH Experimenter (`misc/iohrun.py`)

Run comprehensive benchmarks on IOH problem suite:

```bash
python misc/iohrun.py
```

## 🔧 Configuration

### Using benchmark_config.yaml

The `benchmark_config.yaml` file contains pre-configured setups:

```yaml
quick_test:
  llm_provider: "dummy"
  budget: 5
  n_parents: 2
  n_offspring: 2

openai_gpt4:
  llm_provider: "openai"
  llm_model: "gpt-4-turbo"
  budget: 100
  evaluation_strategy: "bbob"
```

Load a configuration:

```python
import yaml
with open('benchmark_config.yaml') as f:
    config = yaml.safe_load(f)['quick_test']
```

## 🧪 Core Components

### LLaMEA Framework (`llamea/`)

The core framework consists of:

- **llamea.py** - Main evolutionary algorithm engine
  - Population initialization and evolution
  - Selection strategies (elitism, comma)
  - Niching (fitness sharing, clearing)
  
- **llm.py** - LLM provider abstraction
  - Supports: OpenAI, Gemini, Ollama, Custom endpoints
  - Handles code extraction and retry logic
  
- **solution.py** - Solution data structure
  - Stores code, fitness, feedback, genealogy
  
- **utils.py** - Utility functions
  - Diff patching, code distance metrics
  
- **loggers.py** - Experiment logging
  - JSONL conversation logs
  - Code versioning and metrics

### Basic Usage

```python
from llamea import LLaMEA
from llamea.llm import OpenAI_LLM

# Initialize LLM
llm = OpenAI_LLM(model="gpt-4-turbo")

# Define evaluation function
def evaluate(solution):
    # Your evaluation logic here
    return fitness_score

# Create and run LLaMEA
llamea = LLaMEA(
    llm=llm,
    f=evaluate,
    n_parents=5,
    n_offspring=5,
    budget=100
)

best_solution = llamea.run()
```

## 📈 Benchmarking with IOH

### BBOB Benchmark Suite

The Black-Box Optimization Benchmarking (BBOB) suite is used for evaluation:

- 24 test functions
- Multiple dimensions (5, 10, 20)
- Multiple instances per function
- Standard in optimization research

### Running IOH Experiments

```python
from ioh import get_problem, logger
from misc.iohrun import run_ioh_benchmark

# Run on specific problem
problem = get_problem(fid=1, iid=1, dim=10)
algorithm = YourAlgorithm(budget=2000)
algorithm(problem)
```

## 🎯 Enhancement Ideas

This experimentation setup is designed to be extended. Consider:

1. **New LLM Providers**
   - Add custom LLM implementations in `llamea/llm.py`
   
2. **Custom Evaluation Functions**
   - Modify evaluation logic in benchmark scripts
   
3. **Advanced Selection Strategies**
   - Implement new selection methods in `llamea/llamea.py`
   
4. **Visualization Tools**
   - Use `misc/plot_aucs.py` as starting point
   
5. **Hyperparameter Optimization**
   - Enable HPO in main-evolutionary.py with `--hpo`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Make sure you're in the right directory
   cd llamea-experimentation
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **API Key Issues**
   - Check `.env` file exists and has correct keys
   - Verify environment variables are loaded
   ```python
   import os
   print(os.getenv('OPENAI_API_KEY'))
   ```

3. **IOH Not Available**
   ```bash
   pip install ioh
   ```

4. **Budget Exceeded Errors**
   - Reduce budget in configuration
   - Check evaluation function isn't calling too many times

## 📚 Documentation

For detailed documentation on each component, see the original `docs/` folder in the main repository.

Key documentation files:
- `docs/ai_context_summary.md` - Quick overview
- `docs/architecture.md` - System architecture
- `docs/codebase/llamea/README.md` - Framework details

## 🤝 Contributing

When enhancing this experimentation setup:

1. Keep it simple (KISS principle)
2. Document your changes
3. Test with dummy LLM first
4. Add examples for new features
5. Update this README

## 📝 License

See LICENSE file for details.

## 🎓 Citation

If you use LLaMEA in your research, please cite:

```bibtex
@software{llamea2024,
  title={LLaMEA: Large Language Model Evolutionary Algorithm},
  author={van Stein, Niki and others},
  year={2024},
  url={https://github.com/XAI-liacs/LLaMEA}
}
```

---

**Happy Experimenting! 🚀**






