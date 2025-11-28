# LLaMEA Experimentation - Folder Structure

## Complete File Tree

```
llamea-experimentation/
│
├── 📦 llamea/                          # Core LLAMEA Framework
│   ├── __init__.py                     # Package initialization
│   ├── llamea.py                       # Main evolutionary engine (LLaMEA class)
│   ├── llm.py                          # LLM provider abstraction (OpenAI, Gemini, Ollama)
│   ├── solution.py                     # Solution data structure
│   ├── utils.py                        # Utility functions (diff, code distance)
│   ├── loggers.py                      # Experiment logging
│   └── bbobalgs/                       # Example algorithms
│       ├── __init__.py
│       └── ERADS_QuantumFluxUltraRefined.py
│
├── 📚 examples/                        # Example Scripts
│   ├── minimum_example.py              # Simplest LLAMEA usage
│   ├── simple_benchmark.py             # Basic benchmarking
│   ├── black-box-optimization.py       # Black-box optimization
│   ├── black-box-opt-with-HPO.py      # With hyperparameter optimization
│   └── automl_example.py               # AutoML use case
│
├── 🔬 benchmarks/                      # Benchmarking Scripts
│   ├── main.py                         # Legacy (1+1) evolution
│   ├── main-thesis.py                  # Enhanced CLI script (1+1)
│   ├── main-evolutionary.py            # Production framework (μ+λ) ⭐
│   ├── managers.py                     # Custom manager classes
│   └── utils.py                        # BBOB-specific utilities
│
├── 🛠️ misc/                            # Utilities and Tools
│   ├── __init__.py
│   ├── iohrun.py                       # IOH experimenter runner
│   ├── utils.py                        # General utilities
│   ├── plot_aucs.py                    # Visualization tools
│   ├── ast.py                          # AST analysis
│   ├── python_ast_analysis.py          # Python AST utilities
│   ├── transform_to_stn.py             # STN transformation
│   └── visualize_graphs.py             # Graph visualization
│
├── 📖 docs/                            # Documentation (empty, for your notes)
│
├── 📄 Configuration Files
│   ├── requirements.txt                # Python dependencies
│   ├── setup.py                        # Package setup
│   ├── benchmark_config.yaml           # Benchmark configurations
│   └── .env.example                    # Environment variables template
│
├── 📋 Documentation Files
│   ├── README.md                       # Complete documentation
│   ├── QUICKSTART.md                   # 5-minute quick start
│   ├── STRUCTURE.md                    # This file
│   └── LICENSE                         # MIT License
│
└── .env                                # Your API keys (create this!)
```

## File Descriptions

### Core Framework (`llamea/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `llamea.py` | Main evolutionary algorithm | `LLaMEA` class |
| `llm.py` | LLM provider interface | `OpenAI_LLM`, `Gemini_LLM`, `Ollama_LLM` |
| `solution.py` | Solution representation | `Solution` class |
| `utils.py` | Helper functions | `apply_unified_diff()`, `code_distance()` |
| `loggers.py` | Experiment tracking | `ExperimentLogger` class |

### Examples (`examples/`)

| File | Complexity | Use Case | API Keys Required |
|------|-----------|----------|-------------------|
| `minimum_example.py` | ⭐ | Learning basics | Optional |
| `simple_benchmark.py` | ⭐⭐ | Simple benchmarking | Optional |
| `black-box-optimization.py` | ⭐⭐⭐ | BBOB benchmarks | Yes |
| `black-box-opt-with-HPO.py` | ⭐⭐⭐⭐ | With HPO | Yes |
| `automl_example.py` | ⭐⭐⭐⭐ | AutoML tasks | Yes |

### Benchmarks (`benchmarks/`)

| File | Evolution Type | Features | Recommended For |
|------|---------------|----------|-----------------|
| `main.py` | (1+1) | Basic | Learning |
| `main-thesis.py` | (1+1) | CLI, .env | Quick experiments |
| `main-evolutionary.py` | (μ+λ) | Full framework | Production ⭐ |

### Utilities (`misc/`)

| File | Purpose |
|------|---------|
| `iohrun.py` | Run IOH experimenter benchmarks |
| `plot_aucs.py` | Visualize Area Under Curve |
| `ast.py` | Abstract Syntax Tree analysis |
| `utils.py` | General utility functions |
| `visualize_graphs.py` | Graph visualization |

## Essential Files for Running LLAMEA

### Minimum Setup (No API Keys)
```
llamea/
examples/minimum_example.py
requirements.txt
```

### Basic Experimentation
```
llamea/
benchmarks/main-thesis.py
benchmarks/managers.py
benchmarks/utils.py
requirements.txt
.env (with API keys)
```

### Full Production Setup
```
llamea/
benchmarks/main-evolutionary.py
benchmarks/utils.py
misc/iohrun.py
requirements.txt
benchmark_config.yaml
.env (with API keys)
```

## Dependencies Overview

### Core Dependencies
- `numpy` - Numerical operations
- `pandas` - Data handling
- `tqdm` - Progress bars
- `joblib` - Parallel processing

### LLM Providers
- `openai` - OpenAI API
- `google-generativeai` - Gemini API
- `ollama` - Local LLM support

### Benchmarking
- `ioh` - IOH Experimenter
- `configspace` - Configuration spaces

### Optional
- `smac` - Hyperparameter optimization
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - ML utilities

## Quick Reference

### Import Paths
```python
from llamea import LLaMEA
from llamea.llm import OpenAI_LLM, Gemini_LLM, Ollama_LLM
from llamea.solution import Solution
from llamea.loggers import ExperimentLogger
```

### Running Scripts
```bash
# From llamea-experimentation/ directory

# Examples
python examples/minimum_example.py

# Benchmarks
python benchmarks/main-evolutionary.py --llm openai --budget 50

# IOH
python misc/iohrun.py
```

## Enhancement Checklist

When enhancing this setup, you'll typically work with:

- [ ] `llamea/llamea.py` - Modify evolutionary algorithm
- [ ] `llamea/llm.py` - Add new LLM providers
- [ ] `benchmarks/main-evolutionary.py` - Adjust benchmark settings
- [ ] `misc/iohrun.py` - Customize IOH experiments
- [ ] `requirements.txt` - Add new dependencies
- [ ] `benchmark_config.yaml` - Add new configurations

## Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Copy `.env.example` to `.env` and add API keys
3. ✅ Test with: `python examples/minimum_example.py`
4. ✅ Run benchmark: `python benchmarks/main-evolutionary.py --llm dummy --budget 5`
5. 🚀 Start enhancing!

---

**Note:** This is a self-contained experimentation setup. All essential files for running LLAMEA, benchmarking, and IOH experiments are included.



