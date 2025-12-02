# LLAMEA-MADA

**L**arge **L**anguage **M**odel-based **E**volutionary **A**lgorithm Design - **M**etaheuristic **A**lgorithm **D**esign and **A**daptation

A framework for designing and evolving metaheuristic optimization algorithms using Large Language Models (LLMs).

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1   # Windows
source venv/bin/activate       # Linux/Mac

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Set up API keys
cp config/.env.example .env
# Edit .env and add your API keys

# 4. Run a simple example
python experiments/examples/minimum_example.py

# 5. Run evolutionary benchmark
python experiments/benchmarks/main-thesis.py --evolutionary-mode --budget 20
```

## 📁 Project Structure

```
llamea-experimentation/
├── src/                          # Core source code
│   ├── llamea/                   # LLAMEA framework
│   │   ├── __init__.py
│   │   ├── llamea.py            # Main LLAMEA engine
│   │   ├── llm.py               # LLM provider interface
│   │   ├── solution.py          # Solution data structures
│   │   ├── utils.py             # Utility functions
│   │   ├── loggers.py           # Experiment logging
│   │   └── bbobalgs/            # Example algorithms
│   └── misc/                     # Utilities and helpers
│       ├── iohrun.py            # IOH experimenter runner
│       ├── plot_aucs.py         # Visualization tools
│       └── utils.py             # General utilities
│
├── experiments/                  # Experiments and benchmarks
│   ├── benchmarks/              # Benchmark scripts
│   │   ├── main-thesis.py       # Main evolutionary benchmark
│   │   ├── managers.py          # Algorithm managers
│   │   ├── utils.py             # BBOB utilities
│   │   └── visualize_results.py # Result visualization
│   └── examples/                # Example scripts
│       ├── minimum_example.py   # Simplest example
│       ├── simple_benchmark.py  # Basic benchmark
│       └── automl_example.py    # AutoML example
│
├── config/                      # Configuration files
│   ├── benchmark_config.yaml    # Benchmark configurations
│   └── .env.example            # Environment template
│
├── docs/                        # Documentation
│   ├── README.md               # Comprehensive guide
│   ├── QUICKSTART.md           # Quick start guide
│   ├── GETTING_STARTED.md      # Detailed setup
│   └── STRUCTURE.md            # Project structure
│
├── assets/                      # Images and static files
├── tests/                       # Unit tests (to be added)
├── venv/                        # Virtual environment
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🎯 Key Features

- **LLM-Driven Evolution**: Use GPT-4, Gemini, or other LLMs to generate algorithms
- **Population-Based**: Support for (μ+λ) and (μ,λ) evolutionary strategies
- **BBOB Benchmarking**: Comprehensive testing on 24 BBOB functions
- **IOH Integration**: Full IOH Experimenter support
- **Multiple Modes**: Iterative refinement or population-based evolution
- **Flexible Configuration**: YAML-based experiment configuration

## 🔧 Installation

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Setup

```bash
# Clone and navigate to the project
cd llamea-experimentation

# Create virtual environment (if not exists)
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1   # Windows PowerShell
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/.env.example .env
# Edit .env with your API keys
```

## 📚 Usage

### Basic Example

```python
from src.llamea.llamea import LLAMEA
from src.llamea.llm import OpenAILLM

# Initialize LLM
llm = OpenAILLM(api_key="your_key")

# Create LLAMEA instance
llamea = LLAMEA(llm=llm, budget=100)

# Run optimization
result = llamea.run()
```

### Running Benchmarks

**Iterative Mode** (simple refinement):

```bash
python experiments/benchmarks/main-thesis.py --budget 20
```

**Evolutionary Mode** (population-based):

```bash
python experiments/benchmarks/main-thesis.py \
  --evolutionary-mode \
  --n-parents 4 \
  --n-offspring 16 \
  --budget 100 \
  --elitism \
  --detailed-feedback
```

### Command Line Options

| Option                | Description                       | Default          |
| --------------------- | --------------------------------- | ---------------- |
| `--evolutionary-mode` | Enable population-based evolution | False            |
| `--budget`            | Total API call budget             | 100              |
| `--eval-budget`       | Evaluations per algorithm         | 10000            |
| `--n-parents`         | Number of parent algorithms (μ)   | 4                |
| `--n-offspring`       | Offspring per generation (λ)      | 16               |
| `--elitism`           | Enable (μ+λ) strategy             | False            |
| `--detailed-feedback` | Detailed performance feedback     | False            |
| `--model`             | LLM model to use                  | gemini-2.0-flash |
| `--api-key`           | API key                           | From .env        |
| `--base-url`          | Custom API endpoint               | From .env        |

## 🧪 Examples

See `experiments/examples/` for complete examples:

- `minimum_example.py` - Simplest possible usage
- `simple_benchmark.py` - Basic BBOB benchmarking
- `black-box-optimization.py` - Full BBOB suite
- `automl_example.py` - AutoML application

## 📊 Output

Experiments create timestamped directories:

```
exp-MM-DD_HHMMSS-<model>-<name>/
├── code/                    # Generated algorithm code
│   └── try-*-*.py
├── ioh/                     # IOH data
├── conversationlog.txt      # LLM conversations
├── try-*-aucs.txt          # Performance metrics
└── BEST_ALGORITHM.py       # Best found algorithm
```

## 🔬 Extending LLAMEA-MADA

### Adding New LLM Providers

Edit `src/llamea/llm.py`:

```python
class CustomLLM(BaseLLM):
    def generate(self, prompt):
        # Your implementation
        pass
```

### Adding New Benchmarks

Create a new file in `experiments/benchmarks/`:

```python
from src.llamea.llamea import LLAMEA
# Your benchmark code
```

### Modifying Evolution Strategy

Edit `experiments/benchmarks/main-thesis.py`:

- Modify `run_evolutionary_mode()` for different selection
- Change `selection()` function for custom strategies

## 🐛 Troubleshooting

**Import Errors:**

```bash
# Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)\src"              # Windows
```

**API Key Issues:**

- Ensure `.env` file exists in project root
- Check API key format: `OPENAI_API_KEY=sk-...`
- Verify `python-dotenv` is installed

**Virtual Environment:**

```bash
# Recreate if needed
rm -rf venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📖 Documentation

- **Quick Start**: `docs/QUICKSTART.md` - Get running in 5 minutes
- **Getting Started**: `docs/GETTING_STARTED.md` - Detailed walkthrough
- **Project Structure**: `docs/STRUCTURE.md` - File organization
- **Main Thesis Guide**: `docs/MAIN_THESIS_GUIDE.md` - Research context

## 🤝 Contributing

1. Create a feature branch
2. Make your changes in appropriate directories:
   - Core code → `src/llamea/`
   - Experiments → `experiments/`
   - Tests → `tests/`
   - Docs → `docs/`
3. Add tests if applicable
4. Submit a pull request

## 📄 License

MIT License - See `LICENSE` file

## 🔗 References

- IOH Experimenter: https://iohprofiler.github.io/
- BBOB Benchmark: https://coco.gforge.inria.fr/
- OpenAI API: https://platform.openai.com/

## 🎓 Citation

If you use LLAMEA-MADA in your research, please cite:

```bibtex
@software{llamea_mada,
  title={LLAMEA-MADA: LLM-based Metaheuristic Algorithm Design and Adaptation},
  author={LLAMEA Research Team},
  year={2025},
  url={https://github.com/yourrepo/llamea-mada}
}
```

---

**Maintained with ❤️ following the KISS principle**
