# Baseline LLAMEA Project Structure

A clean, organized structure following the KISS principle and matching LLAMEA-MADA layout.

## 📁 Complete Directory Tree

```
llamea-experimentation/                 # 🏠 Project root
│
├── 📦 src/                            # Core source code
│   ├── __init__.py                    # Package initialization
│   │
│   ├── 🧬 llamea/                     # LLAMEA framework
│   │   ├── __init__.py
│   │   ├── llamea.py                 # 🎯 Main engine
│   │   ├── llm.py                    # 🤖 LLM providers
│   │   ├── solution.py               # 📊 Data structures
│   │   ├── utils.py                  # 🔧 Utilities
│   │   ├── loggers.py                # 📝 Experiment logging
│   │   ├── ERADS_QuantumFluxUltraRefined.py
│   │   └── bbobalgs/                 # Example algorithms
│   │       ├── __init__.py
│   │       └── ERADS_QuantumFluxUltraRefined.py
│   │
│   └── 🛠️ misc/                       # Additional utilities
│       ├── __init__.py
│       ├── iohrun.py                 # IOH experimenter
│       ├── plot_aucs.py              # Visualization
│       ├── python_ast_analysis.py    # AST analysis
│       ├── transform_to_stn.py
│       ├── visualize_graphs.py
│       ├── ast.py
│       └── utils.py
│
├── 🔬 experiments/                    # All experiments and examples
│   ├── __init__.py
│   │
│   ├── 📊 benchmarks/                # Main benchmarking scripts
│   │   ├── main-thesis.py            # ⭐ Main evolutionary benchmark
│   │   ├── managers.py               # Algorithm managers
│   │   ├── utils.py                  # BBOB utilities
│   │   ├── plot_llamea_style.py      # Plotting
│   │   └── visualize_results.py      # Result visualization
│   │
│   └── 📚 examples/                  # Learning examples
│       ├── minimum_example.py        # 🌟 Start here!
│       ├── simple_benchmark.py       # Basic benchmark
│       ├── black-box-optimization.py # BBOB example
│       ├── black-box-opt-with-HPO.py # With HPO
│       └── automl_example.py         # AutoML example
│
├── ⚙️ config/                         # Configuration files
│   └── benchmark_config.yaml         # Benchmark presets
│
├── 📖 docs/                          # Documentation
│   ├── INDEX.md                      # Documentation index
│   ├── QUICKSTART.md                 # 5-minute guide
│   ├── GETTING_STARTED.md            # Detailed setup
│   ├── GETTING_STARTED_MAIN_THESIS.md
│   ├── MAIN_THESIS_GUIDE.md          # Research context
│   ├── STRUCTURE.md                  # Structure doc
│   ├── QUICK_REFERENCE.md            # Quick ref
│   ├── CHANGELOG_MAIN_THESIS.md      # Changelog
│   ├── DEPLOYMENT_SUMMARY.md         # Deployment notes
│   └── TREE.txt                      # Tree view
│
├── 🖼️ assets/                        # Images and media
│   ├── comparison.png
│   ├── llamea_best_trajectory.png
│   └── llamea_comparison.png
│
├── 🧪 tests/                         # Test suite
│   └── README.md                     # Testing guide
│
├── 📄 Core Files (Root)
│   ├── README.md                     # ⭐ Main documentation
│   ├── LICENSE                       # ⚖️ MIT License
│   ├── setup.py                      # 📦 Package setup
│   └── requirements.txt              # 📚 Dependencies
│
└── 🚫 Excluded from Git (.gitignore)
    ├── __pycache__/
    ├── *.pyc
    ├── .env
    ├── exp-*/                        # Experiment results
    └── *.egg-info/
```

## 🎯 Key Directories Explained

### `src/` - Core Source Code

**Purpose**: All reusable framework code  
**When to use**: Adding core functionality, new LLM providers, utilities

```python
# Example imports after pip install -e .
from llamea import LLaMEA
from llamea.llm import OpenAI_LLM
from src.misc.plot_aucs import plot_results
```

### `experiments/` - Experiments & Examples

**Purpose**: Runnable scripts for benchmarking and learning  
**When to use**: Creating new experiments, benchmarks, or examples

**Structure**:

- `benchmarks/` - Production experiments with full configuration
- `examples/` - Simple, educational scripts

### `config/` - Configuration

**Purpose**: All configuration files  
**When to use**: Adding new config options, benchmarks presets

**Files**:

- `benchmark_config.yaml` - Predefined experiment configurations

### `docs/` - Documentation

**Purpose**: All project documentation  
**When to use**: Updating guides, adding tutorials

**Key docs**:

- `INDEX.md` - Documentation index
- `QUICKSTART.md` - Get started fast
- `GETTING_STARTED.md` - Detailed walkthrough

### `assets/` - Static Files

**Purpose**: Images, plots, media files  
**When to use**: Adding visualizations, logos, diagrams

### `tests/` - Test Suite

**Purpose**: Unit tests and integration tests  
**When to use**: Adding tests for new features

## 🚀 Quick Navigation

### I want to...

**Run an experiment**
→ `experiments/benchmarks/main-thesis.py`

**Learn by example**
→ `experiments/examples/minimum_example.py`

**Modify LLAMEA core**
→ `src/llamea/llamea.py`

**Add new LLM provider**
→ `src/llamea/llm.py`

**Configure experiments**
→ `config/benchmark_config.yaml`

**Read documentation**
→ `README.md` or `docs/`

**See visualizations**
→ `assets/`

**Add tests**
→ `tests/`

## 📊 File Count Summary

```
📦 src/llamea/         : 9 files  (Core framework)
🛠️ src/misc/           : 8 files  (Utilities)
📊 experiments/benchmarks/ : 5 files  (Benchmarks)
📚 experiments/examples/   : 5 files  (Examples)
📖 docs/              : 10 files (Documentation)
⚙️ config/            : 1 file   (Configuration)
🖼️ assets/            : 3 files  (Images)
```

**Total**: ~40 organized files + documentation

## 🔄 Common Workflows

### Adding a New Feature

1. **Core feature** → Edit `src/llamea/`
2. **Test it** → Add to `tests/`
3. **Example** → Create in `experiments/examples/`
4. **Document** → Update `docs/`

### Running Experiments

1. **Configure** → Edit `config/benchmark_config.yaml`
2. **Run** → `python experiments/benchmarks/main-thesis.py`
3. **Results** → Check `exp-*` folders (git-ignored)
4. **Visualize** → Use `experiments/benchmarks/visualize_results.py`

### Contributing Code

1. **Read** → `CONTRIBUTING.md` (if exists)
2. **Edit** → Appropriate directory
3. **Test** → `pytest tests/`
4. **Document** → Update `docs/`

## 🎓 Best Practices

### File Placement

- ✅ Core logic → `src/llamea/`
- ✅ Utilities → `src/misc/`
- ✅ Experiments → `experiments/benchmarks/`
- ✅ Examples → `experiments/examples/`
- ✅ Docs → `docs/`
- ✅ Config → `config/`
- ✅ Tests → `tests/`

### Import Convention

```python
# Absolute imports from src/
from src.llamea.llamea import LLaMEA
from src.llamea.llm import OpenAI_LLM

# Or install package and use
from llamea import LLaMEA
from llamea.llm import OpenAI_LLM
```

### Naming Convention

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_CASE`

## 🔍 Finding Things

**By Purpose**:

- Optimization algorithms → `src/llamea/llamea.py`
- LLM integration → `src/llamea/llm.py`
- Benchmarking → `experiments/benchmarks/`
- Learning examples → `experiments/examples/`
- Configuration → `config/`

**By File Type**:

- Python code → `src/`, `experiments/`
- Documentation → `docs/`, `README.md`
- Configuration → `config/`
- Images → `assets/`
- Tests → `tests/`

## 📚 Related Documentation

- **README.md** - Main project documentation
- **docs/QUICKSTART.md** - Get running quickly
- **docs/INDEX.md** - Documentation index
- **docs/STRUCTURE.md** - Detailed structure guide

---

**Structure follows the KISS principle: Simple, Clear, Maintainable** 🎯

**Matches LLAMEA-MADA layout for consistency across projects** 🔄
