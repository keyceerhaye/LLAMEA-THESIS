# LLaMEA Experimentation - Index

## 📖 Documentation Quick Access

### Start Here (Choose Your Path)

| If you want to...                   | Read this file                           | Time   |
| ----------------------------------- | ---------------------------------------- | ------ |
| **Get running in 5 minutes**        | [QUICKSTART.md](QUICKSTART.md)           | 5 min  |
| **Understand the complete setup**   | [README.md](README.md)                   | 20 min |
| **See all files and their purpose** | [STRUCTURE.md](STRUCTURE.md)             | 10 min |
| **Follow step-by-step experiments** | [GETTING_STARTED.md](GETTING_STARTED.md) | 30 min |
| **See the folder tree**             | [TREE.txt](TREE.txt)                     | 2 min  |

### Quick Reference

| Topic         | File                                                            | Description           |
| ------------- | --------------------------------------------------------------- | --------------------- |
| Installation  | [requirements.txt](../requirements.txt)                         | All dependencies      |
| Configuration | [config/benchmark_config.yaml](../config/benchmark_config.yaml) | Pre-configured setups |
| Environment   | [.env.example](../.env.example)                                 | API keys template     |
| Setup         | [setup.py](../setup.py)                                         | Package installation  |
| License       | [LICENSE](../LICENSE)                                           | MIT License           |

## 🚀 Quick Commands

```bash
# Installation
pip install -r requirements.txt
pip install -e .

# Test (no API keys)
python experiments/examples/minimum_example.py

# Benchmark (dummy LLM)
python experiments/benchmarks/main-thesis.py --budget 5

# Production (OpenAI)
python experiments/benchmarks/main-thesis.py --evolutionary-mode --budget 50 --model gpt-4.1

# IOH Experimenter
python src/misc/iohrun.py
```

## 📁 Folder Navigation

### Core Framework

- [src/llamea/](../src/llamea/) - Main LLAMEA framework
  - [llamea.py](../src/llamea/llamea.py) - Evolutionary engine
  - [llm.py](../src/llamea/llm.py) - LLM providers
  - [solution.py](../src/llamea/solution.py) - Solution class
  - [utils.py](../src/llamea/utils.py) - Utilities
  - [loggers.py](../src/llamea/loggers.py) - Logging
  - [bbobalgs/](../src/llamea/bbobalgs/) - Example algorithms

### Examples

- [experiments/examples/](../experiments/examples/) - Example scripts
  - [minimum_example.py](../experiments/examples/minimum_example.py) ⭐ Start here
  - [simple_benchmark.py](../experiments/examples/simple_benchmark.py)
  - [black-box-optimization.py](../experiments/examples/black-box-optimization.py)
  - [black-box-opt-with-HPO.py](../experiments/examples/black-box-opt-with-HPO.py)
  - [automl_example.py](../experiments/examples/automl_example.py)

### Benchmarks

- [experiments/benchmarks/](../experiments/benchmarks/) - Benchmark scripts
  - [main-thesis.py](../experiments/benchmarks/main-thesis.py) ⭐ Recommended
  - [managers.py](../experiments/benchmarks/managers.py)
  - [utils.py](../experiments/benchmarks/utils.py)
  - [plot_llamea_style.py](../experiments/benchmarks/plot_llamea_style.py)
  - [visualize_results.py](../experiments/benchmarks/visualize_results.py)

### Utilities

- [src/misc/](../src/misc/) - Utility scripts
  - [iohrun.py](../src/misc/iohrun.py) - IOH runner
  - [plot_aucs.py](../src/misc/plot_aucs.py) - Visualization
  - [utils.py](../src/misc/utils.py) - General utilities
  - [ast.py](../src/misc/ast.py) - AST analysis
  - [python_ast_analysis.py](../src/misc/python_ast_analysis.py)
  - [transform_to_stn.py](../src/misc/transform_to_stn.py)
  - [visualize_graphs.py](../src/misc/visualize_graphs.py)

### Documentation

- [docs/](.) - Guides moved inside this folder

## 🎯 Common Tasks

### Task: Run First Experiment

1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `python experiments/examples/minimum_example.py`

### Task: Benchmark with OpenAI

1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Run `python experiments/benchmarks/main-thesis.py --evolutionary-mode --budget 50 --model gpt-4.1`

### Task: Customize Evaluation

1. Open [experiments/benchmarks/main-thesis.py](../experiments/benchmarks/main-thesis.py)
2. Modify the evaluation logic (e.g., `evaluate_algorithm`)
3. Run your modified script

### Task: Add New LLM Provider

1. Open [src/llamea/llm.py](../src/llamea/llm.py)
2. Create a new class inheriting from `LLM`
3. Implement `query()` method
4. Use in your experiments

### Task: Visualize Results

1. Run an experiment
2. Use [experiments/benchmarks/visualize_results.py](../experiments/benchmarks/visualize_results.py)
3. Check generated plots

## 📊 File Statistics

- **Total Files:** 36
- **Python Files:** 28
- **Documentation Files:** 8
- **Configuration Files:** 3

## 🔗 External Resources

- [Original Repository](https://github.com/XAI-liacs/LLaMEA)
- [IOH Experimenter](https://github.com/IOHprofiler/IOHexperimenter)
- [BBOB Benchmark](https://coco.gforge.inria.fr/)

## 📝 Notes

- This is a self-contained setup
- All essential files included
- Ready for experimentation
- Follows KISS principle
- Well-documented

## 🆘 Need Help?

1. Check [README.md](README.md) for detailed documentation
2. Check [GETTING_STARTED.md](GETTING_STARTED.md) for troubleshooting
3. Check original `docs/` in main repository
4. Open an issue on GitHub

---

**Last Updated:** December 2, 2025  
**Version:** 1.0.0  
**Status:** Ready for experimentation ✅
