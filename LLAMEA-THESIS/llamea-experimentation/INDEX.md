# LLaMEA Experimentation - Index

## 📖 Documentation Quick Access

### Start Here (Choose Your Path)

| If you want to... | Read this file | Time |
|-------------------|---------------|------|
| **Get running in 5 minutes** | [QUICKSTART.md](QUICKSTART.md) | 5 min |
| **Understand the complete setup** | [README.md](README.md) | 20 min |
| **See all files and their purpose** | [STRUCTURE.md](STRUCTURE.md) | 10 min |
| **Follow step-by-step experiments** | [GETTING_STARTED.md](GETTING_STARTED.md) | 30 min |
| **See the folder tree** | [TREE.txt](TREE.txt) | 2 min |

### Quick Reference

| Topic | File | Description |
|-------|------|-------------|
| Installation | [requirements.txt](requirements.txt) | All dependencies |
| Configuration | [benchmark_config.yaml](benchmark_config.yaml) | Pre-configured setups |
| Environment | [.env.example](.env.example) | API keys template |
| Setup | [setup.py](setup.py) | Package installation |
| License | [LICENSE](LICENSE) | MIT License |

## 🚀 Quick Commands

```bash
# Installation
pip install -r requirements.txt

# Test (no API keys)
python examples/minimum_example.py

# Benchmark (dummy LLM)
python benchmarks/main-evolutionary.py --llm dummy --budget 5

# Production (OpenAI)
python benchmarks/main-evolutionary.py --llm openai --budget 50

# IOH Experimenter
python misc/iohrun.py
```

## 📁 Folder Navigation

### Core Framework
- [llamea/](llamea/) - Main LLAMEA framework
  - [llamea.py](llamea/llamea.py) - Evolutionary engine
  - [llm.py](llamea/llm.py) - LLM providers
  - [solution.py](llamea/solution.py) - Solution class
  - [utils.py](llamea/utils.py) - Utilities
  - [loggers.py](llamea/loggers.py) - Logging
  - [bbobalgs/](llamea/bbobalgs/) - Example algorithms

### Examples
- [examples/](examples/) - Example scripts
  - [minimum_example.py](examples/minimum_example.py) ⭐ Start here
  - [simple_benchmark.py](examples/simple_benchmark.py)
  - [black-box-optimization.py](examples/black-box-optimization.py)
  - [black-box-opt-with-HPO.py](examples/black-box-opt-with-HPO.py)
  - [automl_example.py](examples/automl_example.py)

### Benchmarks
- [benchmarks/](benchmarks/) - Benchmark scripts
  - [main-evolutionary.py](benchmarks/main-evolutionary.py) ⭐ Recommended
  - [main-thesis.py](benchmarks/main-thesis.py)
  - [main.py](benchmarks/main.py)
  - [managers.py](benchmarks/managers.py)
  - [utils.py](benchmarks/utils.py)

### Utilities
- [misc/](misc/) - Utility scripts
  - [iohrun.py](misc/iohrun.py) - IOH runner
  - [plot_aucs.py](misc/plot_aucs.py) - Visualization
  - [utils.py](misc/utils.py) - General utilities
  - [ast.py](misc/ast.py) - AST analysis
  - [python_ast_analysis.py](misc/python_ast_analysis.py)
  - [transform_to_stn.py](misc/transform_to_stn.py)
  - [visualize_graphs.py](misc/visualize_graphs.py)

### Documentation
- [docs/](docs/) - Empty folder for your notes

## 🎯 Common Tasks

### Task: Run First Experiment
1. Read [QUICKSTART.md](QUICKSTART.md)
2. Run `python examples/minimum_example.py`

### Task: Benchmark with OpenAI
1. Copy `.env.example` to `.env`
2. Add your OpenAI API key
3. Run `python benchmarks/main-evolutionary.py --llm openai --budget 50`

### Task: Customize Evaluation
1. Open [benchmarks/main-evolutionary.py](benchmarks/main-evolutionary.py)
2. Modify the `evaluate()` function
3. Run your modified script

### Task: Add New LLM Provider
1. Open [llamea/llm.py](llamea/llm.py)
2. Create a new class inheriting from `LLM`
3. Implement `query()` method
4. Use in your experiments

### Task: Visualize Results
1. Run an experiment
2. Use [misc/plot_aucs.py](misc/plot_aucs.py)
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

**Last Updated:** November 22, 2025  
**Version:** 1.0.0  
**Status:** Ready for experimentation ✅






