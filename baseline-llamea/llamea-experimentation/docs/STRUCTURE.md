# LLaMEA Experimentation - Folder Structure

## Updated Tree

```
llamea-experimentation/
│
├── src/                         # Installable package (pip install -e .)
│   ├── llamea/                  # Core LLAMEA framework
│   │   ├── llamea.py            # Evolutionary engine
│   │   ├── llm.py               # LLM provider abstraction
│   │   ├── solution.py          # Solution data structure
│   │   ├── utils.py             # Patch + distance utilities
│   │   ├── loggers.py           # Logging helpers
│   │   └── bbobalgs/            # Example algorithms
│   └── misc/                    # IOH + visualization helpers
│       ├── iohrun.py
│       ├── plot_aucs.py
│       ├── python_ast_analysis.py
│       └── visualize_graphs.py
│
├── experiments/
│   ├── benchmarks/              # Runnable benchmark suite
│   │   ├── main-thesis.py       # CLI entry point (iterative & evolutionary)
│   │   ├── managers.py          # Algorithm manager + logger
│   │   ├── utils.py             # BBOB helpers
│   │   ├── plot_llamea_style.py # Publication plots
│   │   └── visualize_results.py # Quick progress charts
│   └── examples/                # Learning-oriented samples
│       ├── minimum_example.py
│       ├── simple_benchmark.py
│       ├── black-box-optimization.py
│       ├── black-box-opt-with-HPO.py
│       └── automl_example.py
│
├── config/
│   └── benchmark_config.yaml    # Reusable experiment presets
│
├── docs/                        # This documentation bundle
├── assets/                      # Images for reports
├── tests/                       # Placeholder for test suite
├── requirements.txt             # Dependency pinning
├── setup.py                     # Points setuptools at src/
├── README.md                    # Main usage guide
├── LICENSE
└── .env.example                 # Template for API keys
```

## Directory Highlights

### `src/llamea/`

- **Purpose:** Reusable LLAMEA framework.
- **Import path:** After `pip install -e .`, use `from llamea import LLaMEA`.
- **Modify when:** Extending the core algorithm, adding LLM providers, altering solution or logging behavior.

### `src/misc/`

- IOH runner (`iohrun.py`), plotting utilities, AST helpers, and transformation scripts live here.
- These modules assume `llamea` is available on `PYTHONPATH`.

### `experiments/benchmarks/`

- **main-thesis.py:** Orchestrates both iterative and evolutionary experiments with CLI arguments.
- **managers.py:** Handles API calls to OpenAI-compatible endpoints, logging each attempt.
- **plot_llamea_style.py / visualize_results.py:** Produce figures from `exp-*` result folders.

### `experiments/examples/`

- Self-contained scripts that show how to integrate LLAMEA in different scenarios (dummy LLM, BBOB, AutoML, etc.).
- Each script bootstraps `src/` onto `sys.path` so you can run them directly via `python experiments/examples/...`.

### `config/benchmark_config.yaml`

- Central place for storing named experiment presets (budgets, LLM providers, etc.).
- Load via:

```python
import yaml
with open("config/benchmark_config.yaml") as fh:
    profile = yaml.safe_load(fh)["quick_test"]
```

### `docs/`

- Contains INDEX, QUICKSTART, STRUCTURE, TREE, and other guides that explain the reorganized layout.

### `assets/`

- Shared figures such as `llamea_best_trajectory.png` for papers and presentations.

### `tests/`

- Currently ships with a README so you can drop pytest suites in-place (`pytest tests`).

## Minimal Setups

| Scenario         | Required Paths                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Hello World**  | `src/llamea/`, `experiments/examples/minimum_example.py`, `requirements.txt`                                                                     |
| **Benchmarking** | `experiments/benchmarks/main-thesis.py`, `experiments/benchmarks/managers.py`, `experiments/benchmarks/utils.py`, `config/benchmark_config.yaml` |
| **IOH Studies**  | Everything above + `src/misc/iohrun.py` and a populated `.env`                                                                                   |

## Typical Workflow

1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `pip install -e .` (exposes `llamea` from `src/`)
4. `cp .env.example .env` and fill in API keys
5. Run experiments:
   - `python experiments/examples/minimum_example.py`
   - `python experiments/benchmarks/main-thesis.py --evolutionary-mode --budget 50`
6. Visualize: `python experiments/benchmarks/visualize_results.py exp-*/`

## Quick Command Reference

```bash
# Examples
python experiments/examples/minimum_example.py

# Benchmarks
python experiments/benchmarks/main-thesis.py --budget 20 --model gemini-2.0-flash

# IOH
python src/misc/iohrun.py
```

## Enhancement Checklist

- [ ] `src/llamea/llm.py` – add or tweak providers.
- [ ] `experiments/benchmarks/main-thesis.py` – new CLI flags or evaluation logic.
- [ ] `config/benchmark_config.yaml` – capture frequently used settings.
- [ ] `tests/` – add pytest coverage before shipping features.
- [ ] `docs/` – update INDEX/QUICKSTART whenever workflows change.

---

**KISS Reminder:** keep changes scoped to the directory that owns the concern (core logic → `src/`, runnable scripts → `experiments/`, docs → `docs/`). This mirrors the LLAMEA-MADA layout for easier cross-project navigation.
