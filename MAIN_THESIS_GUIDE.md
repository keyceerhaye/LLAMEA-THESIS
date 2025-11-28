# main-thesis.py - Enhanced LLAMEA Implementation Guide

## 🎉 What's New

`main-thesis.py` now implements **full population-based evolutionary optimization** using the original LLAMEA methodology! You can choose between two modes:

1. **Evolutionary Mode** (NEW!) - Population-based with μ parents and λ offspring
2. **Iterative Mode** (Legacy) - Simple (1+1) iterative refinement

## 🚀 Quick Start

### Evolutionary Mode (Recommended)

```bash
# Basic evolutionary run
python main-thesis.py --evolutionary-mode --budget 100

# With custom population sizes
python main-thesis.py --evolutionary-mode --n-parents 4 --n-offspring 16 --budget 100

# With elitism (μ+λ strategy)
python main-thesis.py --evolutionary-mode --elitism --budget 100

# Full featured
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback \
  --n-parents 5 --n-offspring 20 --budget 150
```

### Iterative Mode (Legacy)

```bash
# Simple iterative refinement (original behavior)
python main-thesis.py --budget 50

# With elitism
python main-thesis.py --elitism --budget 50
```

## 📊 Modes Comparison

| Feature | Evolutionary Mode | Iterative Mode |
|---------|------------------|----------------|
| **Population** | ✅ Yes (μ parents) | ❌ No (single algorithm) |
| **Offspring** | ✅ Yes (λ per generation) | ❌ No |
| **Selection** | ✅ (μ+λ) or (μ,λ) | ⚠️ Simple best tracking |
| **Diversity** | ✅ High | ❌ Low |
| **Exploration** | ✅ Excellent | ⚠️ Limited |
| **API Efficiency** | ✅ Better | ⚠️ Sequential |
| **Complexity** | ⚠️ Higher | ✅ Simple |

## 🔬 Evolutionary Mode Details

### How It Works

1. **Initialization Phase**
   - Generate `n_parents` initial algorithms
   - Evaluate each on BBOB benchmark
   - Select best performers

2. **Evolution Phase** (repeated for N generations)
   - Randomly select parents
   - Generate `n_offspring` by mutating parents
   - Evaluate all offspring
   - Select best `n_parents` for next generation

3. **Selection Strategies**
   - **(μ+λ) Elitism**: Select from parents + offspring (preserves best)
   - **(μ,λ) Comma**: Select only from offspring (more exploration)

### Population Parameters

```bash
--n-parents 4        # Number of parent algorithms (μ)
--n-offspring 16     # Number of offspring per generation (λ)
--elitism            # Enable (μ+λ), otherwise uses (μ,λ)
--generations 10     # Fixed generations (optional)
```

### Budget Calculation

**Budget-driven** (default):
```
generations = (budget - n_parents) / n_offspring
total_api_calls = n_parents + (generations × n_offspring)
```

**Example:**
- Budget: 100
- Parents: 4
- Offspring: 16
- Generations: (100 - 4) / 16 = 6
- Total calls: 4 + (6 × 16) = 100

**Fixed generations**:
```bash
--generations 10  # Overrides budget calculation
```

## 📋 All Parameters

### API Configuration
```bash
--api-key <key>           # API key (or use .env)
--base-url <url>          # Custom API endpoint
--max-tokens <int>        # Max tokens per response
--model <name>            # AI model (default: gemini-2.0-flash)
```

### Experiment Configuration
```bash
--experiment-name <name>  # Experiment identifier
--budget <int>            # Total API calls (default: 100)
--eval-budget <int>       # Evaluations per algorithm (default: 10000)
```

### Evolutionary Configuration
```bash
--evolutionary-mode       # Enable population-based evolution
--n-parents <int>         # Number of parents μ (default: 4)
--n-offspring <int>       # Number of offspring λ (default: 16)
--generations <int>       # Fixed generations (optional)
--elitism                 # Enable (μ+λ) strategy
--detailed-feedback       # Detailed performance feedback
```

## 💡 Example Commands

### Quick Test (5 minutes)
```bash
python main-thesis.py --evolutionary-mode --n-parents 2 --n-offspring 4 --budget 10
```

### Small Experiment (1-2 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --budget 50
```

### Medium Experiment (3-5 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback \
  --n-parents 4 --n-offspring 16 --budget 100
```

### Large Experiment (6-12 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback \
  --n-parents 5 --n-offspring 20 --budget 200 --max-tokens 8192
```

### Custom API (e.g., AI/ML API)
```bash
python main-thesis.py --evolutionary-mode --elitism \
  --base-url https://api.aimlapi.com/v1 \
  --model gpt-4o --max-tokens 16384 --budget 100
```

### Fixed Generations
```bash
python main-thesis.py --evolutionary-mode --elitism \
  --n-parents 4 --n-offspring 16 --generations 10
```

## 📈 Understanding Output

### Evolutionary Mode Output

```
INITIALIZATION: Generating 4 parent algorithms
Initializing Parent 1/4 (API call 1)
  Evaluating RandomSearch...
  Fitness: 0.3245

[... more parents ...]

Initialization complete. Best initial fitness: 0.4521

GENERATION 1/6
API Calls: 4/100
Best so far: 0.4521 (AdaptiveSearch)

Offspring 1/16 (API call 5)
  Parent: AdaptiveSearch (fitness: 0.4521)
  Evaluating ImprovedAdaptiveSearch...
  Fitness: 0.4789
  🎉 NEW BEST! 0.4789 > 0.4521

[... more offspring ...]

Selection: (μ+λ) - Best 4 from 20 individuals
New population fitness: ['0.4789', '0.4521', '0.4312']

[... more generations ...]

EVOLUTIONARY OPTIMIZATION COMPLETED
Total API calls: 100
Generations completed: 6
Best algorithm: ImprovedAdaptiveSearch
Best fitness: 0.5234
Best generation: 4
```

### Output Files

```
exp-<timestamp>-<model>-<experiment>-evolutionary-elitism/
├── code/
│   ├── try-0-RandomSearch.py
│   ├── try-1-AdaptiveSearch.py
│   └── ...
├── ioh/
├── try-0-aucs.txt
├── try-1-aucs.txt
├── ...
├── conversationlog.txt
└── BEST_ALGORITHM.py  # 🎉 NEW! Best algorithm found
```

## 🎯 Strategy Recommendations

### For Exploration (Finding Novel Solutions)
```bash
python main-thesis.py --evolutionary-mode \
  --n-parents 5 --n-offspring 20 --budget 150
# No elitism = (μ,λ) = more exploration
```

### For Exploitation (Refining Best Solutions)
```bash
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback \
  --n-parents 3 --n-offspring 12 --budget 100
# Elitism = (μ+λ) = preserves best, detailed feedback guides refinement
```

### Balanced Approach
```bash
python main-thesis.py --evolutionary-mode --elitism \
  --n-parents 4 --n-offspring 16 --budget 100
# Standard configuration, good balance
```

## 🔍 Comparison with Other Scripts

| Script | Population | Selection | Use Case |
|--------|-----------|-----------|----------|
| **main-thesis.py** (Evolutionary) | ✅ Yes | ✅ (μ+λ)/(μ,λ) | **Best for research experiments** |
| **main-thesis.py** (Iterative) | ❌ No | ⚠️ Simple | Quick testing, simple refinement |
| **main-evolutionary.py** | ✅ Yes | ✅ Full LLAMEA | Production, full framework features |
| **main.py** | ❌ No | ❌ None | Legacy, basic testing |

## 🐛 Troubleshooting

### Budget Exhausted Early
```
WARNING: Total calls (120) exceed budget (100)
```
**Solution:** Reduce parents or offspring, or increase budget
```bash
--n-parents 3 --n-offspring 10 --budget 100
```

### Out of Memory
**Solution:** Reduce eval-budget or run sequentially
```bash
--eval-budget 5000
```

### API Rate Limits
**Solution:** The script handles rate limits automatically, but you can:
- Use a model with higher rate limits
- Reduce offspring count for slower generation

## 📚 Key Concepts

### μ (mu) - Parents
Number of algorithms kept in the population each generation.
- Higher μ = more diversity, slower convergence
- Lower μ = less diversity, faster convergence
- Typical: 3-5

### λ (lambda) - Offspring
Number of new algorithms generated per generation.
- Higher λ = more exploration, more API calls
- Lower λ = less exploration, fewer API calls
- Typical: 3-5× the number of parents

### (μ+λ) vs (μ,λ)
- **(μ+λ) Elitism**: Parents can survive (preserves best)
- **(μ,λ) Comma**: Parents die (forces innovation)

## 🎓 Research Tips

1. **Start small**: Test with `--budget 20` first
2. **Use elitism**: Almost always better with `--elitism`
3. **Detailed feedback**: Use `--detailed-feedback` for better refinement
4. **Track best**: Check `BEST_ALGORITHM.py` for the winner
5. **Compare modes**: Run both evolutionary and iterative to compare

## 📝 Citation

If you use this implementation in your research, please cite the original LLAMEA paper and mention the enhanced main-thesis.py implementation.

---

**Happy Evolving! 🧬🤖**

