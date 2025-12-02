# Getting Started with Enhanced main-thesis.py

## 🎯 What is main-thesis.py?

An enhanced benchmark script that implements **population-based evolutionary optimization** using LLMs to generate and evolve algorithms for black-box optimization.

**Two modes available:**
1. **Evolutionary Mode** (NEW!) - Population-based with multiple parents and offspring
2. **Iterative Mode** (Legacy) - Simple iterative refinement

## 🚀 5-Minute Quick Start

### Step 1: Navigate to Directory
```bash
cd LLAMEA-THESIS/llamea-experimentation/benchmarks
```

### Step 2: Create .env File
```bash
# Create .env with your API key
echo "OPENAI_API_KEY=your_api_key_here" > .env

# For custom endpoints (optional)
echo "BASE_URL=https://api.aimlapi.com/v1" >> .env
```

### Step 3: Run Your First Experiment
```bash
# Quick test (10 minutes, uses minimal budget)
python main-thesis.py --evolutionary-mode \
  --n-parents 2 --n-offspring 4 \
  --budget 10 --eval-budget 1000
```

### Step 4: Check Results
```bash
# Results are in exp-<timestamp>-<model>-<experiment>-evolutionary/
ls -la exp-*

# Check the best algorithm found
cat exp-*/BEST_ALGORITHM.py
```

## 🎓 Understanding the Modes

### Evolutionary Mode (Recommended)

**What it does:**
- Maintains a population of μ parent algorithms
- Generates λ offspring each generation
- Selects best algorithms to survive
- Tracks best algorithm across generations

**When to use:**
- Research experiments
- You want better results
- You need diversity
- You have time/budget

**Command:**
```bash
python main-thesis.py --evolutionary-mode --elitism --budget 100
```

### Iterative Mode (Legacy)

**What it does:**
- Generates one initial algorithm
- Refines it iteratively
- Simple (1+1) strategy

**When to use:**
- Quick testing
- Simple experiments
- Limited budget
- You want simplicity

**Command:**
```bash
python main-thesis.py --budget 50
```

## 📋 Essential Parameters

| Parameter | What It Does | Example |
|-----------|-------------|---------|
| `--evolutionary-mode` | Enable population evolution | `--evolutionary-mode` |
| `--budget` | Total API calls | `--budget 100` |
| `--n-parents` | Number of parents (μ) | `--n-parents 4` |
| `--n-offspring` | Offspring per generation (λ) | `--n-offspring 16` |
| `--elitism` | Keep best algorithms | `--elitism` |
| `--eval-budget` | Evaluations per algorithm | `--eval-budget 10000` |
| `--model` | AI model to use | `--model gemini-2.0-flash` |

## 💡 Common Use Cases

### Use Case 1: Quick Test
**Goal:** Test if everything works (5-10 minutes)

```bash
python main-thesis.py --evolutionary-mode \
  --n-parents 2 --n-offspring 4 \
  --budget 10 --eval-budget 1000
```

**What happens:**
- Creates 2 initial algorithms
- Generates 4 offspring
- Runs for ~2 generations
- Takes ~5-10 minutes

### Use Case 2: Thesis Experiment
**Goal:** Get good results for thesis chapter (3-5 hours)

```bash
python main-thesis.py --evolutionary-mode --elitism \
  --detailed-feedback --budget 100
```

**What happens:**
- Creates 4 initial algorithms
- Generates 16 offspring per generation
- Runs for ~6 generations
- Keeps best algorithms (elitism)
- Takes ~3-5 hours

### Use Case 3: Publication Quality
**Goal:** Best possible results (6-12 hours)

```bash
python main-thesis.py --evolutionary-mode --elitism \
  --detailed-feedback \
  --n-parents 5 --n-offspring 20 \
  --budget 200 --max-tokens 8192
```

**What happens:**
- Creates 5 initial algorithms
- Generates 20 offspring per generation
- Runs for ~9 generations
- Maximum exploration
- Takes ~6-12 hours

### Use Case 4: Custom API (e.g., AI/ML API)
**Goal:** Use different API provider

```bash
python main-thesis.py --evolutionary-mode --elitism \
  --base-url https://api.aimlapi.com/v1 \
  --model gpt-4o --max-tokens 16384 \
  --budget 100
```

## 📊 Understanding Output

### Console Output
```
INITIALIZATION: Generating 4 parent algorithms
  Initializing Parent 1/4 (API call 1)
  Evaluating RandomSearch...
  Fitness: 0.3245
  
GENERATION 1/6
  Offspring 1/16 (API call 5)
  Parent: AdaptiveSearch (fitness: 0.4521)
  Evaluating ImprovedAdaptiveSearch...
  Fitness: 0.4789
  🎉 NEW BEST! 0.4789 > 0.4521
```

**What it means:**
- **Fitness**: AUC score (0.0 to 1.0, higher is better)
- **NEW BEST**: Found better algorithm than previous best
- **Generation**: Current evolutionary generation
- **API call**: Number of LLM calls used

### Output Files
```
exp-11-27_143052-gemini-2.0-flash-thesis-experiment-evolutionary-elitism/
├── code/
│   ├── try-0-RandomSearch.py          # First algorithm
│   ├── try-1-AdaptiveSearch.py        # Second algorithm
│   └── ...
├── try-0-aucs.txt                      # Performance data
├── try-1-aucs.txt
├── ...
├── conversationlog.txt                 # LLM conversations
└── BEST_ALGORITHM.py                   # 🏆 Best algorithm found!
```

**What to check:**
1. **BEST_ALGORITHM.py** - The winner! Use this.
2. **conversationlog.txt** - See what the LLM said
3. **try-*-aucs.txt** - Performance metrics

## 🎯 Budget Planning

### How Budget Works
```
Total API calls = n_parents + (generations × n_offspring)
Generations = (budget - n_parents) / n_offspring
```

### Examples

| Budget | Parents | Offspring | Generations | Time (est) |
|--------|---------|-----------|-------------|------------|
| 10 | 2 | 4 | 2 | 10 min |
| 50 | 4 | 16 | 2-3 | 1-2 hours |
| 100 | 4 | 16 | 6 | 3-5 hours |
| 200 | 5 | 20 | 9 | 6-12 hours |

## 🐛 Troubleshooting

### Problem: "API key is required"
**Solution:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key" > .env
```

### Problem: "Budget exhausted early"
**Solution:**
```bash
# Reduce parents or offspring
python main-thesis.py --evolutionary-mode \
  --n-parents 3 --n-offspring 10 --budget 100
```

### Problem: "Out of memory"
**Solution:**
```bash
# Reduce evaluation budget
python main-thesis.py --evolutionary-mode \
  --eval-budget 5000 --budget 100
```

### Problem: "API rate limit"
**Solution:**
The script handles this automatically. Just wait, it will retry.

## 📚 Next Steps

### 1. Read More Documentation
- **QUICK_REFERENCE.md** - Quick commands
- **MAIN_THESIS_GUIDE.md** - Complete guide
- **CHANGELOG_MAIN_THESIS.md** - What's new

### 2. Try Different Configurations
```bash
# More exploration (no elitism)
python main-thesis.py --evolutionary-mode \
  --n-parents 5 --n-offspring 20 --budget 150

# More exploitation (with elitism)
python main-thesis.py --evolutionary-mode --elitism \
  --n-parents 3 --n-offspring 12 --budget 100
```

### 3. Compare Modes
```bash
# Run evolutionary mode
python main-thesis.py --evolutionary-mode --budget 50

# Run iterative mode
python main-thesis.py --budget 50

# Compare BEST_ALGORITHM.py from both runs
```

### 4. Analyze Results
```python
# Load and analyze AUC data
import numpy as np

aucs = np.loadtxt('exp-*/try-0-aucs.txt')
print(f"Mean: {np.mean(aucs):.4f}")
print(f"Std: {np.std(aucs):.4f}")
```

## ✅ Success Checklist

Before running:
- [ ] Created `.env` with API key
- [ ] Navigated to `benchmarks/` directory
- [ ] Chosen mode (evolutionary recommended)
- [ ] Set appropriate budget
- [ ] Decided on elitism

After running:
- [ ] Check `BEST_ALGORITHM.py`
- [ ] Review console output
- [ ] Analyze AUC files
- [ ] Save experiment folder

## 🎓 Tips for Thesis Research

1. **Start Small**: Always test with `--budget 10` first
2. **Use Elitism**: Almost always better results
3. **Document Everything**: Keep notes on configurations
4. **Compare**: Run multiple configurations
5. **Save Results**: Keep all experiment folders

## 💡 Pro Tips

### Tip 1: Naming Experiments
```bash
python main-thesis.py --evolutionary-mode \
  --experiment-name "test-run-1" --budget 100
```

### Tip 2: Maximum Tokens
```bash
python main-thesis.py --evolutionary-mode \
  --max-tokens 8192 --budget 100
```

### Tip 3: Detailed Feedback
```bash
python main-thesis.py --evolutionary-mode \
  --detailed-feedback --budget 100
# Gives LLM more info about performance
```

### Tip 4: Fixed Generations
```bash
python main-thesis.py --evolutionary-mode \
  --generations 10 --n-parents 4 --n-offspring 16
# Runs exactly 10 generations
```

## 🎉 You're Ready!

Start with this command:
```bash
python main-thesis.py --evolutionary-mode --elitism --budget 100
```

This will:
- ✅ Use population-based evolution
- ✅ Keep best algorithms (elitism)
- ✅ Run for ~6 generations
- ✅ Take ~3-5 hours
- ✅ Give you good results

## 📞 Need Help?

1. **Quick answers**: Check QUICK_REFERENCE.md
2. **Detailed info**: Read MAIN_THESIS_GUIDE.md
3. **What's new**: See CHANGELOG_MAIN_THESIS.md
4. **Troubleshooting**: See this file's troubleshooting section

---

**Happy Evolving! 🧬🤖**

**You're now ready to run population-based evolutionary experiments for your thesis!** 🎓

