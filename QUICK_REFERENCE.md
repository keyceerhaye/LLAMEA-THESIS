# main-thesis.py Quick Reference Card

## 🚀 Quick Start

### Evolutionary Mode (Recommended)
```bash
python main-thesis.py --evolutionary-mode --elitism --budget 100
```

### Iterative Mode (Legacy)
```bash
python main-thesis.py --budget 50
```

## 📋 Essential Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--evolutionary-mode` | Off | Enable population-based evolution |
| `--budget` | 100 | Total API calls |
| `--n-parents` | 4 | Number of parents (μ) |
| `--n-offspring` | 16 | Offspring per generation (λ) |
| `--elitism` | Off | Use (μ+λ) instead of (μ,λ) |
| `--eval-budget` | 10000 | Evaluations per algorithm |
| `--model` | gemini-2.0-flash | AI model to use |
| `--max-tokens` | Auto | Max tokens per response |

## 💡 Common Commands

### Quick Test (5-10 min)
```bash
python main-thesis.py --evolutionary-mode --n-parents 2 --n-offspring 4 --budget 10 --eval-budget 1000
```

### Small Experiment (1-2 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --budget 50
```

### Medium Experiment (3-5 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback --budget 100
```

### Large Experiment (6-12 hours)
```bash
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback --n-parents 5 --n-offspring 20 --budget 200
```

### Custom API
```bash
python main-thesis.py --evolutionary-mode --base-url https://api.aimlapi.com/v1 --model gpt-4o --budget 100
```

## 🎯 Strategy Guide

### For Exploration
```bash
# No elitism = (μ,λ) = more diversity
python main-thesis.py --evolutionary-mode --n-parents 5 --n-offspring 20 --budget 150
```

### For Exploitation
```bash
# Elitism + detailed feedback = refine best
python main-thesis.py --evolutionary-mode --elitism --detailed-feedback --budget 100
```

### Balanced
```bash
# Standard configuration
python main-thesis.py --evolutionary-mode --elitism --n-parents 4 --n-offspring 16 --budget 100
```

## 📊 Budget Calculator

**Formula**: `total_calls = n_parents + (generations × n_offspring)`

**Generations**: `generations = (budget - n_parents) / n_offspring`

### Examples

| Budget | Parents | Offspring | Generations | Total Calls |
|--------|---------|-----------|-------------|-------------|
| 50 | 2 | 8 | 6 | 50 |
| 100 | 4 | 16 | 6 | 100 |
| 150 | 5 | 20 | 7 | 145 |
| 200 | 5 | 20 | 9 | 185 |

## 🔍 Output Files

```
exp-<timestamp>-<model>-<experiment>-evolutionary-elitism/
├── code/
│   ├── try-0-Algorithm1.py      # Generated algorithms
│   └── ...
├── try-0-aucs.txt                # Performance metrics
├── ...
├── conversationlog.txt           # LLM conversations
└── BEST_ALGORITHM.py             # 🏆 Best algorithm found
```

## 🐛 Troubleshooting

### Budget Exceeded Warning
```bash
# Reduce parents or offspring
--n-parents 3 --n-offspring 10
```

### API Rate Limits
```bash
# Script handles automatically, but you can:
# - Use model with higher limits
# - Reduce offspring count
```

### Out of Memory
```bash
# Reduce evaluation budget
--eval-budget 5000
```

## 📈 Performance Tips

1. **Always use `--elitism`** for better results
2. **Use `--detailed-feedback`** for faster convergence
3. **Start small** (budget 20) to test
4. **Check `BEST_ALGORITHM.py`** for the winner
5. **Compare modes**: Run both evolutionary and iterative

## 🎓 Research Configurations

### Minimal (Testing)
```bash
--evolutionary-mode --budget 20 --eval-budget 1000
```

### Standard (Thesis Chapter)
```bash
--evolutionary-mode --elitism --detailed-feedback --budget 100
```

### Extensive (Publication)
```bash
--evolutionary-mode --elitism --detailed-feedback --n-parents 5 --n-offspring 20 --budget 200
```

## 🔄 Mode Comparison

| Feature | Evolutionary | Iterative |
|---------|-------------|-----------|
| Population | ✅ Yes | ❌ No |
| Diversity | ✅ High | ❌ Low |
| Exploration | ✅ Excellent | ⚠️ Limited |
| Speed | ⚠️ Slower | ✅ Faster |
| Results | ✅ Better | ⚠️ Good |
| Complexity | ⚠️ Higher | ✅ Simple |

## 📚 Documentation

- **MAIN_THESIS_GUIDE.md** - Complete guide
- **EVOLUTIONARY_FEATURES.md** - Feature details
- **IMPLEMENTATION_SUMMARY.md** - Technical details

## ⚡ One-Liners

```bash
# Best default configuration
python main-thesis.py --evolutionary-mode --elitism --budget 100

# Maximum exploration
python main-thesis.py --evolutionary-mode --n-parents 8 --n-offspring 32 --budget 200

# Quick test
python main-thesis.py --evolutionary-mode --budget 10 --eval-budget 1000

# Legacy mode
python main-thesis.py --budget 50
```

## 🎯 Key Concepts

- **μ (mu)**: Number of parents (survivors each generation)
- **λ (lambda)**: Number of offspring (new algorithms per generation)
- **(μ+λ)**: Elitism - select from parents + offspring
- **(μ,λ)**: Comma - select only from offspring
- **Fitness**: AUC score on BBOB benchmark (0.0 to 1.0)
- **Generation**: One cycle of offspring creation and selection

## ✅ Checklist

Before running:
- [ ] Created `.env` with API key
- [ ] Chosen mode (evolutionary or iterative)
- [ ] Set appropriate budget
- [ ] Configured population sizes (if evolutionary)
- [ ] Decided on elitism

After running:
- [ ] Check `BEST_ALGORITHM.py`
- [ ] Review `conversationlog.txt`
- [ ] Analyze AUC files
- [ ] Compare with other runs

---

**Happy Evolving! 🧬🤖**

For detailed information, see **MAIN_THESIS_GUIDE.md**

