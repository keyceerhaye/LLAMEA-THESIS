# LLAMEA-MADA Quick Start Guide

Get started with LLAMEA-MADA in 5 minutes!

## ⚡ Super Quick Start

```bash
# 1. Activate venv
.\venv\Scripts\Activate.ps1

# 2. Run example (no API key needed)
python experiments/examples/minimum_example.py

# 3. Run benchmark with your API key
python experiments/benchmarks/main-thesis.py --api-key your_key --budget 20
```

## 📂 Project Layout

```
llamea-experimentation/
├── src/llamea/              ← Core framework code
├── experiments/
│   ├── benchmarks/          ← Run experiments here
│   └── examples/            ← Start learning here
├── config/                  ← Configuration files
├── docs/                    ← Full documentation
└── README.md               ← Complete guide
```

## 🎯 Common Commands

### Run Examples
```bash
# Simplest example
python experiments/examples/minimum_example.py

# BBOB benchmark
python experiments/examples/black-box-optimization.py
```

### Run Experiments

**Quick test (20 API calls):**
```bash
python experiments/benchmarks/main-thesis.py --budget 20
```

**Full evolutionary mode:**
```bash
python experiments/benchmarks/main-thesis.py \
  --evolutionary-mode \
  --n-parents 4 \
  --n-offspring 16 \
  --budget 100 \
  --elitism
```

## 🔑 API Key Setup

**Option 1: .env file (recommended)**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add:
OPENAI_API_KEY=your_key_here
BASE_URL=https://api.openai.com/v1
```

**Option 2: Command line**
```bash
python experiments/benchmarks/main-thesis.py --api-key your_key
```

## 📊 Understanding Results

Results are saved in timestamped folders:
```
exp-12-01_235959-gemini-2.0-flash-thesis-experiment/
├── code/               # Generated algorithms
├── BEST_ALGORITHM.py   # Best solution found
├── try-*-aucs.txt     # Performance scores
└── conversationlog.txt # LLM conversations
```

## 🛠️ Modifying LLAMEA

### Add New LLM Provider

Edit `src/llamea/llm.py`:
```python
class MyLLM(BaseLLM):
    def generate(self, prompt):
        # Your implementation
        return response
```

### Create Custom Benchmark

Create `experiments/benchmarks/my_benchmark.py`:
```python
from src.llamea.llamea import LLAMEA

# Your benchmark code
```

### Modify Evolution Strategy

Edit `experiments/benchmarks/main-thesis.py`:
- Change `selection()` function
- Modify `run_evolutionary_mode()`

## 🐛 Quick Fixes

**Import errors:**
```bash
# Add src to path (Windows)
$env:PYTHONPATH += ";$(pwd)\src"
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**Venv issues:**
```bash
# Recreate venv
rm -rf venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 📖 Learn More

- **README.md** - Complete documentation
- **docs/GETTING_STARTED.md** - Detailed walkthrough
- **experiments/examples/** - Example scripts
- **CONTRIBUTING.md** - How to contribute

## 💡 Pro Tips

1. **Start small**: Use `--budget 20` for testing
2. **Use .env files**: Easier than command-line args
3. **Check examples first**: Learn from working code
4. **Read the logs**: `conversationlog.txt` shows LLM thinking
5. **Experiment freely**: Results are saved, nothing is lost

## 🎓 Next Steps

1. ✅ Run an example
2. 📖 Read `docs/GETTING_STARTED.md`
3. 🔧 Modify a benchmark
4. 🚀 Create your own experiment
5. 🤝 Contribute back!

---

**Need help? Check `docs/` or open an issue!**

