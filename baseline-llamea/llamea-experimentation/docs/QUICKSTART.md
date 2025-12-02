# LLaMEA Experimentation - Quick Start Guide

## 🎯 Goal

Get LLAMEA running in under 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
pip install -e .
```

## Step 2: Test Without API Keys (1 minute)

```bash
# Run with dummy LLM (no API keys needed)
python experiments/examples/minimum_example.py
```

Expected output:

```
Initializing LLaMEA...
Generation 1/10
...
Best solution found!
```

## Step 3: Add API Keys (Optional, 1 minute)

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your keys:
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
```

## Step 4: Run Real Experiment (1 minute)

```bash
# With OpenAI (evolutionary mode)
python experiments/benchmarks/main-thesis.py --evolutionary-mode --budget 10 --model gpt-4.1

# Or with Gemini
python experiments/benchmarks/main-thesis.py --budget 10 --model gemini-2.0-flash
```

## 🎉 You're Ready!

Now explore:

- `experiments/examples/` - More example scripts
- `experiments/benchmarks/` - Full benchmark scripts
- `README.md` - Complete documentation

## Common Commands

```bash
# Quick test (dummy LLM)
python experiments/benchmarks/main-thesis.py --budget 5

# Full benchmark (OpenAI, μ+λ)
python experiments/benchmarks/main-thesis.py --evolutionary-mode --elitism --budget 50 --model gpt-4.1

# With hyperparameter tweaks
python experiments/benchmarks/main-thesis.py --evolutionary-mode --n-parents 6 --n-offspring 18 --budget 72

# IOH experimenter
python src/misc/iohrun.py
```

## Need Help?

Check `README.md` for detailed documentation and troubleshooting.
