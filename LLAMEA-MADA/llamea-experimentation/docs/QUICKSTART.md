# LLaMEA Experimentation - Quick Start Guide

## 🎯 Goal
Get LLAMEA running in under 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

## Step 2: Test Without API Keys (1 minute)

```bash
# Run with dummy LLM (no API keys needed)
python examples/minimum_example.py
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
# With OpenAI
python benchmarks/main-thesis.py --llm openai --budget 10

# Or with Gemini
python benchmarks/main-thesis.py --llm gemini --budget 10
```

## 🎉 You're Ready!

Now explore:
- `examples/` - More example scripts
- `benchmarks/` - Full benchmark scripts
- `README.md` - Complete documentation

## Common Commands

```bash
# Quick test (dummy LLM)
python benchmarks/main-evolutionary.py --llm dummy --budget 5

# Full benchmark (OpenAI)
python benchmarks/main-evolutionary.py --llm openai --budget 50 --n-parents 5

# With hyperparameter optimization
python benchmarks/main-evolutionary.py --llm openai --budget 50 --hpo

# IOH experimenter
python misc/iohrun.py
```

## Need Help?

Check `README.md` for detailed documentation and troubleshooting.






