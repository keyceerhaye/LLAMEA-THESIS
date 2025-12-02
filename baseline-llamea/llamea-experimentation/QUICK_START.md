# Baseline LLAMEA - Quick Start

Get up and running with the reorganized Baseline LLAMEA in under 5 minutes!

## 🚀 Installation

```bash
cd llamea-experimentation

# Install dependencies
pip install -r requirements.txt

# Optional: Install as editable package
pip install -e .
```

## 🎯 Quick Test (No API Keys)

```bash
# Run minimal example with dummy LLM
python experiments/examples/minimum_example.py
```

## 🔑 Setup API Keys (Optional)

Create a `.env` file in the root:

```bash
# .env
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## 📊 Run Benchmarks

### Simple Test Run

```bash
python experiments/benchmarks/main-thesis.py --budget 5 --model gemini-2.0-flash
```

### Evolutionary Mode (Recommended)

```bash
python experiments/benchmarks/main-thesis.py \
  --evolutionary-mode \
  --elitism \
  --n-parents 4 \
  --n-offspring 16 \
  --budget 50 \
  --model gemini-2.0-flash
```

## 📁 Project Structure

```
llamea-experimentation/
├── src/                    # Core framework
│   ├── llamea/            # LLAMEA engine
│   └── misc/              # Utilities
├── experiments/           # Runnable scripts
│   ├── benchmarks/       # Main experiments
│   └── examples/         # Learning examples
├── config/               # Configuration
├── docs/                 # Documentation
└── assets/               # Images
```

## 📚 Next Steps

- **Learn more**: Read `README.md`
- **Full guide**: Check `docs/INDEX.md`
- **Examples**: Explore `experiments/examples/`
- **Customize**: Edit `experiments/benchmarks/main-thesis.py`

## 🆘 Need Help?

See `docs/GETTING_STARTED.md` for detailed troubleshooting and guides.

---

**Happy experimenting! 🎉**
