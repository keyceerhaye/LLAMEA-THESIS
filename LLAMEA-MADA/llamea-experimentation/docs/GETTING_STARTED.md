# Getting Started with LLaMEA Experimentation

## ✅ Pre-flight Checklist

### 1. Environment Setup
- [ ] Python 3.11+ installed
- [ ] Virtual environment created (recommended)
- [ ] Dependencies installed: `pip install -r requirements.txt`

### 2. API Keys (Optional for testing)
- [ ] `.env` file created from `.env.example`
- [ ] OpenAI API key added (if using OpenAI)
- [ ] Gemini API key added (if using Gemini)

### 3. Test Installation
- [ ] Run: `python examples/minimum_example.py`
- [ ] Verify no import errors
- [ ] Check output shows evolutionary process

## 🎯 Your First Experiments

### Experiment 1: Dummy LLM Test (No API Keys)
**Goal:** Understand the framework without spending money

```bash
cd llamea-experimentation
python benchmarks/main-evolutionary.py --llm dummy --budget 5 --n-parents 2 --n-offspring 2
```

**What to observe:**
- Population initialization
- Generation-by-generation evolution
- Fitness improvements (or not - it's random!)
- Log files created in experiment directory

**Expected time:** 1-2 minutes

---

### Experiment 2: Simple OpenAI Test
**Goal:** See real LLM-generated algorithms

```bash
python benchmarks/main-thesis.py --llm openai --budget 10
```

**What to observe:**
- LLM generates actual Python code
- Code is executed and evaluated
- Fitness feedback influences next generation
- Check `conversationlog.jsonl` for prompts/responses

**Expected time:** 5-10 minutes
**Cost:** ~$0.10-0.50 (depending on model)

---

### Experiment 3: Full Population Evolution
**Goal:** True evolutionary algorithm with population

```bash
python benchmarks/main-evolutionary.py \
  --llm openai \
  --model gpt-4-turbo \
  --budget 50 \
  --n-parents 5 \
  --n-offspring 5 \
  --elitism
```

**What to observe:**
- Multiple algorithms evolving in parallel
- Parent selection and recombination
- Elitism keeping best solutions
- Population diversity

**Expected time:** 20-30 minutes
**Cost:** ~$2-5

---

### Experiment 4: With Hyperparameter Optimization
**Goal:** Optimize both algorithm structure and hyperparameters

```bash
python benchmarks/main-evolutionary.py \
  --llm openai \
  --budget 50 \
  --n-parents 3 \
  --n-offspring 3 \
  --hpo
```

**What to observe:**
- ConfigSpace definitions in generated code
- Hyperparameter tuning per algorithm
- Better performance due to tuned parameters

**Expected time:** 30-45 minutes
**Cost:** ~$3-7

---

### Experiment 5: IOH Comprehensive Benchmark
**Goal:** Evaluate on full BBOB suite

```bash
# Edit misc/iohrun.py to specify your algorithms
python misc/iohrun.py
```

**What to observe:**
- Evaluation across 24 BBOB functions
- Multiple dimensions (5, 10, 20)
- IOH data files for analysis
- Use IOHanalyzer for visualization

**Expected time:** Hours (depending on algorithms)

## 📊 Understanding the Output

### Experiment Directory Structure
After running an experiment, you'll find:

```
experiments/TIMESTAMP_experimentname/
├── conversationlog.jsonl      # All LLM interactions
├── code/                       # Generated algorithm code
│   ├── generation_0/
│   ├── generation_1/
│   └── ...
├── individuals.jsonl           # Population history
├── aucs.csv                    # Fitness over time
└── config_spaces/              # ConfigSpace definitions (if HPO)
```

### Key Files to Check

1. **conversationlog.jsonl** - See what prompts were sent and responses received
   ```bash
   # View last 5 conversations
   tail -n 5 experiments/*/conversationlog.jsonl | jq
   ```

2. **aucs.csv** - Track fitness improvements
   ```python
   import pandas as pd
   df = pd.read_csv('experiments/*/aucs.csv')
   df.plot(x='generation', y='best_fitness')
   ```

3. **code/** - Inspect generated algorithms
   ```bash
   # View best algorithm from generation 10
   cat experiments/*/code/generation_10/best_*.py
   ```

## 🔧 Common Enhancements

### Enhancement 1: Custom Evaluation Function
**File:** `benchmarks/main-evolutionary.py`

```python
def evaluate(solution):
    # Your custom evaluation logic
    algorithm = solution.algorithm
    
    # Example: Test on your own problem
    result = algorithm.optimize(your_problem)
    fitness = calculate_fitness(result)
    
    return fitness
```

### Enhancement 2: New LLM Provider
**File:** `llamea/llm.py`

```python
class Custom_LLM(LLM):
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model
    
    def query(self, prompt):
        # Your API call here
        response = your_api_call(prompt)
        return response
```

### Enhancement 3: Custom Mutation Operator
**File:** `llamea/llamea.py`

Look for the `_construct_mutation_prompt()` method and modify:

```python
def _construct_mutation_prompt(self, parent):
    # Your custom mutation strategy
    prompt = f"""
    Improve this algorithm:
    {parent.code}
    
    Previous fitness: {parent.fitness}
    
    Your custom instructions here...
    """
    return prompt
```

### Enhancement 4: Different Selection Strategy
**File:** `llamea/llamea.py`

Modify the `_select_survivors()` method:

```python
def _select_survivors(self, population):
    # Your custom selection logic
    # E.g., tournament selection, roulette wheel, etc.
    selected = your_selection_method(population)
    return selected
```

## 🐛 Troubleshooting

### Issue: Import Errors
```bash
# Solution 1: Install in development mode
pip install -e .

# Solution 2: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
$env:PYTHONPATH += ";$(pwd)"              # Windows PowerShell
```

### Issue: API Rate Limits
```python
# In llamea/llm.py, adjust retry logic:
max_retries = 5
retry_delay = 60  # seconds
```

### Issue: Budget Exceeded
```python
# In benchmarks/utils.py, check budget_logger
# Reduce evaluation budget or increase algorithm budget
```

### Issue: Poor Algorithm Quality
- Increase budget (more generations)
- Increase population size (more diversity)
- Enable HPO (better hyperparameters)
- Improve evaluation function (better feedback)
- Adjust mutation prompts (better guidance)

## 📈 Measuring Success

### Metrics to Track

1. **Best Fitness Over Time**
   - Should generally improve
   - Plateaus indicate convergence or local optima

2. **Population Diversity**
   - Use code distance metrics
   - Too low = premature convergence
   - Too high = no exploitation

3. **LLM Token Usage**
   - Track in conversationlog.jsonl
   - Optimize prompts to reduce costs

4. **Evaluation Efficiency**
   - Time per generation
   - Parallelization effectiveness

### Success Criteria

✅ **Minimum Success:**
- Algorithm runs without errors
- Generates valid Python code
- Fitness improves over baseline

✅ **Good Success:**
- Consistent fitness improvement
- Diverse population maintained
- Reasonable token usage

✅ **Excellent Success:**
- Discovers novel algorithms
- Outperforms hand-designed baselines
- Publishable results!

## 🎓 Learning Path

### Week 1: Understanding
- [ ] Read README.md and STRUCTURE.md
- [ ] Run all examples
- [ ] Inspect generated code
- [ ] Understand evaluation pipeline

### Week 2: Experimentation
- [ ] Run benchmarks with different LLMs
- [ ] Try different population sizes
- [ ] Enable/disable HPO
- [ ] Compare results

### Week 3: Enhancement
- [ ] Modify evaluation function
- [ ] Adjust mutation prompts
- [ ] Implement custom selection
- [ ] Add new features

### Week 4: Research
- [ ] Design novel experiments
- [ ] Analyze results statistically
- [ ] Compare with state-of-the-art
- [ ] Write up findings

## 🚀 Next Steps

1. **Complete the checklist above**
2. **Run Experiment 1 (Dummy LLM)**
3. **Read through generated code**
4. **Run Experiment 2 (OpenAI)**
5. **Start planning your enhancements**

## 💡 Tips for Success

1. **Start Small:** Use dummy LLM and small budgets first
2. **Keep It Simple:** Follow KISS principle in modifications
3. **Log Everything:** You'll thank yourself later
4. **Version Control:** Git commit after each successful experiment
5. **Document Changes:** Update this file with your learnings
6. **Ask Questions:** Check issues on GitHub or documentation

---

**You're ready to start experimenting! Good luck! 🎉**

For detailed documentation, see:
- `README.md` - Complete guide
- `QUICKSTART.md` - 5-minute start
- `STRUCTURE.md` - File organization
- Original `docs/` folder - Deep dive into framework






