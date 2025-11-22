# main-thesis.py

## Purpose
Enhanced version of `main.py` with command-line argument support, flexible configuration, and better error handling for thesis experiments.

## Role in System
Production-ready script for running single-algorithm evolution experiments with:
- Configurable API endpoints (OpenAI, custom providers)
- Command-line interface for parameter tuning
- Support for .env files
- Detailed feedback options
- Population configuration (for future use)

## Key Features

### Command-Line Interface
Full argparse-based CLI with categories:

**API Configuration**
- `--api-key` - API key (or from env)
- `--base-url` - Custom API endpoint (or from BASE_URL env)
- `--max-tokens` - Response token limit

**Model Configuration**
- `--model` - Model selection (default: gemini-2.0-flash)
- `--experiment-name` - Custom experiment naming

**Experimental Configuration**
- `--budget` - API call budget (default: 100)
- `--eval-budget` - Evaluation budget per algorithm (default: 10000)
- `--elitism` - Enable elitism flag
- `--detailed-feedback` - Function-group performance feedback

**Population Configuration** (prepared for future evolutionary mode)
- `--n-parents` - Parent count (default: 4)
- `--n-offspring` - Offspring count (default: 16)
- `--generations` - Fixed generation count
- `--evolutionary-mode` - Enable true evolutionary mode (planned)

### Environment Variable Support
Loads from `.env` file using `python-dotenv`:
- `OPENAI_API_KEY` or `AIML_KEY`
- `BASE_URL`

### Debugging Support
- Prints environment variable status
- Shows effective configuration before running
- Detailed error messages with traceback

### Enhanced Error Handling
- Validates API key from multiple sources
- Catches `NoCodeException` separately
- Prints algorithm excerpt on error
- Full traceback for debugging

## Script Flow
1. Parse command-line arguments
2. Load environment variables from .env
3. Validate and select API key (arg > OPENAI_API_KEY > AIML_KEY)
4. Validate and select base URL
5. Create `AlgorithmManager` with all settings
6. Calculate evolutionary potential (if generations specified)
7. Run main optimization loop (same as main.py)
8. Log all results

## Configuration Examples

### Using OpenAI
```bash
python main-thesis.py --api-key sk-... --model gpt-4-turbo --budget 50
```

### Using Custom Endpoint (e.g., AI/ML API)
```bash
python main-thesis.py --base-url https://api.aimlapi.com/v1 --budget 100
```

### Using .env File
```bash
# Create .env file with:
# OPENAI_API_KEY=sk-...
# BASE_URL=https://api.aimlapi.com/v1

python main-thesis.py --model gemini-2.0-flash --elitism
```

### Detailed Feedback Mode
```bash
python main-thesis.py --detailed-feedback --eval-budget 5000
```

## Key Differences from main.py

### Improvements
- ✅ CLI arguments for all parameters
- ✅ Environment variable support
- ✅ Custom API endpoints
- ✅ Token limit control
- ✅ Detailed feedback option
- ✅ Better error messages
- ✅ Evolutionary mode preparation

### Similarities
- Same evaluation pipeline (BBOB functions)
- Same single-algorithm evolution (no populations yet)
- Same logging structure
- Same AUC calculation

## Dependencies
- `os` - Environment variables
- `numpy` - Numerical operations
- `ioh` - IOH experimenter
- `re` - Class name extraction
- `argparse` - CLI parsing
- `managers` - Custom manager classes
- `utils` - Utility functions
- `llamea.utils` - NoCodeException
- `dotenv` (optional) - .env file loading

## Data Flow
Same as main.py, but with configurable parameters:
```
CLI args + .env → Configuration
  ↓
AlgorithmManager(custom config)
  ↓
Evolution loop (configurable budget)
  ↓
BBOB evaluation (configurable eval_budget)
  ↓
Logging with custom experiment name
```

## Population Configuration Logic
Prepares for evolutionary mode (not yet active):
- If `--generations` set: Calculate estimated API calls
- Else: Calculate max generations from budget
- Displays info to user
- **Note**: Currently not used, evolution still (1+1)-style

## Output
- Experiment directory: `exp-{date}_{time}-{model}-{experiment_name}[-elitism]/`
- Same structure as main.py
- Console: More verbose with config summary

## Error Messages

### Missing API Key
```
ValueError: API key is required. Options:
  1. Use --api-key argument
  2. Set OPENAI_API_KEY environment variable
  3. Set AIML_KEY environment variable
  4. Create a .env file with OPENAI_API_KEY=your_key
```

### Code Extraction Error
```
Error: No code extracted from LLM response
```

### Execution Error
```
Error: {exception details}
Algorithm name: {name}
Generated algorithm excerpt: {first 200 chars}...
{full traceback}
```

## Usage Patterns

### Quick Test
```bash
python main-thesis.py --budget 5 --eval-budget 1000
```

### Production Run
```bash
python main-thesis.py \
  --model gpt-4-turbo \
  --budget 200 \
  --eval-budget 10000 \
  --elitism \
  --detailed-feedback \
  --experiment-name my_experiment
```

### Custom Provider
```bash
python main-thesis.py \
  --base-url https://custom-api.com/v1 \
  --api-key custom_key \
  --model custom-model \
  --max-tokens 4096
```

## Integration with LLaMEA Framework
This script uses `AlgorithmManager` (custom wrapper) instead of full `LLaMEA` class:
- **Advantage**: Simpler, more direct control
- **Disadvantage**: Misses LLaMEA features (populations, niching, HPO, parallelization)
- **Recommended**: Use `main-evolutionary.py` for full framework

## When to Use
- Thesis experiments requiring fine-grained control
- Testing new LLM providers
- Comparing single-algorithm vs population evolution
- Custom API endpoint integration
- Detailed feedback experiments

## Limitations
- Still single-algorithm evolution (no populations)
- Manual evolutionary loop
- No HPO support
- No niching
- No parallelization
- `--evolutionary-mode` flag present but not implemented

## Future Work
To enable true evolutionary mode:
1. Replace `AlgorithmManager` with `LLaMEA` class
2. Pass `n_parents`, `n_offspring` to `LLaMEA`
3. Remove manual loop
4. Call `llamea.run()`
5. See `main-evolutionary.py` for implementation


