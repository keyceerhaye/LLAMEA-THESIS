# llm.py

## Purpose
Provides abstraction layer for interacting with various Large Language Model (LLM) providers and extracts structured information (code, descriptions, configuration spaces) from LLM responses.

## Role in System
Acts as the interface between LLaMEA and multiple LLM providers (OpenAI, Gemini, Ollama, DeepSeek). Handles:
- API authentication and communication
- Prompt submission and response retrieval
- Code extraction from markdown-formatted responses
- Description and metadata extraction
- ConfigSpace extraction for HPO
- Retry logic and rate limit handling
- Unified diff patch application

## Key Classes

### `LLM` (Abstract Base Class)
Base class defining the interface all LLM implementations must follow.

**Key Methods:**
- `query(session: list) -> str` - Send messages to LLM, get response (abstract)
- `sample_solution()` - Generate/mutate a solution, extract code and metadata
- `extract_algorithm_code()` - Parse code blocks from LLM response
- `extract_algorithm_description()` - Parse algorithm description
- `extract_configspace()` - Parse ConfigSpace definition for HPO
- `set_logger()` - Attach logger for conversation tracking

**Regex Patterns:**
- `code_pattern`: Extracts code from triple-backtick blocks
- `name_pattern`: Extracts class name from Python code
- `desc_pattern`: Extracts description from comment headers
- `cs_pattern`: Extracts ConfigSpace from response

### `OpenAI_LLM`
Implementation for OpenAI models (GPT-3.5, GPT-4, GPT-4o, etc.)

**Features:**
- Uses official `openai` Python client
- Retry logic with exponential backoff for rate limits
- Respects `Retry-After` headers
- Configurable temperature
- Handles pickling for parallel processing

**Special Methods:**
- `__getstate__()` / `__setstate__()` - Custom pickle support (client not serializable)
- `__deepcopy__()` - Deep copy support for parallel workers

### `Gemini_LLM`
Implementation for Google Gemini models (Gemini 1.5/2.0 Flash, etc.)

**Features:**
- Uses `google-generativeai` library
- System instruction support
- Advanced generation config (temperature, top_p, top_k)
- Retry logic with structured delay parsing
- Handles chat history conversion

### `Ollama_LLM`
Implementation for locally-hosted Ollama models

**Features:**
- No API key required (local models)
- Concatenates session messages for context
- Handles server unavailability gracefully
- Supports any Ollama-compatible model

### `Multi_LLM`
Wrapper for randomly alternating between multiple LLM instances

**Features:**
- Diversity in LLM responses across evolution
- Can mix different model providers
- Randomly selects one LLM per query
- Unified logger across all wrapped LLMs

### `DeepSeek_LLM`
Implementation for DeepSeek chat models (extends OpenAI_LLM)

**Features:**
- Custom base URL for DeepSeek API
- OpenAI-compatible interface
- Inherits all OpenAI_LLM retry logic

### `Dummy_LLM`
Mock LLM for testing (no API calls)

**Features:**
- Returns hardcoded example algorithm
- Useful for unit tests and debugging
- Zero API cost

## Key Operations

### Solution Sampling
The `sample_solution()` method orchestrates:
1. Send prompt to LLM via `query()`
2. Extract code from response
3. Apply unified diff if in diff_mode
4. Extract class name from code
5. Extract description
6. Extract ConfigSpace if HPO enabled
7. Create `Solution` object with all metadata
8. Log conversation if logger attached

### Retry Logic
All implementations handle:
- **Rate Limits**: Exponential backoff, respect retry-after
- **Timeouts**: Retry with increasing delays
- **Connection Errors**: Retry with backoff
- **Generic Errors**: Limited retries before failure

### Diff Mode
When `diff_mode=True`:
- LLM returns unified diff patch instead of full code
- `apply_unified_diff()` patches base code
- Reduces token usage for mutations
- Preserves unchanged parts of code

## Dependencies
- `openai` - OpenAI API client
- `google.generativeai` - Gemini API client
- `ollama` - Ollama API client
- `ConfigSpace` - Hyperparameter configuration
- Internal: `Solution`, `apply_unified_diff()`, `NoCodeException`

## Data Flow
```
Prompt → query() → LLM API → Response Text
                                ↓
                        extract_algorithm_code()
                                ↓
                   [diff_mode? → apply_unified_diff()]
                                ↓
                        extract class name
                        extract description
                        [HPO? → extract_configspace()]
                                ↓
                        Create Solution object
```

## Configuration
- `api_key` - Authentication for cloud LLMs
- `model` - Model identifier (e.g., "gpt-4-turbo")
- `base_url` - Custom API endpoint (optional)
- `temperature` - Sampling temperature (OpenAI)
- `code_pattern`, `name_pattern`, etc. - Custom regex patterns

## Error Handling
- **NoCodeException**: Raised when no code block found in response
- **Rate limit errors**: Automatic retry with delays
- **API errors**: Retry up to `max_retries` times
- **Parsing errors**: Gracefully return None or empty string

## Risks and Quirks
- **Regex fragility**: Pattern matching may fail on unexpected formats
- **API changes**: Provider updates can break compatibility
- **Rate limits**: Aggressive use can exhaust quotas
- **Pickle issues**: OpenAI client not serializable (workaround implemented)
- **Context limits**: Long conversations may exceed model context
- **Diff parsing**: Unified diff requires exact line matching

## Performance Considerations
- Retry logic adds latency on failures
- Temperature affects output diversity and consistency
- Model choice impacts speed and cost
- Diff mode saves tokens but requires careful formatting

## Extension Points
To add new LLM provider:
1. Subclass `LLM`
2. Implement `query(session)` method
3. Handle provider-specific authentication
4. Implement retry logic for provider errors
5. Override regex patterns if needed


