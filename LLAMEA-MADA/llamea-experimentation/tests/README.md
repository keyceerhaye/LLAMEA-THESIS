# Tests

This directory contains unit tests and integration tests for LLAMEA-MADA.

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_llamea.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v
```

## Test Structure

```
tests/
├── test_llamea.py          # Core LLAMEA functionality
├── test_llm.py             # LLM provider tests
├── test_solution.py        # Solution data structure tests
├── test_utils.py           # Utility function tests
├── test_loggers.py         # Logger tests
├── test_integration.py     # End-to-end integration tests
└── fixtures/               # Test fixtures and data
```

## Writing Tests

### Example Test Structure

```python
import pytest
from src.llamea.llamea import LLAMEA

class TestLLAMEA:
    """Test suite for LLAMEA core functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.llamea = LLAMEA()
    
    def test_initialization(self):
        """Test LLAMEA initialization"""
        assert self.llamea is not None
    
    def test_run(self):
        """Test LLAMEA run method"""
        result = self.llamea.run(budget=10)
        assert result is not None
```

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    class MockLLM:
        def generate(self, prompt):
            return "mock response"
    return MockLLM()

def test_with_fixture(mock_llm):
    """Test using fixture"""
    response = mock_llm.generate("test")
    assert response == "mock response"
```

### Mocking External APIs

Always mock external API calls:

```python
from unittest.mock import Mock, patch

@patch('openai.ChatCompletion.create')
def test_llm_call(mock_create):
    """Test LLM API call with mock"""
    mock_create.return_value = Mock(
        choices=[Mock(message=Mock(content="test"))]
    )
    # Your test code
```

## Test Coverage Goals

- **Core modules**: >90% coverage
- **Utilities**: >80% coverage
- **Integration**: Key workflows covered

## CI/CD Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Release tags

## Test Categories

### Unit Tests
- Test individual functions/classes
- Fast execution
- No external dependencies
- High isolation

### Integration Tests
- Test component interactions
- May use test databases
- Slower execution
- More realistic scenarios

### End-to-End Tests
- Test complete workflows
- Use mock APIs
- Validate full system behavior

## Best Practices

1. **Naming**: Use descriptive test names
   - `test_llamea_initializes_with_valid_config`
   - `test_llm_handles_api_error_gracefully`

2. **Arrange-Act-Assert**: Structure tests clearly
   ```python
   def test_example():
       # Arrange
       setup_code()
       
       # Act
       result = function_under_test()
       
       # Assert
       assert result == expected
   ```

3. **Test Edge Cases**: Don't just test happy paths
   - Null inputs
   - Empty collections
   - Large inputs
   - Error conditions

4. **Independent Tests**: Tests should not depend on each other
   - Use fixtures for common setup
   - Clean up after tests
   - Don't share state

5. **Fast Tests**: Keep tests quick
   - Mock external services
   - Use in-memory databases
   - Minimize file I/O

## TODO: Tests to Write

- [ ] `test_llamea.py` - Core LLAMEA engine tests
- [ ] `test_llm.py` - LLM provider tests
- [ ] `test_solution.py` - Solution class tests
- [ ] `test_utils.py` - Utility function tests
- [ ] `test_loggers.py` - Experiment logger tests
- [ ] `test_managers.py` - Algorithm manager tests
- [ ] `test_integration.py` - Full workflow tests
- [ ] `test_benchmarks.py` - BBOB benchmark tests

## Contributing

When adding features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain coverage >80%
4. Add integration tests for new features

---

**Test with confidence! 🧪**

