# Contributing to LLAMEA-MADA

Thank you for your interest in contributing to LLAMEA-MADA! This guide will help you get started.

## 🎯 Development Principles

We follow the **KISS principle** (Keep It Simple, Stupid):
- Write clear, readable code
- Prefer simplicity over cleverness
- Document your changes
- Test thoroughly

## 🚀 Getting Started

### 1. Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourname/llamea-mada.git
cd llamea-mada/llamea-experimentation

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate     # Linux/Mac

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 2. Project Structure

```
llamea-experimentation/
├── src/llamea/          # Core framework - main development area
├── experiments/         # Experiments and benchmarks
├── config/             # Configuration files
├── docs/               # Documentation
├── tests/              # Unit tests
└── assets/             # Static files
```

## 📝 Contribution Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines based on what you're modifying:

#### Core Framework (`src/llamea/`)
- **llamea.py**: Main evolutionary algorithm logic
- **llm.py**: LLM provider integrations
- **solution.py**: Data structures
- **utils.py**: Utility functions
- **loggers.py**: Experiment logging

**Guidelines:**
- Maintain backward compatibility
- Add docstrings for all functions
- Update type hints
- Write unit tests in `tests/`

#### Experiments (`experiments/`)
- **benchmarks/**: New benchmark scripts
- **examples/**: Example usage scripts

**Guidelines:**
- Include command-line interface
- Add documentation in script docstring
- Provide usage examples

#### Documentation (`docs/`)
- Update relevant docs for your changes
- Follow markdown best practices
- Include code examples

### 3. Code Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black src/ experiments/

# Check style with flake8
flake8 src/ experiments/ --max-line-length=100

# Type check with mypy
mypy src/
```

**Key Points:**
- Line length: 100 characters
- Use type hints
- Docstrings: Google style
- Imports: sorted alphabetically

### 4. Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_llamea.py

# Run with coverage
pytest --cov=src tests/
```

**Testing Guidelines:**
- Write tests for new features
- Maintain >80% code coverage
- Test edge cases
- Mock external API calls

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new LLM provider support"
```

**Commit Message Format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat: add Anthropic Claude LLM provider

- Implement ClaudeLLM class in llm.py
- Add Claude API configuration
- Update documentation

Closes #123
```

```
fix: correct AUC calculation in BBOB benchmark

The previous calculation didn't handle edge cases
where evaluations stopped early.
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear description of changes
- Link to related issues
- Screenshots if UI-related
- Test results

## 🔍 Code Review Process

1. **Automated Checks**: CI/CD runs tests and linters
2. **Peer Review**: At least one maintainer reviews
3. **Feedback**: Address review comments
4. **Merge**: Maintainer merges after approval

## 📋 Contribution Ideas

### Beginner-Friendly
- [ ] Add more example scripts
- [ ] Improve documentation
- [ ] Fix typos and formatting
- [ ] Add unit tests

### Intermediate
- [ ] Implement new LLM providers
- [ ] Add new benchmark problems
- [ ] Improve visualization tools
- [ ] Optimize performance

### Advanced
- [ ] Implement new evolutionary strategies
- [ ] Add multi-objective optimization
- [ ] Integrate new meta-learning approaches
- [ ] Develop AutoML capabilities

## 🐛 Reporting Bugs

**Before submitting:**
1. Check existing issues
2. Verify bug in latest version
3. Create minimal reproduction

**Bug Report Template:**
```markdown
**Description:**
Clear description of the bug

**To Reproduce:**
1. Step 1
2. Step 2
3. See error

**Expected Behavior:**
What should happen

**Actual Behavior:**
What actually happens

**Environment:**
- OS: [e.g., Windows 11]
- Python: [e.g., 3.11.9]
- LLAMEA-MADA version: [e.g., 1.0.0]

**Additional Context:**
Error logs, screenshots, etc.
```

## 💡 Feature Requests

**Feature Request Template:**
```markdown
**Problem:**
What problem does this solve?

**Proposed Solution:**
Describe your proposed feature

**Alternatives:**
Alternative solutions considered

**Additional Context:**
Use cases, examples, etc.
```

## 📖 Documentation

When adding documentation:
- Use clear, concise language
- Include code examples
- Add diagrams if helpful
- Link to related docs

## ✅ Checklist Before Submitting PR

- [ ] Code follows style guidelines
- [ ] Added/updated tests
- [ ] All tests pass
- [ ] Updated documentation
- [ ] Added docstrings
- [ ] Checked for breaking changes
- [ ] Added changelog entry
- [ ] Linked related issues

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

## 📞 Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: Search existing issues
- **Discussions**: GitHub Discussions
- **Contact**: Open an issue or discussion

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## 🎓 Learning Resources

- [Python Best Practices](https://docs.python-guide.org/)
- [Git Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows)
- [Testing in Python](https://realpython.com/python-testing/)
- [Documentation Best Practices](https://www.writethedocs.org/)

---

**Thank you for contributing to LLAMEA-MADA! 🚀**

