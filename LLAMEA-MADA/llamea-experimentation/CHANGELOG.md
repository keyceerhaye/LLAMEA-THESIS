# Changelog

All notable changes to LLAMEA-MADA will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-01

### Added
- Complete reorganization of project structure
- New organized directory layout:
  - `src/` for core source code
  - `experiments/` for benchmarks and examples
  - `config/` for configuration files
  - `docs/` for documentation
  - `assets/` for static files
  - `tests/` for unit tests
- Comprehensive README.md with usage examples
- CONTRIBUTING.md guide for contributors
- .gitignore for Python projects
- Enhanced setup.py with package metadata
- .env.example template for environment variables
- CHANGELOG.md (this file)

### Changed
- Moved llamea/ to src/llamea/
- Moved misc/ to src/misc/
- Moved examples/ to experiments/examples/
- Moved benchmark scripts to experiments/benchmarks/
- Moved documentation files to docs/
- Moved images to assets/
- Moved configuration files to config/
- Updated import paths to reflect new structure

### Improved
- Better separation of concerns
- Clearer project organization
- Easier navigation and discoverability
- Simplified contribution workflow
- Enhanced maintainability

## [0.9.0] - 2025-11-22

### Added
- Initial LLAMEA experimentation setup
- Core LLAMEA framework
- BBOB benchmark integration
- IOH experimenter support
- Multiple LLM provider support (OpenAI, Google, Ollama)
- Population-based evolutionary mode
- Iterative refinement mode
- Example scripts and documentation

---

## Unreleased

### Planned
- [ ] Unit tests for core components
- [ ] Integration tests for benchmarks
- [ ] Performance optimization
- [ ] Multi-objective optimization support
- [ ] Advanced visualization tools
- [ ] CI/CD pipeline
- [ ] Docker support
- [ ] Web interface for experiments

