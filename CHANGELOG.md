# Changelog

All notable changes to the LLaMEA thesis project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive codebase documentation for LLM ingestion
- Architecture design document
- Coding rules and conventions document
- Known bugs and technical debt tracking
- AI context summary for efficient LLM understanding

## [1.1.2] - 2025 (from pyproject.toml)

Current release of core LLaMEA framework.

### Framework Features
- Population-based evolutionary algorithm
- Multiple LLM provider support (OpenAI, Gemini, Ollama, DeepSeek)
- Niching strategies (fitness sharing, clearing)
- Hyperparameter optimization (HPO) integration with SMAC
- Diff mode for efficient code mutations
- Adaptive mutation and prompt strategies
- Parallel evaluation with Joblib
- Population evaluation mode for batch processing
- Comprehensive experiment logging

## Thesis Project Additions

### Added - Thesis Scripts
- `main.py` - Original simple evolutionary script
- `main-thesis.py` - Enhanced script with CLI arguments and configuration
- `main-evolutionary.py` - Full framework implementation with populations
- `managers.py` - Custom manager classes for legacy scripts
- `utils.py` - BBOB-specific utilities (AOC logging, budget enforcement)
- `benchmark_config.yaml` - Configuration templates

### Added - Examples
- BBOB benchmark examples (with and without HPO)
- AutoML example on breast cancer dataset
- Minimum example with Dummy LLM

### Added - Analysis Tools
- `logreader/` - Web app for browsing conversation logs
- `misc/` - Utilities for visualization and analysis

### Modified - Framework Integration
- Adapted core LLaMEA for BBOB benchmarking
- Added population evaluation wrappers
- Extended logging for thesis experiments

## Historical Releases (from README)

### [1.0.0] - 2025-03

**LLaMEA v1.0.0 released**

Major stable release of the framework.

### Earlier Milestones

- **2025-07**: Won Silver award at Humies @GECCO2025
- **2025-06**: LLaMEA-BO paper published on arXiv
- **2025-05**: BLADE paper accepted at GECCO 2025
- **2025-05**: Photonic structures paper accepted at GECCO 2025
- **2025-04**: LLaMEA-HPO paper accepted in ACM TELO
- **2025-04**: Code Evolution Graphs paper accepted at GECCO 2025
- **2025-01**: LLaMEA paper accepted in IEEE TEVC
- **2024-11**: Initial release

## Version History Summary

### Major Versions

#### v1.x Series (Current)
- Stable framework with all core features
- HPO integration
- Multiple LLM providers
- Niching strategies
- Diff mode

#### v0.x Series (Pre-release)
- Initial development
- Basic evolutionary algorithm
- OpenAI integration
- Simple logging

## Feature Evolution

### Core Algorithm
- v0.1: Basic (1+1) evolution
- v0.5: Population-based evolution
- v1.0: Niching, adaptive strategies
- v1.1: Diff mode, HPO integration

### LLM Support
- v0.1: OpenAI only
- v0.5: Gemini added
- v1.0: Ollama added
- v1.1: Multi_LLM, DeepSeek added

### Evaluation
- v0.1: Sequential evaluation
- v0.5: Parallel evaluation
- v1.0: Population evaluation mode
- v1.1: Improved timeout handling

### Logging
- v0.1: Basic text logs
- v0.5: JSONL format
- v1.0: Structured logging
- v1.1: Comprehensive metadata

## Breaking Changes

### v1.0.0
- Changed from plain text to JSONL logging format
- Renamed some parameters for consistency
- Modified Solution API (added metadata dict)

### v0.5.0
- Changed evaluation function signature
- Reorganized package structure
- Updated dependencies

## Dependencies Evolution

### Python Version
- v0.1: Python 3.8+
- v1.0: Python 3.10+
- v1.1: Python 3.11+

### Major Dependencies
- numpy: 1.26.3+
- openai: 1.99.1+
- ioh: 0.3.18+
- joblib: 1.4.2+
- ConfigSpace: 1.2.0+

## Migration Guides

### From v0.x to v1.0
1. Update Python to 3.10+
2. Modify evaluation function signature
3. Update logging to JSONL format
4. Rename old parameters (check deprecation warnings)

### From Custom Scripts to Framework
1. Replace AlgorithmManager with LLaMEA
2. Adapt evaluation function signature
3. Use framework logging instead of custom
4. Configure LLM via LLM classes instead of direct API

## Thesis Project Specific Changes

### Configuration Enhancements
- Added .env file support
- CLI arguments for all parameters
- Custom API endpoint support
- Flexible model selection

### BBOB Integration
- Custom AOC logger for BBOB
- Budget enforcement logger
- Early-stop correction
- Detailed function group feedback

### Execution Scripts
- Three-tier script complexity (simple → enhanced → framework)
- Population-based evolution support
- Configurable generation calculations
- Max tokens support for custom providers

## Known Issues by Version

### v1.1.2 (Current)
- Windows: No signal-based timeout support
- ConfigSpace serialization errors silent
- Model names with special characters break paths
- Context window can exceed LLM limits in long runs

See `docs/bugs.md` for complete list.

## Deprecations

### Deprecated Features
- None in current version

### Planned Deprecations
- Plain text logging in managers.py (use JSONL)
- Global exec() pattern (move to isolated namespaces)

## Security Updates

### v1.1.2
- No critical security issues

### General Security Notes
- exec() usage is inherent design choice (trust-based)
- API keys should be environment variables
- Logs may contain sensitive information

## Performance Improvements

### v1.1
- Improved parallel evaluation efficiency
- Reduced token usage with diff mode
- Better memory management

### v1.0
- Added parallel evaluation (major speedup)
- Optimized logging I/O
- Reduced API call overhead

## Contributors

### Core Framework
- Niki van Stein (primary author)
- Thomas Bäck
- Diederick Vermetten (LLaMEA-HPO)
- Additional contributors (see GitHub)

### Thesis Project
- [Your contributions here]

## Reproducibility

### Zenodo Repositories
- v1.1: https://doi.org/10.5281/zenodo.14917719 (LLaMEA-HPO)
- v1.0: https://doi.org/10.5281/zenodo.13842144 (LLaMEA)
- Additional: See README.md for full list

## Related Publications

### 2025
- IEEE TEVC: LLaMEA main paper
- ACM TELO: LLaMEA-HPO paper
- GECCO 2025: Multiple papers (BLADE, Code Evolution Graphs, etc.)

### 2024
- EvoApps 2025: Controlling mutation paper

## Future Roadmap

### Planned for Next Versions
- Checkpoint/resume functionality
- Multi-objective optimization support
- Distributed evaluation
- Transfer learning capabilities
- Constraint handling
- Better progress tracking

### Research Directions
- Coevolutionary algorithms
- Neural architecture search applications
- Scientific discovery applications
- Hybrid methods with other optimizers

## Acknowledgments

- OpenAI for GPT models
- Google for Gemini models
- IOH Profiler team
- SMAC development team
- All contributors and users

---

## How to Update This Changelog

When making changes:

1. Add entry under [Unreleased]
2. Categorize: Added/Changed/Deprecated/Removed/Fixed/Security
3. Be specific and link issues/PRs
4. On release: Move [Unreleased] to new version section
5. Update version numbers and dates

## Changelog Standards

Follow these guidelines:
- Write for humans, not machines
- One entry per significant change
- Group similar changes
- Link to issues/PRs when relevant
- Include breaking changes prominently
- Note deprecations early


