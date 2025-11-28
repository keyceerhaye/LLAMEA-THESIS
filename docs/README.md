# LLaMEA Documentation Index

## Overview
Comprehensive documentation for the LLaMEA (Large Language Model Evolutionary Algorithm) thesis codebase, optimized for LLM ingestion and human understanding.

## Quick Start for AI Assistants

**Start here:** [`ai_context_summary.md`](ai_context_summary.md)

High-density overview optimized for LLM token efficiency. Read this FIRST before editing code.

## Documentation Structure

### 1. High-Level Documents

#### [`ai_context_summary.md`](ai_context_summary.md) ⭐ **START HERE**
- Token-optimized summary for AI assistants
- Critical facts about architecture
- Common editing scenarios
- Quick reference tables
- Emergency fixes
- **Reading time:** 2-3 minutes | **Tokens:** ~2K

#### [`architecture.md`](architecture.md)
- System architecture overview
- Component relationships
- Data flow diagrams
- Design patterns
- Scalability considerations
- Integration points
- **Reading time:** 15-20 minutes | **Tokens:** ~5K

#### [`rules.md`](rules.md)
- Coding style guidelines
- Naming conventions
- Module organization rules
- Design patterns (allowed and forbidden)
- Error handling strategies
- Testing conventions
- **Reading time:** 15-20 minutes | **Tokens:** ~5K

#### [`bugs.md`](bugs.md)
- Known bugs and issues
- Technical debt tracking
- Risky areas of code
- Unimplemented features
- Platform-specific issues
- Workarounds and priorities
- **Reading time:** 10-15 minutes | **Tokens:** ~4K

### 2. Codebase Documentation

#### Directory: [`codebase/`](codebase/)

Detailed file-by-file documentation of the entire codebase.

##### [`codebase/llamea/`](codebase/llamea/) - Core Framework
- [`llamea.py.md`](codebase/llamea/llamea.py.md) - Main evolutionary engine
- [`llm.py.md`](codebase/llamea/llm.py.md) - LLM provider abstraction
- [`solution.py.md`](codebase/llamea/solution.py.md) - Solution data structure
- [`utils.py.md`](codebase/llamea/utils.py.md) - Utility functions
- [`loggers.py.md`](codebase/llamea/loggers.py.md) - Experiment logging
- [`README.md`](codebase/llamea/README.md) - Package overview

##### [`codebase/root/`](codebase/root/) - Execution Scripts
- [`main.py.md`](codebase/root/main.py.md) - Legacy simple script
- [`main-thesis.py.md`](codebase/root/main-thesis.py.md) - Enhanced thesis script
- [`main-evolutionary.py.md`](codebase/root/main-evolutionary.py.md) - Production framework script
- [`managers.py.md`](codebase/root/managers.py.md) - Custom manager classes
- [`utils.py.md`](codebase/root/utils.py.md) - BBOB utilities
- [`README.md`](codebase/root/README.md) - Root files overview

##### [`codebase/examples/`](codebase/examples/) - Usage Examples
- [`README.md`](codebase/examples/README.md) - Examples overview, usage patterns, feature matrix

##### [`codebase/misc/`](codebase/misc/) - Analysis Tools
- [`README.md`](codebase/misc/README.md) - Post-processing utilities

##### [`codebase/logreader/`](codebase/logreader/) - Web Application
- [`README.md`](codebase/logreader/README.md) - Log viewer application

### 3. Project Documentation

#### [`../CHANGELOG.md`](../CHANGELOG.md)
- Version history
- Feature evolution
- Breaking changes
- Migration guides
- **Reading time:** 5-10 minutes

#### [`../README.md`](../README.md)
- Project overview
- Installation instructions
- Quick start guide
- Examples
- Contributing guidelines
- **Reading time:** 10 minutes

#### [`../CONTRIBUTING.md`](../CONTRIBUTING.md)
- Contribution guidelines
- Development setup
- Testing requirements
- Pull request process
- **Reading time:** 5 minutes

## Documentation Usage Guide

### For AI Coding Assistants

**Workflow:**
1. **Always start with** [`ai_context_summary.md`](ai_context_summary.md)
2. **For architecture questions**, read [`architecture.md`](architecture.md)
3. **For code style**, read [`rules.md`](rules.md)
4. **For specific files**, read `codebase/{directory}/{file}.md`
5. **For known issues**, check [`bugs.md`](bugs.md)

**Token Budget Strategy:**
- **Tight budget (<10K tokens)**: Read only `ai_context_summary.md`
- **Medium budget (10-30K tokens)**: Add `architecture.md` + relevant file docs
- **Large budget (>30K tokens)**: Read comprehensively

### For Human Developers

**First-Time Contributors:**
1. Read [`../README.md`](../README.md) - Project overview
2. Read [`ai_context_summary.md`](ai_context_summary.md) - Quick understanding
3. Read [`architecture.md`](architecture.md) - Deep dive
4. Read [`rules.md`](rules.md) - Coding standards
5. Read [`../CONTRIBUTING.md`](../CONTRIBUTING.md) - How to contribute

**Experienced Contributors:**
- Refer to `codebase/` for specific file details
- Check [`bugs.md`](bugs.md) before fixing issues
- Update [`../CHANGELOG.md`](../CHANGELOG.md) with changes

**Researchers:**
- Focus on [`architecture.md`](architecture.md) - System design
- Read `codebase/llamea/llamea.py.md` - Core algorithm
- Check [`../README.md`](../README.md) - Citations and papers

### For Specific Tasks

#### Adding New LLM Provider
1. Read `codebase/llamea/llm.py.md`
2. Check `rules.md` for design patterns
3. See examples in `llm.py` itself

#### Modifying Evaluation
1. Read `codebase/root/utils.py.md` (BBOB-specific)
2. Read `codebase/llamea/llamea.py.md` (evaluation flow)
3. Check `architecture.md` for evaluation interface

#### Creating New Example
1. Read `codebase/examples/README.md`
2. Follow patterns from existing examples
3. Check `rules.md` for file organization

#### Debugging Issues
1. Check `bugs.md` for known issues
2. Read relevant file documentation in `codebase/`
3. Use `logreader/` app for log inspection

#### Understanding Data Flow
1. Read `architecture.md` - data flow diagrams
2. Read `ai_context_summary.md` - quick reference
3. Trace through `codebase/llamea/llamea.py.md`

## Documentation Quality Metrics

### Coverage
- ✅ All core framework files documented
- ✅ All execution scripts documented
- ✅ All major directories documented
- ✅ Architecture comprehensively documented
- ✅ Coding conventions documented
- ✅ Known issues documented
- ✅ AI-optimized summary provided

### Information Density
- **Total documentation:** ~50K tokens
- **AI summary:** ~2K tokens (4% of total)
- **Critical path:** ~10K tokens (20% of total)
- **Comprehensive:** Full set for deep understanding

### Maintainability
- **Format:** Markdown (readable, version-controllable)
- **Structure:** Hierarchical, easy navigation
- **Updates:** Living documentation, update with code changes

## How to Update Documentation

### Adding New Documentation
1. Create `.md` file in appropriate `codebase/` subdirectory
2. Follow structure of existing documentation
3. Update this index (`docs/README.md`)
4. Link from relevant documents

### Modifying Existing Documentation
1. Update corresponding `.md` file
2. Keep consistent with code changes
3. Update [`../CHANGELOG.md`](../CHANGELOG.md) if significant
4. Increment version if needed

### Documentation Standards
- Use Markdown format
- Include code examples where relevant
- Link between related documents
- Maintain consistent structure:
  - Purpose
  - Role in System
  - Key Components
  - Data Flow
  - Dependencies
  - Usage Patterns
  - Risks/Quirks
  - Integration Points

## Documentation Versioning

This documentation corresponds to:
- **LLaMEA Version:** 1.1.2
- **Documentation Generated:** 2025-11-20
- **Last Updated:** 2025-11-20

## Feedback and Contributions

To improve documentation:
1. Open GitHub issue with label `documentation`
2. Describe what's unclear or missing
3. Suggest improvements
4. Submit pull request with changes

## FAQ

**Q: Where do I start as a new AI assistant?**
A: Read [`ai_context_summary.md`](ai_context_summary.md) first, then specific file docs as needed.

**Q: I need to understand the overall system design?**
A: Read [`architecture.md`](architecture.md) for comprehensive overview.

**Q: What are the coding standards?**
A: See [`rules.md`](rules.md) for all conventions.

**Q: Is there a known bug I'm encountering?**
A: Check [`bugs.md`](bugs.md) for known issues and workarounds.

**Q: How do I find documentation for a specific file?**
A: Navigate to `codebase/{directory}/{filename}.md`

**Q: What's the difference between the three main scripts?**
A: See `codebase/root/README.md` for comparison matrix.

**Q: How do I run the examples?**
A: See `codebase/examples/README.md` for instructions.

**Q: Where are the unit tests documented?**
A: Tests follow standard pytest conventions, see `tests/test_*.py` files.

**Q: How do I visualize experiment logs?**
A: See `codebase/logreader/README.md` for web app usage.

**Q: What's the KISS principle mentioned everywhere?**
A: "Keep It Simple, Stupid" - favor simplicity over cleverness in all design decisions.

## Documentation Statistics

```
Total Files Documented: 20+
Total Documentation Size: ~50K tokens
Markdown Files: 22
Code Examples: 100+
Diagrams: 5+
Cross-References: 150+
```

## License

This documentation is part of the LLaMEA project and follows the same MIT License.

---

**Last Updated:** 2025-11-20
**Maintainer:** [Your Name/Team]
**Status:** ✅ Complete and comprehensive





