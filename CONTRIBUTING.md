<!-- SPDX-License-Identifier: MIT
Copyright (c) 2024 Perday Labs -->

# Contributing to Fuzzy Entity Matcher

Thank you for your interest in contributing to Fuzzy Entity Matcher! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/fuzzy-entity-matcher.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e ".[dev,benchmark]"`
6. Create a feature branch: `git checkout -b feature/your-feature-name`

## 🧪 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/perday/fuzzy-entity-matcher.git
cd fuzzy-entity-matcher

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

## 🔧 Development Workflow

### Code Style

We use several tools to maintain code quality:

- **ruff**: Fast Python linter and formatter
- **black**: Code formatting
- **mypy**: Static type checking
- **pytest**: Testing framework

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fuzzy_entity_matcher

# Run only fast tests (skip slow integration tests)
pytest -m "not slow"

# Run specific test file
pytest tests/test_core.py -v

# Run benchmarks
pytest tests/test_benchmarks.py -v
```

### Code Quality Checks

```bash
# Lint code
ruff check src tests

# Format code
ruff format src tests

# Type checking
mypy src

# Security scanning
bandit -r src

# All checks at once
make check  # If Makefile is available
```

### Running Benchmarks

```bash
# Run comprehensive benchmarks
python -m fuzzy_entity_matcher.benchmarks

# Run specific benchmark
python -c "
from fuzzy_entity_matcher import FuzzyMatcherBenchmark
benchmark = FuzzyMatcherBenchmark()
results = benchmark.run_comprehensive_benchmark(sample_size=1000)
benchmark.print_summary()
"
```

## 📝 Contribution Guidelines

### Code Standards

1. **Type Hints**: All functions must have complete type hints
2. **Docstrings**: All public functions need comprehensive docstrings
3. **Tests**: New features require corresponding tests
4. **Performance**: Consider performance implications of changes
5. **Backwards Compatibility**: Maintain API compatibility when possible

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom similarity algorithms
fix: handle empty candidate lists gracefully
docs: update README with new algorithm examples
test: add edge case tests for Unicode handling
perf: optimize cache lookup performance
```

### Pull Request Process

1. **Create Feature Branch**: `git checkout -b feature/your-feature-name`
2. **Make Changes**: Implement your feature with tests
3. **Run Tests**: Ensure all tests pass
4. **Update Documentation**: Update README, docstrings, etc.
5. **Submit PR**: Create a pull request with clear description

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (explain why acceptable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions in isolation
2. **Integration Tests**: Test interaction between components
3. **Performance Tests**: Verify performance characteristics
4. **Edge Case Tests**: Test boundary conditions and error cases

### Writing Tests

```python
import pytest
from fuzzy_entity_matcher import find_matches, FuzzyMatcherError

class TestYourFeature:
    def setup_method(self):
        """Setup before each test method."""
        # Clear caches, reset state, etc.
        pass

    def test_basic_functionality(self):
        """Test basic functionality with clear assertions."""
        result = find_matches("query", ["candidate1", "candidate2"])
        assert isinstance(result, list)
        assert len(result) >= 0

    def test_error_conditions(self):
        """Test error handling."""
        with pytest.raises(FuzzyMatcherError, match="specific error message"):
            find_matches("", [])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with empty inputs, very long strings, Unicode, etc.
        pass
```

### Test Data

- Use realistic music industry data in tests
- Avoid hardcoded test data when possible
- Create helper functions for generating test data
- Mock external dependencies (databases, APIs)

## 📊 Performance Considerations

### Benchmarking

When making performance-related changes:

1. **Baseline**: Measure performance before changes
2. **Implementation**: Make your changes
3. **Measurement**: Measure performance after changes
4. **Documentation**: Document performance impact

```python
from fuzzy_entity_matcher import FuzzyMatcherBenchmark

# Before changes
benchmark = FuzzyMatcherBenchmark()
before_results = benchmark.benchmark_matching(queries, candidates)

# After changes
after_results = benchmark.benchmark_matching(queries, candidates)

# Compare results
improvement = (after_results.matches_per_second / before_results.matches_per_second - 1) * 100
print(f"Performance improvement: {improvement:.1f}%")
```

### Performance Guidelines

- **Cache Wisely**: Use caching for expensive operations
- **Avoid Premature Optimization**: Profile before optimizing
- **Memory Efficiency**: Consider memory usage with large datasets
- **Algorithm Selection**: Choose appropriate algorithms for use cases

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Verify with latest version
3. Create minimal reproduction case

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Reproduction Steps
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Python version:
- Package version:
- Operating system:
- Dependencies:

## Minimal Code Example
```python
# Minimal code that reproduces the issue
```

## Additional Context
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed API
```python
# Example of how the feature would be used
```

## Implementation Ideas
Any thoughts on how this could be implemented

## Alternatives Considered
Other approaches you've considered
```

## 📚 Documentation

### Documentation Standards

1. **README**: Keep README up-to-date with examples
2. **Docstrings**: Use Google-style docstrings
3. **Type Hints**: Comprehensive type annotations
4. **Examples**: Include practical examples
5. **Performance Notes**: Document performance characteristics

### Docstring Format

```python
def find_matches(
    query: str,
    candidates: List[str],
    threshold: float = 0.8,
    include_scores: bool = False
) -> Union[List[str], List[Tuple[str, float]]]:
    """
    Find fuzzy matches for a query string against candidate strings.

    Uses multiple similarity algorithms with intelligent weighting for
    optimal results on music industry entity names.

    Args:
        query: String to find matches for
        candidates: List of candidate strings to match against
        threshold: Minimum similarity score (0.0-1.0, default 0.8)
        include_scores: If True, return (match, score) tuples

    Returns:
        List of matching strings or (string, score) tuples if include_scores=True
        Returns empty list if no matches above threshold

    Raises:
        FuzzyMatcherError: If query is empty or candidates is empty

    Examples:
        >>> find_matches("Sony Music", ["Sony Music Entertainment", "Warner Music"])
        ["Sony Music Entertainment"]

        >>> find_matches("Sony Music", candidates, include_scores=True)
        [("Sony Music Entertainment", 0.95)]
    """
```

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Code Review**: All contributions are reviewed

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Maintain professional communication

## 🏆 Recognition

Contributors are recognized in:

- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **GitHub**: Contributor statistics and recognition

## 📋 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release tag
5. GitHub Actions handles PyPI publication

## 🔗 Resources

- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)
- [ruff Documentation](https://docs.astral.sh/ruff/)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

Thank you for contributing to Fuzzy Entity Matcher! 🎵
