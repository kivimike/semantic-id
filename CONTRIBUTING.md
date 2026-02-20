# Contributing to semantic-id

Thank you for your interest in contributing! This guide covers the development
setup, coding standards, and pull request workflow.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-org>/semantic-id.git
cd semantic-id

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Full test suite with coverage
pytest

# Specific test file
pytest tests/test_rq_kmeans.py -v

# Run with verbose output
pytest -v --tb=long
```

Coverage is enforced in CI. The current threshold is 70% and the goal is 90%+.

## Type Checking

```bash
mypy src/
```

All code under `src/` must pass mypy with the settings in `pyproject.toml`.

## Code Style

We use **black** for formatting, **isort** for import ordering, and **flake8**
for linting. If pre-commit hooks are installed they run automatically; otherwise
check manually:

```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503
```

## Pre-commit Hooks

Install once after cloning:

```bash
pip install pre-commit
pre-commit install
```

Hooks run automatically on `git commit`.

## Pull Request Workflow

1. Create a feature branch from `dev`.
2. Make your changes — include tests for new functionality.
3. Ensure `pytest`, `mypy src/`, and the linters all pass.
4. Open a PR against `dev` with a clear description of what changed and why.
5. CI must be green before merging.

## Adding a New Feature

- Put implementation code in `src/semantic_id/`.
- Add tests in `tests/`.
- Add docstrings to all public classes and functions.
- Update `__init__.py` exports and `__all__` if adding public API surface.
- Use `_validate_embeddings()` from `core.py` for input validation in
  encoders.
- Raise `NotFittedError` / `ShapeMismatchError` from `semantic_id.exceptions`
  instead of bare `RuntimeError` / `ValueError`.

## Reporting Issues

Open a GitHub issue with:

- A minimal reproducing example.
- Python version and OS.
- The full traceback (if applicable).
