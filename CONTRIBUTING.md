# Contributing to Spellcrafting

Thank you for your interest in contributing to Spellcrafting! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sammcvicker/spellcrafting.git
   cd spellcrafting
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Verify the installation:
   ```bash
   uv run python -c "from spellcrafting import spell, guard, Config; print('OK')"
   ```

## Running Tests

```bash
# Run all tests (stop on first failure)
uv run pytest tests/ -x -q

# Run all tests
uv run pytest tests/ -q

# Skip smoke tests (tests that call real LLM APIs)
uv run pytest tests/ -m "not smoke"
```

**Note:** Smoke tests require API keys for the LLM providers. Set the appropriate environment variables:
```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## Code Style

- Follow existing code patterns in the codebase
- Use type hints for all public functions and classes
- Add docstrings to all public APIs
- Keep imports organized (standard library, third-party, local)

### Mutable Default Arguments

Never use mutable defaults directly. Use factory patterns:

**Pydantic models:**
```python
from pydantic import Field

class MyConfig(BaseModel):
    items: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
```

**Dataclasses:**
```python
from dataclasses import dataclass, field

@dataclass
class MyData:
    items: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
```

**Function arguments:**
```python
def my_func(items: list[str] | None = None) -> None:
    items = items or []
```

For truly optional values that shouldn't be passed downstream if unset, use `None`. For collections that should always exist (even if empty), use `default_factory`.

## Testing Requirements

- All new features must include tests
- All bug fixes must include a regression test
- Tests should be fast and deterministic (mock LLM calls when possible)
- Smoke tests (real API calls) should be marked with `@pytest.mark.smoke`

## Submitting Changes

### Issues

Before starting work on a feature or bug fix:
1. Check existing issues to avoid duplicates
2. For new features, open an issue to discuss the design
3. For bugs, include steps to reproduce

### Pull Requests

1. Create a feature branch from `main`:
   ```bash
   git checkout -b issue-{number}-{description}
   ```

2. Make your changes with clear, focused commits

3. Run tests before submitting:
   ```bash
   uv run pytest tests/ -q
   ```

4. Push and open a pull request

### Commit Message Format

Use clear, descriptive commit messages:
```
Complete #{issue_number}: Brief description

Longer explanation if needed.
```

Examples:
- `Complete #42: Add retry support to guards`
- `Fix #17: Handle empty input in summarize spell`

## Project Structure

```
src/spellcrafting/       # Main package
  spell.py           # @spell decorator - core LLM function wrapper
  guard.py           # @guard.input/output decorators - validation
  config.py          # Model alias configuration (Config context manager)
  on_fail.py         # Failure strategies (OnFail.retry, .escalate, etc.)
  validator.py       # LLM-powered validation helpers
  logging.py         # Observability/tracing (SpellExecutionLog)
  result.py          # SpellResult wrapper for metadata
  __init__.py        # Public API exports
tests/               # Test suite
docs/                # Documentation
```

## Getting Help

If you have questions:
- Check existing issues and discussions
- Open a new issue with the `question` label

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
