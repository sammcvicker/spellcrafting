# Spellcrafting

LLMs as a Python language feature. Provides `@spell` and `@guard` decorators to turn functions into LLM-powered operations with input/output validation.

## Package Structure

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
```

## Commands

```bash
# Run tests (stop on first failure)
uv run pytest tests/ -x -q

# Run all tests
uv run pytest tests/ -q

# Skip smoke tests (real LLM calls)
uv run pytest tests/ -m "not smoke"

# Verify imports work
uv run python -c "from spellcrafting import spell, guard, Config; print('OK')"
```

## Core Patterns

### Basic spell
```python
@spell
def summarize(text: str) -> str:
    """Summarize the text in one sentence."""
    ...
```

### Guards (must be INSIDE @spell)
```python
@spell                          # Outermost
@guard.input(validate_input)    # Runs before LLM
@guard.output(check_output)     # Runs after LLM
def my_spell(text: str) -> str:
    """Process text."""
    ...
```

### Model configuration
```python
with Config(model="gpt-4o"):
    result = my_spell("hello")
```

## Issue Labels (priority order)

1. `broken` - Features not working as documented
2. `architecture` - Structural problems
3. `type-safety` - Type system improvements
4. `tech-debt` - Code quality
5. `testing` - Test coverage gaps
6. `documentation` - Docs updates
7. `enhancement` - New features

Quick wins: `dead-code`, `cleanup`, `chore` (batch 3-5 together)

## Git Workflow

- Feature branches: `issue-{number}-{description}`
- Commit format: `Complete #{number}: description`
- Always run tests before merge
- Close issues with: `gh issue close {n} --comment "Completed in $(git rev-parse --short HEAD)"`
