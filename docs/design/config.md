# Config Design

## Overview

Spellcrafting config separates **intent** from **implementation** through model aliases. Developers declare what kind of model they need semantically; operators define what those aliases resolve to.

## Goals

1. **Developers** declare intent: `@spell(model="reasoning")` not `@spell(model="anthropic:claude-opus-4")`
2. **Operators** control implementation: swap models, adjust settings, manage costs—without code changes
3. **No global mutable state**: Config is either file-based (loaded once) or context-scoped (explicit lifetime)
4. **Type-safe**: Pydantic validation for config, clear errors for missing aliases

## Non-Goals

- Environment variable overrides (keep it simple)
- Hot-reloading config files
- Per-request config from decorators (use context managers instead)

---

## Model Aliases

An alias bundles a model identifier with its settings:

```toml
# pyproject.toml
[tool.spellcrafting.models.reasoning]
model = "anthropic:claude-opus-4"
temperature = 0.7
max_tokens = 8192

[tool.spellcrafting.models.fast]
model = "anthropic:claude-haiku"
temperature = 0.2
max_tokens = 1024

[tool.spellcrafting.models.default]
model = "anthropic:claude-sonnet"
```

### Alias Resolution

```python
@spell(model="reasoning")        # Alias lookup
@spell(model="anthropic:claude-sonnet")  # Literal (contains ":")
@spell                           # Uses "default" alias
```

Detection: If `model` contains `:`, treat as literal provider string. Otherwise, resolve as alias.

### Missing Alias Behavior

**Error immediately.** No fallback to `default`, no fallback to treating as literal. Explicit is better than implicit.

```python
@spell(model="typo-alias")
def my_spell(...): ...

# Raises: SpellcraftingConfigError("Unknown model alias 'typo-alias'.
#         Available: reasoning, fast, default")
```

---

## Config Schema

```python
from pydantic import BaseModel

class ModelConfig(BaseModel):
    """Configuration for a model alias."""
    model: str  # provider:model-name format
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float | None = None
    retries: int | None = None
    # Passthrough for provider-specific settings
    extra: dict[str, Any] = {}

class SpellcraftingConfig(BaseModel):
    """Root configuration."""
    models: dict[str, ModelConfig] = {}
```

### Validation

- **Config load time**: Validate structure, types, known fields. Warn on unknown fields (forward compat).
- **Spell execution time**: pydantic-ai validates the actual provider/model. We don't try to enumerate valid models.

```python
# Warning at config load (unknown field, might be typo)
[tool.spellcrafting.models.fast]
model = "anthropic:claude-haiku"
temprature = 0.2  # typo → warning

# Error at runtime (pydantic-ai rejects)
[tool.spellcrafting.models.fast]
model = "anthropic:not-a-real-model"  # fails when spell executes
```

---

## Config Sources & Resolution

### Resolution Order (highest priority first)

```
1. Active Config context (contextvars)
2. Process default (set_as_default)
3. pyproject.toml [tool.spellcrafting]
4. Built-in defaults (just "default" → None, letting pydantic-ai pick)
```

### File Config

Loaded once at import time from `pyproject.toml`. Immutable after load.

```python
# Automatic on first spell execution
# Or explicit:
from spellcrafting import config
print(config.current().models)
```

### Runtime Config (Context Manager)

For operators who need dynamic config (database, secrets, per-tenant):

```python
from spellcrafting import Config

# Load from wherever
db_settings = fetch_from_vault()
config = Config(models={
    "fast": ModelConfig(model="openai:gpt-4o-mini", temperature=0.2),
    "reasoning": ModelConfig(model="openai:o1", temperature=1),
})

# Scoped override
async def handle_request(tenant_id: str):
    tenant_config = get_tenant_config(tenant_id)
    with tenant_config:
        return await my_spell(...)  # uses tenant's config

# Outside context: back to file/default config
```

**Implementation**: `contextvars.ContextVar` for async/thread safety.

### Process Default

For applications that load config once at startup:

```python
# startup.py
config = Config(models=fetch_config())
config.set_as_default()

# Elsewhere in the app—no context manager needed
result = my_spell(...)  # uses startup config
```

This is still explicit—you call `set_as_default()` once. Not hidden global mutation.

---

## API

```python
# spellcrafting/config.py

class Config:
    """Immutable configuration container."""

    def __init__(
        self,
        models: dict[str, ModelConfig | dict] | None = None,
    ) -> None:
        """Create config. Dicts are coerced to ModelConfig."""
        ...

    def __enter__(self) -> Self:
        """Push this config onto the context stack."""
        ...

    def __exit__(self, *exc) -> None:
        """Pop this config from the context stack."""
        ...

    def set_as_default(self) -> None:
        """Set as process-level default (below context, above file)."""
        ...

    def resolve(self, alias: str) -> ModelConfig:
        """Resolve alias to ModelConfig. Raises SpellcraftingConfigError if missing."""
        ...

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load from TOML file."""
        ...

    @classmethod
    def current(cls) -> Config:
        """Get the currently active config (context → default → file)."""
        ...


# Convenience access
def current_config() -> Config:
    """Get active config."""
    return Config.current()
```

---

## Spell Integration

The `@spell` decorator resolves its model at **definition time** for validation, but uses the **active config at call time** for actual settings.

```python
@spell(model="fast")
def classify(text: str) -> Category:
    """Classify the text."""
    ...

# Definition time: validate "fast" exists in some config
# Call time: resolve "fast" from Config.current()
```

### Implementation Sketch

```python
# Agent cache: (spell_id, model_config_hash) → Agent
_agent_cache: dict[tuple[int, int], Agent] = {}

def spell(
    func: Callable[P, T] | None = None,
    *,
    model: str | None = None,
    # ... other params
) -> ...:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        model_alias = model
        spell_id = id(fn)

        # Definition-time warning if alias missing from file config
        if model_alias and ":" not in model_alias:
            file_config = Config.from_file()
            if model_alias not in file_config.models:
                warnings.warn(
                    f"Model alias '{model_alias}' not found in pyproject.toml. "
                    f"Available: {list(file_config.models.keys())}",
                    SpellcraftingConfigWarning,
                )

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Resolve at call time
            config = Config.current()
            model_config = config.resolve(model_alias or "default")

            # Cache key: spell identity + resolved config
            cache_key = (spell_id, hash(model_config))
            if cache_key not in _agent_cache:
                _agent_cache[cache_key] = Agent(
                    model=model_config.model,
                    model_settings=model_config.to_model_settings(),
                    system_prompt=inspect.getdoc(fn) or "",
                    output_type=fn.__annotations__.get("return", str),
                    # ...
                )

            agent = _agent_cache[cache_key]
            user_prompt = _build_user_prompt(fn, args, kwargs)
            return agent.run_sync(user_prompt).output
```

Cache is keyed by `(spell_id, model_config_hash)`. Same spell + same resolved config = reuse agent. Different config context = new agent.

---

## Testing

```python
def test_spell_with_custom_config():
    test_config = Config(models={
        "fast": ModelConfig(model="test:mock", temperature=0),
    })

    with test_config:
        result = classify("hello")

    # No cleanup needed—context exited

def test_missing_alias_raises():
    empty_config = Config(models={})

    with empty_config:
        with pytest.raises(SpellcraftingConfigError, match="Unknown model alias"):
            classify("hello")
```

---

## File Format

Using `pyproject.toml` under `[tool.spellcrafting]`:

```toml
[tool.spellcrafting]
# Future: other spellcrafting settings

[tool.spellcrafting.models.default]
model = "anthropic:claude-sonnet"

[tool.spellcrafting.models.reasoning]
model = "anthropic:claude-opus-4"
temperature = 0.7
max_tokens = 8192

[tool.spellcrafting.models.fast]
model = "anthropic:claude-haiku"
temperature = 0.2
max_tokens = 1024
retries = 1

[tool.spellcrafting.models.creative]
model = "anthropic:claude-sonnet"
temperature = 0.9
top_p = 0.95
```

---

## Design Decisions

1. **Definition-time validation**: Warn at `@spell` decoration if alias missing from file config. Developers will want test coverage, and early warnings catch typos. Runtime config can still add aliases later (warning, not error).

2. **Agent caching**: Cache agents keyed by `(resolved_model_config, spell_identity)`. Invalidate when config context changes. Worth the complexity upfront to avoid repeated agent construction overhead.

3. **Config merging**: Context and file configs merge, with context taking precedence per-alias. If context defines `fast` and file defines `reasoning`, both are available.

4. **Literal model strings**: `@spell(model="anthropic:claude-sonnet")` bypasses alias resolution entirely. Useful escape hatch for one-offs, testing, or when you really do want a specific model regardless of config.
