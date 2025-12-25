# Implementation Order

Blockers analyzed, ordered by dependencies:

## 1. ModelConfig + MagicallyConfigError
**Blockers:** None - foundation types

- `ModelConfig` Pydantic model (model, temperature, max_tokens, etc.)
- `MagicallyConfigError` exception class
- No dependencies on other new code

## 2. Config class core
**Blockers:** ModelConfig

- `__init__(models: dict)` with dict → ModelConfig coercion
- `resolve(alias: str) → ModelConfig` with error on missing
- Context manager via `contextvars.ContextVar`
- `set_as_default()` for process-level default
- `current()` classmethod (context → default → empty)

## 3. Config.from_file() - pyproject.toml loading
**Blockers:** Config core

- Parse `[tool.magically.models.*]` from pyproject.toml
- Validate against ModelConfig schema
- Warn on unknown fields
- Cache result (loaded once at import)

## 4. Spell integration
**Blockers:** Config.from_file()

- Update `@spell` to resolve aliases via `Config.current()`
- Literal detection (contains `:` → skip alias lookup)
- Definition-time warning if alias missing from file config
- Call-time error if alias can't be resolved

## 5. Agent caching
**Blockers:** Spell integration

- Cache key: `(spell_id, hash(model_config))`
- Reuse agents when same spell + same resolved config
- Different config context → new agent

---

Each step is independently testable and shippable.
