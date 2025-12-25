# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `llm_validator` factory for creating Pydantic validators powered by LLM calls
  - Natural language rules: `llm_validator("Must be family-friendly")`
  - Works with Pydantic's `BeforeValidator` for field validation
  - `on_fail="raise"` (default) or `on_fail="fix"` to auto-correct values
  - Uses configurable model aliases for cost control
- `ValidationResult` model for structured LLM validation responses
- `@guard` decorator for composable input/output validation on spells
  - `guard.input(fn)` - validate/transform inputs before LLM call
  - `guard.output(fn)` - validate/transform outputs after LLM call
  - `guard.max_length(input=N, output=M)` - built-in character length limits
- `GuardError` exception raised when guards reject input/output
- `OnFail` enum for guard failure behavior (RAISE for now, RETRY planned)
- `InputGuard`, `OutputGuard` protocols for type-safe guard functions
- `GuardContext` dataclass with spell metadata for guards
- Guardrails design document (`docs/design/guardrails.md`)
- Research documents for guardrails patterns:
  - `docs/research/guardrails-frameworks.md`
  - `docs/research/input_validation_guardrails.md`
  - `docs/design/output_validation.md`
  - `docs/research/enterprise-guardrails-patterns.md`

## [0.1.0] - 2024-12-24

### Added
- `@spell` decorator that turns Python functions into LLM calls
  - Docstring becomes system prompt
  - Return type annotation becomes output schema (via Pydantic)
  - Function arguments become user message
- `Config` class for model configuration with context manager support
- `ModelConfig` for defining model aliases with settings (temperature, max_tokens, etc.)
- Model alias resolution from `pyproject.toml` under `[tool.magically.models]`
- Async support (`async def` functions automatically use async LLM calls)
- Agent caching to avoid recreating PydanticAI agents
- Comprehensive logging system:
  - `SpellExecutionLog` with timing, tokens, cost estimates
  - Distributed tracing with W3C Trace Context compatible IDs
  - `PythonLoggingHandler`, `JSONFileHandler`, `OpenTelemetryHandler`
  - Cost estimation with configurable pricing table
  - Zero overhead when logging is disabled
- `setup_logging()`, `setup_logfire()`, `setup_datadog()` quick setup functions
- `trace_context()` context manager for trace propagation
- `with_trace_id()` for correlating with external trace IDs
