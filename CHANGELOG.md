# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SpellResult.cost_estimate` field for estimated execution cost in USD (None if pricing unavailable)
- `SpellResult.trace_id` field for correlation with logs (None if no trace context active)
- `SpellResult.total_tokens` property for total token count (input + output)
- `SpellResult[T]` generic wrapper for accessing execution metadata alongside spell output
  - `output` - The actual spell output (same type as normal call)
  - `input_tokens` - Number of input tokens used
  - `output_tokens` - Number of output tokens used
  - `model_used` - Actual model used (may differ from alias if escalation occurred)
  - `attempt_count` - Number of execution attempts (1 = no retries)
  - `duration_ms` - Execution duration in milliseconds
- `.with_metadata()` method on spell-decorated functions to get `SpellResult` instead of raw output
  ```python
  @spell(model="fast")
  def classify(text: str) -> Category:
      '''Classify the text.'''
      ...

  # Normal call - just returns Category
  result = classify("some text")

  # With metadata - returns SpellResult[Category]
  result = classify.with_metadata("some text")
  print(result.output)  # Category
  print(result.input_tokens)  # 50
  print(result.model_used)  # "openai:gpt-4o-mini"
  ```
- `docs/pydantic-validation-patterns.md` - Comprehensive guide to Pydantic validation patterns for spell outputs
  - Constrained values with `Literal` and `Enum`
  - Numeric bounds and confidence scores
  - List constraints and field validators
  - Model validators for cross-field rules
  - Nested models and common patterns
  - Retry behavior guidance
  - Integration with `on_fail` strategies and `llm_validator`
- `ValidationMetrics` dataclass for tracking validation-related metrics during spell execution
  - `attempt_count` - Total execution attempts (1 = no retries)
  - `retry_reasons` - List of reasons for retries
  - `input_guards_passed` / `input_guards_failed` - Names of guards that passed/failed
  - `output_guards_passed` / `output_guards_failed` - Names of guards that passed/failed
  - `pydantic_errors` - Validation errors encountered before success or final failure
  - `on_fail_triggered` - Which on_fail strategy was used ("escalate", "fallback", "custom", "retry")
  - `escalated_to_model` - Model used when escalation strategy was triggered
- `SpellExecutionLog.validation` field containing `ValidationMetrics` when logging is enabled
- Guard tracking functions for capturing guard names that pass/fail during execution
- `on_fail` parameter for `@spell` decorator to handle validation failures after retries
  - `OnFail.escalate("model")` - retry with a more capable model on failure
  - `OnFail.fallback(default)` - return a default value instead of raising
  - `OnFail.custom(handler)` - use a custom handler for domain-specific fixes
  - `OnFail.retry()` - default behavior, re-raise after retries exhausted
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

## [0.1.0] - 2025-12-24

### Added
- `@spell` decorator that turns Python functions into LLM calls
  - Docstring becomes system prompt
  - Return type annotation becomes output schema (via Pydantic)
  - Function arguments become user message
- `Config` class for model configuration with context manager support
- `ModelConfig` for defining model aliases with settings (temperature, max_tokens, etc.)
- Model alias resolution from `pyproject.toml` under `[tool.spellcrafting.models]`
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
