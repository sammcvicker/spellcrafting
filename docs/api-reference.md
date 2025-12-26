# API Reference

This document provides a reference for all public exports from the `spellcrafting` package.

## Core

### `spell`

The main decorator for creating LLM-powered functions.

```python
from spellcrafting import spell

@spell(model="anthropic:claude-sonnet-4-20250514")
def summarize(text: str) -> str:
    """Summarize the given text."""
    ...
```

**Parameters:**
- `model` (str): Model identifier. Either a provider:model string (e.g., `"anthropic:claude-sonnet-4-20250514"`) or an alias defined in `pyproject.toml`.
- `tools` (list[Callable], optional): List of tool functions the LLM can call.
- `retries` (int, optional): Number of retries on validation failure. Default: 1.
- `model_settings` (dict, optional): Model-specific settings like `temperature`, `max_tokens`.
- `end_strategy` (str, optional): Tool call behavior - `"early"` (default) or `"exhaustive"`.
- `on_fail` (OnFailStrategy, optional): Failure handling strategy.

**Returns:** A decorated function that calls the LLM when invoked.

### `SpellResult`

Result wrapper that includes execution metadata alongside the output.

```python
from spellcrafting import spell, SpellResult

@spell(model="fast")
def classify(text: str) -> str:
    """Classify the text."""
    ...

# Get result with metadata
result: SpellResult[str] = classify.with_metadata("some text")
print(result.output)        # The actual output
print(result.input_tokens)  # Token usage
print(result.output_tokens)
print(result.total_tokens)
print(result.model_used)    # Actual model used
print(result.duration_ms)   # Execution time
print(result.cost_estimate) # Estimated cost in USD
print(result.trace_id)      # For log correlation
```

**Attributes:**
- `output` (T): The spell output value
- `input_tokens` (int): Input tokens used
- `output_tokens` (int): Output tokens used
- `total_tokens` (int): Total tokens (property)
- `model_used` (str): The model that was used
- `attempt_count` (int): Number of execution attempts
- `duration_ms` (float): Execution duration in milliseconds
- `cost_estimate` (float | None): Estimated cost in USD
- `trace_id` (str | None): Trace ID for log correlation

### `SyncSpell` / `AsyncSpell`

Protocol types for type-hinting spell functions.

```python
from spellcrafting import SyncSpell, AsyncSpell, SpellResult

def run_sync(spell_fn: SyncSpell[str]) -> SpellResult[str]:
    return spell_fn.with_metadata("input")

async def run_async(spell_fn: AsyncSpell[str]) -> SpellResult[str]:
    return await spell_fn.with_metadata("input")
```

---

## Configuration

### `Config`

Context manager for runtime configuration.

```python
from spellcrafting import Config, ModelConfig

# Define model aliases
config = Config(models={
    "fast": ModelConfig(
        model="anthropic:claude-haiku",
        temperature=0.2
    ),
    "reasoning": ModelConfig(
        model="anthropic:claude-sonnet-4-20250514",
        temperature=0.7
    )
})

# Use as context manager
with config:
    result = my_spell("input")

# Or set as process default
config.set_as_default()
```

### `ModelConfig`

Configuration for a model alias.

```python
from spellcrafting import ModelConfig

config = ModelConfig(
    model="anthropic:claude-sonnet-4-20250514",
    temperature=0.7,
    max_tokens=4096
)
```

**Attributes:**
- `model` (str): The provider:model identifier
- `temperature` (float, optional): Sampling temperature
- `max_tokens` (int, optional): Maximum output tokens

### `current_config`

Get the currently active configuration.

```python
from spellcrafting import current_config

config = current_config()
```

---

## Guards

### `guard`

Namespace for guard decorators. Guards provide validation that runs before (input) or after (output) spell execution.

**Important:** Guards must be applied INSIDE the `@spell` decorator:

```python
from spellcrafting import spell, guard

@spell(model="fast")           # <-- Outermost
@guard.input(validate_input)   # <-- Inside
@guard.output(check_output)    # <-- Inside
def my_spell(text: str) -> str:
    """Process text."""
    ...
```

### `guard.input()`

Add an input guard that runs before the LLM call.

```python
def validate_not_empty(input_args: dict, context: dict) -> dict:
    if not input_args.get("text", "").strip():
        raise ValueError("Input text cannot be empty")
    return input_args

@spell(model="fast")
@guard.input(validate_not_empty)
def summarize(text: str) -> str:
    """Summarize the text."""
    ...
```

**Parameters:**
- `guard_fn` (InputGuard): Function that validates/transforms inputs
- `on_fail` (RaiseStrategy, optional): Action on failure (default: `OnFail.RAISE`)

### `guard.output()`

Add an output guard that runs after the LLM call.

```python
def no_competitors(output: str, context: dict) -> str:
    competitors = {"acme", "globex"}
    if any(c in output.lower() for c in competitors):
        raise ValueError("Response mentions competitor")
    return output

@spell(model="fast")
@guard.output(no_competitors)
def respond(query: str) -> str:
    """Respond to the customer."""
    ...
```

**Parameters:**
- `guard_fn` (OutputGuard): Function that validates/transforms output
- `on_fail` (RaiseStrategy, optional): Action on failure (default: `OnFail.RAISE`)

### `guard.max_length()`

Convenience guard for input/output length limits.

```python
@spell(model="fast")
@guard.max_length(input=10000, output=5000)
def summarize(text: str) -> str:
    """Summarize the text."""
    ...
```

### `GuardConfig`

Configuration attached to functions with guards. Access via `get_guard_config()`.

### `get_guard_config()`

Get the guard configuration for a decorated function.

```python
from spellcrafting import get_guard_config

config = get_guard_config(my_spell)
if config:
    print(f"Input guards: {len(config.input_guards)}")
    print(f"Output guards: {len(config.output_guards)}")
```

### `InputGuard` / `OutputGuard`

Protocol types for guard functions.

```python
from spellcrafting import InputGuard, OutputGuard

def my_input_guard(input_args: dict, context: dict) -> dict:
    # Validate/transform inputs
    return input_args

def my_output_guard(output: T, context: dict) -> T:
    # Validate/transform output
    return output
```

### `GuardError`

Exception raised when a guard fails.

```python
from spellcrafting import GuardError

try:
    result = my_spell("invalid input")
except GuardError as e:
    print(f"Guard failed: {e}")
```

---

## Failure Strategies

### `OnFail`

Factory for on_fail strategies.

```python
from spellcrafting import spell, OnFail

# Escalate to a better model on failure
@spell(model="fast", on_fail=OnFail.escalate("reasoning"))
def complex_task(query: str) -> Analysis:
    """Analyze the query."""
    ...

# Return a default value on failure
@spell(on_fail=OnFail.fallback(default=EmptyResponse()))
def optional_enrichment(data: str) -> Enriched:
    """Optional enrichment."""
    ...

# Custom error handling
@spell(on_fail=OnFail.custom(my_handler))
def with_custom_handling(text: str) -> Result:
    """Custom handling."""
    ...
```

**Methods:**
- `OnFail.retry()`: Default behavior - retry with validation error in context
- `OnFail.escalate(model, retries=1)`: Retry with a more capable model
- `OnFail.fallback(default)`: Return a default value instead of raising
- `OnFail.custom(handler)`: Use a custom handler function

**Constants:**
- `OnFail.RAISE`: Raise on failure (used for guards)

---

## Validators

### `llm_validator()`

Create a Pydantic validator from a natural language rule.

```python
from spellcrafting import llm_validator
from pydantic import BaseModel, BeforeValidator
from typing import Annotated

professional = llm_validator(
    "Must be professional and appropriate for business communication",
    model="fast"
)

class Email(BaseModel):
    body: Annotated[str, BeforeValidator(professional)]
```

**Parameters:**
- `rule` (str): Natural language description of the validation rule
- `model` (str, optional): Model alias to use. Default: `"fast"`
- `on_fail` (str, optional): Action on failure - `"raise"` (default) or `"fix"`

**Returns:** A validator function for use with Pydantic's `BeforeValidator`.

### `ValidationResult`

Result from LLM validation check.

```python
from spellcrafting import ValidationResult

# Returned by internal validation spells
result = ValidationResult(
    valid=True,
    reason=None,
    fixed_value=None
)
```

**Attributes:**
- `valid` (bool): Whether validation passed
- `reason` (str | None): Reason for failure
- `fixed_value` (str | None): Fixed value if `on_fail="fix"`

---

## Logging

### `LoggingConfig`

Configuration for spellcrafting logging.

```python
from spellcrafting import LoggingConfig, LogLevel, configure_logging

configure_logging(LoggingConfig(
    enabled=True,
    level=LogLevel.INFO,
    handlers=[PythonLoggingHandler()],
    redact_content=False,
    cost_tracking=True,
))
```

**Attributes:**
- `enabled` (bool): Whether logging is enabled. Default: `False`
- `level` (LogLevel): Minimum log level. Default: `LogLevel.INFO`
- `handlers` (list[LogHandler]): Log handlers to use
- `redact_content` (bool): Redact input/output content. Default: `False`
- `include_input` (bool): Include input in logs. Default: `True`
- `include_output` (bool): Include output in logs. Default: `True`
- `cost_tracking` (bool): Track cost estimates. Default: `True`
- `default_tags` (dict[str, str]): Default tags for all logs

### `LogLevel`

Log levels enumeration.

```python
from spellcrafting import LogLevel

LogLevel.DEBUG
LogLevel.INFO
LogLevel.WARNING
LogLevel.ERROR
```

### `configure_logging()`

Set the logging configuration for the current process.

```python
from spellcrafting import configure_logging, LoggingConfig

configure_logging(LoggingConfig(enabled=True))
```

### `get_logging_config()`

Get the current logging configuration.

```python
from spellcrafting import get_logging_config

config = get_logging_config()
```

### `setup_logging()`

Quick setup for logging.

```python
from spellcrafting import setup_logging, LogLevel

# Development
setup_logging(level=LogLevel.DEBUG)

# Production with OpenTelemetry
setup_logging(
    level=LogLevel.INFO,
    otel=True,
    redact_content=True,
)
```

**Parameters:**
- `level` (LogLevel, optional): Minimum log level. Default: `LogLevel.INFO`
- `otel` (bool, optional): Enable OpenTelemetry export. Default: `False`
- `redact_content` (bool, optional): Redact content. Default: `False`
- `json_file` (str | Path, optional): Path to JSON log file

### `setup_logfire()` / `setup_datadog()`

Convenience aliases for `setup_logging(otel=True)`.

---

## Tracing

### `TraceContext`

W3C Trace Context compatible correlation IDs.

```python
from spellcrafting import TraceContext

ctx = TraceContext.new()
print(ctx.trace_id)   # 32 hex chars
print(ctx.span_id)    # 16 hex chars
print(ctx.to_w3c_traceparent())  # "00-{trace_id}-{span_id}-01"
```

### `trace_context()`

Context manager for trace context propagation.

```python
from spellcrafting import trace_context

with trace_context() as ctx:
    # All spell calls share this trace
    result1 = spell_one("input")
    result2 = spell_two("input")
```

### `current_trace()`

Get the current trace context.

```python
from spellcrafting import current_trace

ctx = current_trace()
if ctx:
    print(f"Trace ID: {ctx.trace_id}")
```

### `with_trace_id()`

Context manager to set a specific trace ID for external correlation.

```python
from spellcrafting import with_trace_id

# Correlate with incoming HTTP request
trace_id = request.headers.get("X-Trace-ID")
with with_trace_id(trace_id):
    result = my_spell("input")
```

---

## Handlers

### `LogHandler`

Protocol for synchronous log handlers.

```python
from spellcrafting import LogHandler, SpellExecutionLog

class MyHandler:
    def handle(self, log: SpellExecutionLog) -> None:
        print(log.to_json())

    def flush(self) -> None:
        pass
```

### `AsyncLogHandler`

Protocol for asynchronous log handlers.

```python
from spellcrafting import AsyncLogHandler, SpellExecutionLog

class MyAsyncHandler:
    async def handle(self, log: SpellExecutionLog) -> None:
        await send_to_service(log.to_dict())

    async def flush(self) -> None:
        pass
```

### `PythonLoggingHandler`

Handler that emits logs to Python's stdlib logging.

```python
from spellcrafting import PythonLoggingHandler

handler = PythonLoggingHandler(logger_name="spellcrafting")
```

### `JSONFileHandler`

Handler that writes JSON logs to a file.

```python
from spellcrafting import JSONFileHandler

handler = JSONFileHandler("/var/log/spellcrafting.jsonl", buffer_size=100)
```

### `OpenTelemetryHandler`

Handler that exports to OpenTelemetry.

```python
from spellcrafting import OpenTelemetryHandler

handler = OpenTelemetryHandler(service_name="my-app")
```

Requires `opentelemetry-api` and `opentelemetry-sdk` packages.

---

## Types

### `SpellExecutionLog`

Complete log of a spell execution.

```python
from spellcrafting import SpellExecutionLog

# Access log data
log.spell_name      # Function name
log.spell_id        # Unique decorator instance ID
log.trace_id        # Correlation ID
log.span_id         # Execution span ID
log.start_time      # UTC datetime
log.end_time        # UTC datetime
log.duration_ms     # Execution time
log.model           # Resolved model
log.model_alias     # Original alias
log.input_args      # Function arguments (or "[REDACTED]")
log.output          # Result (or "[REDACTED]")
log.token_usage     # TokenUsage instance
log.cost_estimate   # CostEstimate instance
log.tool_calls      # List of ToolCallLog
log.success         # Boolean
log.error           # Error message if failed
log.tags            # Custom metadata
log.validation      # ValidationMetrics instance

# Serialize
log.to_dict()       # Dict representation
log.to_json()       # JSON string
```

### `TokenUsage`

Token usage metrics.

```python
from spellcrafting import TokenUsage

usage = TokenUsage(
    input_tokens=100,
    output_tokens=50,
    cache_read_tokens=0,
    cache_write_tokens=0
)
print(usage.total_tokens)  # 150
```

### `CostEstimate`

Estimated cost of execution.

```python
from spellcrafting import CostEstimate

cost = CostEstimate(
    input_cost=0.0003,
    output_cost=0.00075,
    total_cost=0.00105,
    currency="USD",
    model="claude-sonnet-4-20250514"
)
```

### `ToolCallLog`

Log of a single tool invocation.

```python
from spellcrafting import ToolCallLog

tool_log = ToolCallLog(
    tool_name="get_weather",
    arguments={"city": "Tokyo"},
    result="72F and sunny",
    duration_ms=150.5,
    success=True
)
```

### `ValidationMetrics`

Metrics about validation during spell execution.

```python
from spellcrafting import ValidationMetrics

metrics = ValidationMetrics(
    attempt_count=2,
    retry_reasons=["Pydantic validation failed"],
    input_guards_passed=["validate_length"],
    output_guards_passed=["check_format"],
    on_fail_triggered="escalate",
    escalated_to_model="reasoning"
)
```

---

## Exceptions

All exceptions inherit from `SpellcraftingError`.

### `SpellcraftingError`

Base exception for all spellcrafting errors.

### `SpellcraftingConfigError`

Raised for configuration errors.

### `GuardError`

Raised when a guard fails validation.

### `ValidationError`

Raised for validation errors.

---

## Cache Management

### `clear_agent_cache()`

Clear the agent cache.

```python
from spellcrafting import clear_agent_cache

clear_agent_cache()
```

### `get_cache_stats()`

Get cache statistics.

```python
from spellcrafting import get_cache_stats

stats = get_cache_stats()
print(stats.size)
print(stats.hits)
print(stats.misses)
```

### `set_cache_max_size()`

Set the maximum cache size.

```python
from spellcrafting import set_cache_max_size

set_cache_max_size(100)
```

### `CacheStats`

Cache statistics dataclass.

```python
from spellcrafting import CacheStats

stats: CacheStats = get_cache_stats()
```
