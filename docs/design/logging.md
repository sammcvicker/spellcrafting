# Logging Design

## Overview

Enterprise-grade observability for spellcrafting: structured logging, distributed tracing, cost tracking, and performance metrics. Zero overhead when disabled, rich telemetry when enabled.

## Goals

1. **Zero overhead when disabled**: Fast path check at wrapper entry, no allocations or context management
2. **Structured JSON logs**: Consistent schema, machine-parseable, grep-friendly
3. **Distributed tracing**: W3C Trace Context compatible correlation IDs across spell calls
4. **Cost visibility**: Token usage and estimated USD cost per execution
5. **Platform agnostic**: Works with OpenTelemetry, Datadog, Logfire, or plain Python logging
6. **Sensitive data aware**: Built-in redaction for PII/secrets

## Non-Goals

- Automatic PII detection (user specifies redaction)
- Log aggregation/storage (use external tools)
- Real-time alerting (integrate with existing monitoring)
- Sampling strategies (keep it simple, log everything or nothing)

---

## Log Schema

Every spell execution produces a `SpellExecutionLog`:

```python
@dataclass
class SpellExecutionLog:
    # Identity
    spell_name: str           # Function name
    spell_id: int             # Unique decorator instance ID
    trace_id: str             # Correlation ID (32 hex chars)
    span_id: str              # This execution (16 hex chars)
    parent_span_id: str | None

    # Timing
    start_time: datetime      # UTC
    end_time: datetime | None
    duration_ms: float | None

    # Model
    model: str                # Resolved provider:model
    model_alias: str | None   # Original alias if used

    # I/O (redactable)
    input_args: dict | str    # Function arguments or "[REDACTED]"
    output: Any | str         # Result or "[REDACTED]"
    output_type: str          # Type name of output

    # Metrics
    token_usage: TokenUsage   # input, output, cache tokens
    cost_estimate: CostEstimate | None

    # Tools
    tool_calls: list[ToolCallLog]

    # Status
    success: bool
    error: str | None
    error_type: str | None
    retry_count: int

    # Custom
    tags: dict[str, str]      # User-defined metadata
```

### JSON Output

```json
{
  "spell_name": "analyze",
  "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
  "span_id": "00f067aa0ba902b7",
  "start_time": "2024-01-15T10:30:00.123Z",
  "end_time": "2024-01-15T10:30:02.456Z",
  "duration_ms": 2333.0,
  "model": "anthropic:claude-sonnet-4-20250514",
  "model_alias": "fast",
  "token_usage": {
    "input_tokens": 150,
    "output_tokens": 75,
    "cache_read_tokens": 0,
    "cache_write_tokens": 0
  },
  "cost_estimate": {
    "input_cost": 0.00045,
    "output_cost": 0.000225,
    "total_cost": 0.000675,
    "currency": "USD"
  },
  "tool_calls": [{
    "tool_name": "get_weather",
    "duration_ms": 150.5,
    "success": true
  }],
  "success": true,
  "tags": {"environment": "production"}
}
```

---

## Trace Context

W3C Trace Context compatible for interoperability with existing observability stacks.

### Propagation

```python
# Automatic: nested spells inherit trace_id
@spell(model="fast")
def step_one(data: str) -> str: ...

@spell(model="reasoning")
def step_two(data: str) -> str: ...

def pipeline(data: str) -> str:
    # Both calls share same trace_id automatically
    intermediate = step_one(data)
    return step_two(intermediate)
```

### External Correlation

```python
from spellcrafting import with_trace_id

# Correlate with incoming HTTP request
@app.post("/analyze")
async def handle(request: Request):
    trace_id = request.headers.get("X-Trace-ID", uuid4().hex)
    with with_trace_id(trace_id):
        return await analyze(request.body)
```

### Implementation

```python
from contextvars import ContextVar

@dataclass(frozen=True)
class TraceContext:
    trace_id: str      # 32 hex chars, shared across spans
    span_id: str       # 16 hex chars, unique per execution
    parent_span_id: str | None

    @classmethod
    def new(cls, parent: TraceContext | None = None) -> TraceContext:
        return cls(
            trace_id=parent.trace_id if parent else uuid4().hex,
            span_id=uuid4().hex[:16],
            parent_span_id=parent.span_id if parent else None,
        )

    def to_w3c_traceparent(self) -> str:
        """Format: 00-{trace_id}-{span_id}-01"""
        return f"00-{self.trace_id}-{self.span_id}-01"

_trace_context: ContextVar[TraceContext | None] = ContextVar("trace", default=None)
```

---

## Configuration

### Quick Setup

```python
from spellcrafting import setup_logging, LogLevel

# Development: console output
setup_logging(level=LogLevel.DEBUG)

# Production: OTEL + redaction
setup_logging(
    level=LogLevel.INFO,
    otel=True,
    redact_content=True,
)
```

### Full Configuration

```python
from spellcrafting import LoggingConfig, configure_logging

configure_logging(LoggingConfig(
    enabled=True,
    level=LogLevel.INFO,
    handlers=[
        PythonLoggingHandler(logger_name="spellcrafting"),
        JSONFileHandler("/var/log/spellcrafting.jsonl"),
        OpenTelemetryHandler(),
    ],
    redact_content=False,
    include_input=True,
    include_output=True,
    cost_tracking=True,
    default_tags={"service": "my-app", "version": "1.0"},
))
```

### pyproject.toml

```toml
[tool.spellcrafting.logging]
enabled = true
level = "info"
redact_content = false
cost_tracking = true

[tool.spellcrafting.logging.handlers.python]
type = "python"
logger_name = "spellcrafting"

[tool.spellcrafting.logging.handlers.json]
type = "json_file"
path = "logs/spellcrafting.jsonl"
```

---

## Handlers

### Protocol

```python
class LogHandler(Protocol):
    def handle(self, log: SpellExecutionLog) -> None: ...
    def flush(self) -> None: ...
```

### Built-in Handlers

| Handler | Output | Use Case |
|---------|--------|----------|
| `PythonLoggingHandler` | stdlib logging | Development, existing logging setup |
| `JSONFileHandler` | JSONL file | Log aggregation, debugging |
| `OpenTelemetryHandler` | OTEL spans + metrics | Datadog, Honeycomb, Jaeger |

### Custom Handler Example

```python
class CloudWatchHandler:
    def __init__(self, log_group: str):
        self.client = boto3.client('logs')
        self.log_group = log_group
        self.buffer: list[SpellExecutionLog] = []

    def handle(self, log: SpellExecutionLog) -> None:
        self.buffer.append(log)
        if len(self.buffer) >= 100:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.client.put_log_events(
            logGroupName=self.log_group,
            logStreamName="spellcrafting",
            logEvents=[{
                "timestamp": int(log.start_time.timestamp() * 1000),
                "message": log.to_json(),
            } for log in self.buffer],
        )
        self.buffer.clear()
```

---

## Cost Tracking

### Token Usage

Extracted from PydanticAI's `RunUsage`:

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int   # Anthropic prompt caching
    cache_write_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```

### Cost Estimation

Model-specific pricing (user-configurable):

```python
@dataclass
class CostEstimate:
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "USD"
    model: str = ""
```

Default pricing table (overridable):

```python
PRICING = {
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},  # per 1M tokens
    "claude-haiku": {"input": 0.25, "output": 1.25},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}
```

---

## Tool Call Logging

Each tool invocation is logged:

```python
@dataclass
class ToolCallLog:
    tool_name: str
    arguments: dict | str    # Redactable
    result: Any | str        # Redactable
    duration_ms: float
    success: bool
    error: str | None
    timestamp: datetime
```

### Integration

Requires wrapping tool functions to capture timing:

```python
def _wrap_tool(tool: Callable, log: SpellExecutionLog) -> Callable:
    @functools.wraps(tool)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = tool(*args, **kwargs)
            log.tool_calls.append(ToolCallLog(
                tool_name=tool.__name__,
                arguments=kwargs or dict(zip(tool.__code__.co_varnames, args)),
                result=result,
                duration_ms=(time.perf_counter() - start) * 1000,
                success=True,
            ))
            return result
        except Exception as e:
            log.tool_calls.append(ToolCallLog(
                tool_name=tool.__name__,
                arguments=kwargs,
                result=None,
                duration_ms=(time.perf_counter() - start) * 1000,
                success=False,
                error=str(e),
            ))
            raise
    return wrapped
```

---

## Integration with spell.py

### Wrapper Modification

```python
def sync_wrapper(*args, **kwargs) -> T:
    config = get_logging_config()

    # Fast path: no logging overhead
    if not config.enabled:
        resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
        agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)
        result = agent.run_sync(_build_user_prompt(fn, args, kwargs))
        return result.output

    # Logging enabled
    with trace_context() as ctx:
        log = SpellExecutionLog(
            spell_name=fn.__name__,
            spell_id=spell_id,
            trace_id=ctx.trace_id,
            span_id=ctx.span_id,
            parent_span_id=ctx.parent_span_id,
        )

        try:
            # ... execution ...
            log.token_usage = TokenUsage(
                input_tokens=result.usage().input_tokens,
                output_tokens=result.usage().output_tokens,
                # ...
            )
            log.finalize(success=True)
        except Exception as e:
            log.finalize(success=False, error=e)
            raise
        finally:
            _emit_log(log)

    return result.output
```

### Emit to Handlers

```python
def _emit_log(log: SpellExecutionLog) -> None:
    config = get_logging_config()
    for handler in config.handlers:
        try:
            handler.handle(log)
        except Exception:
            pass  # Never let logging break execution
```

---

## API Surface

### Public Exports

```python
# spellcrafting/__init__.py additions

# Configuration
from spellcrafting.logging import (
    LoggingConfig,
    LogLevel,
    configure_logging,
    get_logging_config,
)

# Quick setup (also in spellcrafting.logging)
from spellcrafting.logging import (
    setup_logging,
    setup_logfire,
    setup_datadog,
)

# Tracing
from spellcrafting.logging import (
    TraceContext,
    trace_context,
    current_trace,
    with_trace_id,
)

# Handlers
from spellcrafting.logging import (
    LogHandler,
    PythonLoggingHandler,
    JSONFileHandler,
    OpenTelemetryHandler,
)

# Types (for custom handlers)
from spellcrafting.logging import (
    SpellExecutionLog,
    TokenUsage,
    CostEstimate,
    ToolCallLog,
)
```

---

## Dependencies

**Required**: None (logging module is pure Python)

**Optional**:
```toml
[project.optional-dependencies]
observability = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
]
```

OpenTelemetry handler gracefully degrades if packages not installed.

---

## Performance

1. **Disabled path**: Single `if not config.enabled` check, ~1ns overhead
2. **Enabled path**:
   - TraceContext creation: ~500ns (uuid generation)
   - Log dataclass allocation: ~200ns
   - JSON serialization: ~10μs (deferred to handler)
3. **Handler isolation**: Exceptions in handlers never propagate
4. **Buffered I/O**: JSONFileHandler batches writes

---

## Testing

### Unit Tests

```python
class LogCollector:
    """Test handler that collects logs in memory."""
    def __init__(self):
        self.logs: list[SpellExecutionLog] = []

    def handle(self, log: SpellExecutionLog) -> None:
        self.logs.append(log)

    def flush(self) -> None:
        pass

def test_spell_logs_execution():
    collector = LogCollector()
    configure_logging(LoggingConfig(
        enabled=True,
        handlers=[collector],
    ))

    @spell(model="test:model")
    def greet(name: str) -> str:
        """Greet someone."""
        ...

    with patch("spellcrafting.spell.Agent") as mock:
        mock.return_value.run_sync.return_value.output = "Hello"
        mock.return_value.run_sync.return_value.usage.return_value = Usage(...)
        greet("World")

    assert len(collector.logs) == 1
    log = collector.logs[0]
    assert log.spell_name == "greet"
    assert log.success is True
    assert log.token_usage.input_tokens > 0
```

### Smoke Tests

```python
@pytest.mark.smoke
def test_logging_with_real_api():
    collector = LogCollector()
    configure_logging(LoggingConfig(enabled=True, handlers=[collector]))

    @spell(model="anthropic:claude-sonnet-4-20250514")
    def analyze(text: str) -> str:
        """Analyze text."""
        ...

    result = analyze("Hello")

    log = collector.logs[0]
    assert log.duration_ms > 0
    assert log.token_usage.total_tokens > 0
    assert log.cost_estimate.total_cost > 0
```

---

## Implementation Order

1. **Core types** (`logging.py`): TraceContext, TokenUsage, SpellExecutionLog, LoggingConfig
2. **Context propagation**: trace_context contextvar and context manager
3. **Handlers**: PythonLoggingHandler, JSONFileHandler
4. **spell.py integration**: Modify wrappers to emit logs
5. **OpenTelemetry handler**: Optional OTEL export
6. **Observability helpers**: setup_logging(), platform shortcuts
7. **pyproject.toml loading**: File-based logging config
8. **Cost estimation**: Pricing table and calculation

---

## Design Decisions

1. **Dataclasses over Pydantic**: Logging types use dataclasses for minimal overhead. Validation isn't needed for internal log structures.

2. **Handler errors are silent**: A broken handler should never crash production. Log the handler error to stderr, continue execution.

3. **No sampling**: Keep it simple. If you need sampling, implement it in a custom handler or use OTEL's sampling.

4. **Redaction is opt-in**: Default to logging everything. Operators explicitly enable redaction for sensitive environments.

5. **Cost tracking is best-effort**: Pricing tables go stale. Log token counts (accurate) and estimated cost (approximate). Users can recalculate with current pricing.

6. **Tool logging requires wrapping**: To capture tool call timing, we wrap tool functions. This is done transparently when logging is enabled.
