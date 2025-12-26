# Spellcrafting

**LLMs as a Python language feature.**

Spellcrafting lets you write Python functions that are powered by LLMs using a simple decorator. Your docstring becomes the prompt, your type hints become the schema, and structured outputs just work.

```python
from spellcrafting import spell
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    key_points: list[str]
    confidence: float

@spell(model="anthropic:claude-sonnet-4-20250514")
def analyze(text: str) -> Analysis:
    """Analyze the text for sentiment and extract key points."""
    ...

result = analyze("Python is fantastic for AI development!")
# Analysis(sentiment='positive', key_points=['Python is praised', ...], confidence=0.95)
```

## Installation

```bash
pip install spellcrafting
```

Requires Python 3.10+.

### Optional Dependencies

```bash
# For .env file support (loading API keys from .env)
pip install spellcrafting[dotenv]

# For OpenTelemetry tracing
pip install spellcrafting[otel]

# For Logfire integration
pip install spellcrafting[logfire]

# For Datadog integration
pip install spellcrafting[datadog]

# Install all optional dependencies
pip install spellcrafting[all]
```

## Quick Start

### Basic Usage

```python
from spellcrafting import spell

@spell(model="anthropic:claude-sonnet-4-20250514")
def summarize(text: str) -> str:
    """Summarize the given text in one sentence."""
    ...

summary = summarize("Long article content here...")
```

### Structured Output

Return Pydantic models for validated, structured responses:

```python
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    ingredients: list[str]
    steps: list[str]
    prep_time_minutes: int

@spell(model="anthropic:claude-sonnet-4-20250514")
def create_recipe(dish: str, dietary_restrictions: list[str]) -> Recipe:
    """Create a recipe for the given dish respecting dietary restrictions."""
    ...

recipe = create_recipe("pasta carbonara", ["vegetarian"])
print(recipe.ingredients)  # ['spaghetti', 'eggs', 'parmesan', ...]
```

### Async Support

Use `async def` for non-blocking execution:

```python
@spell(model="anthropic:claude-sonnet-4-20250514")
async def translate(text: str, target_language: str) -> str:
    """Translate the text to the target language."""
    ...

# In an async context
result = await translate("Hello, world!", "Spanish")
```

### Tools

Give your spell access to tools for more capable agents:

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Your weather API call here
    return f"72°F and sunny in {city}"

def get_time(timezone: str) -> str:
    """Get current time in a timezone."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")

@spell(model="anthropic:claude-sonnet-4-20250514", tools=[get_weather, get_time])
def travel_assistant(query: str) -> str:
    """Help the user with travel-related questions. Use tools when needed."""
    ...

response = travel_assistant("What's the weather like in Tokyo?")
```

The `end_strategy` parameter controls tool call behavior:
- `"early"` (default): Stop as soon as the model produces a final response
- `"exhaustive"`: Continue until all tool calls are processed

```python
@spell(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[search_database, fetch_url],
    end_strategy="exhaustive",  # Process all tool calls
)
def research(topic: str) -> Report:
    """Research the topic thoroughly using all available tools."""
    ...
```

## Configuration

### Direct Model Specification

Specify models directly using the `provider:model` format:

```python
@spell(model="anthropic:claude-sonnet-4-20250514")
def fast_task(text: str) -> str:
    """Quick task."""
    ...

@spell(model="openai:gpt-4o")
def openai_task(text: str) -> str:
    """Using OpenAI."""
    ...
```

### Model Aliases via pyproject.toml

Define reusable model configurations:

```toml
# pyproject.toml
[tool.spellcrafting.models.fast]
model = "anthropic:claude-3-5-haiku-latest"
temperature = 0.2
max_tokens = 1024

[tool.spellcrafting.models.reasoning]
model = "anthropic:claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 8192
```

> **Note**: Model names follow the provider's format (e.g., `anthropic:claude-sonnet-4-20250514`). Use versioned model names for reproducibility, or `-latest` suffixes for automatic updates.

Then use the alias:

```python
@spell(model="fast")
def quick_task(text: str) -> str:
    """A quick task using the fast model."""
    ...
```

### Programmatic Configuration

Override configuration at runtime:

```python
from spellcrafting import Config, ModelConfig

config = Config(models={
    "fast": ModelConfig(
        model="anthropic:claude-3-5-haiku-latest",
        temperature=0.3
    )
})

# Use as context manager
with config:
    result = quick_task("Hello")

# Or set as process default
config.set_as_default()
```

### Model Settings

Fine-tune model behavior:

```python
@spell(
    model="anthropic:claude-sonnet-4-20250514",
    model_settings={"temperature": 0.9, "max_tokens": 2000},
    retries=3,  # Retry on validation failures
)
def creative_writing(prompt: str) -> str:
    """Write creative content."""
    ...
```

## LLM-Powered Validation

Use `llm_validator` to create Pydantic validators powered by natural language rules:

```python
from spellcrafting import llm_validator
from pydantic import BaseModel, BeforeValidator
from typing import Annotated

# Create a validator from a natural language rule
family_friendly = llm_validator(
    "Content must be appropriate for all ages with no profanity",
    model="fast"
)

class Response(BaseModel):
    content: Annotated[str, BeforeValidator(family_friendly)]

# Use FIX strategy to auto-correct values
professional = llm_validator(
    "Must be professional business communication",
    model="fast",
    on_fail="fix"  # Attempt to fix invalid values
)

class Email(BaseModel):
    body: Annotated[str, BeforeValidator(professional)]
```

The `on_fail` parameter controls behavior when validation fails:
- `"raise"` (default): Raise `ValueError` with the reason
- `"fix"`: Attempt to fix the value to satisfy the rule

> **Note**: LLM validators add latency and cost. Use fast/cheap models for validation checks.

## Guardrails

Use `@guard` decorators to add input/output validation around your spells:

```python
from spellcrafting import spell, guard, GuardError

def validate_not_empty(input_args: dict, context: dict) -> dict:
    """Validate that input text is not empty."""
    if not input_args.get("text", "").strip():
        raise ValueError("Input text cannot be empty")
    return input_args

def check_no_competitors(output: str, context: dict) -> str:
    """Ensure output doesn't mention competitor names."""
    competitors = {"acme", "globex"}
    if any(c in output.lower() for c in competitors):
        raise ValueError("Response mentions competitor")
    return output

@spell(model="fast")
@guard.input(validate_not_empty)
@guard.output(check_no_competitors)
def summarize(text: str) -> str:
    """Summarize the given text."""
    ...
```

**Important**: Guards must be applied *inside* `@spell` (spell is the outermost decorator).

### Built-in Guards

```python
# Limit input and output character lengths
@spell(model="fast")
@guard.max_length(input_max=10000, output_max=5000)
def summarize(text: str) -> str:
    """Summarize the text."""
    ...
```

### Guard Context

Guard functions receive a context dict with execution metadata:

```python
def my_guard(input_args: dict, context: dict) -> dict:
    print(f"Spell: {context['spell_name']}")
    print(f"Model: {context['model']}")
    print(f"Attempt: {context['attempt_number']}")
    return input_args
```

### Async Guards

Guards can be async functions when used with async spells:

```python
async def async_validator(input_args: dict, context: dict) -> dict:
    result = await some_async_validation(input_args)
    return input_args

@spell(model="fast")
@guard.input(async_validator)
async def my_async_spell(text: str) -> str:
    """Process text."""
    ...
```

## Handling Validation Failures

The `on_fail` parameter controls what happens when the LLM output fails Pydantic validation after all retries are exhausted:

```python
from spellcrafting import spell, OnFail

# Escalate to a more capable model on failure
@spell(model="fast", on_fail=OnFail.escalate("reasoning"))
def complex_task(query: str) -> Analysis:
    """Complex analysis that may need a better model."""
    ...

# Return a default value instead of raising
@spell(on_fail=OnFail.fallback(default=DefaultResponse()))
def optional_enrichment(data: str) -> Enriched:
    """Optionally enrich the data."""
    ...

# Custom handler for domain-specific fixes
def fix_dates(error: Exception, attempt: int, context: dict) -> Dates:
    if "date format" in str(error):
        return parse_dates_manually(context["input_args"]["text"])
    raise error

@spell(on_fail=OnFail.custom(fix_dates))
def extract_dates(text: str) -> Dates:
    """Extract dates from text."""
    ...
```

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `OnFail.retry()` | Default. Retry with validation error in context. |
| `OnFail.escalate(model)` | Try a more capable model after retries exhausted. |
| `OnFail.fallback(default)` | Return a default value instead of raising. |
| `OnFail.custom(handler)` | Call a custom handler function. |

## Execution Metadata with SpellResult

Use `.with_metadata()` to get detailed execution information alongside the spell output:

```python
from spellcrafting import spell

@spell(model="fast")
def classify(text: str) -> Category:
    """Classify the text."""
    ...

# Normal call - just returns Category
result = classify("some text")

# With metadata - returns SpellResult[Category]
result = classify.with_metadata("some text")

# Access the output and metadata
print(result.output)        # Category instance
print(result.input_tokens)  # 50
print(result.output_tokens) # 25
print(result.total_tokens)  # 75
print(result.model_used)    # "openai:gpt-4o-mini"
print(result.duration_ms)   # 234.5
print(result.cost_estimate) # 0.00015 (USD)
print(result.attempt_count) # 1 (no retries)
print(result.trace_id)      # "abc123..." (for log correlation)
```

### SpellResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | `T` | The spell's return value |
| `input_tokens` | `int` | Number of input tokens used |
| `output_tokens` | `int` | Number of output tokens generated |
| `total_tokens` | `int` | Sum of input and output tokens |
| `model_used` | `str` | The actual model that was used |
| `duration_ms` | `float` | Execution time in milliseconds |
| `cost_estimate` | `float \| None` | Estimated cost in USD |
| `attempt_count` | `int` | Number of attempts (1 = no retries) |
| `trace_id` | `str \| None` | Trace ID for log correlation |

## Observability

Spellcrafting provides comprehensive logging, tracing, and cost tracking.

### Quick Setup

```python
from spellcrafting import setup_logging, LogLevel

# Enable logging with default settings
setup_logging(level=LogLevel.INFO)

# With OpenTelemetry export
setup_logging(level=LogLevel.INFO, otel=True)

# Write to JSON file
setup_logging(level=LogLevel.INFO, json_file="spells.jsonl")

# Redact sensitive content
setup_logging(level=LogLevel.INFO, redact_content=True)
```

### Provider Integrations

```python
from spellcrafting import setup_logfire, setup_datadog

# Logfire (requires: pip install spellcrafting[logfire])
setup_logfire()

# Datadog (requires: pip install spellcrafting[datadog])
setup_datadog()
```

### Distributed Tracing

Propagate trace context across spell calls:

```python
from spellcrafting import with_trace_id

# Correlate with external request trace
with with_trace_id(request.headers["X-Trace-ID"]):
    result = my_spell("input")
```

### Configuration via pyproject.toml

```toml
[tool.spellcrafting.logging]
enabled = true
level = "info"
redact_content = false

[tool.spellcrafting.logging.handlers.python]
type = "python"
logger_name = "spellcrafting"

[tool.spellcrafting.logging.handlers.file]
type = "json_file"
path = "logs/spells.jsonl"
```

## How It Works

1. **Docstring to System Prompt**: Your function's docstring becomes the LLM's system prompt
2. **Arguments to User Message**: Function arguments are formatted as the user message
3. **Return Type to Schema**: The return type annotation defines the expected output structure
4. **Validation**: Pydantic validates the LLM's response matches your schema

## Supported Providers

Spellcrafting uses [PydanticAI](https://ai.pydantic.dev/) under the hood, supporting:

- Anthropic (`anthropic:claude-*`)
- OpenAI (`openai:gpt-*`)
- Google (`google:gemini-*`)
- Groq (`groq:*`)
- And more...

Set the appropriate API key environment variable for your provider:

```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
```

## License

MIT
