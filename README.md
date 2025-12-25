# Magically

**LLMs as a Python language feature.**

Magically lets you write Python functions that are powered by LLMs using a simple decorator. Your docstring becomes the prompt, your type hints become the schema, and structured outputs just work.

```python
from magically import spell
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
pip install magically
```

Requires Python 3.13+.

### Optional Dependencies

```bash
# For .env file support (loading API keys from .env)
pip install magically[dotenv]

# For OpenTelemetry tracing
pip install magically[otel]

# For Logfire integration
pip install magically[logfire]

# For Datadog integration
pip install magically[datadog]

# Install all optional dependencies
pip install magically[all]
```

## Quick Start

### Basic Usage

```python
from magically import spell

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
[tool.magically.models.fast]
model = "anthropic:claude-3-5-haiku-latest"
temperature = 0.2
max_tokens = 1024

[tool.magically.models.reasoning]
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
from magically import Config, ModelConfig

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
from magically import llm_validator
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

## Observability

Magically provides comprehensive logging, tracing, and cost tracking.

### Quick Setup

```python
from magically import setup_logging, LogLevel

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
from magically import setup_logfire, setup_datadog

# Logfire (requires: pip install magically[logfire])
setup_logfire()

# Datadog (requires: pip install magically[datadog])
setup_datadog()
```

### Distributed Tracing

Propagate trace context across spell calls:

```python
from magically import with_trace_id

# Correlate with external request trace
with with_trace_id(request.headers["X-Trace-ID"]):
    result = my_spell("input")
```

### Execution Metadata

Access token usage and cost estimates:

```python
result = my_spell.with_metadata("input")

print(f"Tokens: {result.input_tokens} in, {result.output_tokens} out")
print(f"Cost: ${result.cost_estimate.total_cost:.4f}")
print(f"Duration: {result.duration_ms}ms")
print(f"Output: {result.output}")
```

### Configuration via pyproject.toml

```toml
[tool.magically.logging]
enabled = true
level = "info"
redact_content = false

[tool.magically.logging.handlers.python]
type = "python"
logger_name = "magically"

[tool.magically.logging.handlers.file]
type = "json_file"
path = "logs/spells.jsonl"
```

## How It Works

1. **Docstring to System Prompt**: Your function's docstring becomes the LLM's system prompt
2. **Arguments to User Message**: Function arguments are formatted as the user message
3. **Return Type to Schema**: The return type annotation defines the expected output structure
4. **Validation**: Pydantic validates the LLM's response matches your schema

## Supported Providers

Magically uses [PydanticAI](https://ai.pydantic.dev/) under the hood, supporting:

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
