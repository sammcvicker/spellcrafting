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
model = "anthropic:claude-haiku"
temperature = 0.2
max_tokens = 1024

[tool.magically.models.reasoning]
model = "anthropic:claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 8192
```

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
        model="anthropic:claude-haiku",
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

## How It Works

1. **Docstring → System Prompt**: Your function's docstring becomes the LLM's system prompt
2. **Arguments → User Message**: Function arguments are formatted as the user message
3. **Return Type → Schema**: The return type annotation defines the expected output structure
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
