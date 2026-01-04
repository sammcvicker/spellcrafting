# Examples

Quick examples to get started with spellcrafting.

## Running

```bash
# Set your API key
export ANTHROPIC_API_KEY="your-key"

# Run any example
uv run python examples/content_moderation.py
uv run python examples/image_analysis.py           # or: examples/image_analysis.py photo.jpg
uv run python examples/multi_env_config.py
uv run python examples/research_assistant.py
```

## What's Here

| Example | What it shows |
|---------|---------------|
| `content_moderation.py` | Guards for input validation and output safety, `OnFail.escalate` |
| `image_analysis.py` | Multi-modal `Image` type, automatic optimization, vision spells |
| `multi_env_config.py` | Model aliases, `Config` context manager, dev/prod switching |
| `research_assistant.py` | Tools, `.with_metadata()` for token tracking |
