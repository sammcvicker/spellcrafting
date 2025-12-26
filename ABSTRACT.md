# Spellcrafting: Design Philosophy

Spellcrafting makes LLMs a Python language feature. This document explains the design decisions behind the library.

## Core Concept: Spells, Not Agents

The central abstraction is the **spell** - a function decorated with `@spell` that uses an LLM to produce its return value. Unlike agent frameworks that emphasize autonomy and multi-step reasoning, spells are:

- **Deterministic in structure**: The function signature defines inputs and outputs
- **Composable**: Spells are regular functions that can be called from anywhere
- **Observable**: Each invocation produces traceable, structured output

This design choice prioritizes predictability over flexibility. When you call a spell, you know exactly what shape of data you'll get back.

## Intent vs Implementation

Configuration in Spellcrafting separates **intent** (what you want) from **implementation** (how to get it):

```python
# Intent: Use a "fast" model
@spell(model="fast")
def summarize(text: str) -> str: ...

# Implementation: pyproject.toml maps "fast" to a specific provider/model
# [tool.spellcrafting.models.fast]
# model = "anthropic:claude-haiku"
```

This separation allows:
- Swapping models without code changes
- Environment-specific configuration (dev/staging/prod)
- Centralized model management across a codebase

## Relationship with PydanticAI

Spellcrafting is built on [PydanticAI](https://ai.pydantic.dev/), which provides:
- Multi-provider LLM support (Anthropic, OpenAI, etc.)
- Structured output via Pydantic models
- Retry logic with validation feedback

Spellcrafting adds:
- Declarative function-based API (`@spell` decorator)
- Configuration system (pyproject.toml, env vars, runtime config)
- Guards for input/output validation
- Failure strategies (escalation, fallback, custom handlers)

## Target Use Cases

Spellcrafting is designed for:

1. **Structured data extraction**: Parse unstructured text into typed objects
2. **Content transformation**: Summarization, translation, reformatting
3. **Classification and routing**: Categorize inputs for downstream processing
4. **Validation and enrichment**: Augment data with LLM-powered insights

## Non-Goals

Spellcrafting explicitly does NOT aim to:

- **Replace agent frameworks**: For complex multi-step reasoning with tool use, use PydanticAI directly or dedicated agent frameworks
- **Provide chat interfaces**: Spells are single-turn, stateless operations
- **Handle streaming**: Output is returned as complete structured objects
- **Manage conversation history**: Each spell call is independent

## Guards: Semantic Validation

Guards extend Pydantic's structural validation with semantic checks:

```python
@spell(model="fast")
@guard.input(validate_not_harmful)   # Semantic: Is the input safe?
@guard.output(check_no_competitors)  # Semantic: Does output mention competitors?
def respond(query: str) -> str: ...
```

Guards always raise on failure (RaiseStrategy) - they are validation, not error recovery. For error recovery, use spell-level `on_fail` strategies.

## Failure Strategies

When LLM output fails validation, spells support multiple recovery strategies:

- **Retry** (default): PydanticAI retries with validation error context
- **Escalate**: Try a more capable model
- **Fallback**: Return a default value
- **Custom**: Application-specific error handling

This allows graceful degradation without cluttering business logic with try/catch blocks.
