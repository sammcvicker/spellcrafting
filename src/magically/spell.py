from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, ParamSpec, Sequence, TypeVar, overload

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

P = ParamSpec("P")
T = TypeVar("T")


def _build_user_prompt(func: Callable[..., Any], args: tuple, kwargs: dict) -> str:
    """Build user prompt from function arguments."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    parts = []
    for name, value in bound.arguments.items():
        parts.append(f"{name}: {value!r}")

    return "\n".join(parts)


@overload
def spell(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: str = "early",
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def spell(
    func: Callable[P, T] | None = None,
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: str = "early",
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that turns a function into an LLM-powered spell.

    The function's docstring becomes the system prompt.
    The function's return type annotation becomes the output schema.
    Function arguments are passed to the LLM as the user message.

    Args:
        func: The function to decorate (when used without parentheses)
        model: LLM model to use (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet')
        model_settings: Model settings like temperature, max_tokens
        retries: Number of retries for output validation failures
        tools: Additional tool functions the agent can use
        end_strategy: 'early' (default) or 'exhaustive' for tool call handling

    Example:
        @spell
        def summarize(text: str) -> Summary:
            '''Summarize the given text into key points.'''
            ...

        @spell(model="anthropic:claude-sonnet", retries=2)
        def analyze(data: str) -> Analysis:
            '''Analyze the data and return structured insights.'''
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        # Extract system prompt from docstring
        system_prompt = inspect.getdoc(fn) or ""

        # Extract return type for output validation
        hints = fn.__annotations__
        output_type = hints.get("return", str)

        # Handle None return type
        if output_type is type(None):
            output_type = str

        # Create the agent
        agent: Agent[None, Any] = Agent(
            model=model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=retries,
            tools=list(tools),
            end_strategy=end_strategy,  # type: ignore[arg-type]
            model_settings=model_settings,
        )

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            user_prompt = _build_user_prompt(fn, args, kwargs)
            result = agent.run_sync(user_prompt)
            return result.output  # type: ignore[return-value]

        # Store agent for testing/introspection
        wrapper._agent = agent  # type: ignore[attr-defined]
        wrapper._original_func = fn  # type: ignore[attr-defined]

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
