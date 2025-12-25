from __future__ import annotations

import asyncio
import functools
import inspect
import warnings
from collections.abc import Awaitable
from typing import Any, Callable, ParamSpec, Sequence, TypeVar, overload

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from magically.config import Config, MagicallyConfigError, ModelConfig
from magically.logging import (
    SpellExecutionLog,
    TokenUsage,
    _emit_log,
    estimate_cost,
    get_logging_config,
    trace_context,
)

P = ParamSpec("P")
T = TypeVar("T")

# Agent cache: (spell_id, config_hash) -> Agent
_agent_cache: dict[tuple[int, int], Agent[None, Any]] = {}


def _is_literal_model(model: str) -> bool:
    """Check if model is a literal (provider:model) vs an alias."""
    return ":" in model


def _settings_hash(settings: ModelSettings | None) -> int:
    """Create a hash for ModelSettings."""
    if settings is None:
        return 0
    # Convert to dict and hash the sorted items
    items = tuple(sorted((k, v) for k, v in settings.items() if v is not None))
    return hash(items)


def _build_user_prompt(func: Callable[..., Any], args: tuple, kwargs: dict) -> str:
    """Build user prompt from function arguments."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    parts = []
    for name, value in bound.arguments.items():
        parts.append(f"{name}: {value!r}")

    return "\n".join(parts)


def _extract_input_args(func: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    """Extract function arguments as a dictionary for logging."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def _extract_token_usage(result: Any) -> TokenUsage:
    """Extract token usage from PydanticAI result."""
    try:
        usage = result.usage()
        return TokenUsage(
            input_tokens=usage.request_tokens or 0,
            output_tokens=usage.response_tokens or 0,
            cache_read_tokens=0,  # PydanticAI doesn't expose cache tokens yet
            cache_write_tokens=0,
        )
    except Exception:
        return TokenUsage()


# Overloads for sync functions
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


# Overloads for async functions
@overload
def spell(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: str = "early",
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]: ...


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

        # Definition-time warning: check if alias exists in file config
        if model is not None and not _is_literal_model(model):
            file_config = Config.from_file()
            if model not in file_config.models:
                warnings.warn(
                    f"Model alias '{model}' not found in pyproject.toml. "
                    f"It must be provided via Config context or set_as_default().",
                    stacklevel=3,
                )

        def _resolve_model_and_settings() -> tuple[str | None, ModelSettings | None, int]:
            """Resolve model alias and merge settings at call time.

            Returns:
                Tuple of (resolved_model, resolved_settings, config_hash)
            """
            if model is None:
                config_hash = hash((None, _settings_hash(model_settings)))
                return None, model_settings, config_hash

            # Literal model (contains :) - use as-is
            if _is_literal_model(model):
                config_hash = hash((model, _settings_hash(model_settings)))
                return model, model_settings, config_hash

            # Alias - resolve via current config
            config = Config.current()
            try:
                model_config = config.resolve(model)
            except MagicallyConfigError:
                raise MagicallyConfigError(
                    f"Model alias '{model}' could not be resolved. "
                    f"Define it in pyproject.toml or provide via Config context."
                )

            # Build ModelSettings from ModelConfig
            resolved_settings: dict[str, Any] = {}
            if model_config.temperature is not None:
                resolved_settings["temperature"] = model_config.temperature
            if model_config.max_tokens is not None:
                resolved_settings["max_tokens"] = model_config.max_tokens
            if model_config.top_p is not None:
                resolved_settings["top_p"] = model_config.top_p
            if model_config.timeout is not None:
                resolved_settings["timeout"] = model_config.timeout

            # Merge with explicit model_settings (explicit takes precedence)
            if model_settings:
                for key, value in model_settings.items():
                    if value is not None:
                        resolved_settings[key] = value

            final_settings = ModelSettings(**resolved_settings) if resolved_settings else None

            # Hash based on resolved ModelConfig + explicit overrides
            config_hash = hash((hash(model_config), _settings_hash(model_settings)))
            return model_config.model, final_settings, config_hash

        # Unique ID for this spell (using id of the wrapper after creation)
        spell_id: int = 0  # Will be set after wrapper is created

        def _get_or_create_agent(config_hash: int, resolved_model: str | None, resolved_settings: ModelSettings | None) -> Agent[None, Any]:
            """Get agent from cache or create a new one."""
            cache_key = (spell_id, config_hash)
            agent = _agent_cache.get(cache_key)

            if agent is None:
                agent = Agent(
                    model=resolved_model,
                    output_type=output_type,
                    system_prompt=system_prompt,
                    retries=retries,
                    tools=list(tools),
                    end_strategy=end_strategy,  # type: ignore[arg-type]
                    model_settings=resolved_settings,
                )
                _agent_cache[cache_key] = agent

            return agent

        # Check if the decorated function is async
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
                agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)
                user_prompt = _build_user_prompt(fn, args, kwargs)

                # Fast path: no logging overhead
                logging_config = get_logging_config()
                if not logging_config.enabled:
                    result = await agent.run(user_prompt)
                    return result.output  # type: ignore[return-value]

                # Logging enabled
                with trace_context() as ctx:
                    log = SpellExecutionLog(
                        spell_name=fn.__name__,
                        spell_id=spell_id,
                        trace_id=ctx.trace_id,
                        span_id=ctx.span_id,
                        parent_span_id=ctx.parent_span_id,
                        model=resolved_model or "",
                        model_alias=model,
                        input_args=_extract_input_args(fn, args, kwargs),
                    )

                    try:
                        result = await agent.run(user_prompt)
                        log.token_usage = _extract_token_usage(result)
                        log.cost_estimate = estimate_cost(resolved_model or "", log.token_usage)
                        log.finalize(success=True, output=result.output)
                        return result.output  # type: ignore[return-value]
                    except Exception as e:
                        log.finalize(success=False, error=e)
                        raise
                    finally:
                        _emit_log(log)

            wrapper = async_wrapper  # type: ignore[assignment]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
                agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)
                user_prompt = _build_user_prompt(fn, args, kwargs)

                # Fast path: no logging overhead
                logging_config = get_logging_config()
                if not logging_config.enabled:
                    result = agent.run_sync(user_prompt)
                    return result.output  # type: ignore[return-value]

                # Logging enabled
                with trace_context() as ctx:
                    log = SpellExecutionLog(
                        spell_name=fn.__name__,
                        spell_id=spell_id,
                        trace_id=ctx.trace_id,
                        span_id=ctx.span_id,
                        parent_span_id=ctx.parent_span_id,
                        model=resolved_model or "",
                        model_alias=model,
                        input_args=_extract_input_args(fn, args, kwargs),
                    )

                    try:
                        result = agent.run_sync(user_prompt)
                        log.token_usage = _extract_token_usage(result)
                        log.cost_estimate = estimate_cost(resolved_model or "", log.token_usage)
                        log.finalize(success=True, output=result.output)
                        return result.output  # type: ignore[return-value]
                    except Exception as e:
                        log.finalize(success=False, error=e)
                        raise
                    finally:
                        _emit_log(log)

            wrapper = sync_wrapper

        # Assign unique spell ID based on wrapper's id
        spell_id = id(wrapper)

        # Store for testing/introspection
        wrapper._original_func = fn  # type: ignore[attr-defined]
        wrapper._model_alias = model  # type: ignore[attr-defined]
        wrapper._system_prompt = system_prompt  # type: ignore[attr-defined]
        wrapper._output_type = output_type  # type: ignore[attr-defined]
        wrapper._retries = retries  # type: ignore[attr-defined]
        wrapper._spell_id = spell_id  # type: ignore[attr-defined]
        wrapper._is_async = is_async  # type: ignore[attr-defined]
        wrapper._resolve_model_and_settings = _resolve_model_and_settings  # type: ignore[attr-defined]

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
