from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, ParamSpec, Sequence, TypeVar, overload

from magically._pydantic_ai import (
    Agent,
    EndStrategy,
    ModelSettings,
    UnexpectedModelBehavior,
    ValidationError,
)

from magically.config import Config, MagicallyConfigError, ModelConfig
from magically.on_fail import (
    OnFailStrategy,
    RetryStrategy,
    EscalateStrategy,
    FallbackStrategy,
    CustomStrategy,
)
from magically.guard import GuardError, GuardExecutor
from magically.logging import (
    SpellExecutionLog,
    ToolCallLog,
    TokenUsage,
    ValidationMetrics,
    _emit_log,
    current_trace,
    estimate_cost,
    get_logging_config,
    trace_context,
)
from magically.result import SpellResult

P = ParamSpec("P")
T = TypeVar("T")

# Type aliases for improved readability
# The cache maps (spell_id, config_hash) -> Agent instance
# This enables reusing agents when the same spell is called with the same config
SpellId = int
ConfigHash = int
AgentCacheKey = tuple[SpellId, ConfigHash]
CachedAgent = Agent[None, Any]

# Default maximum cache size (can be configured via set_cache_max_size)
_DEFAULT_CACHE_MAX_SIZE = 100


@dataclass
class CacheStats:
    """Statistics about the agent cache."""

    size: int
    """Current number of agents in the cache."""
    max_size: int
    """Maximum number of agents allowed in the cache."""
    hits: int
    """Number of cache hits (agent reused)."""
    misses: int
    """Number of cache misses (new agent created)."""
    evictions: int
    """Number of agents evicted due to cache full."""


class _LRUAgentCache:
    """Thread-safe LRU cache for Agent instances.

    This cache automatically evicts least-recently-used agents when
    the maximum size is reached, preventing unbounded memory growth.
    """

    def __init__(self, max_size: int = _DEFAULT_CACHE_MAX_SIZE):
        self._cache: OrderedDict[AgentCacheKey, CachedAgent] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: AgentCacheKey) -> CachedAgent | None:
        """Get an agent from the cache, moving it to most-recently-used."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def set(self, key: AgentCacheKey, agent: CachedAgent) -> None:
        """Add an agent to the cache, evicting LRU if necessary."""
        with self._lock:
            # If max_size is 0, caching is disabled
            if self._max_size == 0:
                return

            if key in self._cache:
                # Update existing and move to end
                self._cache.move_to_end(key)
                self._cache[key] = agent
                return

            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

            self._cache[key] = agent

    def clear(self) -> int:
        """Clear all agents from the cache. Returns number of agents cleared."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def remove_by_spell_id(self, spell_id: int) -> int:
        """Remove all cached agents for a specific spell ID.

        Args:
            spell_id: The spell ID to remove entries for.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            keys_to_remove = [k for k in self._cache if k[0] == spell_id]
            for key in keys_to_remove:
                del self._cache[key]
            return len(keys_to_remove)

    def stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            return CacheStats(
                size=len(self._cache),
                max_size=self._max_size,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    def set_max_size(self, max_size: int) -> None:
        """Set the maximum cache size. Evicts entries if necessary."""
        if max_size < 0:
            raise ValueError("max_size must be non-negative")
        with self._lock:
            self._max_size = max_size
            # Evict if necessary
            while len(self._cache) > self._max_size and self._max_size > 0:
                self._cache.popitem(last=False)
                self._evictions += 1

    @property
    def max_size(self) -> int:
        """Get the current maximum cache size."""
        with self._lock:
            return self._max_size


# Global agent cache instance
_agent_cache = _LRUAgentCache()


def clear_agent_cache() -> int:
    """Clear all cached agents.

    This is useful for:
    - Freeing memory in long-running processes
    - Resetting state between tests
    - Forcing agent recreation after config changes

    Returns:
        Number of agents that were cleared from the cache.

    Example:
        >>> from magically import clear_agent_cache
        >>> cleared = clear_agent_cache()
        >>> print(f"Cleared {cleared} agents from cache")
    """
    return _agent_cache.clear()


def get_cache_stats() -> CacheStats:
    """Get statistics about the agent cache.

    Returns a CacheStats object with:
    - size: Current number of cached agents
    - max_size: Maximum cache capacity
    - hits: Number of cache hits
    - misses: Number of cache misses
    - evictions: Number of agents evicted due to cache full

    Example:
        >>> from magically import get_cache_stats
        >>> stats = get_cache_stats()
        >>> print(f"Cache: {stats.size}/{stats.max_size} agents")
        >>> print(f"Hit rate: {stats.hits / (stats.hits + stats.misses):.1%}")
    """
    return _agent_cache.stats()


def set_cache_max_size(max_size: int) -> None:
    """Set the maximum number of agents to cache.

    When the cache reaches this limit, the least-recently-used agents
    are evicted to make room for new ones.

    Args:
        max_size: Maximum number of agents to cache. Must be non-negative.
            Set to 0 to effectively disable caching.

    Raises:
        ValueError: If max_size is negative.

    Example:
        >>> from magically import set_cache_max_size
        >>> set_cache_max_size(50)  # Reduce cache size
        >>> set_cache_max_size(0)   # Disable caching
    """
    _agent_cache.set_max_size(max_size)


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
    except (AttributeError, TypeError):
        # Result may not have usage() method, or usage may have unexpected format
        return TokenUsage()


def _wrap_tool(tool: Callable[..., Any], log: SpellExecutionLog) -> Callable[..., Any]:
    """Wrap a tool function to capture call logs.

    Args:
        tool: The tool function to wrap
        log: The SpellExecutionLog to append tool calls to

    Returns:
        Wrapped function that logs tool calls
    """
    @functools.wraps(tool)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        tool_name = getattr(tool, "__name__", repr(tool))

        # Build arguments dict for logging
        try:
            sig = inspect.signature(tool)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except (ValueError, TypeError):
            # Fallback if signature inspection fails
            arguments = {"args": args, "kwargs": kwargs} if args or kwargs else {}

        try:
            result = tool(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            log.tool_calls.append(ToolCallLog(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                duration_ms=duration_ms,
                success=True,
            ))
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            log.tool_calls.append(ToolCallLog(
                tool_name=tool_name,
                arguments=arguments,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            ))
            raise

    return wrapped


def _wrap_tool_async(tool: Callable[..., Any], log: SpellExecutionLog) -> Callable[..., Any]:
    """Wrap an async tool function to capture call logs.

    Args:
        tool: The async tool function to wrap
        log: The SpellExecutionLog to append tool calls to

    Returns:
        Wrapped async function that logs tool calls
    """
    @functools.wraps(tool)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        tool_name = getattr(tool, "__name__", repr(tool))

        # Build arguments dict for logging
        try:
            sig = inspect.signature(tool)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
        except (ValueError, TypeError):
            # Fallback if signature inspection fails
            arguments = {"args": args, "kwargs": kwargs} if args or kwargs else {}

        try:
            result = await tool(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            log.tool_calls.append(ToolCallLog(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                duration_ms=duration_ms,
                success=True,
            ))
            return result
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            log.tool_calls.append(ToolCallLog(
                tool_name=tool_name,
                arguments=arguments,
                duration_ms=duration_ms,
                success=False,
                error=str(e),
            ))
            raise

    return wrapped


def _wrap_tools_for_logging(
    tools: Sequence[Callable[..., Any]], log: SpellExecutionLog
) -> list[Callable[..., Any]]:
    """Wrap tools to capture call logs.

    Args:
        tools: Sequence of tool functions to wrap
        log: The SpellExecutionLog to append tool calls to

    Returns:
        List of wrapped tool functions
    """
    wrapped_tools = []
    for tool in tools:
        if asyncio.iscoroutinefunction(tool):
            wrapped_tools.append(_wrap_tool_async(tool, log))
        else:
            wrapped_tools.append(_wrap_tool(tool, log))
    return wrapped_tools


def _resolve_escalation_model(
    escalate_model: str,
) -> tuple[str, ModelSettings | None]:
    """Resolve the escalation model alias to actual model and settings."""
    if _is_literal_model(escalate_model):
        return escalate_model, None

    # Alias - resolve via current config
    config = Config.current()
    try:
        model_config = config.resolve(escalate_model)
    except MagicallyConfigError:
        raise MagicallyConfigError(
            f"Escalation model alias '{escalate_model}' could not be resolved. "
            f"Define it in pyproject.toml or provide via Config context."
        )

    return model_config.model, model_config.to_model_settings()


def _handle_on_fail_sync(
    error: Exception,
    on_fail: OnFailStrategy,
    user_prompt: str,
    output_type: type,
    system_prompt: str,
    tools: list,
    end_strategy: EndStrategy,
    input_args: dict[str, Any],
    spell_name: str,
    model_alias: str | None,
) -> Any:
    """Handle on_fail strategy for sync execution."""
    if isinstance(on_fail, RetryStrategy):
        # Default behavior - just re-raise wrapped, PydanticAI already handled retries
        raise ValidationError(str(error), original_error=error) from error

    if isinstance(on_fail, FallbackStrategy):
        # Return the default value
        return on_fail.default

    if isinstance(on_fail, CustomStrategy):
        # Call the custom handler
        context = {
            "spell_name": spell_name,
            "model": model_alias,
            "input_args": input_args,
        }
        return on_fail.handler(error, 1, context)

    if isinstance(on_fail, EscalateStrategy):
        # Create a new agent with the escalated model
        escalated_model, escalated_settings = _resolve_escalation_model(on_fail.model)
        escalated_agent = Agent(
            model=escalated_model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=on_fail.retries,
            tools=tools,
            end_strategy=end_strategy,
            model_settings=escalated_settings,
        )
        result = escalated_agent.run_sync(user_prompt)
        return result.output

    # Unknown strategy type - re-raise wrapped
    raise ValidationError(str(error), original_error=error) from error


async def _handle_on_fail_async(
    error: Exception,
    on_fail: OnFailStrategy,
    user_prompt: str,
    output_type: type,
    system_prompt: str,
    tools: list,
    end_strategy: EndStrategy,
    input_args: dict[str, Any],
    spell_name: str,
    model_alias: str | None,
) -> Any:
    """Handle on_fail strategy for async execution."""
    if isinstance(on_fail, RetryStrategy):
        # Default behavior - just re-raise wrapped, PydanticAI already handled retries
        raise ValidationError(str(error), original_error=error) from error

    if isinstance(on_fail, FallbackStrategy):
        # Return the default value
        return on_fail.default

    if isinstance(on_fail, CustomStrategy):
        # Call the custom handler
        context = {
            "spell_name": spell_name,
            "model": model_alias,
            "input_args": input_args,
        }
        result = on_fail.handler(error, 1, context)
        # Support async handlers
        if asyncio.iscoroutine(result):
            return await result
        return result

    if isinstance(on_fail, EscalateStrategy):
        # Create a new agent with the escalated model
        escalated_model, escalated_settings = _resolve_escalation_model(on_fail.model)
        escalated_agent = Agent(
            model=escalated_model,
            output_type=output_type,
            system_prompt=system_prompt,
            retries=on_fail.retries,
            tools=tools,
            end_strategy=end_strategy,
            model_settings=escalated_settings,
        )
        result = await escalated_agent.run(user_prompt)
        return result.output

    # Unknown strategy type - re-raise wrapped
    raise ValidationError(str(error), original_error=error) from error


# Overloads for sync functions
@overload
def spell(func: Callable[P, T]) -> Callable[P, T]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: EndStrategy = "early",
    on_fail: OnFailStrategy | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


# Overloads for async functions
@overload
def spell(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: EndStrategy = "early",
    on_fail: OnFailStrategy | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]: ...


def spell(
    func: Callable[P, T] | None = None,
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: Sequence[Callable[..., Any]] = (),
    end_strategy: EndStrategy = "early",
    on_fail: OnFailStrategy | None = None,
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator that turns a function into an LLM-powered spell.

    The function's docstring becomes the system prompt (unless overridden
    via the system_prompt parameter).
    The function's return type annotation becomes the output schema.
    Function arguments are passed to the LLM as the user message.

    IMPORTANT: Decorator order matters! @spell must be the OUTERMOST decorator,
    with guards applied INSIDE:

        @spell                    # <-- Always outermost
        @guard.input(validate)    # Input guards run before LLM
        @guard.output(check)      # Output guards run after LLM
        def my_spell(...):
            ...

    Args:
        func: The function to decorate (when used without parentheses)
        model: LLM model to use (e.g., 'openai:gpt-4o', 'anthropic:claude-sonnet')
        model_settings: Model settings like temperature, max_tokens
        system_prompt: Override the system prompt (default: use docstring).
            Useful for dynamically generated prompts or when docstring is
            used for documentation purposes.
        retries: Number of retries for output validation failures
        tools: Additional tool functions the agent can use
        end_strategy: 'early' (default) or 'exhaustive' for tool call handling
        on_fail: Strategy for handling validation failures after retries exhausted.
            - OnFail.retry() - default, retry with error in context
            - OnFail.escalate("model") - try a more capable model
            - OnFail.fallback(default) - return default value
            - OnFail.custom(handler) - custom handler function

    Example:
        @spell
        def summarize(text: str) -> Summary:
            '''Summarize the given text into key points.'''
            ...

        @spell(model="anthropic:claude-sonnet", retries=2)
        def analyze(data: str) -> Analysis:
            '''Analyze the data and return structured insights.'''
            ...

        @spell(model="fast", on_fail=OnFail.escalate("reasoning"))
        def complex_task(query: str) -> Analysis:
            '''Complex analysis that may need a better model.'''
            ...

        # Dynamic system prompt (for programmatic use cases like llm_validator)
        @spell(model="fast", system_prompt="Check if value satisfies: must be positive")
        def validate(value: str) -> ValidationResult:
            '''Docstring used for documentation, not as prompt.'''
            ...
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        # Use explicit system_prompt if provided, otherwise extract from docstring
        effective_system_prompt = system_prompt if system_prompt is not None else (inspect.getdoc(fn) or "")

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

            # Get base settings from ModelConfig
            config_settings = model_config.to_model_settings()

            # Merge with explicit model_settings (explicit takes precedence)
            if model_settings:
                resolved_settings: dict[str, Any] = dict(config_settings) if config_settings else {}
                for key, value in model_settings.items():
                    if value is not None:
                        resolved_settings[key] = value
                final_settings = ModelSettings(**resolved_settings) if resolved_settings else None
            else:
                final_settings = config_settings

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
                    system_prompt=effective_system_prompt,
                    retries=retries,
                    tools=list(tools),
                    end_strategy=end_strategy,
                    model_settings=resolved_settings,
                )
                _agent_cache.set(cache_key, agent)

            return agent

        # Check if the decorated function is async
        is_async = asyncio.iscoroutinefunction(fn)

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
                agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Check logging config early to decide which guard runners to use
                logging_config = get_logging_config()
                logging_enabled = logging_config.enabled

                # Initialize validation metrics for tracking
                validation_metrics: ValidationMetrics | None = None
                if logging_enabled:
                    validation_metrics = ValidationMetrics()

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model  # Use alias, not resolved
                    if logging_enabled and validation_metrics:
                        guard_result = await GuardExecutor.run_input_guards_tracked_async(
                            guard_config, input_args, guard_context
                        )
                        input_args = guard_result.result
                        validation_metrics.input_guards_passed = guard_result.passed
                        validation_metrics.input_guards_failed = guard_result.failed
                    else:
                        input_args = await GuardExecutor.run_input_guards_async(
                            guard_config, input_args, guard_context
                        )
                    # Rebuild user prompt with potentially transformed args
                    user_prompt = "\n".join(f"{k}: {v!r}" for k, v in input_args.items())
                else:
                    user_prompt = _build_user_prompt(fn, args, kwargs)

                # Fast path: no logging overhead
                if not logging_enabled:
                    try:
                        result = await agent.run(user_prompt)
                        output = result.output
                    except UnexpectedModelBehavior as e:
                        if on_fail is not None:
                            output = await _handle_on_fail_async(
                                e,
                                on_fail,
                                user_prompt,
                                output_type,
                                effective_system_prompt,
                                list(tools),
                                end_strategy,
                                input_args,
                                fn.__name__,
                                model,
                            )
                        else:
                            raise ValidationError(str(e), original_error=e) from e

                    # Run output guards if present
                    if guard_config and guard_config.output_guards:
                        guard_context = GuardExecutor.build_context(fn)
                        guard_context["model"] = model
                        output = await GuardExecutor.run_output_guards_async(
                            guard_config, output, guard_context
                        )

                    return output  # type: ignore[return-value]

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
                        input_args=input_args,
                        validation=validation_metrics,
                    )

                    # When logging is enabled and we have tools, create a temporary
                    # agent with wrapped tools to capture tool call logs
                    if tools:
                        wrapped_tools = _wrap_tools_for_logging(tools, log)
                        logging_agent = Agent(
                            model=resolved_model,
                            output_type=output_type,
                            system_prompt=effective_system_prompt,
                            retries=retries,
                            tools=wrapped_tools,
                            end_strategy=end_strategy,
                            model_settings=resolved_settings,
                        )
                    else:
                        logging_agent = agent

                    result = None
                    try:
                        try:
                            result = await logging_agent.run(user_prompt)
                            output = result.output
                        except UnexpectedModelBehavior as e:
                            if on_fail is not None:
                                # Track validation error and on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                    if isinstance(on_fail, EscalateStrategy):
                                        validation_metrics.on_fail_triggered = "escalate"
                                        validation_metrics.escalated_to_model = on_fail.model
                                    elif isinstance(on_fail, FallbackStrategy):
                                        validation_metrics.on_fail_triggered = "fallback"
                                    elif isinstance(on_fail, CustomStrategy):
                                        validation_metrics.on_fail_triggered = "custom"
                                    elif isinstance(on_fail, RetryStrategy):
                                        validation_metrics.on_fail_triggered = "retry"

                                output = await _handle_on_fail_async(
                                    e,
                                    on_fail,
                                    user_prompt,
                                    output_type,
                                    effective_system_prompt,
                                    list(tools),
                                    end_strategy,
                                    input_args,
                                    fn.__name__,
                                    model,
                                )
                            else:
                                # Track validation error even when no on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                raise ValidationError(str(e), original_error=e) from e

                        # Run output guards if present (with tracking)
                        if guard_config and guard_config.output_guards:
                            guard_context = GuardExecutor.build_context(fn)
                            guard_context["model"] = model
                            if validation_metrics:
                                guard_result = await GuardExecutor.run_output_guards_tracked_async(
                                    guard_config, output, guard_context
                                )
                                output = guard_result.result
                                validation_metrics.output_guards_passed = guard_result.passed
                                validation_metrics.output_guards_failed = guard_result.failed
                            else:
                                output = await GuardExecutor.run_output_guards_async(
                                    guard_config, output, guard_context
                                )

                        if result is not None:
                            log.token_usage = _extract_token_usage(result)
                            log.cost_estimate = estimate_cost(resolved_model or "", log.token_usage)
                        log.finalize(success=True, output=output)
                        return output  # type: ignore[return-value]
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

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Check logging config early to decide which guard runners to use
                logging_config = get_logging_config()
                logging_enabled = logging_config.enabled

                # Initialize validation metrics for tracking
                validation_metrics: ValidationMetrics | None = None
                if logging_enabled:
                    validation_metrics = ValidationMetrics()

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model  # Use alias, not resolved
                    if logging_enabled and validation_metrics:
                        guard_result = GuardExecutor.run_input_guards_tracked(
                            guard_config, input_args, guard_context
                        )
                        input_args = guard_result.result
                        validation_metrics.input_guards_passed = guard_result.passed
                        validation_metrics.input_guards_failed = guard_result.failed
                    else:
                        input_args = GuardExecutor.run_input_guards(
                            guard_config, input_args, guard_context
                        )
                    # Rebuild user prompt with potentially transformed args
                    user_prompt = "\n".join(f"{k}: {v!r}" for k, v in input_args.items())
                else:
                    user_prompt = _build_user_prompt(fn, args, kwargs)

                # Fast path: no logging overhead
                if not logging_enabled:
                    try:
                        result = agent.run_sync(user_prompt)
                        output = result.output
                    except UnexpectedModelBehavior as e:
                        if on_fail is not None:
                            output = _handle_on_fail_sync(
                                e,
                                on_fail,
                                user_prompt,
                                output_type,
                                effective_system_prompt,
                                list(tools),
                                end_strategy,
                                input_args,
                                fn.__name__,
                                model,
                            )
                        else:
                            raise ValidationError(str(e), original_error=e) from e

                    # Run output guards if present
                    if guard_config and guard_config.output_guards:
                        guard_context = GuardExecutor.build_context(fn)
                        guard_context["model"] = model
                        output = GuardExecutor.run_output_guards(
                            guard_config, output, guard_context
                        )

                    return output  # type: ignore[return-value]

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
                        input_args=input_args,
                        validation=validation_metrics,
                    )

                    # When logging is enabled and we have tools, create a temporary
                    # agent with wrapped tools to capture tool call logs
                    if tools:
                        wrapped_tools = _wrap_tools_for_logging(tools, log)
                        logging_agent = Agent(
                            model=resolved_model,
                            output_type=output_type,
                            system_prompt=effective_system_prompt,
                            retries=retries,
                            tools=wrapped_tools,
                            end_strategy=end_strategy,
                            model_settings=resolved_settings,
                        )
                    else:
                        logging_agent = agent

                    result = None
                    try:
                        try:
                            result = logging_agent.run_sync(user_prompt)
                            output = result.output
                        except UnexpectedModelBehavior as e:
                            if on_fail is not None:
                                # Track validation error and on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                    if isinstance(on_fail, EscalateStrategy):
                                        validation_metrics.on_fail_triggered = "escalate"
                                        validation_metrics.escalated_to_model = on_fail.model
                                    elif isinstance(on_fail, FallbackStrategy):
                                        validation_metrics.on_fail_triggered = "fallback"
                                    elif isinstance(on_fail, CustomStrategy):
                                        validation_metrics.on_fail_triggered = "custom"
                                    elif isinstance(on_fail, RetryStrategy):
                                        validation_metrics.on_fail_triggered = "retry"

                                output = _handle_on_fail_sync(
                                    e,
                                    on_fail,
                                    user_prompt,
                                    output_type,
                                    effective_system_prompt,
                                    list(tools),
                                    end_strategy,
                                    input_args,
                                    fn.__name__,
                                    model,
                                )
                            else:
                                # Track validation error even when no on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                raise ValidationError(str(e), original_error=e) from e

                        # Run output guards if present (with tracking)
                        if guard_config and guard_config.output_guards:
                            guard_context = GuardExecutor.build_context(fn)
                            guard_context["model"] = model
                            if validation_metrics:
                                guard_result = GuardExecutor.run_output_guards_tracked(
                                    guard_config, output, guard_context
                                )
                                output = guard_result.result
                                validation_metrics.output_guards_passed = guard_result.passed
                                validation_metrics.output_guards_failed = guard_result.failed
                            else:
                                output = GuardExecutor.run_output_guards(
                                    guard_config, output, guard_context
                                )

                        if result is not None:
                            log.token_usage = _extract_token_usage(result)
                            log.cost_estimate = estimate_cost(resolved_model or "", log.token_usage)
                        log.finalize(success=True, output=output)
                        return output  # type: ignore[return-value]
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

        # Store marker to detect if guards are applied outside @spell
        wrapper._is_spell_wrapper = True  # type: ignore[attr-defined]
        wrapper._model_alias = model  # type: ignore[attr-defined]
        wrapper._system_prompt = effective_system_prompt  # type: ignore[attr-defined]
        wrapper._output_type = output_type  # type: ignore[attr-defined]
        wrapper._retries = retries  # type: ignore[attr-defined]
        wrapper._spell_id = spell_id  # type: ignore[attr-defined]
        wrapper._is_async = is_async  # type: ignore[attr-defined]
        wrapper._on_fail = on_fail  # type: ignore[attr-defined]
        wrapper._resolve_model_and_settings = _resolve_model_and_settings  # type: ignore[attr-defined]

        # Add with_metadata method for accessing execution metadata
        if is_async:
            async def with_metadata_async(*args: P.args, **kwargs: P.kwargs) -> SpellResult[T]:
                """Run the spell and return output with execution metadata."""
                start_time = time.perf_counter()

                resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
                agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model
                    input_args = await GuardExecutor.run_input_guards_async(
                        guard_config, input_args, guard_context
                    )
                    user_prompt = "\n".join(f"{k}: {v!r}" for k, v in input_args.items())
                else:
                    user_prompt = _build_user_prompt(fn, args, kwargs)

                attempt_count = 1
                actual_model = resolved_model or ""

                try:
                    result = await agent.run(user_prompt)
                    output = result.output
                except UnexpectedModelBehavior as e:
                    if on_fail is not None:
                        if isinstance(on_fail, EscalateStrategy):
                            actual_model = on_fail.model
                            attempt_count += 1
                        output = await _handle_on_fail_async(
                            e,
                            on_fail,
                            user_prompt,
                            output_type,
                            effective_system_prompt,
                            list(tools),
                            end_strategy,
                            input_args,
                            fn.__name__,
                            model,
                        )
                        result = None  # No result object when on_fail handles it
                    else:
                        raise ValidationError(str(e), original_error=e) from e

                # Run output guards if present
                if guard_config and guard_config.output_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model
                    output = await GuardExecutor.run_output_guards_async(
                        guard_config, output, guard_context
                    )

                # Extract token usage from result
                input_tokens = 0
                output_tokens = 0
                if result is not None:
                    try:
                        usage = result.usage()
                        input_tokens = usage.request_tokens or 0
                        output_tokens = usage.response_tokens or 0
                    except (AttributeError, TypeError):
                        # Result may not have usage() method or unexpected format
                        pass

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Estimate cost if we have token data
                cost = None
                if actual_model and (input_tokens > 0 or output_tokens > 0):
                    cost_estimate_obj = estimate_cost(
                        actual_model,
                        TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    )
                    if cost_estimate_obj:
                        cost = cost_estimate_obj.total_cost

                # Get trace ID if available
                trace = current_trace()
                trace_id = trace.trace_id if trace else None

                return SpellResult(
                    output=output,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_used=actual_model,
                    attempt_count=attempt_count,
                    duration_ms=duration_ms,
                    cost_estimate=cost,
                    trace_id=trace_id,
                )

            wrapper.with_metadata = with_metadata_async  # type: ignore[attr-defined]
        else:
            def with_metadata_sync(*args: P.args, **kwargs: P.kwargs) -> SpellResult[T]:
                """Run the spell and return output with execution metadata."""
                start_time = time.perf_counter()

                resolved_model, resolved_settings, config_hash = _resolve_model_and_settings()
                agent = _get_or_create_agent(config_hash, resolved_model, resolved_settings)

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model
                    input_args = GuardExecutor.run_input_guards(
                        guard_config, input_args, guard_context
                    )
                    user_prompt = "\n".join(f"{k}: {v!r}" for k, v in input_args.items())
                else:
                    user_prompt = _build_user_prompt(fn, args, kwargs)

                attempt_count = 1
                actual_model = resolved_model or ""

                try:
                    result = agent.run_sync(user_prompt)
                    output = result.output
                except UnexpectedModelBehavior as e:
                    if on_fail is not None:
                        if isinstance(on_fail, EscalateStrategy):
                            actual_model = on_fail.model
                            attempt_count += 1
                        output = _handle_on_fail_sync(
                            e,
                            on_fail,
                            user_prompt,
                            output_type,
                            effective_system_prompt,
                            list(tools),
                            end_strategy,
                            input_args,
                            fn.__name__,
                            model,
                        )
                        result = None  # No result object when on_fail handles it
                    else:
                        raise ValidationError(str(e), original_error=e) from e

                # Run output guards if present
                if guard_config and guard_config.output_guards:
                    guard_context = GuardExecutor.build_context(fn)
                    guard_context["model"] = model
                    output = GuardExecutor.run_output_guards(
                        guard_config, output, guard_context
                    )

                # Extract token usage from result
                input_tokens = 0
                output_tokens = 0
                if result is not None:
                    try:
                        usage = result.usage()
                        input_tokens = usage.request_tokens or 0
                        output_tokens = usage.response_tokens or 0
                    except (AttributeError, TypeError):
                        # Result may not have usage() method or unexpected format
                        pass

                duration_ms = (time.perf_counter() - start_time) * 1000

                # Estimate cost if we have token data
                cost = None
                if actual_model and (input_tokens > 0 or output_tokens > 0):
                    cost_estimate_obj = estimate_cost(
                        actual_model,
                        TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    )
                    if cost_estimate_obj:
                        cost = cost_estimate_obj.total_cost

                # Get trace ID if available
                trace = current_trace()
                trace_id = trace.trace_id if trace else None

                return SpellResult(
                    output=output,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_used=actual_model,
                    attempt_count=attempt_count,
                    duration_ms=duration_ms,
                    cost_estimate=cost,
                    trace_id=trace_id,
                )

            wrapper.with_metadata = with_metadata_sync  # type: ignore[attr-defined]

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
