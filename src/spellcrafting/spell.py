from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import threading
import time
import warnings
from collections import OrderedDict
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import Any, Callable, ParamSpec, Protocol, TypeVar, overload, runtime_checkable

# Module-level logger for debug messages about internal operations
_logger = logging.getLogger(__name__)

from spellcrafting._pydantic_ai import (
    Agent,
    EndStrategy,
    ModelSettings,
    UnexpectedModelBehavior,
    ValidationError,
)

from spellcrafting.config import Config, SpellcraftingConfigError, ModelConfig
from spellcrafting.on_fail import (
    OnFailStrategy,
    RetryStrategy,
    EscalateStrategy,
    FallbackStrategy,
    CustomStrategy,
)
from spellcrafting.guard import GuardContext, GuardError, GuardExecutor
from spellcrafting.logging import (
    SpellExecutionLog,
    ToolCallLog,
    TokenUsage,
    ValidationMetrics,
    _emit_log,
    capture_execution_log,
    current_trace,
    estimate_cost,
    get_logging_config,
    is_capture_mode,
    trace_context,
)
from spellcrafting.result import SpellResult

P = ParamSpec("P")
T = TypeVar("T")

# Type aliases for improved readability
# The cache maps (spell_id, config_hash) -> Agent instance
# This enables reusing agents when the same spell is called with the same config
SpellId = int
ConfigHash = int
AgentCacheKey = tuple[SpellId, ConfigHash]
# Note: CachedAgent uses Any for the output type because the cache holds agents
# with heterogeneous output types (each spell can have a different return type).
# Using a TypeVar would require the cache to be generic over output type, which
# is not possible since a single cache instance stores agents for all spells.
# The type safety for individual spell outputs is maintained at the spell wrapper
# level through the T TypeVar in SpellWrapper/AsyncSpellWrapper protocols.
CachedAgent = Agent[None, Any]


# ---------------------------------------------------------------------------
# Protocol types for spell wrappers (issue #26, #27)
# ---------------------------------------------------------------------------


@runtime_checkable
class SpellWrapper(Protocol[P, T]):
    """Protocol for sync functions decorated with @spell.

    This protocol provides type safety for the additional attributes and methods
    added by the @spell decorator to synchronous functions.

    Attributes:
        _original_func: The original undecorated function
        _is_spell_wrapper: Marker to detect if function is a spell wrapper
        _model_alias: The model alias specified in @spell (may be None)
        _system_prompt: The effective system prompt (from docstring or explicit)
        _output_type: The output type extracted from return annotation
        _retries: Number of retries configured
        _spell_id: Unique identifier for this spell instance
        _is_async: Whether the wrapped function is async (False for sync)
        _on_fail: The configured on_fail strategy

    Methods:
        __call__: Call the spell with arguments
        with_metadata: Call the spell and return SpellResult with execution metadata
        _resolve_model_and_settings: Resolve model alias at call time
    """

    _original_func: Callable[P, T]
    _is_spell_wrapper: bool
    _model_alias: str | None
    _system_prompt: str
    _output_type: type[T]
    _retries: int
    _spell_id: int
    _is_async: bool
    _on_fail: OnFailStrategy | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Execute the spell with the given arguments."""
        ...

    def with_metadata(self, *args: P.args, **kwargs: P.kwargs) -> SpellResult[T]:
        """Execute the spell and return output with execution metadata."""
        ...

    def _resolve_model_and_settings(self) -> tuple[str | None, ModelSettings | None, int]:
        """Resolve model alias and merge settings at call time."""
        ...


@runtime_checkable
class AsyncSpellWrapper(Protocol[P, T]):
    """Protocol for async functions decorated with @spell.

    This protocol provides type safety for the additional attributes and methods
    added by the @spell decorator to asynchronous functions.

    Attributes:
        _original_func: The original undecorated function
        _is_spell_wrapper: Marker to detect if function is a spell wrapper
        _model_alias: The model alias specified in @spell (may be None)
        _system_prompt: The effective system prompt (from docstring or explicit)
        _output_type: The output type extracted from return annotation
        _retries: Number of retries configured
        _spell_id: Unique identifier for this spell instance
        _is_async: Whether the wrapped function is async (True for async)
        _on_fail: The configured on_fail strategy

    Methods:
        __call__: Call the spell with arguments (returns Awaitable)
        with_metadata: Call the spell and return SpellResult with execution metadata
        _resolve_model_and_settings: Resolve model alias at call time
    """

    _original_func: Callable[P, Awaitable[T]]
    _is_spell_wrapper: bool
    _model_alias: str | None
    _system_prompt: str
    _output_type: type[T]
    _retries: int
    _spell_id: int
    _is_async: bool
    _on_fail: OnFailStrategy | None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[T]:
        """Execute the spell with the given arguments."""
        ...

    def with_metadata(self, *args: P.args, **kwargs: P.kwargs) -> Awaitable[SpellResult[T]]:
        """Execute the spell and return output with execution metadata."""
        ...

    def _resolve_model_and_settings(self) -> tuple[str | None, ModelSettings | None, int]:
        """Resolve model alias and merge settings at call time."""
        ...

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
        >>> from spellcrafting import clear_agent_cache
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
        >>> from spellcrafting import get_cache_stats
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
        >>> from spellcrafting import set_cache_max_size
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


def _build_user_prompt(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> str:
    """Build user prompt from function arguments."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    parts = []
    for name, value in bound.arguments.items():
        parts.append(f"{name}: {value!r}")

    return "\n".join(parts)


def _extract_input_args(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
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
    except (AttributeError, TypeError) as e:
        # Result may not have usage() method, or usage may have unexpected format
        _logger.debug("Could not extract token usage from result: %s", e)
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
    tools: list[Callable[..., Any]], log: SpellExecutionLog
) -> list[Callable[..., Any]]:
    """Wrap tools to capture call logs.

    Args:
        tools: List of tool functions to wrap
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
    except SpellcraftingConfigError:
        raise SpellcraftingConfigError(
            f"Escalation model alias '{escalate_model}' could not be resolved. "
            f"Define it in pyproject.toml or provide via Config context."
        )

    return model_config.model, model_config.to_model_settings()


# ---------------------------------------------------------------------------
# OnFailContext dataclass to reduce parameter count (issue #8, #119)
# ---------------------------------------------------------------------------


@dataclass
class OnFailContext:
    """Context for handling on_fail strategies.

    This dataclass bundles the parameters needed for on_fail handling,
    reducing the parameter count of _handle_on_fail_sync/async from 10 to 3.
    """

    error: Exception
    """The error that triggered the on_fail strategy."""
    on_fail: OnFailStrategy
    """The on_fail strategy to execute."""
    user_prompt: str
    """The user prompt that was sent to the LLM."""
    output_type: type[Any]
    """The expected output type for the spell."""
    system_prompt: str
    """The system prompt for the spell."""
    tools: list[Callable[..., Any]]
    """Tools available for the spell."""
    end_strategy: EndStrategy
    """The end strategy for tool calls."""
    input_args: dict[str, Any]
    """The input arguments passed to the spell."""
    spell_name: str
    """The name of the spell function."""
    model_alias: str | None
    """The model alias specified in @spell (may be None)."""


# ---------------------------------------------------------------------------
# Shared helper functions for sync/async wrapper DRY (issue #24)
# ---------------------------------------------------------------------------


def _track_on_fail_strategy(
    validation_metrics: ValidationMetrics | None,
    error: Exception,
    on_fail: OnFailStrategy,
) -> None:
    """Track validation error and on_fail strategy in metrics.

    This helper reduces duplication between sync and async wrappers by
    centralizing the validation tracking logic.
    """
    if validation_metrics is None:
        return

    validation_metrics.pydantic_errors.append(str(error))
    if isinstance(on_fail, EscalateStrategy):
        validation_metrics.on_fail_triggered = "escalate"
        validation_metrics.escalated_to_model = on_fail.model
    elif isinstance(on_fail, FallbackStrategy):
        validation_metrics.on_fail_triggered = "fallback"
    elif isinstance(on_fail, CustomStrategy):
        validation_metrics.on_fail_triggered = "custom"
    elif isinstance(on_fail, RetryStrategy):
        validation_metrics.on_fail_triggered = "retry"


def _create_execution_log(
    fn: Callable[..., Any],
    spell_id: int,
    ctx: Any,  # TraceContext
    resolved_model: str | None,
    model_alias: str | None,
    input_args: dict[str, Any],
    validation_metrics: ValidationMetrics | None,
) -> SpellExecutionLog:
    """Create a SpellExecutionLog with common parameters.

    This helper reduces duplication between sync and async wrappers.
    """
    return SpellExecutionLog(
        spell_name=fn.__name__,
        spell_id=spell_id,
        trace_id=ctx.trace_id,
        span_id=ctx.span_id,
        parent_span_id=ctx.parent_span_id,
        model=resolved_model or "",
        model_alias=model_alias,
        input_args=input_args,
        validation=validation_metrics,
    )


def _create_logging_agent(
    tools: list[Callable[..., Any]] | None,
    log: SpellExecutionLog,
    agent: Agent[None, Any],
    resolved_model: str | None,
    output_type: type,
    effective_system_prompt: str,
    retries: int,
    end_strategy: EndStrategy,
    resolved_settings: ModelSettings | None,
) -> Agent[None, Any]:
    """Create a logging agent with wrapped tools, or return the existing agent.

    This helper reduces duplication between sync and async wrappers.
    """
    if tools:
        wrapped_tools = _wrap_tools_for_logging(tools, log)
        return Agent(
            model=resolved_model,
            output_type=output_type,
            system_prompt=effective_system_prompt,
            retries=retries,
            tools=wrapped_tools,
            end_strategy=end_strategy,
            model_settings=resolved_settings,
        )
    return agent


def _finalize_log_success(
    log: SpellExecutionLog,
    result: Any,
    resolved_model: str | None,
    output: Any,
) -> None:
    """Finalize log on successful execution.

    This helper reduces duplication between sync and async wrappers.
    """
    if result is not None:
        log.token_usage = _extract_token_usage(result)
        log.cost_estimate = estimate_cost(resolved_model or "", log.token_usage)
    log.finalize(success=True, output=output)


def _process_captured_result(
    output: T,
    captured_log: SpellExecutionLog | None,
    caller_trace_id: str | None,
    resolve_model_fn: Callable[[], tuple[str | None, Any, int]],
) -> SpellResult[T]:
    """Process captured execution log into SpellResult.

    This helper reduces duplication between sync and async with_metadata methods.

    Args:
        output: The spell output value
        captured_log: The captured execution log (may be None)
        caller_trace_id: The caller's trace ID (None if no active trace)
        resolve_model_fn: Function to resolve model and settings

    Returns:
        SpellResult populated from the log or with fallback values
    """
    if captured_log:
        result = SpellResult.from_execution_log(output, captured_log)
        # Only include trace_id if caller had an active trace context
        # (the logging path creates its own internal trace context)
        if caller_trace_id is None:
            result = SpellResult(
                output=result.output,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                model_used=result.model_used,
                attempt_count=result.attempt_count,
                duration_ms=result.duration_ms,
                cost_estimate=result.cost_estimate,
                trace_id=None,
            )
        return result

    # Fallback if no log was captured (should not happen in normal use)
    resolved_model, _, _ = resolve_model_fn()
    return SpellResult(
        output=output,
        model_used=resolved_model or "",
    )


# ---------------------------------------------------------------------------
# _SpellConfig: Extracted configuration class to reduce nesting (issue #135)
# ---------------------------------------------------------------------------


@dataclass
class _SpellConfig:
    """Configuration for a spell instance.

    This class extracts configuration from the spell decorator's closure,
    making the code more debuggable by reducing nested function depth.
    All configuration is stored as instance attributes rather than captured
    in closures.

    Attributes:
        fn: The original decorated function
        model: Model alias or literal (may be None)
        model_settings: Additional model settings
        effective_system_prompt: The resolved system prompt
        output_type: The return type of the spell
        retries: Number of retry attempts
        tools: List of tool functions
        end_strategy: Strategy for handling tool calls
        on_fail: Strategy for validation failures
        spell_id: Unique identifier for this spell
        is_async: Whether the decorated function is async
    """

    fn: Callable[..., Any]
    model: str | None
    model_settings: ModelSettings | None
    effective_system_prompt: str
    output_type: type[Any]
    retries: int
    tools: list[Callable[..., Any]] | None
    end_strategy: EndStrategy
    on_fail: OnFailStrategy | None
    spell_id: int = field(default=0)  # Set after wrapper is created
    is_async: bool = field(default=False)

    def resolve_model_and_settings(self) -> tuple[str | None, ModelSettings | None, int]:
        """Resolve model alias and merge settings at call time.

        Returns:
            Tuple of (resolved_model, resolved_settings, config_hash)
        """
        if self.model is None:
            config_hash = hash((None, _settings_hash(self.model_settings)))
            return None, self.model_settings, config_hash

        # Literal model (contains :) - use as-is
        if _is_literal_model(self.model):
            config_hash = hash((self.model, _settings_hash(self.model_settings)))
            return self.model, self.model_settings, config_hash

        # Alias - resolve via current config
        config = Config.current()
        try:
            model_config = config.resolve(self.model)
        except SpellcraftingConfigError:
            raise SpellcraftingConfigError(
                f"Model alias '{self.model}' could not be resolved. "
                f"Define it in pyproject.toml or provide via Config context."
            )

        # Get base settings from ModelConfig
        config_settings = model_config.to_model_settings()

        # Merge with explicit model_settings (explicit takes precedence)
        if self.model_settings:
            resolved_settings: dict[str, Any] = dict(config_settings) if config_settings else {}
            for key, value in self.model_settings.items():
                if value is not None:
                    resolved_settings[key] = value
            final_settings = ModelSettings(**resolved_settings) if resolved_settings else None
        else:
            final_settings = config_settings

        # Hash based on resolved ModelConfig + explicit overrides
        config_hash = hash((hash(model_config), _settings_hash(self.model_settings)))
        return model_config.model, final_settings, config_hash

    def get_or_create_agent(
        self,
        config_hash: int,
        resolved_model: str | None,
        resolved_settings: ModelSettings | None,
    ) -> Agent[None, Any]:
        """Get agent from cache or create a new one."""
        cache_key = (self.spell_id, config_hash)
        agent = _agent_cache.get(cache_key)

        if agent is None:
            agent = Agent(
                model=resolved_model,
                output_type=self.output_type,
                system_prompt=self.effective_system_prompt,
                retries=self.retries,
                tools=self.tools or [],
                end_strategy=self.end_strategy,
                model_settings=resolved_settings,
            )
            _agent_cache.set(cache_key, agent)

        return agent

    def create_on_fail_context(
        self,
        error: Exception,
        user_prompt: str,
        input_args: dict[str, Any],
    ) -> OnFailContext:
        """Create an OnFailContext for handling validation failures."""
        return OnFailContext(
            error=error,
            on_fail=self.on_fail,  # type: ignore[arg-type]
            user_prompt=user_prompt,
            output_type=self.output_type,
            system_prompt=self.effective_system_prompt,
            tools=self.tools or [],
            end_strategy=self.end_strategy,
            input_args=input_args,
            spell_name=self.fn.__name__,
            model_alias=self.model,
        )


def _handle_on_fail_sync(ctx: OnFailContext) -> Any:
    """Handle on_fail strategy for sync execution.

    Args:
        ctx: OnFailContext containing all parameters for on_fail handling.

    Returns:
        The result from the on_fail strategy, or raises ValidationError.
    """
    if isinstance(ctx.on_fail, RetryStrategy):
        # Default behavior - just re-raise wrapped, PydanticAI already handled retries
        raise ValidationError(str(ctx.error), original_error=ctx.error) from ctx.error

    if isinstance(ctx.on_fail, FallbackStrategy):
        # Return the default value
        return ctx.on_fail.default

    if isinstance(ctx.on_fail, CustomStrategy):
        # Call the custom handler
        handler_context = {
            "spell_name": ctx.spell_name,
            "model": ctx.model_alias,
            "input_args": ctx.input_args,
        }
        return ctx.on_fail.handler(ctx.error, 1, handler_context)

    if isinstance(ctx.on_fail, EscalateStrategy):
        # Create a new agent with the escalated model
        escalated_model, escalated_settings = _resolve_escalation_model(ctx.on_fail.model)
        escalated_agent = Agent(
            model=escalated_model,
            output_type=ctx.output_type,
            system_prompt=ctx.system_prompt,
            retries=ctx.on_fail.retries,
            tools=ctx.tools,
            end_strategy=ctx.end_strategy,
            model_settings=escalated_settings,
        )
        result = escalated_agent.run_sync(ctx.user_prompt)
        return result.output

    # Unknown strategy type - re-raise wrapped
    raise ValidationError(str(ctx.error), original_error=ctx.error) from ctx.error


async def _handle_on_fail_async(ctx: OnFailContext) -> Any:
    """Handle on_fail strategy for async execution.

    Args:
        ctx: OnFailContext containing all parameters for on_fail handling.

    Returns:
        The result from the on_fail strategy, or raises ValidationError.
    """
    if isinstance(ctx.on_fail, RetryStrategy):
        # Default behavior - just re-raise wrapped, PydanticAI already handled retries
        raise ValidationError(str(ctx.error), original_error=ctx.error) from ctx.error

    if isinstance(ctx.on_fail, FallbackStrategy):
        # Return the default value
        return ctx.on_fail.default

    if isinstance(ctx.on_fail, CustomStrategy):
        # Call the custom handler
        handler_context = {
            "spell_name": ctx.spell_name,
            "model": ctx.model_alias,
            "input_args": ctx.input_args,
        }
        result = ctx.on_fail.handler(ctx.error, 1, handler_context)
        # Support async handlers
        if asyncio.iscoroutine(result):
            return await result
        return result

    if isinstance(ctx.on_fail, EscalateStrategy):
        # Create a new agent with the escalated model
        escalated_model, escalated_settings = _resolve_escalation_model(ctx.on_fail.model)
        escalated_agent = Agent(
            model=escalated_model,
            output_type=ctx.output_type,
            system_prompt=ctx.system_prompt,
            retries=ctx.on_fail.retries,
            tools=ctx.tools,
            end_strategy=ctx.end_strategy,
            model_settings=escalated_settings,
        )
        result = await escalated_agent.run(ctx.user_prompt)
        return result.output

    # Unknown strategy type - re-raise wrapped
    raise ValidationError(str(ctx.error), original_error=ctx.error) from ctx.error


# Overloads for sync functions - use SpellWrapper Protocol for proper typing
@overload
def spell(func: Callable[P, T]) -> SpellWrapper[P, T]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: list[Callable[..., Any]] | None = None,
    end_strategy: EndStrategy = "early",
    on_fail: OnFailStrategy | None = None,
) -> Callable[[Callable[P, T]], SpellWrapper[P, T]]: ...


# Overloads for async functions - use AsyncSpellWrapper Protocol for proper typing
@overload
def spell(func: Callable[P, Awaitable[T]]) -> AsyncSpellWrapper[P, T]: ...


@overload
def spell(
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: list[Callable[..., Any]] | None = None,
    end_strategy: EndStrategy = "early",
    on_fail: OnFailStrategy | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], AsyncSpellWrapper[P, T]]: ...


def spell(
    func: Callable[P, T] | None = None,
    *,
    model: str | None = None,
    model_settings: ModelSettings | None = None,
    system_prompt: str | None = None,
    retries: int = 1,
    tools: list[Callable[..., Any]] | None = None,
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

    Raises:
        ValueError: If retries is negative or end_strategy is invalid.

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
    # Input validation (issue #176)
    if retries < 0:
        raise ValueError(f"retries must be non-negative, got {retries}")
    if end_strategy not in ("early", "exhaustive"):
        raise ValueError(
            f"Invalid end_strategy {end_strategy!r}. Must be 'early' or 'exhaustive'"
        )

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        # Use explicit system_prompt if provided, otherwise extract from docstring
        effective_system_prompt = system_prompt if system_prompt is not None else (inspect.getdoc(fn) or "")

        # Extract return type for output validation (issue #28: validate output_type)
        hints = fn.__annotations__
        if "return" not in hints:
            warnings.warn(
                f"Function '{fn.__name__}' has no return type annotation, defaulting to str. "
                f"Add a return type annotation (e.g., -> str, -> MyModel) for explicit typing.",
                UserWarning,
                stacklevel=3,
            )
        output_type = hints.get("return", str)

        # Handle None return type - raise error since spells must return a value
        # Check both type(None) (from -> type(None)) and None (from -> None)
        if output_type is type(None) or output_type is None:
            raise TypeError(
                f"Function '{fn.__name__}' has None return type. "
                f"Spells must return a value. Use -> str or a Pydantic model."
            )

        # Definition-time warning: check if alias exists in file config
        if model is not None and not _is_literal_model(model):
            file_config = Config.from_file()
            if model not in file_config.models:
                warnings.warn(
                    f"Model alias '{model}' not found in pyproject.toml. "
                    f"It must be provided via Config context or set_as_default().",
                    stacklevel=3,
                )

        # Create spell configuration object to reduce closure nesting (issue #135)
        # This moves methods out of nested closures into a class, improving debuggability
        spell_config = _SpellConfig(
            fn=fn,
            model=model,
            model_settings=model_settings,
            effective_system_prompt=effective_system_prompt,
            output_type=output_type,
            retries=retries,
            tools=tools,
            end_strategy=end_strategy,
            on_fail=on_fail,
            is_async=asyncio.iscoroutinefunction(fn),
        )

        # Check if the decorated function is async
        is_async = spell_config.is_async

        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                resolved_model, resolved_settings, config_hash = spell_config.resolve_model_and_settings()
                agent = spell_config.get_or_create_agent(config_hash, resolved_model, resolved_settings)

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Check logging config early to decide which guard runners to use
                # Also take logging path if we're in capture mode (for with_metadata)
                logging_config = get_logging_config()
                logging_enabled = logging_config.enabled or is_capture_mode()

                # Initialize validation metrics for tracking
                validation_metrics: ValidationMetrics | None = None
                if logging_enabled:
                    validation_metrics = ValidationMetrics()

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
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
                        if spell_config.on_fail is not None:
                            on_fail_ctx = spell_config.create_on_fail_context(e, user_prompt, input_args)
                            output = await _handle_on_fail_async(on_fail_ctx)
                        else:
                            raise ValidationError(str(e), original_error=e) from e

                    # Run output guards if present
                    if guard_config and guard_config.output_guards:
                        guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
                        output = await GuardExecutor.run_output_guards_async(
                            guard_config, output, guard_context
                        )

                    return output  # type: ignore[return-value]

                # Logging enabled - use helper functions to reduce duplication (issue #24)
                with trace_context() as ctx:
                    log = _create_execution_log(
                        fn, spell_config.spell_id, ctx, resolved_model, spell_config.model,
                        input_args, validation_metrics,
                    )

                    # Create logging agent with wrapped tools if needed
                    logging_agent = _create_logging_agent(
                        spell_config.tools, log, agent, resolved_model, spell_config.output_type,
                        spell_config.effective_system_prompt, spell_config.retries,
                        spell_config.end_strategy, resolved_settings,
                    )

                    result = None
                    try:
                        try:
                            result = await logging_agent.run(user_prompt)
                            output = result.output
                        except UnexpectedModelBehavior as e:
                            if spell_config.on_fail is not None:
                                _track_on_fail_strategy(validation_metrics, e, spell_config.on_fail)
                                on_fail_ctx = spell_config.create_on_fail_context(e, user_prompt, input_args)
                                output = await _handle_on_fail_async(on_fail_ctx)
                            else:
                                # Track validation error even when no on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                raise ValidationError(str(e), original_error=e) from e

                        # Run output guards if present (with tracking)
                        if guard_config and guard_config.output_guards:
                            guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
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

                        _finalize_log_success(log, result, resolved_model, output)
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
                resolved_model, resolved_settings, config_hash = spell_config.resolve_model_and_settings()
                agent = spell_config.get_or_create_agent(config_hash, resolved_model, resolved_settings)

                # Check for guards
                guard_config = GuardExecutor.get_config(fn)
                input_args = _extract_input_args(fn, args, kwargs)

                # Check logging config early to decide which guard runners to use
                # Also take logging path if we're in capture mode (for with_metadata)
                logging_config = get_logging_config()
                logging_enabled = logging_config.enabled or is_capture_mode()

                # Initialize validation metrics for tracking
                validation_metrics: ValidationMetrics | None = None
                if logging_enabled:
                    validation_metrics = ValidationMetrics()

                # Run input guards if present
                if guard_config and guard_config.input_guards:
                    guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
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
                        if spell_config.on_fail is not None:
                            on_fail_ctx = spell_config.create_on_fail_context(e, user_prompt, input_args)
                            output = _handle_on_fail_sync(on_fail_ctx)
                        else:
                            raise ValidationError(str(e), original_error=e) from e

                    # Run output guards if present
                    if guard_config and guard_config.output_guards:
                        guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
                        output = GuardExecutor.run_output_guards(
                            guard_config, output, guard_context
                        )

                    return output  # type: ignore[return-value]

                # Logging enabled - use helper functions to reduce duplication (issue #24)
                with trace_context() as ctx:
                    log = _create_execution_log(
                        fn, spell_config.spell_id, ctx, resolved_model, spell_config.model,
                        input_args, validation_metrics,
                    )

                    # Create logging agent with wrapped tools if needed
                    logging_agent = _create_logging_agent(
                        spell_config.tools, log, agent, resolved_model, spell_config.output_type,
                        spell_config.effective_system_prompt, spell_config.retries,
                        spell_config.end_strategy, resolved_settings,
                    )

                    result = None
                    try:
                        try:
                            result = logging_agent.run_sync(user_prompt)
                            output = result.output
                        except UnexpectedModelBehavior as e:
                            if spell_config.on_fail is not None:
                                _track_on_fail_strategy(validation_metrics, e, spell_config.on_fail)
                                on_fail_ctx = spell_config.create_on_fail_context(e, user_prompt, input_args)
                                output = _handle_on_fail_sync(on_fail_ctx)
                            else:
                                # Track validation error even when no on_fail strategy
                                if validation_metrics:
                                    validation_metrics.pydantic_errors.append(str(e))
                                raise ValidationError(str(e), original_error=e) from e

                        # Run output guards if present (with tracking)
                        if guard_config and guard_config.output_guards:
                            guard_context = GuardExecutor.build_context(fn, model=spell_config.model)
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

                        _finalize_log_success(log, result, resolved_model, output)
                        return output  # type: ignore[return-value]
                    except Exception as e:
                        log.finalize(success=False, error=e)
                        raise
                    finally:
                        _emit_log(log)

            wrapper = sync_wrapper

        # Assign unique spell ID based on wrapper's id
        spell_config.spell_id = id(wrapper)

        # Store for testing/introspection
        wrapper._original_func = fn  # type: ignore[attr-defined]

        # Store marker to detect if guards are applied outside @spell
        wrapper._is_spell_wrapper = True  # type: ignore[attr-defined]
        wrapper._model_alias = spell_config.model  # type: ignore[attr-defined]
        wrapper._system_prompt = spell_config.effective_system_prompt  # type: ignore[attr-defined]
        wrapper._output_type = spell_config.output_type  # type: ignore[attr-defined]
        wrapper._retries = spell_config.retries  # type: ignore[attr-defined]
        wrapper._spell_id = spell_config.spell_id  # type: ignore[attr-defined]
        wrapper._is_async = spell_config.is_async  # type: ignore[attr-defined]
        wrapper._on_fail = spell_config.on_fail  # type: ignore[attr-defined]
        wrapper._resolve_model_and_settings = spell_config.resolve_model_and_settings  # type: ignore[attr-defined]

        # Add with_metadata method for accessing execution metadata
        # This is a thin wrapper around the main execution path that captures
        # metadata via the logging system instead of duplicating execution logic
        if is_async:
            async def with_metadata_async(*args: P.args, **kwargs: P.kwargs) -> SpellResult[T]:
                """Run the spell and return output with execution metadata."""
                caller_trace = current_trace()
                caller_trace_id = caller_trace.trace_id if caller_trace else None

                with capture_execution_log() as captured:
                    output = await async_wrapper(*args, **kwargs)

                return _process_captured_result(
                    output, captured.log, caller_trace_id, spell_config.resolve_model_and_settings
                )

            wrapper.with_metadata = with_metadata_async  # type: ignore[attr-defined]
        else:
            def with_metadata_sync(*args: P.args, **kwargs: P.kwargs) -> SpellResult[T]:
                """Run the spell and return output with execution metadata."""
                caller_trace = current_trace()
                caller_trace_id = caller_trace.trace_id if caller_trace else None

                with capture_execution_log() as captured:
                    output = sync_wrapper(*args, **kwargs)

                return _process_captured_result(
                    output, captured.log, caller_trace_id, spell_config.resolve_model_and_settings
                )

            wrapper.with_metadata = with_metadata_sync  # type: ignore[attr-defined]

        return wrapper

    if func is not None:
        return decorator(func)

    return decorator
