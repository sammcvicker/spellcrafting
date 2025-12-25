"""Guard decorators for input/output validation on spells.

Guards provide composable validation that runs before (input) or after (output)
spell execution. They're designed to work alongside Pydantic structural validation
for semantic/safety checks that need the full context.

Design Note: Pydantic vs Dataclass Usage
----------------------------------------
This module uses dataclasses for internal types (_GuardConfig, GuardRunResult)
because they are internal bookkeeping, not user-provided configuration.
See config.py for contrast - it uses Pydantic for parsing user configuration.

IMPORTANT: Decorator order matters! @spell must be the OUTERMOST decorator,
with guards applied INSIDE:

    @spell                    # <-- Always outermost
    @guard.input(validate)    # Input guards run before LLM
    @guard.output(check)      # Output guards run after LLM
    def my_spell(...):
        ...

Async Guard Support:
    Guards can be either sync or async functions. When used with async spells,
    async guards are properly awaited. Sync guards work in both contexts.

    async def async_validator(input_args: dict, ctx: dict) -> dict:
        result = await some_async_check(input_args)
        return input_args

    @spell(model="fast")
    @guard.input(async_validator)  # Awaited in async spell
    async def my_async_spell(text: str) -> str:
        ...

Example:
    from magically import spell, guard

    def validate_not_empty(input_args: dict, context: dict) -> dict:
        if not input_args.get("text", "").strip():
            raise ValueError("Input text cannot be empty")
        return input_args

    @spell(model="fast")
    @guard.input(validate_not_empty)
    @guard.output(lambda output, ctx: output if len(output) > 10 else raise_error())
    def summarize(text: str) -> str:
        '''Summarize the given text.'''
        ...
"""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, ParamSpec, Protocol, TypeVar

from magically.exceptions import GuardError
from magically.on_fail import OnFail, RaiseStrategy

P = ParamSpec("P")
T = TypeVar("T")


class InputGuard(Protocol):
    """Protocol for input guard functions.

    Input guards validate/transform input arguments before the LLM call.
    They receive the bound arguments as a dict and a context dict.
    They can modify and return the arguments, or raise to reject.

    Guards can be either synchronous or asynchronous:
    - Sync guards: Called directly in both sync and async spell execution
    - Async guards: Awaited when spell is async, run via asyncio when spell is sync

    Note:
        For async guards, define them as async functions::

            async def my_async_guard(input_args: dict, ctx: dict) -> dict:
                result = await some_async_validation(input_args)
                return input_args

        Async guards will be awaited when used with async spells.
    """

    def __call__(self, input_args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Validate/transform inputs. Raise to reject.

        Can also return a coroutine for async guards.
        """
        ...


class OutputGuard(Protocol):
    """Protocol for output guard functions.

    Output guards validate/transform the output after the LLM call.
    They receive the output value and a context dict.
    They can modify and return the output, or raise to reject.

    Guards can be either synchronous or asynchronous:
    - Sync guards: Called directly in both sync and async spell execution
    - Async guards: Awaited when spell is async, run via asyncio when spell is sync

    Note:
        For async guards, define them as async functions::

            async def my_async_guard(output: str, ctx: dict) -> str:
                is_valid = await some_async_check(output)
                if not is_valid:
                    raise ValueError("Invalid output")
                return output

        Async guards will be awaited when used with async spells.
    """

    def __call__(self, output: T, context: dict[str, Any]) -> T:
        """Validate/transform output. Raise to reject.

        Can also return a coroutine for async guards.
        """
        ...


# Internal marker attribute name - use get_guard_config() for public access
_GUARD_MARKER = "_magically_guards"


@dataclass(frozen=True)
class GuardContext:
    """Context passed to guard functions during execution.

    This dataclass provides typed access to execution context, replacing
    the previous untyped dict interface. Guards receive this context
    as their second argument.

    Attributes:
        spell_name: Name of the spell being executed
        model: Model alias used for the spell (may be None)
        attempt_number: Current retry attempt (1-based)

    Example:
        def my_guard(input_args: dict, ctx: GuardContext) -> dict:
            print(f"Running guard for {ctx.spell_name} on attempt {ctx.attempt_number}")
            return input_args
    """

    spell_name: str
    model: str | None = None
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for backwards compatibility with dict-based guards."""
        return {
            "spell_name": self.spell_name,
            "model": self.model,
            "attempt_number": self.attempt_number,
        }


@dataclass
class GuardConfig:
    """Guard configuration attached to decorated functions.

    This class stores the input and output guards configured on a function
    via @guard.input() and @guard.output() decorators.

    Users can introspect guard configuration using get_guard_config():

        @spell
        @guard.input(my_validator)
        def my_spell(text: str) -> str:
            ...

        config = get_guard_config(my_spell)
        if config:
            print(f"Has {len(config.input_guards)} input guards")
    """

    input_guards: list[tuple[InputGuard, RaiseStrategy]] = field(default_factory=list)
    output_guards: list[tuple[OutputGuard, RaiseStrategy]] = field(default_factory=list)


def get_guard_config(func: Callable) -> GuardConfig | None:
    """Get the guard configuration for a decorated function.

    Returns the GuardConfig if the function has guards attached,
    or None if no guards are configured.

    Args:
        func: A function that may have guards attached via @guard decorators.

    Returns:
        GuardConfig with input_guards and output_guards lists, or None.

    Example:
        @spell
        @guard.input(validate_input)
        @guard.output(check_output)
        def my_spell(text: str) -> str:
            ...

        config = get_guard_config(my_spell)
        if config:
            print(f"Input guards: {len(config.input_guards)}")
            print(f"Output guards: {len(config.output_guards)}")
    """
    return getattr(func, _GUARD_MARKER, None)


def _get_or_create_guard_config(func: Callable) -> GuardConfig:
    """Get or create guard config on a function (internal helper)."""
    if not hasattr(func, _GUARD_MARKER):
        setattr(func, _GUARD_MARKER, GuardConfig())
    return getattr(func, _GUARD_MARKER)


def _build_context(func: Callable, attempt: int = 1, model: str | None = None) -> GuardContext:
    """Build context for guard functions.

    Args:
        func: The spell function being executed.
        attempt: Current retry attempt (1-based).
        model: Model alias override. If None, uses func._model_alias if available.
    """
    spell_name = getattr(func, "__name__", "unknown")
    # Use explicit model if provided, otherwise fall back to function attribute
    model_alias = model if model is not None else getattr(func, "_model_alias", None)

    return GuardContext(
        spell_name=spell_name,
        model=model_alias,
        attempt_number=attempt,
    )


def _run_input_guards(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: GuardContext,
) -> dict[str, Any]:
    """Run input guards in order, transforming input_args."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_args = input_args
    for guard_fn, _on_fail in guards:
        try:
            current_args = guard_fn(current_args, context_dict)
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_args


def _run_output_guards(
    guards: list[tuple[OutputGuard, RaiseStrategy]],
    output: T,
    context: GuardContext,
) -> T:
    """Run output guards in order (outermost first), transforming output."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_output = output
    for guard_fn, _on_fail in guards:
        try:
            current_output = guard_fn(current_output, context_dict)
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_output


async def _run_input_guards_async(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: GuardContext,
) -> dict[str, Any]:
    """Run input guards in order, supporting async guard functions."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_args = input_args
    for guard_fn, _on_fail in guards:
        try:
            result = guard_fn(current_args, context_dict)
            if asyncio.iscoroutine(result):
                current_args = await result
            else:
                current_args = result
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_args


async def _run_output_guards_async(
    guards: list[tuple[OutputGuard, RaiseStrategy]],
    output: T,
    context: GuardContext,
) -> T:
    """Run output guards in order, supporting async guard functions."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_output = output
    for guard_fn, _on_fail in guards:
        try:
            result = guard_fn(current_output, context_dict)
            if asyncio.iscoroutine(result):
                current_output = await result
            else:
                current_output = result
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_output


# ---------------------------------------------------------------------------
# Guard runners with tracking (for logging/metrics)
# ---------------------------------------------------------------------------


@dataclass
class GuardRunResult(Generic[T]):
    """Result of running guards with tracking information.

    Type parameter T represents the result type:
    - dict[str, Any] for input guards (transformed input_args)
    - The output type for output guards (transformed output)
    """

    result: T  # The transformed input_args or output
    passed: list[str]  # Names of guards that passed
    failed: list[str]  # Names of guards that failed (if exception caught)


def _get_guard_name(guard_fn: Callable) -> str:
    """Get a human-readable name for a guard function."""
    return getattr(guard_fn, "__name__", None) or getattr(guard_fn, "__qualname__", "unknown")


def _run_input_guards_tracked(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: GuardContext,
) -> GuardRunResult[dict[str, Any]]:
    """Run input guards with tracking. Returns result and guard names that passed/failed."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_args = input_args
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            current_args = guard_fn(current_args, context_dict)
            passed.append(guard_name)
        except Exception as e:
            failed.append(guard_name)
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e

    return GuardRunResult(result=current_args, passed=passed, failed=failed)


def _run_output_guards_tracked(
    guards: list[tuple[OutputGuard, RaiseStrategy]],
    output: T,
    context: GuardContext,
) -> GuardRunResult[T]:
    """Run output guards with tracking. Returns result and guard names that passed/failed."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_output = output
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            current_output = guard_fn(current_output, context_dict)
            passed.append(guard_name)
        except Exception as e:
            failed.append(guard_name)
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e

    return GuardRunResult(result=current_output, passed=passed, failed=failed)


async def _run_input_guards_tracked_async(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: GuardContext,
) -> GuardRunResult[dict[str, Any]]:
    """Run input guards with tracking, supporting async guard functions."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_args = input_args
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            result = guard_fn(current_args, context_dict)
            if asyncio.iscoroutine(result):
                current_args = await result
            else:
                current_args = result
            passed.append(guard_name)
        except Exception as e:
            failed.append(guard_name)
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e

    return GuardRunResult(result=current_args, passed=passed, failed=failed)


async def _run_output_guards_tracked_async(
    guards: list[tuple[OutputGuard, RaiseStrategy]],
    output: T,
    context: GuardContext,
) -> GuardRunResult[T]:
    """Run output guards with tracking, supporting async guard functions."""
    # Convert to dict for backwards compatibility with dict-based guards
    context_dict = context.to_dict()
    current_output = output
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            result = guard_fn(current_output, context_dict)
            if asyncio.iscoroutine(result):
                current_output = await result
            else:
                current_output = result
            passed.append(guard_name)
        except Exception as e:
            failed.append(guard_name)
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e

    return GuardRunResult(result=current_output, passed=passed, failed=failed)


class _GuardNamespace:
    """Namespace for guard decorators.

    Usage:
        from magically import guard

        @spell(model="fast")
        @guard.input(my_input_validator)
        @guard.output(my_output_validator)
        def my_spell(...):
            ...
    """

    @staticmethod
    def input(
        guard_fn: InputGuard,
        *,
        on_fail: RaiseStrategy = OnFail.RAISE,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Add an input guard to a spell.

        Input guards run before the LLM call. They receive the function's
        bound arguments as a dict and can validate, transform, or reject.

        Execution Order:
            Guards execute in decorator order (top to bottom)::

                @spell(model="fast")
                @guard.input(first_guard)   # Runs first
                @guard.input(second_guard)  # Runs second
                def summarize(text: str) -> str:
                    ...

        Args:
            guard_fn: Function that validates/transforms inputs.
                      Signature: (input_args: dict, context: dict) -> dict
                      Raise any exception to reject.
            on_fail: Action on failure (default: RAISE)

        Returns:
            Decorator that adds the input guard.

        Example:
            def validate_length(input_args: dict, context: dict) -> dict:
                if len(input_args.get("text", "")) > 10000:
                    raise ValueError("Input too long")
                return input_args

            @spell(model="fast")
            @guard.input(validate_length)
            def summarize(text: str) -> str:
                '''Summarize.'''
                ...
        """

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            # Check if guards are being applied OUTSIDE @spell (wrong order)
            if getattr(func, "_is_spell_wrapper", False):
                func_name = getattr(func, "__name__", "unknown")
                warnings.warn(
                    f"Guard applied outside @spell decorator on '{func_name}'. "
                    f"Guards must be applied INSIDE @spell:\n"
                    f"    @spell                    # <-- outermost\n"
                    f"    @guard.input(...)         # <-- inside\n"
                    f"    def {func_name}(...):\n"
                    f"        ...\n"
                    f"The guard will NOT be integrated with spell execution.",
                    UserWarning,
                    stacklevel=2,
                )
            config = _get_or_create_guard_config(func)
            # Prepend so guards run in decorator order (innermost first for input)
            config.input_guards.insert(0, (guard_fn, on_fail))
            return func

        return decorator

    @staticmethod
    def output(
        guard_fn: OutputGuard,
        *,
        on_fail: RaiseStrategy = OnFail.RAISE,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Add an output guard to a spell.

        Output guards run after the LLM call. They receive the output
        value and can validate, transform, or reject.

        Execution Order:
            Guards execute in decorator order (top to bottom)::

                @spell(model="fast")
                @guard.output(first_guard)   # Runs first
                @guard.output(second_guard)  # Runs second
                def respond(query: str) -> str:
                    ...

        Args:
            guard_fn: Function that validates/transforms output.
                      Signature: (output: T, context: dict) -> T
                      Raise any exception to reject.
            on_fail: Action on failure (default: RAISE)

        Returns:
            Decorator that adds the output guard.

        Example:
            def no_competitors(output: str, context: dict) -> str:
                competitors = {"acme", "globex"}
                if any(c in output.lower() for c in competitors):
                    raise ValueError("Response mentions competitor")
                return output

            @spell(model="fast")
            @guard.output(no_competitors)
            def respond(query: str) -> str:
                '''Respond to the customer.'''
                ...
        """

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            # Check if guards are being applied OUTSIDE @spell (wrong order)
            if getattr(func, "_is_spell_wrapper", False):
                func_name = getattr(func, "__name__", "unknown")
                warnings.warn(
                    f"Guard applied outside @spell decorator on '{func_name}'. "
                    f"Guards must be applied INSIDE @spell:\n"
                    f"    @spell                    # <-- outermost\n"
                    f"    @guard.output(...)        # <-- inside\n"
                    f"    def {func_name}(...):\n"
                    f"        ...\n"
                    f"The guard will NOT be integrated with spell execution.",
                    UserWarning,
                    stacklevel=2,
                )
            config = _get_or_create_guard_config(func)
            # Append so guards run in decorator order (outermost first for output)
            config.output_guards.append((guard_fn, on_fail))
            return func

        return decorator

    @staticmethod
    def max_length(
        *,
        input_max: int | None = None,
        output_max: int | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Add character length limits for inputs and/or outputs.

        Args:
            input_max: Maximum character length for input text (checks all str args)
            output_max: Maximum character length for output text

        Returns:
            Decorator that adds length guards.

        Example:
            @spell(model="fast")
            @guard.max_length(input_max=10000, output_max=5000)
            def summarize(text: str) -> str:
                '''Summarize.'''
                ...
        """

        def decorator(func: Callable[P, T]) -> Callable[P, T]:
            # Check if guards are being applied OUTSIDE @spell (wrong order)
            if getattr(func, "_is_spell_wrapper", False):
                func_name = getattr(func, "__name__", "unknown")
                warnings.warn(
                    f"Guard applied outside @spell decorator on '{func_name}'. "
                    f"Guards must be applied INSIDE @spell:\n"
                    f"    @spell                         # <-- outermost\n"
                    f"    @guard.max_length(...)         # <-- inside\n"
                    f"    def {func_name}(...):\n"
                    f"        ...\n"
                    f"The guard will NOT be integrated with spell execution.",
                    UserWarning,
                    stacklevel=2,
                )
            config = _get_or_create_guard_config(func)

            if input_max is not None:
                # Capture in local variable to avoid late-binding issues
                max_input_chars = input_max

                def check_input_length(input_args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                    # Check all string arguments
                    for key, value in input_args.items():
                        if isinstance(value, str) and len(value) > max_input_chars:
                            raise GuardError(
                                f"Input '{key}' exceeds maximum length of {max_input_chars} characters "
                                f"(got {len(value)})"
                            )
                    return input_args

                config.input_guards.insert(0, (check_input_length, OnFail.RAISE))

            if output_max is not None:
                # Capture in local variable to avoid late-binding issues
                max_output_chars = output_max

                def check_output_length(out: Any, context: dict[str, Any]) -> Any:
                    if isinstance(out, str) and len(out) > max_output_chars:
                        raise GuardError(
                            f"Output exceeds maximum length of {max_output_chars} characters "
                            f"(got {len(out)})"
                        )
                    return out

                config.output_guards.append((check_output_length, OnFail.RAISE))

            return func

        return decorator


# Create singleton instance
guard = _GuardNamespace()


# ---------------------------------------------------------------------------
# GuardExecutor: Internal API for spell.py integration
# ---------------------------------------------------------------------------


class GuardExecutor:
    """Internal API for running guards from spell.py.

    This class provides a clean interface for spell.py to execute guards
    without needing to import private helper functions directly.

    NOT for external use - this is an internal implementation detail.
    """

    # Expose the guard marker constant for spell.py
    MARKER = _GUARD_MARKER

    @staticmethod
    def get_config(func: Callable) -> GuardConfig | None:
        """Get the guard config attached to a function, if any."""
        return get_guard_config(func)

    @staticmethod
    def build_context(func: Callable, attempt: int = 1, model: str | None = None) -> GuardContext:
        """Build context for guard functions.

        Args:
            func: The spell function being executed.
            attempt: Current retry attempt (1-based).
            model: Model alias override. If None, uses func._model_alias if available.
        """
        return _build_context(func, attempt, model)

    @staticmethod
    def run_input_guards(
        guard_config: GuardConfig,
        input_args: dict[str, Any],
        context: GuardContext,
    ) -> dict[str, Any]:
        """Run input guards synchronously."""
        return _run_input_guards(guard_config.input_guards, input_args, context)

    @staticmethod
    async def run_input_guards_async(
        guard_config: GuardConfig,
        input_args: dict[str, Any],
        context: GuardContext,
    ) -> dict[str, Any]:
        """Run input guards asynchronously."""
        return await _run_input_guards_async(guard_config.input_guards, input_args, context)

    @staticmethod
    def run_input_guards_tracked(
        guard_config: GuardConfig,
        input_args: dict[str, Any],
        context: GuardContext,
    ) -> GuardRunResult[dict[str, Any]]:
        """Run input guards with tracking for metrics."""
        return _run_input_guards_tracked(guard_config.input_guards, input_args, context)

    @staticmethod
    async def run_input_guards_tracked_async(
        guard_config: GuardConfig,
        input_args: dict[str, Any],
        context: GuardContext,
    ) -> GuardRunResult[dict[str, Any]]:
        """Run input guards asynchronously with tracking."""
        return await _run_input_guards_tracked_async(guard_config.input_guards, input_args, context)

    @staticmethod
    def run_output_guards(
        guard_config: GuardConfig,
        output: T,
        context: GuardContext,
    ) -> T:
        """Run output guards synchronously."""
        return _run_output_guards(guard_config.output_guards, output, context)

    @staticmethod
    async def run_output_guards_async(
        guard_config: GuardConfig,
        output: T,
        context: GuardContext,
    ) -> T:
        """Run output guards asynchronously."""
        return await _run_output_guards_async(guard_config.output_guards, output, context)

    @staticmethod
    def run_output_guards_tracked(
        guard_config: GuardConfig,
        output: T,
        context: GuardContext,
    ) -> GuardRunResult[T]:
        """Run output guards with tracking for metrics."""
        return _run_output_guards_tracked(guard_config.output_guards, output, context)

    @staticmethod
    async def run_output_guards_tracked_async(
        guard_config: GuardConfig,
        output: T,
        context: GuardContext,
    ) -> GuardRunResult[T]:
        """Run output guards asynchronously with tracking."""
        return await _run_output_guards_tracked_async(guard_config.output_guards, output, context)


# Export for use in spell.py
__all__ = [
    "guard",
    "GuardConfig",
    "GuardContext",
    "get_guard_config",
    "GuardError",
    "GuardExecutor",
    "GuardRunResult",
    "OnFail",
    "InputGuard",
    "OutputGuard",
]
