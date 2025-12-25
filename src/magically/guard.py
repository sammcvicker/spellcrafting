"""Guard decorators for input/output validation on spells.

Guards provide composable validation that runs before (input) or after (output)
spell execution. They're designed to work alongside Pydantic structural validation
for semantic/safety checks that need the full context.

IMPORTANT: Decorator order matters! @spell must be the OUTERMOST decorator,
with guards applied INSIDE:

    @spell                    # <-- Always outermost
    @guard.input(validate)    # Input guards run before LLM
    @guard.output(check)      # Output guards run after LLM
    def my_spell(...):
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
from typing import Any, Callable, ParamSpec, Protocol, TypeVar

from magically.exceptions import GuardError
from magically.on_fail import OnFail, RaiseStrategy

P = ParamSpec("P")
T = TypeVar("T")


class InputGuard(Protocol):
    """Protocol for input guard functions.

    Input guards validate/transform input arguments before the LLM call.
    They receive the bound arguments as a dict and a context dict.
    They can modify and return the arguments, or raise to reject.
    """

    def __call__(self, input_args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """Validate/transform inputs. Raise to reject."""
        ...


class OutputGuard(Protocol):
    """Protocol for output guard functions.

    Output guards validate/transform the output after the LLM call.
    They receive the output value and a context dict.
    They can modify and return the output, or raise to reject.
    """

    def __call__(self, output: T, context: dict[str, Any]) -> T:
        """Validate/transform output. Raise to reject."""
        ...


# Marker to detect if a function is guarded
_GUARD_MARKER = "_magically_guards"


@dataclass
class _GuardConfig:
    """Internal configuration for guards on a function."""

    input_guards: list[tuple[InputGuard, RaiseStrategy]] = field(default_factory=list)
    output_guards: list[tuple[OutputGuard, RaiseStrategy]] = field(default_factory=list)


def _get_or_create_guard_config(func: Callable) -> _GuardConfig:
    """Get or create guard config on a function."""
    if not hasattr(func, _GUARD_MARKER):
        setattr(func, _GUARD_MARKER, _GuardConfig())
    return getattr(func, _GUARD_MARKER)


def _build_context(func: Callable, attempt: int = 1) -> dict[str, Any]:
    """Build context dict for guard functions."""
    spell_name = getattr(func, "__name__", "unknown")
    model_alias = getattr(func, "_model_alias", None)

    return {
        "spell_name": spell_name,
        "model": model_alias,
        "attempt_number": attempt,
    }


def _run_input_guards(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run input guards in order, transforming input_args."""
    current_args = input_args
    for guard_fn, _on_fail in guards:
        try:
            current_args = guard_fn(current_args, context)
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_args


def _run_output_guards(
    guards: list[tuple[OutputGuard, RaiseStrategy]],
    output: T,
    context: dict[str, Any],
) -> T:
    """Run output guards in order (outermost first), transforming output."""
    current_output = output
    for guard_fn, _on_fail in guards:
        try:
            current_output = guard_fn(current_output, context)
        except Exception as e:
            # Guards only support RaiseStrategy - wrap non-GuardError exceptions
            if isinstance(e, GuardError):
                raise
            raise GuardError(str(e)) from e
    return current_output


async def _run_input_guards_async(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run input guards in order, supporting async guard functions."""
    current_args = input_args
    for guard_fn, _on_fail in guards:
        try:
            result = guard_fn(current_args, context)
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
    context: dict[str, Any],
) -> T:
    """Run output guards in order, supporting async guard functions."""
    current_output = output
    for guard_fn, _on_fail in guards:
        try:
            result = guard_fn(current_output, context)
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
class GuardRunResult:
    """Result of running guards with tracking information."""

    result: Any  # The transformed input_args or output
    passed: list[str]  # Names of guards that passed
    failed: list[str]  # Names of guards that failed (if exception caught)


def _get_guard_name(guard_fn: Callable) -> str:
    """Get a human-readable name for a guard function."""
    return getattr(guard_fn, "__name__", None) or getattr(guard_fn, "__qualname__", "unknown")


def _run_input_guards_tracked(
    guards: list[tuple[InputGuard, RaiseStrategy]],
    input_args: dict[str, Any],
    context: dict[str, Any],
) -> GuardRunResult:
    """Run input guards with tracking. Returns result and guard names that passed/failed."""
    current_args = input_args
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            current_args = guard_fn(current_args, context)
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
    context: dict[str, Any],
) -> GuardRunResult:
    """Run output guards with tracking. Returns result and guard names that passed/failed."""
    current_output = output
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            current_output = guard_fn(current_output, context)
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
    context: dict[str, Any],
) -> GuardRunResult:
    """Run input guards with tracking, supporting async guard functions."""
    current_args = input_args
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            result = guard_fn(current_args, context)
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
    context: dict[str, Any],
) -> GuardRunResult:
    """Run output guards with tracking, supporting async guard functions."""
    current_output = output
    passed: list[str] = []
    failed: list[str] = []

    for guard_fn, _on_fail in guards:
        guard_name = _get_guard_name(guard_fn)
        try:
            result = guard_fn(current_output, context)
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
        input: int | None = None,
        output: int | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, T]]:
        """Add character length limits for inputs and/or outputs.

        Args:
            input: Maximum character length for input text (checks first str arg)
            output: Maximum character length for output text

        Returns:
            Decorator that adds length guards.

        Example:
            @spell(model="fast")
            @guard.max_length(input=10000, output=5000)
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

            if input is not None:

                def check_input_length(input_args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
                    # Check all string arguments
                    for key, value in input_args.items():
                        if isinstance(value, str) and len(value) > input:
                            raise GuardError(
                                f"Input '{key}' exceeds maximum length of {input} characters "
                                f"(got {len(value)})"
                            )
                    return input_args

                config.input_guards.insert(0, (check_input_length, OnFail.RAISE))

            if output is not None:

                def check_output_length(out: Any, context: dict[str, Any]) -> Any:
                    if isinstance(out, str) and len(out) > output:
                        raise GuardError(
                            f"Output exceeds maximum length of {output} characters "
                            f"(got {len(out)})"
                        )
                    return out

                config.output_guards.append((check_output_length, OnFail.RAISE))

            return func

        return decorator


# Create singleton instance
guard = _GuardNamespace()


# Export for use in spell.py
__all__ = [
    "guard",
    "GuardError",
    "OnFail",
    "InputGuard",
    "OutputGuard",
]
