"""Tests for the @guard decorator."""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from spellcrafting import spell, guard, GuardError, OnFail, GuardContext
from spellcrafting.guard import (
    GuardConfig,
    get_guard_config,
    _get_or_create_guard_config,
    _run_input_guards,
    _run_output_guards,
    _run_input_guards_tracked,
    _run_output_guards_tracked,
    _run_input_guards_async,
    _run_output_guards_async,
    _run_input_guards_tracked_async,
    _run_output_guards_tracked_async,
    _build_context,
)

# Default test context for guard runner tests
TEST_CONTEXT = GuardContext(spell_name="test_spell")


# ---------------------------------------------------------------------------
# Behavior-based tests (#108)
# These tests verify guard behavior without checking internal structure.
# Guards only run when integrated with @spell, so we test via the _run_*
# functions directly or via @spell with mocked Agent.
# ---------------------------------------------------------------------------


class TestGuardBehavior:
    """Behavior-based tests for guards that don't depend on internal structure (#108).

    These tests verify what guards DO, not how they're stored internally.
    They use the guard runner functions directly to test guard behavior.
    """

    def test_input_guard_is_called_via_runner(self):
        """Input guard function should be called when run through runner."""
        called = []

        def tracking_guard(args, ctx):
            called.append(args.copy())
            return args

        guards = [(tracking_guard, OnFail.RAISE)]
        input_args = {"text": "test"}

        result = _run_input_guards(guards, input_args, TEST_CONTEXT)

        # Guard should have been called with correct args
        assert len(called) == 1
        assert called[0] == {"text": "test"}
        assert result == {"text": "test"}

    def test_output_guard_is_called_via_runner(self):
        """Output guard function should be called when run through runner."""
        called = []

        def tracking_guard(output, ctx):
            called.append(output)
            return output

        guards = [(tracking_guard, OnFail.RAISE)]

        result = _run_output_guards(guards, "test output", TEST_CONTEXT)

        assert len(called) == 1
        assert called[0] == "test output"
        assert result == "test output"

    def test_input_guard_can_transform_via_runner(self):
        """Input guard should be able to transform arguments."""
        def uppercase_guard(args, ctx):
            return {k: v.upper() if isinstance(v, str) else v for k, v in args.items()}

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "HELLO"}

    def test_output_guard_can_transform_via_runner(self):
        """Output guard should be able to transform return value."""
        def uppercase_guard(output, ctx):
            return output.upper()

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_output_guards(guards, "hello", TEST_CONTEXT)

        assert result == "HELLO"

    def test_input_guard_can_reject_via_runner(self):
        """Input guard raising should prevent further execution."""
        def rejecting_guard(args, ctx):
            if len(args.get("text", "")) < 5:
                raise ValueError("Input too short")
            return args

        guards = [(rejecting_guard, OnFail.RAISE)]

        # Short input should be rejected
        with pytest.raises(GuardError, match="Input too short"):
            _run_input_guards(guards, {"text": "hi"}, TEST_CONTEXT)

        # Valid input should work
        result = _run_input_guards(guards, {"text": "hello world"}, TEST_CONTEXT)
        assert result == {"text": "hello world"}

    def test_output_guard_can_reject_via_runner(self):
        """Output guard raising should cause error."""
        def rejecting_guard(output, ctx):
            if "bad" in output:
                raise ValueError("Bad output detected")
            return output

        guards = [(rejecting_guard, OnFail.RAISE)]

        # Bad output should be rejected
        with pytest.raises(GuardError, match="Bad output detected"):
            _run_output_guards(guards, "this is bad content", TEST_CONTEXT)

        # Good output should work
        result = _run_output_guards(guards, "this is good content", TEST_CONTEXT)
        assert result == "this is good content"

    def test_multiple_input_guards_all_called_via_runner(self):
        """All input guards should be called in order."""
        call_order = []

        def guard1(args, ctx):
            call_order.append("guard1")
            return args

        def guard2(args, ctx):
            call_order.append("guard2")
            return args

        guards = [(guard1, OnFail.RAISE), (guard2, OnFail.RAISE)]
        _run_input_guards(guards, {"text": "test"}, TEST_CONTEXT)

        assert call_order == ["guard1", "guard2"]

    def test_multiple_output_guards_all_called_via_runner(self):
        """All output guards should be called in order."""
        call_order = []

        def guard1(output, ctx):
            call_order.append("guard1")
            return output

        def guard2(output, ctx):
            call_order.append("guard2")
            return output

        guards = [(guard1, OnFail.RAISE), (guard2, OnFail.RAISE)]
        _run_output_guards(guards, "test", TEST_CONTEXT)

        assert call_order == ["guard1", "guard2"]

    def test_guard_receives_context_via_runner(self):
        """Guard should receive context passed to runner."""
        received_ctx = {}

        def context_guard(args, ctx):
            received_ctx.update(ctx)
            return args

        guards = [(context_guard, OnFail.RAISE)]
        context = GuardContext(spell_name="my_function", model="fast")

        _run_input_guards(guards, {"text": "test"}, context)

        assert received_ctx["spell_name"] == "my_function"
        assert received_ctx["model"] == "fast"

    def test_guard_chain_transforms_accumulate_via_runner(self):
        """Multiple guards should accumulate their transforms."""
        def add_prefix(args, ctx):
            args["text"] = f"PREFIX_{args['text']}"
            return args

        def add_suffix(args, ctx):
            args["text"] = f"{args['text']}_SUFFIX"
            return args

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result["text"] == "PREFIX_hello_SUFFIX"

    def test_input_guard_called_with_spell(self):
        """Input guard should be called when integrated with @spell."""
        called = []

        def tracking_guard(args, ctx):
            called.append(args.copy())
            return args

        @spell
        @guard.input(tracking_guard)
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        assert len(called) == 1
        assert called[0] == {"text": "hello"}

    def test_output_guard_called_with_spell(self):
        """Output guard should be called when integrated with @spell."""
        called = []

        def tracking_guard(output, ctx):
            called.append(output)
            return output

        @spell
        @guard.output(tracking_guard)
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "test output"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("hello")

        assert len(called) == 1
        assert called[0] == "test output"
        assert result == "test output"

    def test_input_guard_transforms_with_spell(self):
        """Input guard should transform args when integrated with @spell."""
        def uppercase_guard(args, ctx):
            return {k: v.upper() if isinstance(v, str) else v for k, v in args.items()}

        @spell
        @guard.input(uppercase_guard)
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            test_spell("hello")

        # Verify the prompt sent to agent contains transformed input
        call_args = mock_agent.run_sync.call_args[0][0]
        assert "HELLO" in call_args

    def test_output_guard_transforms_with_spell(self):
        """Output guard should transform output when integrated with @spell."""
        def uppercase_guard(output, ctx):
            return output.upper()

        @spell
        @guard.output(uppercase_guard)
        def test_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "hello"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = test_spell("test")

        assert result == "HELLO"


# ---------------------------------------------------------------------------
# Internal structure tests
# These tests verify internal implementation details and may break if
# implementation changes. They're marked for documentation purposes.
# ---------------------------------------------------------------------------


class TestGuardDecoratorsBasic:
    """Tests for guard decorator structure and metadata.

    NOTE: These tests check internal structure (GuardConfig). While useful for
    verifying implementation details, the behavior tests in TestGuardBehavior
    are more stable and should be preferred when possible.
    """

    @pytest.mark.internal
    def test_guard_input_adds_to_config(self):
        def my_guard(input_args: dict, context: dict) -> dict:
            return input_args

        @guard.input(my_guard)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.input_guards) == 1
        assert config.input_guards[0][0] is my_guard
        assert config.input_guards[0][1] == OnFail.RAISE

    @pytest.mark.internal
    def test_guard_output_adds_to_config(self):
        def my_guard(output: str, context: dict) -> str:
            return output

        @guard.output(my_guard)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.output_guards) == 1
        assert config.output_guards[0][0] is my_guard
        assert config.output_guards[0][1] == OnFail.RAISE

    @pytest.mark.internal
    def test_multiple_input_guards(self):
        def guard1(args: dict, ctx: dict) -> dict:
            return args

        def guard2(args: dict, ctx: dict) -> dict:
            return args

        @guard.input(guard1)
        @guard.input(guard2)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.input_guards) == 2
        # Decorators apply bottom-up: guard2 first, then guard1
        # With insert(0, ...), guard1 ends up first in the list
        assert config.input_guards[0][0] is guard1
        assert config.input_guards[1][0] is guard2

    @pytest.mark.internal
    def test_multiple_output_guards(self):
        def guard1(out: str, ctx: dict) -> str:
            return out

        def guard2(out: str, ctx: dict) -> str:
            return out

        @guard.output(guard1)
        @guard.output(guard2)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.output_guards) == 2
        # Decorators apply bottom-up: guard2 first, then guard1
        # With append(...), guard2 ends up first in the list, guard1 second
        assert config.output_guards[0][0] is guard2
        assert config.output_guards[1][0] is guard1

    @pytest.mark.internal
    def test_combined_input_and_output_guards(self):
        def in_guard(args: dict, ctx: dict) -> dict:
            return args

        def out_guard(out: str, ctx: dict) -> str:
            return out

        @guard.input(in_guard)
        @guard.output(out_guard)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.input_guards) == 1
        assert len(config.output_guards) == 1

    @pytest.mark.internal
    def test_get_guard_config_returns_none_for_unguarded_function(self):
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is None


class TestInputGuardExecution:
    """Tests for input guard execution logic."""

    def test_input_guard_receives_args(self):
        received_args = {}

        def capture_guard(input_args: dict, context: dict) -> dict:
            received_args.update(input_args)
            return input_args

        guards = [(capture_guard, OnFail.RAISE)]
        input_args = {"text": "hello", "count": 5}

        _run_input_guards(guards, input_args, TEST_CONTEXT)
        assert received_args == {"text": "hello", "count": 5}

    def test_input_guard_receives_context(self):
        received_context = {}

        def capture_guard(input_args: dict, context: dict) -> dict:
            received_context.update(context)
            return input_args

        guards = [(capture_guard, OnFail.RAISE)]
        context = GuardContext(spell_name="test_spell", model="fast")

        _run_input_guards(guards, {}, context)
        assert received_context["spell_name"] == "test_spell"
        assert received_context["model"] == "fast"

    def test_input_guard_can_transform_args(self):
        def uppercase_guard(input_args: dict, context: dict) -> dict:
            return {k: v.upper() if isinstance(v, str) else v for k, v in input_args.items()}

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "HELLO"}

    def test_input_guard_chain_transforms(self):
        def add_prefix(args: dict, ctx: dict) -> dict:
            return {k: f"prefix_{v}" if isinstance(v, str) else v for k, v in args.items()}

        def add_suffix(args: dict, ctx: dict) -> dict:
            return {k: f"{v}_suffix" if isinstance(v, str) else v for k, v in args.items()}

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "prefix_hello_suffix"}

    def test_input_guard_raises_guard_error(self):
        def rejecting_guard(input_args: dict, context: dict) -> dict:
            raise ValueError("Invalid input")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Invalid input"):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_input_guard_preserves_guard_error(self):
        def rejecting_guard(input_args: dict, context: dict) -> dict:
            raise GuardError("Already a guard error")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Already a guard error"):
            _run_input_guards(guards, {}, TEST_CONTEXT)


class TestOutputGuardExecution:
    """Tests for output guard execution logic."""

    def test_output_guard_receives_output(self):
        received_output = []

        def capture_guard(output: str, context: dict) -> str:
            received_output.append(output)
            return output

        guards = [(capture_guard, OnFail.RAISE)]

        _run_output_guards(guards, "hello world", TEST_CONTEXT)
        assert received_output == ["hello world"]

    def test_output_guard_receives_context(self):
        received_context = {}

        def capture_guard(output: str, context: dict) -> str:
            received_context.update(context)
            return output

        guards = [(capture_guard, OnFail.RAISE)]
        context = GuardContext(spell_name="my_spell")

        _run_output_guards(guards, "test", context)
        assert received_context["spell_name"] == "my_spell"

    def test_output_guard_can_transform(self):
        def uppercase_guard(output: str, context: dict) -> str:
            return output.upper()

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_output_guards(guards, "hello", TEST_CONTEXT)

        assert result == "HELLO"

    def test_output_guard_chain_transforms(self):
        def add_prefix(out: str, ctx: dict) -> str:
            return f"prefix_{out}"

        def add_suffix(out: str, ctx: dict) -> str:
            return f"{out}_suffix"

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]
        result = _run_output_guards(guards, "hello", TEST_CONTEXT)

        assert result == "prefix_hello_suffix"

    def test_output_guard_raises_guard_error(self):
        def rejecting_guard(output: str, context: dict) -> str:
            raise ValueError("Invalid output")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Invalid output"):
            _run_output_guards(guards, "test", TEST_CONTEXT)


class TestMaxLengthGuard:
    """Tests for the built-in max_length guard."""

    def test_max_length_input_passes(self):
        @guard.max_length(input_max=100)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        result = _run_input_guards(
            config.input_guards,
            {"text": "short text"},
            TEST_CONTEXT,
        )
        assert result == {"text": "short text"}

    def test_max_length_input_fails(self):
        @guard.max_length(input_max=10)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None

        with pytest.raises(GuardError, match="exceeds maximum length of 10"):
            _run_input_guards(
                config.input_guards,
                {"text": "this is a longer text"},
                TEST_CONTEXT,
            )

    def test_max_length_output_passes(self):
        @guard.max_length(output_max=100)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        result = _run_output_guards(
            config.output_guards,
            "short output",
            TEST_CONTEXT,
        )
        assert result == "short output"

    def test_max_length_output_fails(self):
        @guard.max_length(output_max=5)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None

        with pytest.raises(GuardError, match="exceeds maximum length of 5"):
            _run_output_guards(
                config.output_guards,
                "longer output",
                TEST_CONTEXT,
            )

    def test_max_length_both_input_and_output(self):
        @guard.max_length(input_max=100, output_max=50)
        def fn(text: str) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None
        assert len(config.input_guards) == 1
        assert len(config.output_guards) == 1

    def test_max_length_closure_captures_correctly(self):
        """Verify max_length closures capture the correct limit values (#101).

        This test ensures the closure captures by value, not reference,
        by creating multiple guards with different limits.
        """
        # Create two functions with different limits
        @guard.max_length(input_max=5)
        def short_limit(text: str) -> str:
            return text

        @guard.max_length(input_max=100)
        def long_limit(text: str) -> str:
            return text

        short_config = get_guard_config(short_limit)
        long_config = get_guard_config(long_limit)

        # Short limit should reject text with 10 chars
        with pytest.raises(GuardError, match="exceeds maximum length of 5"):
            _run_input_guards(
                short_config.input_guards,
                {"text": "1234567890"},  # 10 chars
                TEST_CONTEXT,
            )

        # Long limit should accept text with 10 chars
        result = _run_input_guards(
            long_config.input_guards,
            {"text": "1234567890"},  # 10 chars
            TEST_CONTEXT,
        )
        assert result == {"text": "1234567890"}

    def test_max_length_output_closure_captures_correctly(self):
        """Verify output max_length closures capture the correct limit values (#101)."""
        @guard.max_length(output_max=3)
        def short_output(text: str) -> str:
            return text

        @guard.max_length(output_max=50)
        def long_output(text: str) -> str:
            return text

        short_config = get_guard_config(short_output)
        long_config = get_guard_config(long_output)

        # Short output limit should reject 10 chars
        with pytest.raises(GuardError, match="exceeds maximum length of 3"):
            _run_output_guards(
                short_config.output_guards,
                "1234567890",
                TEST_CONTEXT,
            )

        # Long output limit should accept 10 chars
        result = _run_output_guards(
            long_config.output_guards,
            "1234567890",
            TEST_CONTEXT,
        )
        assert result == "1234567890"

    def test_max_length_input_ignores_non_strings(self):
        """Input guard should skip non-string arguments (#58)."""
        @guard.max_length(input_max=10)
        def fn(text: str, count: int, data: list) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None

        # Non-string arguments should not be checked, even if "large"
        result = _run_input_guards(
            config.input_guards,
            {"text": "short", "count": 99999, "data": list(range(1000))},
            TEST_CONTEXT,
        )
        assert result["text"] == "short"
        assert result["count"] == 99999
        assert len(result["data"]) == 1000

    def test_max_length_input_checks_only_strings(self):
        """Input guard should only check string arguments (#58)."""
        @guard.max_length(input_max=5)
        def fn(text: str, number: int) -> str:
            return text

        config = get_guard_config(fn)
        assert config is not None

        # String over limit should fail
        with pytest.raises(GuardError, match="exceeds maximum length of 5"):
            _run_input_guards(
                config.input_guards,
                {"text": "this is too long", "number": 12345678},
                TEST_CONTEXT,
            )

        # String under limit passes, even with large int
        result = _run_input_guards(
            config.input_guards,
            {"text": "ok", "number": 12345678},
            TEST_CONTEXT,
        )
        assert result["text"] == "ok"

    def test_max_length_output_ignores_non_strings(self):
        """Output guard should skip non-string values (#58)."""
        @guard.max_length(output_max=5)
        def fn() -> dict:
            pass

        config = get_guard_config(fn)
        assert config is not None

        # Dict output should not be checked
        result = _run_output_guards(
            config.output_guards,
            {"key": "a very long value that exceeds limit"},
            TEST_CONTEXT,
        )
        assert result == {"key": "a very long value that exceeds limit"}

    def test_max_length_output_ignores_list(self):
        """Output guard should skip list values (#58)."""
        @guard.max_length(output_max=3)
        def fn() -> list:
            pass

        config = get_guard_config(fn)
        assert config is not None

        # List output should not be checked (even though len > 3)
        result = _run_output_guards(
            config.output_guards,
            [1, 2, 3, 4, 5],
            TEST_CONTEXT,
        )
        assert result == [1, 2, 3, 4, 5]

    def test_max_length_output_ignores_int(self):
        """Output guard should skip int values (#58)."""
        @guard.max_length(output_max=2)
        def fn() -> int:
            pass

        config = get_guard_config(fn)
        assert config is not None

        # Int output should not be checked
        result = _run_output_guards(
            config.output_guards,
            12345,
            TEST_CONTEXT,
        )
        assert result == 12345

    def test_max_length_input_mixed_string_limits(self):
        """Input guard checks strings but not other types in mixed args (#58)."""
        @guard.max_length(input_max=10)
        def fn(a: str, b: int, c: str, d: dict) -> str:
            return a

        config = get_guard_config(fn)
        assert config is not None

        # First string under limit, second string over limit - should fail
        with pytest.raises(GuardError, match="'c' exceeds maximum length"):
            _run_input_guards(
                config.input_guards,
                {"a": "short", "b": 999, "c": "this is way too long", "d": {"x": 1}},
                TEST_CONTEXT,
            )

        # Both strings under limit - should pass
        result = _run_input_guards(
            config.input_guards,
            {"a": "short", "b": 999, "c": "also ok", "d": {"x": 1}},
            TEST_CONTEXT,
        )
        assert result["a"] == "short"
        assert result["c"] == "also ok"


class TestBuildContext:
    """Tests for context building helper."""

    def test_build_context_basic(self):
        def my_func():
            pass

        context = _build_context(my_func)
        assert context.spell_name == "my_func"
        assert context.attempt_number == 1

    def test_build_context_with_model_alias(self):
        def my_func():
            pass

        my_func._model_alias = "fast"

        context = _build_context(my_func)
        assert context.model == "fast"

    def test_build_context_with_attempt(self):
        def my_func():
            pass

        context = _build_context(my_func, attempt=3)
        assert context.attempt_number == 3


class TestGuardWithSpell:
    """Tests for guard integration with @spell decorator."""

    def test_input_guard_runs_before_llm(self):
        call_order = []

        def track_input(args: dict, ctx: dict) -> dict:
            call_order.append("input_guard")
            return args

        @spell
        @guard.input(track_input)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        def track_agent(*args, **kwargs):
            call_order.append("agent_run")
            return mock_result

        mock_agent.run_sync = track_agent

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            summarize("test input")

        assert call_order == ["input_guard", "agent_run"]

    def test_output_guard_runs_after_llm(self):
        call_order = []

        def track_output(out: str, ctx: dict) -> str:
            call_order.append("output_guard")
            return out

        @spell
        @guard.output(track_output)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        def track_agent(*args, **kwargs):
            call_order.append("agent_run")
            return mock_result

        mock_agent.run_sync = track_agent

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            summarize("test input")

        assert call_order == ["agent_run", "output_guard"]

    def test_input_guard_can_reject_before_llm_call(self):
        def rejecting_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Rejected!")

        @spell
        @guard.input(rejecting_guard)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Rejected!"):
                summarize("test input")

            # LLM should not be called
            mock_agent.run_sync.assert_not_called()

    def test_output_guard_can_reject_after_llm_call(self):
        def rejecting_guard(out: str, ctx: dict) -> str:
            raise ValueError("Bad output!")

        @spell
        @guard.output(rejecting_guard)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Bad output!"):
                summarize("test input")

            # LLM was called
            mock_agent.run_sync.assert_called_once()

    def test_input_guard_can_transform_input(self):
        def uppercase_guard(args: dict, ctx: dict) -> dict:
            return {"text": args["text"].upper()}

        @spell
        @guard.input(uppercase_guard)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            summarize("hello")

            # Check the prompt was transformed
            call_args = mock_agent.run_sync.call_args[0][0]
            assert "HELLO" in call_args

    def test_output_guard_can_transform_output(self):
        def uppercase_guard(out: str, ctx: dict) -> str:
            return out.upper()

        @spell
        @guard.output(uppercase_guard)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = summarize("test")
            assert result == "SUMMARY"

    def test_max_length_guard_with_spell(self):
        @spell
        @guard.max_length(input_max=10, output_max=100)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="exceeds maximum length"):
                summarize("this is a very long input that exceeds the limit")

    def test_multiple_guards_compose(self):
        transforms = []

        def guard1(args: dict, ctx: dict) -> dict:
            transforms.append("g1")
            args["text"] = f"[G1]{args['text']}"
            return args

        def guard2(args: dict, ctx: dict) -> dict:
            transforms.append("g2")
            args["text"] = f"[G2]{args['text']}"
            return args

        @spell
        @guard.input(guard1)
        @guard.input(guard2)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            summarize("hello")

            # guard1 runs first (outermost decorator applied last via insert(0))
            # then guard2
            assert transforms == ["g1", "g2"]

            # Check the combined transformation
            call_args = mock_agent.run_sync.call_args[0][0]
            assert "[G2][G1]hello" in call_args


class TestAsyncGuardWithSpell:
    """Tests for guard integration with async @spell decorator.

    NOTE: Guard functions themselves are currently synchronous, even when used
    with async spells. These tests verify that sync guards work correctly in
    async spell context. The guards run in the async event loop but do not
    themselves use await.

    For async I/O in guards, you can use asyncio.to_thread() to run blocking
    operations, or create async guard functions that will be awaited if they
    return a coroutine.

    Example of async-compatible guard:
        async def async_guard(args: dict, ctx: dict) -> dict:
            # This will be awaited automatically
            result = await some_async_check(args)
            return args if result else raise ValueError("Failed")
    """

    @pytest.mark.asyncio
    async def test_sync_input_guard_runs_in_async_spell(self):
        """Sync guards work correctly in async spell context."""
        guard_called = []

        def track_guard(args: dict, ctx: dict) -> dict:
            guard_called.append(True)
            return args

        @spell
        @guard.input(track_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            await summarize("test")

        assert guard_called == [True]

    @pytest.mark.asyncio
    async def test_sync_output_guard_runs_in_async_spell(self):
        """Sync output guards work correctly in async spell context."""
        guard_called = []

        def track_guard(out: str, ctx: dict) -> str:
            guard_called.append(True)
            return out

        @spell
        @guard.output(track_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            await summarize("test")

        assert guard_called == [True]

    @pytest.mark.asyncio
    async def test_guard_rejection_works_in_async_spell(self):
        """Guards can reject input/output in async spell context."""

        def rejecting_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Rejected!")

        @spell
        @guard.input(rejecting_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Rejected!"):
                await summarize("test")

    @pytest.mark.asyncio
    async def test_async_guard_function_is_awaited(self):
        """Guard functions that return coroutines are properly awaited."""
        guard_called = []

        async def async_track_guard(args: dict, ctx: dict) -> dict:
            # This is an async guard function
            guard_called.append("async_guard")
            return args

        @spell
        @guard.input(async_track_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            await summarize("test")

        assert guard_called == ["async_guard"]


class TestGuardContext:
    """Tests for context passed to guard functions."""

    def test_context_includes_spell_name(self):
        received_ctx = {}

        def capture_ctx(args: dict, ctx: dict) -> dict:
            received_ctx.update(ctx)
            return args

        @spell
        @guard.input(capture_ctx)
        def my_spell_name(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "out"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_spell_name("test")

        assert received_ctx["spell_name"] == "my_spell_name"

    def test_context_includes_model_alias(self):
        received_ctx = {}

        def capture_ctx(args: dict, ctx: dict) -> dict:
            received_ctx.update(ctx)
            return args

        @spell(model="openai:gpt-4o")
        @guard.input(capture_ctx)
        def my_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "out"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_spell("test")

        assert received_ctx["model"] == "openai:gpt-4o"


class TestGuardPreservesFunction:
    """Tests that guards don't break function metadata."""

    def test_guard_preserves_name(self):
        @guard.input(lambda a, c: a)
        def my_function(text: str) -> str:
            return text

        assert my_function.__name__ == "my_function"

    def test_guard_preserves_docstring(self):
        @guard.input(lambda a, c: a)
        def my_function(text: str) -> str:
            """My docstring."""
            return text

        assert my_function.__doc__ == "My docstring."

    def test_guards_work_with_spell_metadata(self):
        @spell
        @guard.input(lambda a, c: a)
        @guard.output(lambda o, c: o)
        def summarize(text: str) -> str:
            """Summarize text."""
            ...

        assert summarize.__name__ == "summarize"
        assert summarize._system_prompt == "Summarize text."


class TestDecoratorOrderWarning:
    """Tests for decorator order detection and warnings."""

    def test_guard_input_outside_spell_warns(self):
        """Guard applied outside @spell should warn about incorrect order."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @guard.input(lambda a, c: a)
            @spell
            def my_spell(text: str) -> str:
                """Test."""
                ...

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Guard applied outside @spell decorator" in str(w[0].message)
            assert "my_spell" in str(w[0].message)
            assert "Guards must be applied INSIDE @spell" in str(w[0].message)

    def test_guard_output_outside_spell_warns(self):
        """Guard applied outside @spell should warn about incorrect order."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @guard.output(lambda o, c: o)
            @spell
            def my_spell(text: str) -> str:
                """Test."""
                ...

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Guard applied outside @spell decorator" in str(w[0].message)
            assert "my_spell" in str(w[0].message)

    def test_guard_max_length_outside_spell_warns(self):
        """max_length guard applied outside @spell should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @guard.max_length(input_max=100)
            @spell
            def my_spell(text: str) -> str:
                """Test."""
                ...

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Guard applied outside @spell decorator" in str(w[0].message)

    def test_guard_inside_spell_no_warning(self):
        """Guards correctly inside @spell should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @spell
            @guard.input(lambda a, c: a)
            @guard.output(lambda o, c: o)
            def my_spell(text: str) -> str:
                """Test."""
                ...

            # Filter only UserWarnings (may have other deprecation warnings)
            guard_warnings = [x for x in w if "Guard applied outside" in str(x.message)]
            assert len(guard_warnings) == 0

    def test_multiple_guards_outside_spell_warns_each(self):
        """Multiple guards outside @spell should each produce a warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @guard.input(lambda a, c: a)
            @guard.output(lambda o, c: o)
            @spell
            def my_spell(text: str) -> str:
                """Test."""
                ...

            guard_warnings = [x for x in w if "Guard applied outside" in str(x.message)]
            assert len(guard_warnings) == 2

    def test_spell_wrapper_has_marker(self):
        """Spell wrapper should have _is_spell_wrapper marker."""
        @spell
        def my_spell(text: str) -> str:
            """Test."""
            ...

        assert hasattr(my_spell, "_is_spell_wrapper")
        assert my_spell._is_spell_wrapper is True

    def test_guard_outside_spell_not_integrated(self):
        """Guards outside @spell should not run during spell execution."""
        guard_calls = []

        def track_guard(args: dict, ctx: dict) -> dict:
            guard_calls.append("guard_called")
            return args

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            @guard.input(track_guard)
            @spell
            def my_spell(text: str) -> str:
                """Test."""
                ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_spell("test")

        # Guard on outer wrapper is NOT integrated with spell execution
        # The guard marker is on the spell wrapper, not the original function
        # So spell's inner logic doesn't see it
        assert guard_calls == []

    def test_guard_inside_spell_runs(self):
        """Guards inside @spell should run during spell execution."""
        guard_calls = []

        def track_guard(args: dict, ctx: dict) -> dict:
            guard_calls.append("guard_called")
            return args

        @spell
        @guard.input(track_guard)
        def my_spell(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_spell("test")

        # Guard inside @spell IS integrated and runs
        assert guard_calls == ["guard_called"]


class TestGuardContextDataclass:
    """Tests for the GuardContext dataclass passed to guard functions (#167, #80).

    The context passed to guards contains spell metadata like spell_name,
    model alias, and attempt number. These tests verify the GuardContext
    structure and its to_dict() method for backwards compatibility.
    """

    def test_context_has_spell_name(self):
        """Context should include the spell name from the decorated function."""
        context = _build_context(lambda: None)
        assert context.spell_name == "<lambda>"
        # Also verify to_dict for backwards compatibility
        ctx_dict = context.to_dict()
        assert "spell_name" in ctx_dict
        assert ctx_dict["spell_name"] == "<lambda>"

    def test_context_has_model_alias(self):
        """Context should include model alias when set on function."""
        def my_func():
            pass
        my_func._model_alias = "fast"

        context = _build_context(my_func)
        assert context.model == "fast"

    def test_context_model_none_when_not_set(self):
        """Context model should be None when no model alias is set."""
        def my_func():
            pass

        context = _build_context(my_func)
        assert context.model is None

    def test_context_has_attempt_number(self):
        """Context should include attempt number."""
        def my_func():
            pass

        context = _build_context(my_func)
        assert context.attempt_number == 1

    def test_context_custom_attempt_number(self):
        """Context should respect custom attempt number."""
        def my_func():
            pass

        context = _build_context(my_func, attempt=5)
        assert context.attempt_number == 5

    def test_context_to_dict_is_mutable(self):
        """Guards receive a mutable dict via to_dict() for backwards compatibility."""
        def my_func():
            pass

        context = _build_context(my_func)
        ctx_dict = context.to_dict()
        # Guards can add their own data to the dict copy
        ctx_dict["custom_key"] = "custom_value"
        assert ctx_dict["custom_key"] == "custom_value"

    def test_context_passed_to_input_guard_in_spell(self):
        """Input guards receive proper context during spell execution."""
        received_contexts = []

        def capture_context_guard(args: dict, ctx: dict) -> dict:
            received_contexts.append(ctx.copy())
            return args

        @spell(model="openai:gpt-4o")
        @guard.input(capture_context_guard)
        def my_test_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_test_spell("hello")

        assert len(received_contexts) == 1
        ctx = received_contexts[0]
        assert ctx["spell_name"] == "my_test_spell"
        assert ctx["model"] == "openai:gpt-4o"
        assert ctx["attempt_number"] == 1

    def test_context_passed_to_output_guard_in_spell(self):
        """Output guards receive proper context during spell execution."""
        received_contexts = []

        def capture_context_guard(output: str, ctx: dict) -> str:
            received_contexts.append(ctx.copy())
            return output

        @spell(model="openai:gpt-4o")
        @guard.output(capture_context_guard)
        def my_output_spell(text: str) -> str:
            """Test spell."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            my_output_spell("hello")

        assert len(received_contexts) == 1
        ctx = received_contexts[0]
        assert ctx["spell_name"] == "my_output_spell"
        assert ctx["model"] == "openai:gpt-4o"


class TestTrackedGuardsEarlyFailure:
    """Tests for tracked guard functions with early guard failure (#168).

    When guards are run with tracking, the passed/failed lists should correctly
    reflect which guards succeeded before a failure occurred.
    """

    def test_tracked_input_guards_all_pass(self):
        """All guards pass, verify full tracking."""
        def guard1(args: dict, ctx: dict) -> dict:
            return args

        def guard2(args: dict, ctx: dict) -> dict:
            return args

        guards = [
            (guard1, OnFail.RAISE),
            (guard2, OnFail.RAISE),
        ]

        result = _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)

        assert len(result.passed) == 2
        assert "guard1" in result.passed
        assert "guard2" in result.passed
        assert len(result.failed) == 0
        assert result.result == {"text": "hello"}

    def test_tracked_output_guards_all_pass(self):
        """All output guards pass, verify full tracking."""
        def check1(output: str, ctx: dict) -> str:
            return output

        def check2(output: str, ctx: dict) -> str:
            return output

        guards = [
            (check1, OnFail.RAISE),
            (check2, OnFail.RAISE),
        ]

        result = _run_output_guards_tracked(guards, "test output", TEST_CONTEXT)

        assert len(result.passed) == 2
        assert "check1" in result.passed
        assert "check2" in result.passed
        assert len(result.failed) == 0
        assert result.result == "test output"

    def test_tracked_input_guard_early_failure_tracks_passed(self):
        """When second guard fails, first guard should be in passed list."""
        def passing_guard(args: dict, ctx: dict) -> dict:
            return args

        def failing_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Guard failed!")

        def never_called(args: dict, ctx: dict) -> dict:
            raise RuntimeError("Should not be called")

        guards = [
            (passing_guard, OnFail.RAISE),
            (failing_guard, OnFail.RAISE),
            (never_called, OnFail.RAISE),
        ]

        with pytest.raises(GuardError, match="Guard failed!"):
            _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_tracked_input_guard_first_fails(self):
        """When first guard fails, passed list should be empty."""
        def failing_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("First guard failed!")

        def never_called(args: dict, ctx: dict) -> dict:
            raise RuntimeError("Should not be called")

        guards = [
            (failing_guard, OnFail.RAISE),
            (never_called, OnFail.RAISE),
        ]

        with pytest.raises(GuardError, match="First guard failed!"):
            _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_tracked_output_guard_early_failure(self):
        """When output guard fails, tracking should record what passed before."""
        def passing_guard(output: str, ctx: dict) -> str:
            return output

        def failing_guard(output: str, ctx: dict) -> str:
            raise ValueError("Output guard failed!")

        guards = [
            (passing_guard, OnFail.RAISE),
            (failing_guard, OnFail.RAISE),
        ]

        with pytest.raises(GuardError, match="Output guard failed!"):
            _run_output_guards_tracked(guards, "test", TEST_CONTEXT)

    def test_tracked_guard_transforms_before_failure(self):
        """Guards can transform input before a later guard fails."""
        def transform_guard(args: dict, ctx: dict) -> dict:
            args["text"] = args["text"].upper()
            return args

        def failing_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Failed!")

        guards = [
            (transform_guard, OnFail.RAISE),
            (failing_guard, OnFail.RAISE),
        ]

        with pytest.raises(GuardError):
            _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_tracked_guards_empty_list(self):
        """Empty guard list should return empty tracking."""
        result = _run_input_guards_tracked([], {"text": "hello"}, TEST_CONTEXT)

        assert len(result.passed) == 0
        assert len(result.failed) == 0
        assert result.result == {"text": "hello"}

    def test_tracked_guards_single_pass(self):
        """Single passing guard should be tracked."""
        def single_guard(args: dict, ctx: dict) -> dict:
            return args

        guards = [(single_guard, OnFail.RAISE)]

        result = _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result.passed == ["single_guard"]
        assert result.failed == []

    def test_tracked_guards_single_fail(self):
        """Single failing guard should be tracked in failed list."""
        def single_failing_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Failed!")

        guards = [(single_failing_guard, OnFail.RAISE)]

        with pytest.raises(GuardError):
            _run_input_guards_tracked(guards, {"text": "hello"}, TEST_CONTEXT)


class TestAsyncGuardRunners:
    """Tests for async guard runner functions directly (#12).

    These tests verify that the _run_*_async guard runner functions properly
    await async guard functions. This tests the guard module's internal
    implementation separate from spell integration.
    """

    @pytest.mark.asyncio
    async def test_async_input_guard_runner(self):
        """Async guard runner properly awaits async guard functions."""
        import asyncio

        async def async_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return {k: v.upper() if isinstance(v, str) else v for k, v in args.items()}

        guards = [(async_guard, OnFail.RAISE)]

        result = await _run_input_guards_async(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "HELLO"}

    @pytest.mark.asyncio
    async def test_async_output_guard_runner(self):
        """Async output guard runner properly awaits async guard functions."""
        import asyncio

        async def async_guard(output: str, ctx: dict) -> str:
            await asyncio.sleep(0)
            return output.upper()

        guards = [(async_guard, OnFail.RAISE)]

        result = await _run_output_guards_async(guards, "hello", TEST_CONTEXT)

        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_async_input_guard_runner_tracked(self):
        """Async input guard runner with tracking properly awaits async guards."""
        import asyncio

        async def async_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return args

        guards = [(async_guard, OnFail.RAISE)]

        result = await _run_input_guards_tracked_async(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result.result == {"text": "hello"}
        assert "async_guard" in result.passed
        assert len(result.failed) == 0

    @pytest.mark.asyncio
    async def test_async_output_guard_runner_tracked(self):
        """Async output guard runner with tracking properly awaits async guards."""
        import asyncio

        async def async_guard(output: str, ctx: dict) -> str:
            await asyncio.sleep(0)
            return output

        guards = [(async_guard, OnFail.RAISE)]

        result = await _run_output_guards_tracked_async(guards, "test", TEST_CONTEXT)

        assert result.result == "test"
        assert "async_guard" in result.passed
        assert len(result.failed) == 0

    @pytest.mark.asyncio
    async def test_async_guard_runner_rejection(self):
        """Async guard runner properly handles rejection from async guard."""
        import asyncio

        async def rejecting_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            raise ValueError("Async guard rejection")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Async guard rejection"):
            await _run_input_guards_async(guards, {"text": "hello"}, TEST_CONTEXT)

    @pytest.mark.asyncio
    async def test_async_guard_runner_mixed_sync_async(self):
        """Async guard runner handles mix of sync and async guards."""
        import asyncio

        call_order = []

        def sync_guard(args: dict, ctx: dict) -> dict:
            call_order.append("sync")
            return args

        async def async_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            call_order.append("async")
            return args

        guards = [(sync_guard, OnFail.RAISE), (async_guard, OnFail.RAISE)]

        result = await _run_input_guards_async(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "hello"}
        assert call_order == ["sync", "async"]

    @pytest.mark.asyncio
    async def test_async_guard_runner_chain_transforms(self):
        """Async guard runner properly chains transforms from multiple async guards."""
        import asyncio

        async def add_prefix(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return {k: f"PREFIX_{v}" if isinstance(v, str) else v for k, v in args.items()}

        async def add_suffix(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return {k: f"{v}_SUFFIX" if isinstance(v, str) else v for k, v in args.items()}

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]

        result = await _run_input_guards_async(guards, {"text": "hello"}, TEST_CONTEXT)

        assert result == {"text": "PREFIX_hello_SUFFIX"}


class TestAsyncGuardFunctions:
    """Tests for async guard functions with spell integration (#12).

    These tests verify that guard functions which are async (use await) are
    properly awaited during spell execution. This exercises the code path
    where `asyncio.iscoroutine(result)` returns True.
    """

    @pytest.mark.asyncio
    async def test_async_input_guard_with_await(self):
        """Async input guard that does await is properly executed."""
        import asyncio

        guard_calls = []
        captured_prompts = []

        async def async_input_guard(args: dict, ctx: dict) -> dict:
            # Simulate async work
            await asyncio.sleep(0)
            guard_calls.append("async_input")
            return {k: v.upper() if isinstance(v, str) else v for k, v in args.items()}

        @spell
        @guard.input(async_input_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            captured_prompts.append(prompt)
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await summarize("hello")

        assert guard_calls == ["async_input"]
        # Verify the guard transformed the input
        assert len(captured_prompts) == 1
        # The prompt should contain the uppercased text
        assert "HELLO" in captured_prompts[0]

    @pytest.mark.asyncio
    async def test_async_output_guard_with_await(self):
        """Async output guard that does await is properly executed."""
        import asyncio

        guard_calls = []

        async def async_output_guard(output: str, ctx: dict) -> str:
            # Simulate async work
            await asyncio.sleep(0)
            guard_calls.append("async_output")
            return output.upper()

        @spell
        @guard.output(async_output_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            result = await summarize("test")

        assert guard_calls == ["async_output"]
        assert result == "SUMMARY"

    @pytest.mark.asyncio
    async def test_async_input_guard_can_reject(self):
        """Async input guard that raises is properly handled."""
        import asyncio

        async def rejecting_async_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)  # Simulate async work
            raise ValueError("Async rejection!")

        @spell
        @guard.input(rejecting_async_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Async rejection!"):
                await summarize("test")

    @pytest.mark.asyncio
    async def test_async_output_guard_can_reject(self):
        """Async output guard that raises is properly handled."""
        import asyncio

        async def rejecting_async_guard(output: str, ctx: dict) -> str:
            await asyncio.sleep(0)  # Simulate async work
            raise ValueError("Async output rejection!")

        @spell
        @guard.output(rejecting_async_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Async output rejection!"):
                await summarize("test")

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_guards(self):
        """Mix of sync and async guards execute correctly."""
        import asyncio

        call_order = []

        def sync_input_guard(args: dict, ctx: dict) -> dict:
            call_order.append("sync_input")
            return args

        async def async_input_guard(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            call_order.append("async_input")
            return args

        def sync_output_guard(output: str, ctx: dict) -> str:
            call_order.append("sync_output")
            return output

        async def async_output_guard(output: str, ctx: dict) -> str:
            await asyncio.sleep(0)
            call_order.append("async_output")
            return output

        @spell
        @guard.input(sync_input_guard)
        @guard.input(async_input_guard)
        @guard.output(sync_output_guard)
        @guard.output(async_output_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            await summarize("test")

        # Input guards run first (in decorator order), then output guards
        assert "sync_input" in call_order
        assert "async_input" in call_order
        assert "sync_output" in call_order
        assert "async_output" in call_order

    @pytest.mark.asyncio
    async def test_async_guard_chain_transforms(self):
        """Multiple async guards properly chain their transforms."""
        import asyncio

        captured_prompts = []

        async def add_prefix(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return {k: f"PREFIX_{v}" if isinstance(v, str) else v for k, v in args.items()}

        async def add_suffix(args: dict, ctx: dict) -> dict:
            await asyncio.sleep(0)
            return {k: f"{v}_SUFFIX" if isinstance(v, str) else v for k, v in args.items()}

        @spell
        @guard.input(add_prefix)
        @guard.input(add_suffix)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "summary"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            captured_prompts.append(prompt)
            return mock_result

        mock_agent.run = mock_run

        with patch("spellcrafting.spell.Agent", return_value=mock_agent):
            await summarize("hello")

        # Check the prompt contains the transformed input
        assert len(captured_prompts) == 1
        assert "PREFIX_hello_SUFFIX" in captured_prompts[0]


class TestGuardErrorHandling:
    """Negative tests for guard error handling (#103).

    Guards can raise various types of exceptions. These tests verify that
    different exception types are properly wrapped in GuardError.
    """

    def test_input_guard_typeerror_wrapped(self):
        """Guards raising TypeError should be wrapped in GuardError."""
        def type_error_guard(input_args: dict, context: dict) -> dict:
            raise TypeError("Expected string, got int")

        guards = [(type_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Expected string, got int"):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_output_guard_typeerror_wrapped(self):
        """Output guards raising TypeError should be wrapped in GuardError."""
        def type_error_guard(output: str, context: dict) -> str:
            raise TypeError("Invalid output type")

        guards = [(type_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Invalid output type"):
            _run_output_guards(guards, "test", TEST_CONTEXT)

    def test_input_guard_custom_exception_wrapped(self):
        """Guards raising custom exceptions should be wrapped in GuardError."""
        class CustomValidationError(Exception):
            pass

        def custom_error_guard(input_args: dict, context: dict) -> dict:
            raise CustomValidationError("Custom validation failed")

        guards = [(custom_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Custom validation failed"):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_output_guard_custom_exception_wrapped(self):
        """Output guards raising custom exceptions should be wrapped in GuardError."""
        class CustomOutputError(Exception):
            pass

        def custom_error_guard(output: str, context: dict) -> str:
            raise CustomOutputError("Custom output error")

        guards = [(custom_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Custom output error"):
            _run_output_guards(guards, "test", TEST_CONTEXT)

    def test_input_guard_returns_none_passthrough(self):
        """Guards returning None should pass through (None is a valid return).

        Note: Guards returning None can cause downstream issues if the spell
        expects a dict, but the guard runner doesn't enforce return type.
        """
        def none_guard(input_args: dict, context: dict) -> dict:
            return None  # type: ignore

        guards = [(none_guard, OnFail.RAISE)]

        # None is passed through - guard runner doesn't type-check returns
        result = _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)
        assert result is None

    def test_output_guard_returns_none_passthrough(self):
        """Output guards returning None should pass through.

        Note: This can cause downstream issues, but guard runner is permissive.
        """
        def none_guard(output: str, context: dict) -> str:
            return None  # type: ignore

        guards = [(none_guard, OnFail.RAISE)]

        result = _run_output_guards(guards, "test", TEST_CONTEXT)
        assert result is None

    def test_output_guard_wrong_return_type_passthrough(self):
        """Output guards returning wrong type should pass through.

        Guard runners don't enforce return types - this is the spell's
        responsibility via Pydantic validation.
        """
        def wrong_type_guard(output: str, context: dict) -> str:
            return 42  # type: ignore  # Wrong type

        guards = [(wrong_type_guard, OnFail.RAISE)]

        result = _run_output_guards(guards, "test", TEST_CONTEXT)
        assert result == 42  # Passes through without type checking

    def test_input_guard_exception_preserves_original(self):
        """GuardError should preserve the original exception via __cause__."""
        class OriginalError(Exception):
            pass

        def raising_guard(input_args: dict, context: dict) -> dict:
            raise OriginalError("Original error message")

        guards = [(raising_guard, OnFail.RAISE)]

        with pytest.raises(GuardError) as exc_info:
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        # Verify the original exception is preserved in the chain
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, OriginalError)

    def test_guard_error_not_double_wrapped(self):
        """GuardError raised by guard should not be wrapped again."""
        def guard_error_guard(input_args: dict, context: dict) -> dict:
            raise GuardError("Already a guard error")

        guards = [(guard_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Already a guard error") as exc_info:
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        # Should not have a __cause__ since it wasn't wrapped
        assert exc_info.value.__cause__ is None

    def test_input_guard_attribute_error_wrapped(self):
        """AttributeError in guard should be wrapped in GuardError."""
        def attr_error_guard(input_args: dict, context: dict) -> dict:
            # Simulate accessing attribute that doesn't exist
            return input_args["text"].nonexistent_method()

        guards = [(attr_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_input_guard_key_error_wrapped(self):
        """KeyError in guard should be wrapped in GuardError."""
        def key_error_guard(input_args: dict, context: dict) -> dict:
            # Access key that doesn't exist
            return {"result": input_args["missing_key"]}

        guards = [(key_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

    def test_multiple_guards_first_error_stops_chain(self):
        """When first guard raises, subsequent guards should not run."""
        call_order = []

        def first_guard(args: dict, ctx: dict) -> dict:
            call_order.append("first")
            raise ValueError("First guard failed")

        def second_guard(args: dict, ctx: dict) -> dict:
            call_order.append("second")
            return args

        guards = [(first_guard, OnFail.RAISE), (second_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="First guard failed"):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)

        # Only first guard should have been called
        assert call_order == ["first"]

    def test_runtime_error_in_guard_wrapped(self):
        """RuntimeError in guard should be wrapped in GuardError."""
        def runtime_error_guard(input_args: dict, context: dict) -> dict:
            raise RuntimeError("Unexpected runtime error")

        guards = [(runtime_error_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Unexpected runtime error"):
            _run_input_guards(guards, {"text": "hello"}, TEST_CONTEXT)
