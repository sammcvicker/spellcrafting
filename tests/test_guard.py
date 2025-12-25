"""Tests for the @guard decorator."""

import sys
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import magically.config as config_module
from magically import spell, guard, GuardError, OnFail
from magically.guard import (
    _get_or_create_guard_config,
    _run_input_guards,
    _run_output_guards,
    _build_context,
    _GUARD_MARKER,
)

# Access the actual spell module (not the function)
spell_module = sys.modules["magically.spell"]


@pytest.fixture(autouse=True)
def reset_config():
    """Reset config state between tests."""
    config_module._file_config_cache = None
    config_module._process_default = None
    yield
    config_module._file_config_cache = None
    config_module._process_default = None


@pytest.fixture(autouse=True)
def reset_agent_cache():
    """Reset agent cache between tests."""
    spell_module._agent_cache.clear()
    yield
    spell_module._agent_cache.clear()


class TestGuardDecoratorsBasic:
    """Tests for guard decorator structure and metadata."""

    def test_guard_input_adds_to_config(self):
        def my_guard(input_args: dict, context: dict) -> dict:
            return input_args

        @guard.input(my_guard)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.input_guards) == 1
        assert config.input_guards[0][0] is my_guard
        assert config.input_guards[0][1] == OnFail.RAISE

    def test_guard_output_adds_to_config(self):
        def my_guard(output: str, context: dict) -> str:
            return output

        @guard.output(my_guard)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.output_guards) == 1
        assert config.output_guards[0][0] is my_guard
        assert config.output_guards[0][1] == OnFail.RAISE

    def test_multiple_input_guards(self):
        def guard1(args: dict, ctx: dict) -> dict:
            return args

        def guard2(args: dict, ctx: dict) -> dict:
            return args

        @guard.input(guard1)
        @guard.input(guard2)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.input_guards) == 2
        # Decorators apply bottom-up: guard2 first, then guard1
        # With insert(0, ...), guard1 ends up first in the list
        assert config.input_guards[0][0] is guard1
        assert config.input_guards[1][0] is guard2

    def test_multiple_output_guards(self):
        def guard1(out: str, ctx: dict) -> str:
            return out

        def guard2(out: str, ctx: dict) -> str:
            return out

        @guard.output(guard1)
        @guard.output(guard2)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.output_guards) == 2
        # Decorators apply bottom-up: guard2 first, then guard1
        # With append(...), guard2 ends up first in the list, guard1 second
        assert config.output_guards[0][0] is guard2
        assert config.output_guards[1][0] is guard1

    def test_combined_input_and_output_guards(self):
        def in_guard(args: dict, ctx: dict) -> dict:
            return args

        def out_guard(out: str, ctx: dict) -> str:
            return out

        @guard.input(in_guard)
        @guard.output(out_guard)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.input_guards) == 1
        assert len(config.output_guards) == 1


class TestInputGuardExecution:
    """Tests for input guard execution logic."""

    def test_input_guard_receives_args(self):
        received_args = {}

        def capture_guard(input_args: dict, context: dict) -> dict:
            received_args.update(input_args)
            return input_args

        guards = [(capture_guard, OnFail.RAISE)]
        input_args = {"text": "hello", "count": 5}

        _run_input_guards(guards, input_args, {})
        assert received_args == {"text": "hello", "count": 5}

    def test_input_guard_receives_context(self):
        received_context = {}

        def capture_guard(input_args: dict, context: dict) -> dict:
            received_context.update(context)
            return input_args

        guards = [(capture_guard, OnFail.RAISE)]
        context = {"spell_name": "test_spell", "model": "fast"}

        _run_input_guards(guards, {}, context)
        assert received_context["spell_name"] == "test_spell"
        assert received_context["model"] == "fast"

    def test_input_guard_can_transform_args(self):
        def uppercase_guard(input_args: dict, context: dict) -> dict:
            return {k: v.upper() if isinstance(v, str) else v for k, v in input_args.items()}

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, {})

        assert result == {"text": "HELLO"}

    def test_input_guard_chain_transforms(self):
        def add_prefix(args: dict, ctx: dict) -> dict:
            return {k: f"prefix_{v}" if isinstance(v, str) else v for k, v in args.items()}

        def add_suffix(args: dict, ctx: dict) -> dict:
            return {k: f"{v}_suffix" if isinstance(v, str) else v for k, v in args.items()}

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]
        result = _run_input_guards(guards, {"text": "hello"}, {})

        assert result == {"text": "prefix_hello_suffix"}

    def test_input_guard_raises_guard_error(self):
        def rejecting_guard(input_args: dict, context: dict) -> dict:
            raise ValueError("Invalid input")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Invalid input"):
            _run_input_guards(guards, {"text": "hello"}, {})

    def test_input_guard_preserves_guard_error(self):
        def rejecting_guard(input_args: dict, context: dict) -> dict:
            raise GuardError("Already a guard error")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Already a guard error"):
            _run_input_guards(guards, {}, {})


class TestOutputGuardExecution:
    """Tests for output guard execution logic."""

    def test_output_guard_receives_output(self):
        received_output = []

        def capture_guard(output: str, context: dict) -> str:
            received_output.append(output)
            return output

        guards = [(capture_guard, OnFail.RAISE)]

        _run_output_guards(guards, "hello world", {})
        assert received_output == ["hello world"]

    def test_output_guard_receives_context(self):
        received_context = {}

        def capture_guard(output: str, context: dict) -> str:
            received_context.update(context)
            return output

        guards = [(capture_guard, OnFail.RAISE)]

        _run_output_guards(guards, "test", {"spell_name": "my_spell"})
        assert received_context["spell_name"] == "my_spell"

    def test_output_guard_can_transform(self):
        def uppercase_guard(output: str, context: dict) -> str:
            return output.upper()

        guards = [(uppercase_guard, OnFail.RAISE)]
        result = _run_output_guards(guards, "hello", {})

        assert result == "HELLO"

    def test_output_guard_chain_transforms(self):
        def add_prefix(out: str, ctx: dict) -> str:
            return f"prefix_{out}"

        def add_suffix(out: str, ctx: dict) -> str:
            return f"{out}_suffix"

        guards = [(add_prefix, OnFail.RAISE), (add_suffix, OnFail.RAISE)]
        result = _run_output_guards(guards, "hello", {})

        assert result == "prefix_hello_suffix"

    def test_output_guard_raises_guard_error(self):
        def rejecting_guard(output: str, context: dict) -> str:
            raise ValueError("Invalid output")

        guards = [(rejecting_guard, OnFail.RAISE)]

        with pytest.raises(GuardError, match="Invalid output"):
            _run_output_guards(guards, "test", {})


class TestMaxLengthGuard:
    """Tests for the built-in max_length guard."""

    def test_max_length_input_passes(self):
        @guard.max_length(input=100)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        result = _run_input_guards(
            config.input_guards,
            {"text": "short text"},
            {},
        )
        assert result == {"text": "short text"}

    def test_max_length_input_fails(self):
        @guard.max_length(input=10)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)

        with pytest.raises(GuardError, match="exceeds maximum length of 10"):
            _run_input_guards(
                config.input_guards,
                {"text": "this is a longer text"},
                {},
            )

    def test_max_length_output_passes(self):
        @guard.max_length(output=100)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        result = _run_output_guards(
            config.output_guards,
            "short output",
            {},
        )
        assert result == "short output"

    def test_max_length_output_fails(self):
        @guard.max_length(output=5)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)

        with pytest.raises(GuardError, match="exceeds maximum length of 5"):
            _run_output_guards(
                config.output_guards,
                "longer output",
                {},
            )

    def test_max_length_both_input_and_output(self):
        @guard.max_length(input=100, output=50)
        def fn(text: str) -> str:
            return text

        config = getattr(fn, _GUARD_MARKER)
        assert len(config.input_guards) == 1
        assert len(config.output_guards) == 1


class TestBuildContext:
    """Tests for context building helper."""

    def test_build_context_basic(self):
        def my_func():
            pass

        context = _build_context(my_func)
        assert context["spell_name"] == "my_func"
        assert context["attempt_number"] == 1

    def test_build_context_with_model_alias(self):
        def my_func():
            pass

        my_func._model_alias = "fast"

        context = _build_context(my_func)
        assert context["model"] == "fast"

    def test_build_context_with_attempt(self):
        def my_func():
            pass

        context = _build_context(my_func, attempt=3)
        assert context["attempt_number"] == 3


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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = summarize("test")
            assert result == "SUMMARY"

    def test_max_length_guard_with_spell(self):
        @spell
        @guard.max_length(input=10, output=100)
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
            summarize("hello")

            # guard1 runs first (outermost decorator applied last via insert(0))
            # then guard2
            assert transforms == ["g1", "g2"]

            # Check the combined transformation
            call_args = mock_agent.run_sync.call_args[0][0]
            assert "[G2][G1]hello" in call_args


class TestAsyncGuardWithSpell:
    """Tests for guard integration with async @spell decorator."""

    @pytest.mark.asyncio
    async def test_async_input_guard_runs(self):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
            await summarize("test")

        assert guard_called == [True]

    @pytest.mark.asyncio
    async def test_async_output_guard_runs(self):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
            await summarize("test")

        assert guard_called == [True]

    @pytest.mark.asyncio
    async def test_async_guard_can_reject(self):
        def rejecting_guard(args: dict, ctx: dict) -> dict:
            raise ValueError("Rejected!")

        @spell
        @guard.input(rejecting_guard)
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_agent = MagicMock()

        with patch("magically.spell.Agent", return_value=mock_agent):
            with pytest.raises(GuardError, match="Rejected!"):
                await summarize("test")


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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

            @guard.max_length(input=100)
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

        with patch("magically.spell.Agent", return_value=mock_agent):
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

        with patch("magically.spell.Agent", return_value=mock_agent):
            my_spell("test")

        # Guard inside @spell IS integrated and runs
        assert guard_calls == ["guard_called"]
