"""Tests for the @spell decorator."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

import sys

import magically.config as config_module
from magically import spell
from magically.config import Config, MagicallyConfigError, ModelConfig
from magically.spell import _build_user_prompt, _is_literal_model

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


class Summary(BaseModel):
    key_points: list[str]
    sentiment: str


class TestBuildUserPrompt:
    """Tests for user prompt construction from function arguments."""

    def test_single_arg(self):
        def fn(text: str) -> str:
            pass

        result = _build_user_prompt(fn, ("hello world",), {})
        assert result == "text: 'hello world'"

    def test_multiple_args(self):
        def fn(a: str, b: int) -> str:
            pass

        result = _build_user_prompt(fn, ("foo", 42), {})
        assert "a: 'foo'" in result
        assert "b: 42" in result

    def test_kwargs(self):
        def fn(text: str, count: int = 5) -> str:
            pass

        result = _build_user_prompt(fn, (), {"text": "hello", "count": 10})
        assert "text: 'hello'" in result
        assert "count: 10" in result

    def test_default_values(self):
        def fn(text: str, count: int = 5) -> str:
            pass

        result = _build_user_prompt(fn, ("hello",), {})
        assert "text: 'hello'" in result
        assert "count: 5" in result


class TestSpellDecoratorInspection:
    """Tests for function inspection by @spell decorator."""

    def test_extracts_docstring_as_system_prompt(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize the given text."""
            ...

        assert summarize._system_prompt == "Summarize the given text."

    def test_extracts_return_type(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze text."""
            ...

        assert analyze._output_type == Summary

    def test_default_return_type_is_str(self):
        @spell
        def process(text: str):
            """Process text."""
            ...

        assert process._output_type == str

    def test_preserves_function_metadata(self):
        @spell
        def my_spell(text: str) -> str:
            """My docstring."""
            ...

        assert my_spell.__name__ == "my_spell"
        assert my_spell.__doc__ == "My docstring."

    def test_stores_original_function(self):
        def original(text: str) -> str:
            """Original."""
            ...

        decorated = spell(original)
        assert decorated._original_func is original


class TestSpellDecoratorParams:
    """Tests for @spell decorator parameters."""

    def test_literal_model_param(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "openai:gpt-4o"

    def test_retries_param(self):
        @spell(retries=3)
        def fn(text: str) -> str:
            """Test."""
            ...

        assert fn._retries == 3

    def test_decorator_without_parens(self):
        @spell
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_original_func")

    def test_decorator_with_empty_parens(self):
        @spell()
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_original_func")


class TestSpellExecution:
    """Tests for @spell execution with mocked LLM."""

    def test_calls_agent_run_sync(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "Mocked summary"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = summarize("Hello world")

            mock_agent.run_sync.assert_called_once()
            assert "text: 'Hello world'" in mock_agent.run_sync.call_args[0][0]
            assert result == "Mocked summary"

    def test_returns_structured_output(self):
        @spell
        def analyze(text: str) -> Summary:
            """Analyze."""
            ...

        expected = Summary(key_points=["point1"], sentiment="positive")
        mock_result = MagicMock()
        mock_result.output = expected
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = analyze("Test text")
            assert result == expected
            assert isinstance(result, Summary)


class TestLiteralModelDetection:
    """Tests for _is_literal_model helper."""

    def test_literal_with_colon(self):
        assert _is_literal_model("openai:gpt-4o") is True
        assert _is_literal_model("anthropic:claude-sonnet") is True

    def test_alias_without_colon(self):
        assert _is_literal_model("fast") is False
        assert _is_literal_model("reasoning") is False


class TestSpellConfigIntegration:
    """Tests for @spell with Config integration."""

    def test_literal_model_bypasses_config(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "openai:gpt-4o"

    def test_alias_resolves_via_config_context(self):
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku", temperature=0.5)
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        with config:
            resolved_model, resolved_settings, _ = fn._resolve_model_and_settings()
            assert resolved_model == "anthropic:claude-haiku"
            assert resolved_settings.get("temperature") == 0.5

    def test_alias_resolves_via_process_default(self):
        config = Config(models={
            "reasoning": ModelConfig(model="anthropic:claude-opus-4")
        })
        config.set_as_default()

        @spell(model="reasoning")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "anthropic:claude-opus-4"

    def test_alias_resolves_via_file_config(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[tool.magically.models.fast]
model = "file:model"
max_tokens = 2048
""")
        monkeypatch.chdir(tmp_path)

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, resolved_settings, _ = fn._resolve_model_and_settings()
        assert resolved_model == "file:model"
        assert resolved_settings.get("max_tokens") == 2048

    def test_unresolved_alias_raises_at_call_time(self):
        @spell(model="nonexistent")
        def fn(text: str) -> str:
            """Test."""
            ...

        with pytest.raises(MagicallyConfigError, match="nonexistent"):
            fn._resolve_model_and_settings()

    def test_definition_time_warning_for_missing_alias(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")
        monkeypatch.chdir(tmp_path)

        with pytest.warns(UserWarning, match="Model alias 'missing'"):
            @spell(model="missing")
            def fn(text: str) -> str:
                """Test."""
                ...

    def test_no_warning_for_literal_model(self, tmp_path, monkeypatch):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("""
[project]
name = "test"
""")
        monkeypatch.chdir(tmp_path)

        # Should not warn for literal models
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            @spell(model="openai:gpt-4o")
            def fn(text: str) -> str:
                """Test."""
                ...

    def test_model_settings_from_config_merged_with_explicit(self):
        config = Config(models={
            "fast": ModelConfig(
                model="test:model",
                temperature=0.5,
                max_tokens=1000
            )
        })

        @spell(model="fast", model_settings={"temperature": 0.9})
        def fn(text: str) -> str:
            """Test."""
            ...

        with config:
            _, resolved_settings, _ = fn._resolve_model_and_settings()
            # Explicit temperature overrides config
            assert resolved_settings.get("temperature") == 0.9
            # Config max_tokens is preserved
            assert resolved_settings.get("max_tokens") == 1000

    def test_context_overrides_process_default_for_alias(self):
        default = Config(models={
            "fast": ModelConfig(model="default:model")
        })
        default.set_as_default()

        override = Config(models={
            "fast": ModelConfig(model="override:model")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        # Without context, uses process default
        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "default:model"

        # With context, uses context config
        with override:
            resolved_model, _, _ = fn._resolve_model_and_settings()
            assert resolved_model == "override:model"


class TestAgentCaching:
    """Tests for agent caching behavior."""

    def test_same_config_reuses_agent(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
                # First call creates agent
                fn("hello")
                assert mock_agent_class.call_count == 1

                # Second call with same config reuses agent
                fn("world")
                assert mock_agent_class.call_count == 1

                # Verify agent was used twice
                assert mock_agent.run_sync.call_count == 2

    def test_different_config_creates_new_agent(self):
        config1 = Config(models={
            "fast": ModelConfig(model="test:model1")
        })
        config2 = Config(models={
            "fast": ModelConfig(model="test:model2")
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            # Call with first config
            with config1:
                fn("hello")
            assert mock_agent_class.call_count == 1

            # Call with different config creates new agent
            with config2:
                fn("world")
            assert mock_agent_class.call_count == 2

    def test_different_spells_get_different_agents(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        def fn1(text: str) -> str:
            """Test 1."""
            ...

        @spell(model="fast")
        def fn2(text: str) -> str:
            """Test 2."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
                fn1("hello")
                assert mock_agent_class.call_count == 1

                # Different spell creates new agent even with same config
                fn2("world")
                assert mock_agent_class.call_count == 2

    def test_literal_model_caching(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            fn("hello")
            fn("world")
            # Same literal model reuses agent
            assert mock_agent_class.call_count == 1

    def test_cache_key_includes_model_settings(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model", temperature=0.5)
        })

        @spell(model="fast")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            with config:
                fn("hello")
            assert mock_agent_class.call_count == 1

            # Different temperature creates new agent
            config2 = Config(models={
                "fast": ModelConfig(model="test:model", temperature=0.9)
            })
            with config2:
                fn("world")
            assert mock_agent_class.call_count == 2

    def test_spell_id_stored_on_wrapper(self):
        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        assert hasattr(fn, "_spell_id")
        assert fn._spell_id == id(fn)


class TestAsyncSpell:
    """Tests for async @spell support."""

    def test_async_function_detected(self):
        @spell
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        assert summarize._is_async is True

    def test_sync_function_detected(self):
        @spell
        def summarize(text: str) -> str:
            """Summarize."""
            ...

        assert summarize._is_async is False

    def test_async_preserves_metadata(self):
        @spell
        async def my_async_spell(text: str) -> str:
            """My async docstring."""
            ...

        assert my_async_spell.__name__ == "my_async_spell"
        assert my_async_spell.__doc__ == "My async docstring."

    def test_async_extracts_return_type(self):
        @spell
        async def analyze(text: str) -> Summary:
            """Analyze."""
            ...

        assert analyze._output_type == Summary

    def test_async_with_model_param(self):
        @spell(model="openai:gpt-4o", retries=3)
        async def fn(text: str) -> str:
            """Test."""
            ...

        resolved_model, _, _ = fn._resolve_model_and_settings()
        assert resolved_model == "openai:gpt-4o"
        assert fn._retries == 3
        assert fn._is_async is True

    @pytest.mark.asyncio
    async def test_async_calls_agent_run(self):
        @spell
        async def summarize(text: str) -> str:
            """Summarize."""
            ...

        mock_result = MagicMock()
        mock_result.output = "Async mocked summary"
        mock_agent = MagicMock()

        # Make run return a coroutine
        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await summarize("Hello world")
            assert result == "Async mocked summary"

    @pytest.mark.asyncio
    async def test_async_returns_structured_output(self):
        @spell
        async def analyze(text: str) -> Summary:
            """Analyze."""
            ...

        expected = Summary(key_points=["async point"], sentiment="positive")
        mock_result = MagicMock()
        mock_result.output = expected
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with patch("magically.spell.Agent", return_value=mock_agent):
            result = await analyze("Test text")
            assert result == expected
            assert isinstance(result, Summary)

    @pytest.mark.asyncio
    async def test_async_agent_caching(self):
        config = Config(models={
            "fast": ModelConfig(model="test:model")
        })

        @spell(model="fast")
        async def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()

        async def mock_run(prompt):
            return mock_result
        mock_agent.run = mock_run

        with config:
            with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
                # First call creates agent
                await fn("hello")
                assert mock_agent_class.call_count == 1

                # Second call with same config reuses agent
                await fn("world")
                assert mock_agent_class.call_count == 1


class TestCacheManagement:
    """Tests for cache management API."""

    def test_clear_agent_cache(self):
        from magically import clear_agent_cache, get_cache_stats

        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            fn("hello")

        # Cache should have 1 agent
        stats = get_cache_stats()
        assert stats.size == 1

        # Clear and verify
        cleared = clear_agent_cache()
        assert cleared == 1

        stats = get_cache_stats()
        assert stats.size == 0

    def test_get_cache_stats_returns_correct_values(self):
        from magically import clear_agent_cache, get_cache_stats

        # Start fresh
        clear_agent_cache()

        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        with patch("magically.spell.Agent", return_value=mock_agent):
            # First call - cache miss
            fn("hello")

            # Second call - cache hit
            fn("world")

        stats = get_cache_stats()
        assert stats.size == 1
        assert stats.max_size == 100  # default
        assert stats.misses >= 1
        assert stats.hits >= 1

    def test_set_cache_max_size(self):
        from magically import set_cache_max_size, get_cache_stats, clear_agent_cache

        # Start fresh
        clear_agent_cache()

        # Set small max size
        set_cache_max_size(2)

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        # Create 3 spells (more than max size)
        @spell(model="openai:gpt-4o")
        def fn1(text: str) -> str:
            """Test 1."""
            ...

        @spell(model="openai:gpt-4o")
        def fn2(text: str) -> str:
            """Test 2."""
            ...

        @spell(model="openai:gpt-4o")
        def fn3(text: str) -> str:
            """Test 3."""
            ...

        with patch("magically.spell.Agent", return_value=mock_agent):
            fn1("a")
            fn2("b")
            fn3("c")

        stats = get_cache_stats()
        # Should only have max_size agents
        assert stats.size == 2
        assert stats.max_size == 2
        assert stats.evictions >= 1

        # Restore default
        set_cache_max_size(100)

    def test_set_cache_max_size_negative_raises(self):
        from magically import set_cache_max_size

        with pytest.raises(ValueError, match="non-negative"):
            set_cache_max_size(-1)

    def test_lru_eviction_order(self):
        from magically import set_cache_max_size, clear_agent_cache, get_cache_stats

        # Start fresh
        clear_agent_cache()

        # Set small max size
        set_cache_max_size(2)

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        @spell(model="openai:gpt-4o")
        def fn1(text: str) -> str:
            """Test 1."""
            ...

        @spell(model="openai:gpt-4o")
        def fn2(text: str) -> str:
            """Test 2."""
            ...

        @spell(model="openai:gpt-4o")
        def fn3(text: str) -> str:
            """Test 3."""
            ...

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            # Fill cache with fn1, fn2
            fn1("a")  # Creates agent 1
            fn2("b")  # Creates agent 2

            # Access fn1 again (makes it most recently used)
            fn1("c")

            # Add fn3 (should evict fn2, not fn1)
            fn3("d")

            # fn1 should still be cached (was used more recently)
            fn1("e")

            # Count total agent creations
            # fn1: 1 creation, fn2: 1 creation, fn3: 1 creation, fn1 again: 0 (cached)
            assert mock_agent_class.call_count == 3

        stats = get_cache_stats()
        assert stats.evictions >= 1

        # Restore default
        set_cache_max_size(100)

    def test_cache_stats_dataclass(self):
        from magically import CacheStats

        stats = CacheStats(size=5, max_size=100, hits=10, misses=3, evictions=2)
        assert stats.size == 5
        assert stats.max_size == 100
        assert stats.hits == 10
        assert stats.misses == 3
        assert stats.evictions == 2

    def test_disable_caching_with_zero_max_size(self):
        from magically import set_cache_max_size, get_cache_stats, clear_agent_cache

        # Start fresh
        clear_agent_cache()

        # Disable caching
        set_cache_max_size(0)

        mock_result = MagicMock()
        mock_result.output = "result"
        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result

        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        with patch("magically.spell.Agent", return_value=mock_agent) as mock_agent_class:
            fn("a")
            fn("b")
            fn("c")

            # Each call should create a new agent (no caching)
            assert mock_agent_class.call_count == 3

        stats = get_cache_stats()
        assert stats.size == 0
        assert stats.max_size == 0

        # Restore default
        set_cache_max_size(100)


class TestConcurrentSpellExecution:
    """Tests for concurrent spell execution with shared cache (#175).

    The agent cache is shared across all spell invocations. These tests verify
    thread safety and correct behavior under concurrent access.
    """

    def test_concurrent_sync_calls_same_spell(self):
        """Multiple threads calling same spell should share agent."""
        import threading

        @spell(model="openai:gpt-4o")
        def fn(text: str) -> str:
            """Test."""
            ...

        agents_created = []
        lock = threading.Lock()

        def create_agent(*args, **kwargs):
            with lock:
                agents_created.append(True)
            agent = MagicMock()
            agent.run_sync.return_value = MagicMock(output="result")
            return agent

        results = []
        errors = []

        def run_spell():
            try:
                result = fn("test")
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with patch("magically.spell.Agent", side_effect=create_agent):
            threads = [
                threading.Thread(target=run_spell)
                for _ in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        # Due to race conditions, we might create 1-2 agents
        # (threads racing to populate empty cache)
        # The important thing is we don't create 10 agents
        assert len(agents_created) <= 3

    def test_concurrent_sync_calls_different_spells(self):
        """Multiple threads calling different spells should create separate agents."""
        import threading

        @spell(model="openai:gpt-4o")
        def spell1(text: str) -> str:
            """Spell 1."""
            ...

        @spell(model="openai:gpt-4o")
        def spell2(text: str) -> str:
            """Spell 2."""
            ...

        agents_created = []
        lock = threading.Lock()

        def create_agent(*args, **kwargs):
            with lock:
                agents_created.append(True)
            agent = MagicMock()
            agent.run_sync.return_value = MagicMock(output="result")
            return agent

        results = []

        def run_spell1():
            results.append(("s1", spell1("test")))

        def run_spell2():
            results.append(("s2", spell2("test")))

        with patch("magically.spell.Agent", side_effect=create_agent):
            threads = []
            for _ in range(5):
                threads.append(threading.Thread(target=run_spell1))
                threads.append(threading.Thread(target=run_spell2))

            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Should have created at least 2 agents (one per spell)
        # Might create a few more due to race conditions
        assert len(agents_created) >= 2
        assert len(agents_created) <= 6  # Some reasonable upper bound

    @pytest.mark.asyncio
    async def test_concurrent_async_calls_same_spell(self):
        """Multiple async tasks calling same spell should share agent."""
        import asyncio

        @spell(model="openai:gpt-4o")
        async def fn(text: str) -> str:
            """Test."""
            ...

        agents_created = []

        def create_agent(*args, **kwargs):
            agents_created.append(True)
            agent = MagicMock()

            async def mock_run(prompt):
                return MagicMock(output="result")
            agent.run = mock_run
            return agent

        with patch("magically.spell.Agent", side_effect=create_agent):
            tasks = [fn("test") for _ in range(10)]
            results = await asyncio.gather(*tasks)

        assert len(results) == 10
        # In async, first task creates the agent, others might race
        # but should still be limited
        assert len(agents_created) <= 3

    def test_cache_thread_safety_during_eviction(self):
        """Cache should be thread-safe during LRU eviction."""
        import threading
        from magically import set_cache_max_size, clear_agent_cache

        clear_agent_cache()
        set_cache_max_size(2)

        @spell(model="openai:gpt-4o")
        def spell1(text: str) -> str:
            """S1."""
            ...

        @spell(model="openai:gpt-4o")
        def spell2(text: str) -> str:
            """S2."""
            ...

        @spell(model="openai:gpt-4o")
        def spell3(text: str) -> str:
            """S3."""
            ...

        mock_result = MagicMock()
        mock_result.output = "result"

        def create_agent(*args, **kwargs):
            agent = MagicMock()
            agent.run_sync.return_value = mock_result
            return agent

        errors = []

        def run_spells():
            try:
                spell1("a")
                spell2("b")
                spell3("c")
            except Exception as e:
                errors.append(e)

        with patch("magically.spell.Agent", side_effect=create_agent):
            threads = [threading.Thread(target=run_spells) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        # Should complete without errors
        assert len(errors) == 0

        # Restore default cache size
        set_cache_max_size(100)
