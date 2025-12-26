"""Centralized test configuration and fixtures.

This module provides shared fixtures that reset all global state between tests,
addressing test pollution issues from shared module-level state.

Global state that must be reset:
- config_module._file_config_cache
- config_module._process_default
- config_module._config_context (ContextVar)
- spell_module._agent_cache
- logging_module._process_logging_config
- logging_module._file_logging_config_cache
- logging_module._logging_config (ContextVar)
- logging_module._trace_context (ContextVar)
"""

import sys
from contextvars import ContextVar

import pytest
from dotenv import load_dotenv

import spellcrafting.config as config_module
import spellcrafting.logging as logging_module

load_dotenv()


def _get_spell_module():
    """Get the spell module from sys.modules.

    We need to use sys.modules because `import spellcrafting.spell` would get
    the spell function from __init__.py rather than the module itself.
    """
    # Ensure module is imported
    import spellcrafting.spell  # noqa: F401
    return sys.modules["spellcrafting.spell"]


@pytest.fixture(autouse=True)
def reset_all_global_state():
    """Reset all global state before/after each test.

    This single fixture replaces the multiple autouse fixtures that were
    previously scattered across test files, ensuring consistent cleanup
    and reducing test pollution.
    """
    spell_module = _get_spell_module()

    # === BEFORE TEST ===

    # Reset config module state
    config_module._file_config_cache = None
    config_module._process_default = None

    # Reset spell module state
    spell_module._agent_cache.clear()

    # Reset logging module state
    logging_module._process_logging_config = None
    logging_module._file_logging_config_cache = None
    # Recreate ContextVars to ensure clean state (reset to default)
    logging_module._logging_config = ContextVar("logging_config", default=None)
    logging_module._trace_context = ContextVar("trace", default=None)

    yield

    # === AFTER TEST ===

    # Reset config module state
    config_module._file_config_cache = None
    config_module._process_default = None

    # Reset spell module state
    spell_module._agent_cache.clear()

    # Reset logging module state
    logging_module._process_logging_config = None
    logging_module._file_logging_config_cache = None
