import sys

import pytest
from dotenv import load_dotenv

import magically.config as config_module

load_dotenv()

# Access spell module for cache reset
spell_module = sys.modules.get("magically.spell")


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: tests that hit real LLM APIs")


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
    if spell_module:
        spell_module._agent_cache.clear()
    yield
    if spell_module:
        spell_module._agent_cache.clear()
