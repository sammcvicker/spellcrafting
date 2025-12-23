import pytest
from dotenv import load_dotenv

load_dotenv()


def pytest_configure(config):
    config.addinivalue_line("markers", "smoke: tests that hit real LLM APIs")
