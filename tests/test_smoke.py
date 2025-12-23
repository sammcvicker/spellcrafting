"""Smoke tests that hit real LLM APIs."""

import os

import pytest
from pydantic import BaseModel

from magically import spell


pytestmark = pytest.mark.smoke


class Summary(BaseModel):
    key_points: list[str]
    sentiment: str


@pytest.fixture(autouse=True)
def require_api_key():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


def test_basic_spell_returns_string():
    @spell(model="anthropic:claude-sonnet-4-20250514")
    def greet(name: str) -> str:
        """Generate a short, friendly greeting for the given name."""
        ...

    result = greet("Sam")
    assert isinstance(result, str)
    assert len(result) > 0


def test_spell_returns_structured_output():
    @spell(model="anthropic:claude-sonnet-4-20250514")
    def summarize(text: str) -> Summary:
        """Summarize the text into key points and overall sentiment."""
        ...

    result = summarize(
        "Python is a great programming language. It's easy to learn and very powerful. "
        "Many developers love using it for web development, data science, and automation."
    )
    assert isinstance(result, Summary)
    assert len(result.key_points) > 0
    assert result.sentiment in ["positive", "negative", "neutral", "mixed"]
