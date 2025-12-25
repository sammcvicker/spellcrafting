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


async def test_async_spell_returns_string():
    @spell(model="anthropic:claude-sonnet-4-20250514")
    async def greet_async(name: str) -> str:
        """Generate a short, friendly greeting for the given name."""
        ...

    result = await greet_async("Alice")
    assert isinstance(result, str)
    assert len(result) > 0


async def test_async_spell_returns_structured_output():
    @spell(model="anthropic:claude-sonnet-4-20250514")
    async def summarize_async(text: str) -> Summary:
        """Summarize the text into key points and overall sentiment."""
        ...

    result = await summarize_async(
        "Rust is a systems programming language focused on safety and performance. "
        "It prevents memory errors at compile time and has a growing community."
    )
    assert isinstance(result, Summary)
    assert len(result.key_points) > 0


def test_spell_with_tools():
    """Test that spells can use tools to gather information."""

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        # Simulated weather data
        weather_data = {
            "tokyo": "72°F and sunny",
            "london": "55°F and rainy",
            "new york": "65°F and cloudy",
        }
        return weather_data.get(city.lower(), "Weather data not available")

    def get_population(city: str) -> str:
        """Get the population of a city."""
        population_data = {
            "tokyo": "14 million",
            "london": "9 million",
            "new york": "8 million",
        }
        return population_data.get(city.lower(), "Population data not available")

    class CityInfo(BaseModel):
        city: str
        weather: str
        population: str

    @spell(model="anthropic:claude-sonnet-4-20250514", tools=[get_weather, get_population])
    def get_city_info(city: str) -> CityInfo:
        """Get information about a city using the available tools."""
        ...

    result = get_city_info("Tokyo")
    assert isinstance(result, CityInfo)
    assert result.city.lower() == "tokyo"
    assert "72" in result.weather or "sunny" in result.weather.lower()
    assert "14" in result.population or "million" in result.population.lower()


async def test_async_spell_with_tools():
    """Test that async spells can use tools."""

    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression."""
        try:
            # Safe eval for simple math
            allowed = set("0123456789+-*/.(). ")
            if all(c in allowed for c in expression):
                return str(eval(expression))
            return "Invalid expression"
        except Exception:
            return "Error evaluating expression"

    class MathResult(BaseModel):
        expression: str
        result: float
        explanation: str

    @spell(model="anthropic:claude-sonnet-4-20250514", tools=[calculate])
    async def solve_math(problem: str) -> MathResult:
        """Solve a math problem using the calculator tool."""
        ...

    result = await solve_math("What is 15 * 7 + 23?")
    assert isinstance(result, MathResult)
    assert result.result == 128.0  # 15 * 7 + 23 = 128
