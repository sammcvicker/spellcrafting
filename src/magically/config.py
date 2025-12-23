"""Configuration system for magically.

Separates intent (model aliases) from implementation (provider strings).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class MagicallyConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class ModelConfig(BaseModel):
    """Configuration for a model alias.

    Bundles a model identifier with its settings.

    Example:
        ModelConfig(
            model="anthropic:claude-sonnet",
            temperature=0.7,
            max_tokens=4096,
        )
    """

    model_config = ConfigDict(extra="ignore")

    model: str
    """Provider:model-name format (e.g., 'anthropic:claude-sonnet')."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float | None = None
    retries: int | None = None

    extra: dict[str, Any] = {}
    """Passthrough for provider-specific settings."""

    def __hash__(self) -> int:
        """Hash for agent caching."""
        return hash(
            (
                self.model,
                self.temperature,
                self.max_tokens,
                self.top_p,
                self.timeout,
                self.retries,
                tuple(sorted(self.extra.items())),
            )
        )
