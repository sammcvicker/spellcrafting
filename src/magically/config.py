"""Configuration system for magically.

Separates intent (model aliases) from implementation (provider strings).
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


_config_context: ContextVar[Config | None] = ContextVar("magically_config", default=None)
_process_default: Config | None = None


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


class Config:
    """Immutable configuration container.

    Holds model aliases and their configurations. Supports context manager
    for scoped overrides and process-level defaults.

    Example:
        config = Config(models={
            "fast": ModelConfig(model="anthropic:claude-haiku"),
            "reasoning": {"model": "anthropic:claude-opus-4", "temperature": 0.7},
        })

        with config:
            result = my_spell(...)  # uses this config
    """

    def __init__(
        self,
        models: dict[str, ModelConfig | dict[str, Any]] | None = None,
    ) -> None:
        """Create config. Dicts are coerced to ModelConfig."""
        self._models: dict[str, ModelConfig] = {}
        self._token: Token[Config | None] | None = None

        if models:
            for alias, model_config in models.items():
                if isinstance(model_config, dict):
                    self._models[alias] = ModelConfig.model_validate(model_config)
                else:
                    self._models[alias] = model_config

    @property
    def models(self) -> dict[str, ModelConfig]:
        """Get all model aliases."""
        return self._models.copy()

    def resolve(self, alias: str) -> ModelConfig:
        """Resolve alias to ModelConfig.

        Args:
            alias: The model alias to resolve.

        Returns:
            The ModelConfig for the alias.

        Raises:
            MagicallyConfigError: If alias is not found.
        """
        if alias not in self._models:
            available = list(self._models.keys())
            raise MagicallyConfigError(
                f"Unknown model alias '{alias}'. Available: {available}"
            )
        return self._models[alias]

    def __enter__(self) -> Self:
        """Push this config onto the context stack."""
        self._token = _config_context.set(self)
        return self

    def __exit__(self, *exc: object) -> None:
        """Pop this config from the context stack."""
        if self._token is not None:
            _config_context.reset(self._token)
            self._token = None

    def set_as_default(self) -> None:
        """Set as process-level default (below context, above file config)."""
        global _process_default
        _process_default = self

    @classmethod
    def current(cls) -> Config:
        """Get the currently active config.

        Resolution order:
        1. Active context (from `with config:`)
        2. Process default (from `config.set_as_default()`)
        3. Empty config (file loading comes in step 3)
        """
        # Check context first
        context_config = _config_context.get()
        if context_config is not None:
            return context_config

        # Then process default
        if _process_default is not None:
            return _process_default

        # Fall back to empty config (file loading added in step 3)
        return cls()


def current_config() -> Config:
    """Get the currently active config."""
    return Config.current()
