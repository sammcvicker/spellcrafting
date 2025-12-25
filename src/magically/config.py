"""Configuration system for magically.

Separates intent (model aliases) from implementation (provider strings).
"""

from __future__ import annotations

import tomllib
import warnings
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


_config_context: ContextVar[Config | None] = ContextVar("magically_config", default=None)
_process_default: Config | None = None
_file_config_cache: Config | None = None


class MagicallyConfigError(Exception):
    """Raised when configuration is invalid or missing."""


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

    extra: dict[str, Any] = Field(default_factory=dict)
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
    def from_file(cls, path: Path | str | None = None) -> Config:
        """Load config from pyproject.toml.

        Parses [tool.magically.models.*] sections into model aliases.

        Args:
            path: Path to pyproject.toml. If None, searches from cwd upward.

        Returns:
            Config with model aliases from file, or empty Config if not found.

        Example pyproject.toml:
            [tool.magically.models.fast]
            model = "anthropic:claude-haiku"

            [tool.magically.models.reasoning]
            model = "anthropic:claude-opus-4"
            temperature = 0.7
        """
        if path is not None:
            pyproject_path = Path(path)
        else:
            pyproject_path = cls._find_pyproject()

        if pyproject_path is None or not pyproject_path.exists():
            return cls()

        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError:
            return cls()

        tool_config = data.get("tool", {}).get("magically", {})
        models_config = tool_config.get("models", {})

        if not models_config:
            return cls()

        # Validate and warn about unknown fields
        known_fields = {"model", "temperature", "max_tokens", "top_p", "timeout", "retries", "extra"}
        models: dict[str, dict[str, Any]] = {}

        for alias, settings in models_config.items():
            if not isinstance(settings, dict):
                continue
            unknown = set(settings.keys()) - known_fields
            if unknown:
                warnings.warn(
                    f"Unknown fields in [tool.magically.models.{alias}]: {unknown}",
                    stacklevel=2,
                )
            models[alias] = settings

        return cls(models=models)

    @classmethod
    def _find_pyproject(cls) -> Path | None:
        """Search for pyproject.toml from cwd upward."""
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                return candidate
        return None

    @classmethod
    def current(cls) -> Config:
        """Get the currently active config.

        Resolution order:
        1. Active context (from `with config:`)
        2. Process default (from `config.set_as_default()`)
        3. File config (from pyproject.toml, cached)
        """
        global _file_config_cache

        # Check context first
        context_config = _config_context.get()
        if context_config is not None:
            return context_config

        # Then process default
        if _process_default is not None:
            return _process_default

        # Fall back to file config (loaded once and cached)
        if _file_config_cache is None:
            _file_config_cache = cls.from_file()
        return _file_config_cache


def current_config() -> Config:
    """Get the currently active config."""
    return Config.current()
