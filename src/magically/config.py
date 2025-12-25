"""Configuration system for magically.

Separates intent (model aliases) from implementation (provider strings).
"""

from __future__ import annotations

import tomllib
import warnings
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError

from magically._pyproject import find_pyproject
from magically.exceptions import MagicallyConfigError


_config_context: ContextVar[Config | None] = ContextVar("magically_config", default=None)
_process_default: Config | None = None
_file_config_cache: Config | None = None


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

    def merge(self, other: Config) -> Config:
        """Merge another config on top of this one.

        Creates a new Config with aliases from both configs.
        Aliases from `other` take precedence over aliases from `self`.

        Args:
            other: Config to merge on top of this one.

        Returns:
            A new Config with merged aliases.
        """
        merged_models = {**self._models, **other._models}
        return Config(models=merged_models)

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
            pyproject_path = find_pyproject()

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

        # Validate each model config eagerly with helpful error messages
        validated_models: dict[str, ModelConfig] = {}

        for alias, settings in models_config.items():
            if not isinstance(settings, dict):
                raise MagicallyConfigError(
                    f"Invalid config for [tool.magically.models.{alias}] in {pyproject_path}: "
                    f"expected a table/dict, got {type(settings).__name__}"
                )

            # Warn about unknown fields (extra fields are ignored by Pydantic)
            known_fields = {"model", "temperature", "max_tokens", "top_p", "timeout", "retries", "extra"}
            unknown = set(settings.keys()) - known_fields
            if unknown:
                warnings.warn(
                    f"Unknown fields in [tool.magically.models.{alias}]: {unknown}",
                    stacklevel=2,
                )

            # Validate with Pydantic and provide helpful error on failure
            try:
                validated_models[alias] = ModelConfig.model_validate(settings)
            except PydanticValidationError as e:
                # Format the Pydantic errors into a readable message
                error_details = []
                for error in e.errors():
                    loc = ".".join(str(x) for x in error["loc"]) if error["loc"] else "root"
                    msg = error["msg"]
                    error_details.append(f"  - {loc}: {msg}")

                raise MagicallyConfigError(
                    f"Invalid config for [tool.magically.models.{alias}] in {pyproject_path}:\n"
                    + "\n".join(error_details)
                ) from e

        return cls(models=validated_models)

    @classmethod
    def current(cls) -> Config:
        """Get the currently active config.

        Configs are merged with higher-priority sources taking precedence per-alias.
        If context defines `fast` and file defines `reasoning`, both are available.

        Resolution order (highest priority first):
        1. Active context (from `with config:`)
        2. Process default (from `config.set_as_default()`)
        3. File config (from pyproject.toml, cached)
        """
        global _file_config_cache

        # Start with file config as base (loaded once and cached)
        if _file_config_cache is None:
            _file_config_cache = cls.from_file()
        base = _file_config_cache

        # Merge process default on top
        if _process_default is not None:
            base = base.merge(_process_default)

        # Merge context on top
        context_config = _config_context.get()
        if context_config is not None:
            base = base.merge(context_config)

        return base


def current_config() -> Config:
    """Get the currently active config."""
    return Config.current()
