"""Configuration system for magically.

Separates intent (model aliases) from implementation (provider strings).

Design Note: Pydantic vs Dataclass Usage
----------------------------------------
This module uses Pydantic BaseModel for `ModelConfig` because:
- It validates user-provided configuration from pyproject.toml
- It parses and coerces TOML data into typed Python objects
- It provides helpful error messages for invalid config

See logging.py and result.py for contrast - those use dataclasses for
internal telemetry types where validation overhead is not needed.
"""

from __future__ import annotations

import os
import tomllib
import warnings
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Self

import json
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError as PydanticValidationError

from magically._pydantic_ai import ModelSettings
from magically._pyproject import find_pyproject
from magically.exceptions import MagicallyConfigError


# Environment variable names for configuration
ENV_DEFAULT_MODEL = "MAGICALLY_DEFAULT_MODEL"
ENV_DEFAULT_TIMEOUT = "MAGICALLY_DEFAULT_TIMEOUT"
ENV_DEFAULT_TEMPERATURE = "MAGICALLY_DEFAULT_TEMPERATURE"
ENV_DEFAULT_MAX_TOKENS = "MAGICALLY_DEFAULT_MAX_TOKENS"

# Default timeout for LLM calls (in seconds)
# This prevents indefinite hangs when providers are slow or unresponsive
DEFAULT_TIMEOUT = 120.0  # 2 minutes


_config_context: ContextVar[Config | None] = ContextVar("magically_config", default=None)
_process_default: Config | None = None
_file_config_cache: Config | None = None
_env_config_cache: Config | None = None


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

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate and normalize model string format.

        - Strips whitespace
        - Validates provider:model format if colon is present
        - Ensures both provider and model name are non-empty
        """
        v = v.strip()
        if not v:
            raise ValueError("Model string cannot be empty")

        if ":" in v:
            # Split on first colon only to allow model names with colons
            parts = v.split(":", 1)
            provider = parts[0].strip()
            model_name = parts[1].strip()

            if not provider:
                raise ValueError(f"Invalid model format: {v!r}. Provider cannot be empty.")
            if not model_name:
                raise ValueError(f"Invalid model format: {v!r}. Model name cannot be empty.")

            # Return normalized format (stripped)
            return f"{provider}:{model_name}"

        return v

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float | None = None
    retries: int | None = None

    extra: dict[str, Any] = Field(default_factory=dict)
    """Passthrough for provider-specific settings."""

    def __hash__(self) -> int:
        """Hash for agent caching.

        Uses JSON serialization for the extra dict to handle nested
        dicts and lists that would otherwise be unhashable.
        """
        # Serialize extra values to JSON strings to handle nested dicts/lists
        extra_items = tuple(
            sorted((k, json.dumps(v, sort_keys=True)) for k, v in self.extra.items())
        )
        return hash(
            (
                self.model,
                self.temperature,
                self.max_tokens,
                self.top_p,
                self.timeout,
                self.retries,
                extra_items,
            )
        )

    def to_model_settings(self) -> ModelSettings | None:
        """Convert to PydanticAI ModelSettings.

        Builds a ModelSettings dict from the configured values,
        only including non-None values. Uses DEFAULT_TIMEOUT if no
        timeout is explicitly configured.

        Returns:
            ModelSettings dict if any settings are configured, None otherwise.
        """
        settings: dict[str, Any] = {}
        if self.temperature is not None:
            settings["temperature"] = self.temperature
        if self.max_tokens is not None:
            settings["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            settings["top_p"] = self.top_p
        # Use configured timeout or fall back to default
        settings["timeout"] = self.timeout if self.timeout is not None else DEFAULT_TIMEOUT
        return ModelSettings(**settings) if settings else None


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
        except tomllib.TOMLDecodeError as e:
            warnings.warn(
                f"Failed to parse {pyproject_path}: {e}. "
                "Magically configuration will be ignored.",
                stacklevel=2,
            )
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
    def from_env(cls) -> Config:
        """Load config from environment variables.

        Reads environment variables to create a 'default' model alias:
        - MAGICALLY_DEFAULT_MODEL: Model identifier (e.g., 'anthropic:claude-sonnet')
        - MAGICALLY_DEFAULT_TIMEOUT: Request timeout in seconds
        - MAGICALLY_DEFAULT_TEMPERATURE: Temperature (0.0-1.0)
        - MAGICALLY_DEFAULT_MAX_TOKENS: Maximum tokens to generate

        Returns:
            Config with 'default' model alias if MAGICALLY_DEFAULT_MODEL is set,
            or empty Config if not set.

        Example:
            # Set environment variables:
            # export MAGICALLY_DEFAULT_MODEL=anthropic:claude-sonnet
            # export MAGICALLY_DEFAULT_TEMPERATURE=0.7

            config = Config.from_env()
            # config.resolve("default") will return the model config
        """
        env_model = os.environ.get(ENV_DEFAULT_MODEL)
        if not env_model:
            return cls()

        # Build model config from env vars
        model_settings: dict[str, Any] = {"model": env_model}

        env_timeout = os.environ.get(ENV_DEFAULT_TIMEOUT)
        if env_timeout:
            try:
                model_settings["timeout"] = float(env_timeout)
            except ValueError:
                pass  # Ignore invalid timeout values

        env_temp = os.environ.get(ENV_DEFAULT_TEMPERATURE)
        if env_temp:
            try:
                model_settings["temperature"] = float(env_temp)
            except ValueError:
                pass  # Ignore invalid temperature values

        env_max_tokens = os.environ.get(ENV_DEFAULT_MAX_TOKENS)
        if env_max_tokens:
            try:
                model_settings["max_tokens"] = int(env_max_tokens)
            except ValueError:
                pass  # Ignore invalid max_tokens values

        return cls(models={"default": model_settings})

    @classmethod
    def current(cls) -> Config:
        """Get the currently active config.

        Configs are merged with higher-priority sources taking precedence per-alias.
        If context defines `fast` and file defines `reasoning`, both are available.

        Resolution order (highest priority first):
        1. Active context (from `with config:`)
        2. Process default (from `config.set_as_default()`)
        3. Environment variables (from MAGICALLY_* env vars, cached)
        4. File config (from pyproject.toml, cached)
        """
        global _file_config_cache, _env_config_cache

        # Start with file config as base (loaded once and cached)
        if _file_config_cache is None:
            _file_config_cache = cls.from_file()
        base = _file_config_cache

        # Merge env config on top of file config
        if _env_config_cache is None:
            _env_config_cache = cls.from_env()
        if _env_config_cache._models:
            base = base.merge(_env_config_cache)

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


def clear_config_cache() -> None:
    """Clear the cached file and environment configuration.

    This function clears all cached configuration, forcing the next call
    to Config.current() to reload from pyproject.toml and environment
    variables.

    Useful for:
    - Development when pyproject.toml changes (with hot-reload frameworks)
    - Testing when you need to reset config state
    - Forcing re-read of environment variables after they change

    Note:
        This does NOT clear programmatic config set via set_as_default()
        or active context managers. Those must be explicitly managed.

    Example:
        # After modifying pyproject.toml:
        clear_config_cache()
        config = Config.current()  # Will reload from file
    """
    global _file_config_cache, _env_config_cache
    _file_config_cache = None
    _env_config_cache = None
