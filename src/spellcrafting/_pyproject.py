"""Shared pyproject.toml utilities.

This module provides common functionality for finding and loading pyproject.toml,
avoiding duplication between config.py and logging.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Python 3.11+ has tomllib in stdlib; use tomli for 3.10
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def find_pyproject() -> Path | None:
    """Search for pyproject.toml from cwd upward.

    Returns:
        Path to pyproject.toml if found, None otherwise.
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def load_pyproject() -> dict[str, Any] | None:
    """Load and parse pyproject.toml.

    Returns:
        Parsed TOML data as a dict, or None if file not found or parse error.
    """
    path = find_pyproject()
    if path is None:
        return None

    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def get_spellcrafting_config() -> dict[str, Any]:
    """Get the [tool.spellcrafting] section from pyproject.toml.

    Returns:
        The spellcrafting config dict, or empty dict if not found.
    """
    data = load_pyproject()
    if data is None:
        return {}
    return data.get("tool", {}).get("spellcrafting", {})
