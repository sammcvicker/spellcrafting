"""Adapter layer for pydantic_ai internals.

This module isolates all imports from pydantic_ai's internal modules (exceptions,
settings, agent internals) to a single location. If pydantic_ai changes these
internals, only this file needs to be updated.

The stable public API from pydantic_ai (Agent) is imported directly here as well
to provide a single source for all pydantic_ai dependencies.
"""

from __future__ import annotations

# Import Agent - this is stable public API
from pydantic_ai import Agent

# Import ValidationError from our exception hierarchy
from spellcrafting.exceptions import ValidationError

# Import internal modules with fallback handling
try:
    from pydantic_ai.exceptions import UnexpectedModelBehavior
except ImportError:
    # Fallback if pydantic_ai renames/moves this exception
    try:
        from pydantic_ai.errors import UnexpectedModelBehavior  # type: ignore[no-redef]
    except ImportError:
        # Ultimate fallback - create our own exception class
        class UnexpectedModelBehavior(Exception):  # type: ignore[no-redef]
            """Fallback exception when pydantic_ai's exception is unavailable."""
            pass

try:
    from pydantic_ai.settings import ModelSettings
except ImportError:
    # Fallback if pydantic_ai renames/moves ModelSettings
    try:
        from pydantic_ai.models import ModelSettings  # type: ignore[no-redef]
    except ImportError:
        # Ultimate fallback - use dict-like TypedDict
        from typing import TypedDict

        class ModelSettings(TypedDict, total=False):  # type: ignore[no-redef]
            """Fallback ModelSettings when pydantic_ai's is unavailable."""
            temperature: float
            max_tokens: int
            top_p: float
            timeout: float

try:
    from pydantic_ai.agent import EndStrategy
except ImportError:
    # Fallback for EndStrategy
    from typing import Literal
    EndStrategy = Literal["early", "exhaustive"]  # type: ignore[misc]


__all__ = [
    "Agent",
    "UnexpectedModelBehavior",
    "ModelSettings",
    "EndStrategy",
    "ValidationError",
]
