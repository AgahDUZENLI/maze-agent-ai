"""
Compatibility re-export for the canonical environment module.

The project now maintains a single MazeEnvironment implementation in
`environment.py`. This module remains only so existing imports keep working.
"""

from environment import Action, MazeEnvironment, TurnResult

__all__ = ["Action", "MazeEnvironment", "TurnResult"]
