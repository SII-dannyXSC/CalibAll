"""caliball.robots -- Robot type registry with self-contained implementations.

Usage::

    from caliball.robots import build_robot, list_robots

    print(list_robots())          # ['aloha_cobot_magic', 'arx5_robotwin', ...]
    robot = build_robot("franka")
    T = robot.fkine_flange(q)        # (1, 4, 4)

Adding a new robot:
    Create a new .py file in this directory with @register_robot("name").
    It will be auto-discovered — no need to modify this file.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from caliball.robots._base import BaseTF
from caliball.robots._registry import get_robot_cls

__all__ = ["build_robot", "list_robots", "BaseTF"]

_ALL_REGISTERED = False


def _register_all() -> None:
    """Auto-import all robot modules (non-underscore .py files) to trigger @register_robot."""
    global _ALL_REGISTERED
    if _ALL_REGISTERED:
        return
    pkg_dir = Path(__file__).parent
    for py in sorted(pkg_dir.glob("*.py")):
        if py.name.startswith("_") or py.name == "__init__.py":
            continue
        importlib.import_module(f"caliball.robots.{py.stem}")
    _ALL_REGISTERED = True


def build_robot(robot_type: str, **kwargs) -> BaseTF:
    """Construct a robot TF instance by registered type name."""
    _register_all()
    cls = get_robot_cls(robot_type)
    return cls(**kwargs)


def list_robots() -> list[str]:
    """Return sorted list of all registered robot type names."""
    _register_all()
    from caliball.robots._registry import list_robots as _lr
    return _lr()
