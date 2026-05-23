"""Robot type registry with decorator-based registration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from caliball.robots._base import BaseTF

_REGISTRY: dict[str, Type["BaseTF"]] = {}


def register_robot(name: str):
    """Class decorator: ``@register_robot("franka")`` registers the class
    so that ``build_robot("franka", ...)`` can find it."""

    def _wrap(cls: Type["BaseTF"]) -> Type["BaseTF"]:
        if name in _REGISTRY:
            raise ValueError(
                f"Robot type {name!r} already registered "
                f"({_REGISTRY[name].__qualname__}); "
                f"cannot re-register with {cls.__qualname__}"
            )
        _REGISTRY[name] = cls
        cls._robot_type_name = name  # type: ignore[attr-defined]
        return cls

    return _wrap


def get_robot_cls(name: str) -> Type["BaseTF"]:
    """Return the class registered under *name*, or raise ``KeyError``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown robot type {name!r}. "
            f"Available: {sorted(_REGISTRY)}"
        ) from None


def list_robots() -> list[str]:
    """Return sorted list of all registered robot type names."""
    return sorted(_REGISTRY)
