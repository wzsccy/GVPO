from __future__ import annotations

from typing import Dict, Type

from .planner_base import BasePlanner
from .planner_lacam import LaCAMPlanner
from .planner_lns2 import LNS2Planner


class PlannerFactory:
    """Factory helper to hide concrete planner classes behind a string flag."""

    _PLANNERS: Dict[str, Type[BasePlanner]] = {
        "lacam": LaCAMPlanner,
        "lns2": LNS2Planner,
    }

    @classmethod
    def create(cls, planner_type: str, **kwargs) -> BasePlanner:
        key = planner_type.lower()
        if key not in cls._PLANNERS:
            raise ValueError(f"Unknown planner_type '{planner_type}'. Available: {', '.join(cls.choices())}")
        return cls._PLANNERS[key](**kwargs)

    @classmethod
    def choices(cls) -> list[str]:
        return sorted(cls._PLANNERS.keys())


__all__ = ["PlannerFactory"]
