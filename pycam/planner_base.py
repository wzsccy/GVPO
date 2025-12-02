from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .mapf_utils import Config, Configs, Deadline, Grid


@dataclass
class PlannerRequest:
    """Container describing a MAPF planning query shared by all planners."""

    grid: Grid
    starts: Config
    goals: Config
    time_limit_ms: int = 5000
    start_time: Optional[float] = None
    deadline: Optional[Deadline] = None
    seed: int = 0
    verbose: int = 1

    def effective_start_time(self) -> float:
        return self.start_time if self.start_time is not None else time.time()


@dataclass
class PlannerResult:
    """Standardized response emitted by planner backends."""

    solution: Optional[Configs]
    cost: Optional[float] = None
    runtime_s: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


class BasePlanner:
    """Minimal interface shared by planner implementations."""

    name: str = "base"

    def solve(self, request: PlannerRequest) -> PlannerResult:  # pragma: no cover - interface
        raise NotImplementedError("Planner subclasses must implement solve()")


__all__ = ["PlannerRequest", "PlannerResult", "BasePlanner"]
