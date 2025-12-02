from __future__ import annotations

import time

from loguru import logger

from .lacam import LaCAM
from .planner_base import BasePlanner, PlannerRequest, PlannerResult


class LaCAMPlanner(BasePlanner):
    """Thin adapter that exposes the historical LaCAM solver via PlannerRequest."""

    name = "lacam"

    def __init__(self, *, flg_star: bool = True, verbose: int = 1) -> None:
        self.flg_star = flg_star
        self.verbose = verbose

    def solve(self, request: PlannerRequest) -> PlannerResult:
        planner = LaCAM()
        start_time = request.effective_start_time()
        try:
            solution = planner.solve(
                grid=request.grid,
                starts=request.starts,
                goals=request.goals,
                start_time=start_time,
                time_limit_ms=request.time_limit_ms,
                deadline=request.deadline,
                flg_star=self.flg_star,
                seed=request.seed,
                verbose=request.verbose if request.verbose is not None else self.verbose,
                fl00=0,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("LaCAM planner raised an exception: %s", exc)
            return PlannerResult(solution=None, cost=None, runtime_s=0.0, meta={"planner": self.name, "error": str(exc)})

        runtime = time.time() - start_time
        cost = getattr(planner, "total_cost", None)
        meta = {"planner": self.name, "total_cost": cost}

        if not isinstance(solution, list):
            meta["status"] = solution
            return PlannerResult(solution=None, cost=cost, runtime_s=runtime, meta=meta)

        return PlannerResult(solution=solution, cost=cost, runtime_s=runtime, meta=meta)


__all__ = ["LaCAMPlanner"]
