from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from loguru import logger

from LNS2.ReplanCBSSIPPS import LNS2CBS
from LNS2.Utils import get_sum_of_cost

from .mapf_utils import Config, Configs, Grid
from .planner_base import BasePlanner, PlannerRequest, PlannerResult

CoordXY = Tuple[int, int]


def _config_to_xy_list(config: Config) -> List[CoordXY]:
    return [(coord[1], coord[0]) for coord in config.positions]


def _grid_to_map(grid: Grid) -> List[List[bool]]:
    return grid.astype(bool).tolist()


def _paths_to_configs(paths: Sequence[Sequence[CoordXY]]) -> Configs:
    if not paths:
        return []
    max_len = max(len(p) for p in paths)
    configs: Configs = []
    for t in range(max_len):
        config = Config()
        for path in paths:
            if not path:
                config.append((0, 0))
                continue
            loc = path[t] if t < len(path) else path[-1]
            config.append((loc[1], loc[0]))
        configs.append(config)
    return configs


class LNS2Planner(BasePlanner):
    """Adapter around the LNS2 CBS-with-SIPPS solver."""

    name = "lns2"

    def __init__(self, *, num_neighbours: int = 5) -> None:
        self.num_neighbours = max(1, int(num_neighbours))

    def solve(self, request: PlannerRequest) -> PlannerResult:
        instance_map = _grid_to_map(request.grid)
        starts_xy = _config_to_xy_list(request.starts)
        goals_xy = _config_to_xy_list(request.goals)
        if not starts_xy or not goals_xy:
            logger.warning("LNS2 planner received empty starts/goals.")
            return PlannerResult(
                solution=None,
                cost=None,
                runtime_s=0.0,
                meta={"planner": self.name, "status": "failed"},
            )

        num_agents = min(len(starts_xy), len(goals_xy))
        starts_xy = starts_xy[:num_agents]
        goals_xy = goals_xy[:num_agents]

        start_time = request.effective_start_time()
        time_limit_s: Optional[float] = None
        if request.time_limit_ms is not None:
            time_limit_s = max(0.0, float(request.time_limit_ms) / 1000.0)
        try:
            paths, num_replans = LNS2CBS(
                self.num_neighbours,
                len(instance_map[0]),
                len(instance_map),
                instance_map,
                starts_xy,
                goals_xy,
                time_limit_s=time_limit_s,
            )
        except TimeoutError:
            runtime = time.time() - start_time
            meta = {"planner": self.name, "status": "over_time"}
            logger.warning(
                "LNS2 planner exceeded wall-clock budget (%.2fs)", time_limit_s or 0.0
            )
            return PlannerResult(solution=None, cost=None, runtime_s=runtime, meta=meta)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("LNS2 planner raised an exception: %s", exc)
            return PlannerResult(solution=None, cost=None, runtime_s=0.0, meta={"planner": self.name, "error": str(exc)})

        runtime = time.time() - start_time
        meta = {"planner": self.name, "num_replans": num_replans}
        if paths is None:
            meta["status"] = "failed"
            return PlannerResult(solution=None, cost=None, runtime_s=runtime, meta=meta)

        solution = _paths_to_configs(paths)
        cost = get_sum_of_cost(paths)
        meta["total_cost"] = cost
        return PlannerResult(solution=solution, cost=cost, runtime_s=runtime, meta=meta)


__all__ = ["LNS2Planner"]
