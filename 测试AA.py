import argparse
from pathlib import Path

from pycam.mapf_utils import get_scenario00
from pycam.lacam import LaCAM as LaCAM
from pycam import (
    LaCAM,
    get_grid,
    get_scenario,
    save_configs_for_visualizer,
    validate_mapf_solution,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=Path,
        default=Path(__file__).parent / "assets0" / "warehouse-10-20-10-2-1.map",
        # default=Path(__file__).parent / "assets1" / "maze-128-128-1.map",
        # default=Path(__file__).parent / "assets" / "test01.map",
        # default=Path(__file__).parent / "assets" / "random-32-32-20.map",
        # default=Path(__file__).parent / "assets" / "room-64-64-16.map",
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=Path,
        # default=Path(__file__).parent / "assets1" / "maze-128-128-1-even-1.scen",
        default=Path(__file__).parent / "assets0" / "warehouse-10-20-10-2-1-random-1.scen",
        # default=Path(__file__).parent / "assets" / "random-32-32-20-random-1.scen",
        # default=Path(__file__).parent / "assets" / "room-64-64-16-random-1.scen",
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=100,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
    )
    parser.add_argument("-s", "--seed", type=int, default=255)
    parser.add_argument("-t", "--time_limit_ms", type=int, default=1000)
    parser.add_argument(
        "--flg_star", action=argparse.BooleanOptionalAction, default=True
    )

    args = parser.parse_args()

    # define problem instance
    grid,dimension,obstacales = get_grid(args.map_file)
    starts, goals= get_scenario00(args.scen_file, args.num_agents)

    aa=1
    time = 0
    # solve MAPF
    planner = LaCAM()
    solution = planner.solve(
        grid=grid,
        starts=starts,
        goals=goals,
        seed=args.seed,
        start_time = time,
        time_limit_ms=100,
        flg_star=args.flg_star,
        verbose=args.verbose,
        fl00=aa,
    )
    # print(starts,goals)
    # validate_mapf_solution(grid, starts, goals, solution)
    sol=[]
    # for i in solution:
    #     sol.append(i.positions)
    # print(sol)
    # save result
    # save_configs_for_visualizer(solution, args.output_file)