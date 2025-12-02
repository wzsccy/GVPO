import argparse
import os
import time

from pypibt import (
    PIBT,
    get_grid,
    get_scenario,
    is_valid_mapf_solution,
    save_configs_for_visualizer,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--map-file",
        type=str,
        default=os.path.join(
            # os.path.dirname(__file__), "assets", "random-32-32-10.map"
            # os.path.dirname(__file__), "assets", "tunnel.map"
            os.path.dirname(__file__), "assets", "room-32-32-4.map"
            # os.path.dirname(__file__), "assets", "test01.map"
        ),
    )
    parser.add_argument(
        "-i",
        "--scen-file",
        type=str,
        default=os.path.join(
            # os.path.dirname(__file__), "assets", "random-32-32-10-random-1.scen"
            # os.path.dirname(__file__), "assets", "tunnel.scen"
            os.path.dirname(__file__), "assets", "room-32-32-4-random-1.scen"
            # os.path.dirname(__file__), "assets", "test01.scen"
        ),
    )
    parser.add_argument(
        "-N",
        "--num-agents",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="output.txt",
    )
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("--max-timestep", type=int, default=1000)
    args = parser.parse_args()

    # define problem instance
    grid = get_grid(args.map_file)
    starts, goals = get_scenario(args.scen_file, args.num_agents)

    # solve MAPF
    now=time.time()

    pibt = PIBT(grid, starts, goals, seed=args.seed)
    plan,total_cost = pibt.run(max_timestep=args.max_timestep)

    end=time.time()
    print(plan)
    # validation: True -> feasible solution
    print(f"solved: {is_valid_mapf_solution(grid, starts, goals, plan)}")
    print("runtime={0}\ntotal_cost={1}".format(end-now,total_cost))

    # save result
    save_configs_for_visualizer(plan, args.output_file)
