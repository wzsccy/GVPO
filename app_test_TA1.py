# 开发时间：2025/3/25 21:51
# 开发语言：Python
'''---------------------------------------'''
import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from pycam.MAPD_DQN import DQNOptimizer
from CBS_TA.CBS import (
    CBS,
    Environment,
)
from LNS2.ReplanCBSSIPPS import LNS2CBS
from LNS2.ReplanPPSIPPS import LNS2PP
from ML_EECBS.eecbs import EECBSSolver  # 八方向
from ML_EECBS.focalsearch_single_agent_planner import get_sum_of_cost
from ML_EECBS_4.run_experiments import ML_EECBS_4

from MA_CBS.MACBS import MACBS_four, MACBS_eight
import winsound
import pandas as pd
from pycam.allocate import Allocate
from pycam.allo_util_RL import RLAllocator
from pycam.pibt import PIBT
from pycam.lacam0 import LaCAM as LaCAM0
from pycam.lacam_4 import LaCAM as LaCAM_4
from pycam import (
    LaCAM,
    get_grid,
    get_scenario,
    save_configs_for_visualizer,
    validate_mapf_solution,
)

from pycam.lacam0 import LaCAM as lacam0
from pycam.planner_base import PlannerRequest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_initial_plan(num_agents: int, num_tasks: int) -> dict:
    plan = {f"a{i}": [] for i in range(num_agents)}
    for task_idx in range(num_tasks):
        plan[f"a{task_idx % num_agents}"].append(f"t{task_idx}")
    return plan


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_single_agent_cost(agent_idx: int, tasks, agent_dict, task_dict) -> float:
    current = agent_dict.get(agent_idx)
    if current is None:
        return 0.0
    total = 0.0
    for task_name in tasks:
        try:
            task_idx = int(task_name[1:])
        except (TypeError, ValueError):
            continue
        task_entry = task_dict.get(task_idx)
        if not task_entry:
            pickup = dropoff = current
        else:
            pickup, dropoff = task_entry[0], task_entry[1]
        total += manhattan(current, pickup)
        total += manhattan(pickup, dropoff)
        current = dropoff
    return total


def compute_agent_cost_map(plan, agent_dict, task_dict) -> dict:
    return {
        agent_name: compute_single_agent_cost(int(agent_name[1:]), tasks, agent_dict, task_dict)
        for agent_name, tasks in plan.items()
    }


def compute_total_cost(plan, agent_dict, task_dict) -> float:
    total = 0.0
    for agent_name, tasks in plan.items():
        try:
            agent_idx = int(agent_name[1:])
        except (TypeError, ValueError):
            continue
        total += compute_single_agent_cost(agent_idx, tasks, agent_dict, task_dict)
    return total


def build_optimize_funcs(num_agents: int, num_tasks: int, agent_dict, task_dict):
    def move_task(plan):
        mutated = copy.deepcopy(plan)
        src = f"a{random.randrange(num_agents)}"
        dst = f"a{random.randrange(num_agents)}"
        if mutated[src]:
            task = mutated[src].pop(random.randrange(len(mutated[src])))
            mutated[dst].insert(random.randrange(len(mutated[dst]) + 1), task)
        return mutated

    def swap_tasks(plan):
        mutated = copy.deepcopy(plan)
        if num_agents < 2:
            return mutated
        a_idx, b_idx = random.sample(range(num_agents), k=2)
        agent_a, agent_b = f"a{a_idx}", f"a{b_idx}"
        if mutated[agent_a] and mutated[agent_b]:
            i = random.randrange(len(mutated[agent_a]))
            j = random.randrange(len(mutated[agent_b]))
            mutated[agent_a][i], mutated[agent_b][j] = mutated[agent_b][j], mutated[agent_a][i]
        return mutated

    def reinsert_block(plan):
        mutated = copy.deepcopy(plan)
        donors = [agent for agent, tasks in mutated.items() if tasks]
        if not donors:
            return mutated
        src = random.choice(donors)
        src_tasks = mutated[src]
        block_len = random.randint(1, min(3, len(src_tasks)))
        start_idx = random.randrange(0, len(src_tasks) - block_len + 1)
        block = src_tasks[start_idx : start_idx + block_len]
        del src_tasks[start_idx : start_idx + block_len]
        dst = random.choice(list(mutated.keys()))
        insert_idx = random.randrange(len(mutated[dst]) + 1)
        for offset, task in enumerate(block):
            mutated[dst].insert(insert_idx + offset, task)
        return mutated

    def two_opt_within_agent(plan):
        mutated = copy.deepcopy(plan)
        candidates = [agent for agent, tasks in mutated.items() if len(tasks) >= 2]
        if not candidates:
            return mutated
        agent_name = random.choice(candidates)
        tasks = mutated[agent_name]
        i, j = sorted(random.sample(range(len(tasks)), 2))
        before_cost = compute_single_agent_cost(int(agent_name[1:]), tasks, agent_dict, task_dict)
        new_order = tasks[:i] + list(reversed(tasks[i : j + 1])) + tasks[j + 1 :]
        after_cost = compute_single_agent_cost(int(agent_name[1:]), new_order, agent_dict, task_dict)
        if after_cost < before_cost:
            mutated[agent_name] = new_order
        return mutated

    def balance_heavy_to_light(plan):
        mutated = copy.deepcopy(plan)
        cost_map = compute_agent_cost_map(mutated, agent_dict, task_dict)
        if not cost_map:
            return mutated
        heavy = max(cost_map.items(), key=lambda kv: kv[1])[0]
        light = min(cost_map.items(), key=lambda kv: kv[1])[0]
        if heavy == light or not mutated[heavy]:
            return mutated
        task_idx = random.randrange(len(mutated[heavy]))
        task = mutated[heavy].pop(task_idx)
        insert_idx = random.randrange(len(mutated[light]) + 1)
        mutated[light].insert(insert_idx, task)
        return mutated

    return [move_task, swap_tasks, reinsert_block, two_opt_within_agent, balance_heavy_to_light]


def plan_to_pairs(plan: dict, num_agents: int):
    pairs = []
    for agent_idx in range(num_agents):
        tasks = plan.get(f"a{agent_idx}", [])
        tuple_vals = [agent_idx]
        if tasks:
            for task_name in tasks:
                try:
                    tuple_vals.append(int(task_name[1:]))
                except (TypeError, ValueError):
                    continue
        if len(tuple_vals) == 1:
            tuple_vals.append(-1)
        pairs.append(tuple(tuple_vals))
    return pairs


if __name__ == "__main__":
    map_list = ["ost003d.map"]
    # map_list = ["empty-32-32.map"]
    for map in map_list:
        set_map = map
        Num_T = 200
        dl_la = 200 # 限制路径规划的时间

        dl = 900  # 中图b700
        N_list = [20, 40, 60, 80, 100, 120]
        for N in N_list:
            Num_A = N
            Isexcel = True  # 是否导出excel
            data_save_dict = dict()  # 用于存储20实例的数据
            "跑20个实例"
            for ins_num in range(1, 6):
                temp_list = []
                set_scen = ("{0}-random-{1}.scen".format(set_map.replace(".map", ""), ins_num))
                # data_save_dict = dict(ins_num=temp_list)  # 用于存储20实例的数据
                parser = argparse.ArgumentParser()
                parser.add_argument("-m", "--map-file", type=Path,
                                    default=Path(__file__).parent / "assets2" / set_map, )
                parser.add_argument("-i", "--scen-file", type=Path,
                                    default=Path(__file__).parent / "assets2" / set_scen, )
                parser.add_argument("-N", "--num-agents", type=int, default=0)
                parser.add_argument("-o", "--output-file", type=str, default="output.txt", )
                parser.add_argument("-v", "--verbose", type=int, default=1, )
                parser.add_argument("-s", "--seed", type=int, default=0)
                parser.add_argument("-t", "--time_limit_ms", type=int, default=1000)
                parser.add_argument("--flg_star", action=argparse.BooleanOptionalAction, default=True)
                args = parser.parse_args()

                # 设置实例
                print("\033[34mExecute {0} [{1}-{2}]\033[0m".format(set_scen, Num_A, Num_T))
                grid, dimension, obstacles = get_grid(args.map_file)

                # print("71",grid,"\n" ,dimension,"\n", obstacles)
                starts, goals, All_data_list, agent_dict, task_dict = get_scenario(args.scen_file, Num_A, Num_T)

                # 执行任务分配
                TA_start_time = time.time()
                allocate = Allocate(agent_dict, task_dict, dimension, obstacles, grid)

                seed_everything(args.seed + ins_num)
                agent_map = {int(k): v for k, v in agent_dict.items()}
                task_map = {int(k): v for k, v in task_dict.items()}
                num_agents = len(agent_map)
                num_tasks = len(task_map)
                init_plan = build_initial_plan(num_agents, num_tasks)
                optimize_funcs = build_optimize_funcs(num_agents, num_tasks, agent_map, task_map)
                reward_fn = lambda plan, a_map=agent_map, t_map=task_map: -compute_total_cost(plan, a_map, t_map)
                planner_request = PlannerRequest(
                    grid=grid,
                    starts=copy.deepcopy(starts),
                    goals=copy.deepcopy(goals),
                    time_limit_ms=args.time_limit_ms,
                    seed=args.seed,
                    verbose=args.verbose,
                )
                planner_kwargs = {"num_neighbours": 8}
                rl_hparams = {
                    "max_episodes": 120,
                    "batch_size": 192,
                    "num_generations": 6,
                    "learning_rate": 1e-4,
                }

                def run_lns2_allocator(policy_type: str):
                    allocator = RLAllocator(
                        num_agents,
                        num_tasks,
                        optimize_funcs,
                        reward_fn,
                        algo_type=policy_type,
                        planner_type="lns2",
                        planner_kwargs=planner_kwargs,
                        max_episodes=rl_hparams["max_episodes"],
                        batch_size=rl_hparams["batch_size"],
                        num_generations=rl_hparams["num_generations"],
                        learning_rate=rl_hparams["learning_rate"],
                    )
                    plan, _ = allocator.train(copy.deepcopy(init_plan))
                    try:
                        allocator.solve_mapf(planner_request)
                    except Exception as exc:
                        print(f"[WARN] MAPF planner failed for {policy_type}: {exc}")
                    cost = compute_total_cost(plan, agent_map, task_map)
                    return plan_to_pairs(plan, num_agents), round(cost, 2)

                TA_end_time = time.time()

                TA5_start_time = time.time()
                pairs_5, C_5 = run_lns2_allocator("grpo")
                TA5_end_time = time.time()
                print("算法1(lns2+grpo)的任务分配成本:{0}".format(C_5))
                print("算法1(lns2+grpo)运行时间:{0:.2f}s".format(TA5_end_time - TA5_start_time))

                TA51_start_time = time.time()
                pairs_51, C_51 = run_lns2_allocator("gvpo")
                TA51_end_time = time.time()
                print("算法2(lns2+gvpo)的任务分配成本:{0}".format(C_51))
                print("算法2(lns2+gvpo)运行时间:{0:.2f}s".format(TA51_end_time - TA51_start_time))

                TA5_run = (TA5_end_time - TA5_start_time) + (TA_end_time - TA_start_time)
                TA51_run = (TA51_end_time - TA51_start_time) + (TA_end_time - TA_start_time)
                print("\033[33mAllocation_sequence\033[0m : True")
                TA_end_time = time.time()

                prepare_start_time = time.time()
                agent_dict0 = copy.deepcopy(agent_dict)  # 为CBS保留不执行任务的代理
                task_dict0 = copy.deepcopy(task_dict)

                "--TA2--"
                init_AT_2, AT_2, max_agent_2 = allocate.get_AT(pairs_5, agent_dict, task_dict)
                start_Config1 = copy.deepcopy(starts)  # 转Config类型
                goal_Config2 = copy.deepcopy(starts)
                cout_total_cost_2 = 0  # 总成本
                total_sol_2 = dict()  # 总方案
                over_time_flg = False  # 如果路径规划算法超出30s，就截停返回-1
                for pair in pairs_5:
                    if pair[-1] != -1:
                        total_sol_2[pair[0]] = []

                "--TA4--"
                init_AT_4, AT_4, max_agent_4 = allocate.get_AT(pairs_51, agent_dict0, task_dict)
                start_Config1 = copy.deepcopy(starts)  # 转Config类型
                goal_Config2 = copy.deepcopy(starts)
                cout_total_cost_4 = 0  # 总成本
                total_sol_4 = dict()  # 总方案
                over_time_flg = False  # 如果路径规划算法超出30s，就截停返回-1
                for pair in pairs_51:
                    if pair[-1] != -1:
                        total_sol_4[pair[0]] = []

                prepare_end_time = time.time()

                for aaa in [1,2]:
                    # for aaa in ,[0, 2, 4, 5, 6, 7]:
                    "Elacam*_TA5"
                    if aaa == 1:
                        lacam0_start_time = time.time()
                        cout_total_cost = 0  # 总成本
                        dl = dl_la
                        flag = False
                        for i in range(0, len(AT_2[max_agent_2]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT_2[agent][i] for agent in AT_2]
                            goal = [AT_2[agent][i + 1] for agent in AT_2 if i + 1 < len(AT_2[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            planner = lacam0()
                            solution = planner.solve(
                                grid=grid,
                                starts=start_Config1,
                                goals=goal_Config2,
                                seed=args.seed,
                                start_time=lacam0_start_time,
                                time_limit_ms=dl,
                                flg_star=args.flg_star,
                                verbose=args.verbose,
                            )

                            if solution == 'over_time':
                                flag = True
                                break
                            else:
                                # total_sol = allocate.data_transform(i, solution, total_sol_4)  # 转换solution数据为各代理_路径
                                cout_total_cost += planner.total_cost
                                temp_run_de_time0 = time.time()
                        # print("151", solution)
                        # print(total_sol)
                        lacam0_end_time = time.time()
                        if flag == False:
                            TaskA_time, Total_time, Total_cost = (round(TA5_run, 2),
                                                                  round(TA5_run + (
                                                                          lacam0_end_time - lacam0_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                                  -1,
                                                                  round(cout_total_cost, 2))
                        print("\033[33mAlgth_name\033[0m :", "\033[31mElacam*_TA5\033[0m\n"
                                                             "\033[33mTaskA_time\033[0m : {0}"
                                                             "\n\033[33mTotal_time\033[0m : {1}"
                                                             "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))
                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list

                    "RL_lacam*_TA5"
                    if aaa == 2:
                        lacam_start_time = time.time()
                        cout_total_cost = 0  # 总成本
                        total_sol = {}
                        dl = dl_la
                        flag = False
                        # print("183",AT_4)
                        for i in range(0, len(AT_4[max_agent_4]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT_4[agent][i] for agent in AT_4]
                            goal = [AT_4[agent][i + 1] for agent in AT_4 if i + 1 < len(AT_4[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            planner = LaCAM()
                            solution = planner.solve(
                                grid=grid,
                                starts=start_Config1,
                                goals=goal_Config2,
                                seed=args.seed,
                                start_time=lacam_start_time,
                                time_limit_ms=dl,
                                flg_star=args.flg_star,
                                verbose=args.verbose,
                            )
                            if solution == 'over_time':
                                flag = True
                                break
                            else:
                                total_sol = allocate.data_transform(i, solution, total_sol_4)  # 转换solution数据为各代理_路径
                                cout_total_cost += planner.total_cost
                                temp_run_de_time0 = time.time()

                        # print(total_sol)
                        "RL优化解方案"
                        # dq = DQNOptimizer(total_sol,cout_total_cost,AT_4,grid,dimension,obstacles)
                        # new_P,new_cost = dq.optimize()
                        # print("RL新成本",new_cost)
                        # dq = (total_sol,,cout_total_cost,AT_4,griZd,dimension,obstacles)

                        lacam_end_time = time.time()
                        if flag == False:
                            TaskA_time, Total_time, Total_cost = (round(TA51_run, 2),
                                                                  round(TA51_run + (
                                                                          lacam_end_time - lacam_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                                  -1,
                                                                  round(cout_total_cost, 2))
                        print("\033[33mAlgth_name\033[0m :", "\033[31mRL_Elacam*_TA5\033[0m\n"
                                                             "\033[33mTaskA_time\033[0m : {0}"
                                                             "\n\033[33mTotal_time\033[0m : {1}"
                                                             "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))
                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    # "ML-EECBS_8_TA2"
                    # if aaa == 3 and not over_time_flg:
                    #     ML_EECBS_start_time = time.time()
                    #     cout_total_cost = 0
                    #     # dl = 100
                    #     over_time_flg_0 = False  # 当前算法是否超过30s
                    #     for i in range(0, len(AT_2[max_agent_2]) - 1):
                    #         # 取出各代理的第i个元素和第i+1个元素
                    #         start = [AT_2[agent][i] for agent in AT_2]
                    #         goal = [AT_2[agent][i + 1] for agent in AT_2 if i + 1 < len(AT_2[agent])]
                    #         c_start = copy.deepcopy(start)
                    #         c_goal = copy.deepcopy(goal)
                    #         # 不需要对已到达的代理进行搜索
                    #         j = 0
                    #         while j < len(c_start):
                    #             if c_start[j] == c_goal[j]: 
                    #                 del c_start[j], c_goal[j]
                    #             j += 1
                    #         map = copy.deepcopy(grid)
                    #         map = (~ map).tolist()
                    #         cbs = EECBSSolver(map, c_start, c_goal)
                    #         paths = cbs.find_solution(dl, ML_EECBS_start_time)
                    #         if paths == 'over_time':
                    #             over_time_flg_0 = True
                    #             break
                    #         else:
                    #             total_cost = get_sum_of_cost(paths)
                    #             cout_total_cost += total_cost
                    #
                    #     ML_EECBS_end_time = time.time()
                    #     if not over_time_flg_0:
                    #         TaskA_time, Total_time, Total_cost = (round(TA2_run, 2), round(
                    #             (TA2_run) + (ML_EECBS_end_time - ML_EECBS_start_time), 2),
                    #                                               round(cout_total_cost, 2))
                    #     else:
                    #         TaskA_time, Total_time, Total_cost = (
                    #             round(TA2_run, 2), -1, round(cout_total_cost, 2))
                    #
                    #     print("\n\033[33mAlgth_name\033[0m :", "\033[31mML_EECBS_8_TA2\033[0m"
                    #                                            "\n\033[33mTaskA_time\033[0m : {0}"
                    #                                            "\n\033[33mTotal_time\033[0m : {1}"
                    #                                            "\n\033[33mTotal_cost\033[0m : {2}".
                    #           format(TaskA_time, Total_time, Total_cost))
                    #
                    #     data = (Total_time), (Total_cost)
                    #     temp_list.append(data)
                    #     data_save_dict['{0}'.format(ins_num)] = temp_list
                    # "ML-EECBS_8_TA4"
                    # if aaa == 4 and not over_time_flg:
                    #     ML_EECBS_start_time = time.time()
                    #     cout_total_cost = 0
                    #     # dl = 100
                    #     over_time_flg_0 = False  # 当前算法是否超过30s
                    #     for i in range(0, len(AT_4[max_agent_4]) - 1):
                    #         # 取出各代理的第i个元素和第i+1个元素
                    #         start = [AT_4[agent][i] for agent in AT_4]
                    #         goal = [AT_4[agent][i + 1] for agent in AT_4 if i + 1 < len(AT_4[agent])]
                    #         c_start = copy.deepcopy(start)
                    #         c_goal = copy.deepcopy(goal)
                    #         # 不需要对已到达的代理进行搜索
                    #         j = 0
                    #         while j < len(c_start):
                    #             if c_start[j] == c_goal[j]:
                    #                 del c_start[j], c_goal[j]
                    #             j += 1
                    #         map = copy.deepcopy(grid)
                    #         map = (~ map).tolist()
                    #         cbs = EECBSSolver(map, c_start, c_goal)
                    #         paths = cbs.find_solution(dl, ML_EECBS_start_time)
                    #         if paths == 'over_time':
                    #             over_time_flg_0 = True
                    #             break
                    #         else:
                    #             total_cost = get_sum_of_cost(paths)
                    #             cout_total_cost += total_cost
                    #
                    #     ML_EECBS_end_time = time.time()
                    #     if not over_time_flg_0:
                    #         TaskA_time, Total_time, Total_cost = (round(TA4_run, 2), round(
                    #             (TA4_run) + (ML_EECBS_end_time - ML_EECBS_start_time), 2),
                    #                                               round(cout_total_cost, 2))
                    #     else:
                    #         TaskA_time, Total_time, Total_cost = (
                    #             round(TA4_run, 2), -1, round(cout_total_cost, 2))
                    #
                    #     print("\n\033[33mAlgth_name\033[0m :", "\033[31mML_EECBS_8_TA4\033[0m"
                    #                                            "\n\033[33mTaskA_time\033[0m : {0}"
                    #                                            "\n\033[33mTotal_time\033[0m : {1}"
                    #                                            "\n\033[33mTotal_cost\033[0m : {2}".
                    #           format(TaskA_time, Total_time, Total_cost))
                    #
                    #     data = (Total_time), (Total_cost)
                    #     temp_list.append(data)
                    #     data_save_dict['{0}'.format(ins_num)] = temp_list
                    # "MA_CBS_8_TA2"
                    # if aaa == 5 and not over_time_flg:
                    #     MA_CBS_8_start_time = time.time()
                    #     cout_total_cost = 0
                    #     # dl = 100
                    #     over_time_flg_0 = False  # 当前算法是否超过30s
                    #     for i in range(0, len(AT_2[max_agent_2]) - 1):
                    #         # 取出各代理的第i个元素和第i+1个元素
                    #         start = [AT_2[agent][i] for agent in AT_2]
                    #         goal = [AT_2[agent][i + 1] for agent in AT_2 if i + 1 < len(AT_2[agent])]
                    #         c_start = copy.deepcopy(start)
                    #         c_goal = copy.deepcopy(goal)
                    #         # 不需要对已到达的代理进行搜索
                    #         j = 0
                    #         while j < len(c_start):
                    #             if c_start[j] == c_goal[j]:
                    #                 del c_start[j], c_goal[j]
                    #             j += 1
                    #         map = copy.deepcopy(grid)
                    #
                    #         cbs = MACBS_eight(dl, MA_CBS_8_start_time, map, c_start, c_goal)
                    #         paths, cost = cbs.get_result()
                    #         if paths == 'over_time':
                    #             over_time_flg_0 = True
                    #             break
                    #         else:
                    #             total_cost = cost
                    #             cout_total_cost += total_cost
                    #         # print(paths)
                    #
                    #     MA_CBS_8_end_time = time.time()
                    #     if not over_time_flg_0:
                    #         TaskA_time, Total_time, Total_cost = (round(TA2_run, 2), round(
                    #             (TA2_run) + (MA_CBS_8_end_time - MA_CBS_8_start_time), 2),
                    #                                               round(cout_total_cost, 2))
                    #     else:
                    #         TaskA_time, Total_time, Total_cost = (
                    #             round(TA2_run, 2), -1, round(cout_total_cost, 2))
                    #
                    #     print("\n\033[33mAlgth_name\033[0m :", "\033[31mMA_CBS_8_TA2\033[0m"
                    #                                            "\n\033[33mTaskA_time\033[0m : {0}"
                    #                                            "\n\033[33mTotal_time\033[0m : {1}"
                    #                                            "\n\033[33mTotal_cost\033[0m : {2}".
                    #           format(TaskA_time, Total_time, Total_cost))
                    #
                    #     data = (Total_time), (Total_cost)
                    #     temp_list.append(data)
                    #     data_save_dict['{0}'.format(ins_num)] = temp_list
                    # "MA_CBS_8_TA4"
                    # if aaa == 6 and not over_time_flg:
                    #     MA_CBS_8_start_time = time.time()
                    #     cout_total_cost = 0
                    #     # dl = 100
                    #     over_time_flg_0 = False  # 当前算法是否超过30s
                    #     for i in range(0, len(AT_4[max_agent_4]) - 1):
                    #         # 取出各代理的第i个元素和第i+1个元素
                    #         start = [AT_4[agent][i] for agent in AT_4]
                    #         goal = [AT_4[agent][i + 1] for agent in AT_4 if i + 1 < len(AT_4[agent])]
                    #         c_start = copy.deepcopy(start)
                    #         c_goal = copy.deepcopy(goal)
                    #         # 不需要对已到达的代理进行搜索
                    #         j = 0
                    #         while j < len(c_start):
                    #             if c_start[j] == c_goal[j]:
                    #                 del c_start[j], c_goal[j]
                    #             j += 1
                    #         map = copy.deepcopy(grid)
                    #
                    #         cbs = MACBS_eight(dl, MA_CBS_8_start_time, map, c_start, c_goal)
                    #         paths, cost = cbs.get_result()
                    #         if paths == 'over_time':
                    #             over_time_flg_0 = True
                    #             break
                    #         else:
                    #             total_cost = cost
                    #             cout_total_cost += total_cost
                    #         # print(paths)
                    #
                    #     MA_CBS_8_end_time = time.time()
                    #     if not over_time_flg_0:
                    #         TaskA_time, Total_time, Total_cost = (round(TA4_run, 2), round(
                    #             (TA4_run) + (MA_CBS_8_end_time - MA_CBS_8_start_time), 2),
                    #                                               round(cout_total_cost, 2))
                    #     else:
                    #         TaskA_time, Total_time, Total_cost = (
                    #             round(TA4_run, 2), -1, round(cout_total_cost, 2))
                    #
                    #     print("\n\033[33mAlgth_name\033[0m :", "\033[31mMA_CBS_8_TA4\033[0m"
                    #                                            "\n\033[33mTaskA_time\033[0m : {0}"
                    #                                            "\n\033[33mTotal_time\033[0m : {1}"
                    #                                            "\n\033[33mTotal_cost\033[0m : {2}".
                    #           format(TaskA_time, Total_time, Total_cost))
                    #
                    #     data = (Total_time), (Total_cost)
                    #     temp_list.append(data)
                    #     data_save_dict['{0}'.format(ins_num)] = temp_list
            "数据导入表中"
            if Isexcel == True:
                # 创建一个空的DataFrame列表
                dfs = []
                # 遍历字典，将数据添加到DataFrame列表中
                df = pd.DataFrame({
                    'scen': ['random-{0}'.format(index) for index in data_save_dict.keys()],
                    'L2_T': [data_save_dict[index][0][0] for index in data_save_dict.keys()],
                    'L2_C': [data_save_dict[index][0][1] for index in data_save_dict.keys()],
                    'L4_T': [data_save_dict[index][1][0] for index in data_save_dict.keys()],
                    'L4_C': [data_save_dict[index][1][1] for index in data_save_dict.keys()],
                })
                dfs.append(df)
                # 使用pd.concat合并所有DataFrame
                if Num_A == 20:
                    df1 = pd.concat(dfs, ignore_index=True)
                elif Num_A == 40:
                    df2 = pd.concat(dfs, ignore_index=True)
                elif Num_A == 60:
                    df3 = pd.concat(dfs, ignore_index=True)
                elif Num_A == 80:
                    df4 = pd.concat(dfs, ignore_index=True)
                elif Num_A == 100:
                    df5 = pd.concat(dfs, ignore_index=True)
                elif Num_A == 120:
                    df6 = pd.concat(dfs, ignore_index=True)
                # 将DataFrame导出到Excel文件
                if Num_A == 120:
                    with pd.ExcelWriter(r'Datas14\{0}.xlsx'.format(set_map.replace(".map", ""))) as writer:
                        df1.to_excel(writer, sheet_name='20_200', index=False)
                        df2.to_excel(writer, sheet_name='40_200', index=False)
                        df3.to_excel(writer, sheet_name='60_200', index=False)
                        df4.to_excel(writer, sheet_name='80_200', index=False)
                        df5.to_excel(writer, sheet_name='100_200', index=False)
                        df6.to_excel(writer, sheet_name='120_200', index=False)
            "绘图"
            # if total_sol != {}:
            #     copy_total_sol=allocate.transform_for_draw(pairs,total_sol,agent_dict,task_dict) #制作绘图数据
            #     # allocate.start_Hamiltonian()          # 展示哈密顿环
            #     allocate.draw_map(copy_total_sol,0)   # 动态MAPD
            "结束提示"
            winsound.Beep(880, 200)  # 880赫兹，持续200毫秒
