import argparse
import copy
import time
from pathlib import Path
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

if __name__ == "__main__":
    map_list = ["random-32-32-10.map"]

    for map in map_list:
        set_map = map
        Num_T = 100
        N_list = [20]

        for N in N_list:
            Num_A = N
            Isexcel = False  # 是否导出excel
            data_save_dict = dict()  # 用于存储20实例的数据
            "跑20个实例"
            for ins_num in range(1,2):
                temp_list = []
                set_scen = ("{0}-random-{1}.scen".format(set_map.replace(".map", ""), ins_num))

                # data_save_dict = dict(ins_num=temp_list)  # 用于存储20实例的数据

                parser = argparse.ArgumentParser()
                parser.add_argument("-m", "--map-file", type=Path,
                                    default=Path(__file__).parent / "assets0" / set_map, )
                parser.add_argument("-i", "--scen-file", type=Path,
                                    default=Path(__file__).parent / "assets0" / set_scen, )
                parser.add_argument("-N", "--num-agents", type=int, default=0)
                parser.add_argument("-o", "--output-file", type=str, default="output.txt", )
                parser.add_argument("-v", "--verbose", type=int, default=1, )
                parser.add_argument("-s", "--seed", type=int, default=0)
                parser.add_argument("-t", "--time_limit_ms", type=int, default=1000)
                parser.add_argument("--flg_star", action=argparse.BooleanOptionalAction, default=True)
                args = parser.parse_args()

                # 设置实例
                print("\033[34mExecute {0} [{1}-100]\033[0m".format(set_scen, Num_A))
                grid, dimension, obstacles = get_grid(args.map_file)
                starts, goals, All_data_list, agent_dict, task_dict = get_scenario(args.scen_file, Num_A, Num_T)

                agent_dict = {int(k): v for k, v in agent_dict.items()}
                task_dict  = {int(k): v for k, v in task_dict.items()}

                # 执行任务分配
                TA_start_time = time.time()
                allocate = Allocate(agent_dict, task_dict, dimension, obstacles, grid)
                # ② 正确“解包”返回值
                pairs, min_cost = allocate.allocating_task('TA5')
                # ③ 统一 pairs 的 (agent_id, task_id) 为 int，并过滤无任务/不在字典的项
                pairs = [ (int(p[0]), int(p[1])) + tuple(p[2:]) for p in pairs ]
                valid_agent_ids = set(agent_dict.keys())
                valid_task_ids  = set(task_dict.keys())
                pairs = [p for p in pairs if len(p) >= 2 and p[1] != -1 and p[0] in valid_agent_ids and p[1] in valid_task_ids]
 

                if pairs:
                    print("\033[33mAllocation_sequence\033[0m : True")
                else:
                    print("\033[33mAllocation_sequence\033[0m : False")
                # print("\033[33mAllocation_sequence:\033[0m\n{0}".format(pairs))
                TA_end_time = time.time()

                prepare_start_time = time.time()
                agent_dict0 = copy.deepcopy(agent_dict)  # 为CBS保留不执行任务的代理
                task_dict0 = copy.deepcopy(task_dict)

                init_AT, AT, max_agent = allocate.get_AT(pairs, agent_dict, task_dict)
                # ④ 防呆：若 AT 为空或 max_agent 异常，给出明确提示
                if not AT or max_agent is None or max_agent not in AT:
                    raise RuntimeError("AT is empty after allocation. ""Check scen capacity / Num_A / Num_T, or whether pairs were all filtered.")
                
                
                start_Config1 = copy.deepcopy(starts)  # 转Config类型
                goal_Config2 = copy.deepcopy(starts)
                cout_total_cost = 0  # 总成本
                total_sol = dict()  # 总方案
                over_time_flg = False  # 如果路径规划算法超出30s，就截停返回-1
                for pair in pairs:
                    if pair[-1] != -1:
                        total_sol[pair[0]] = []
                prepare_end_time = time.time()

                for aaa in [0]:
                    # for aaa in ,[0, 2, 4, 5, 6, 7]:
                    "lacam*_4"
                    if aaa == -1:
                        lacam_start_time = time.time()
                        cout_total_cost = 0  # 总成本
                        flag = False
                        dl = 30
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            planner = LaCAM_4()
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
                            total_sol = allocate.data_transform(i, solution, total_sol)  # 转换solution数据为各代理_路径
                            cout_total_cost += 0.9*planner.total_cost
                            temp_run_de_time0 = time.time()

                        lacam_end_time = time.time()
                        if flag == False:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                                  round((TA_end_time - TA_start_time) + (
                                                                          lacam_end_time - lacam_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                                  -1,
                                                                  round(cout_total_cost, 2))

                        print("\033[33mAlgth_name\033[0m :", "\033[31mlacam*_4\033[0m\n"
                                                             "\033[33mTaskA_time\033[0m : {0}"
                                                             "\n\033[33mTotal_time\033[0m : {1}"
                                                             "\n\033[33mTotal_cost\033[0m : {2}\n".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "lacam*_8"
                    if aaa == 0:
                        lacam_start_time = time.time()
                        cout_total_cost = 0  # 总成本
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            planner = LaCAM()
                            solution = planner.solve(
                                start_time=lacam_start_time,
                                grid=grid,
                                starts=start_Config1,
                                goals=goal_Config2,
                                seed=args.seed,
                                time_limit_ms=args.time_limit_ms,
                                flg_star=args.flg_star,
                                verbose=args.verbose,
                            )
                            total_sol = allocate.data_transform(i, solution, total_sol)  # 转换solution数据为各代理_路径
                            cout_total_cost += planner.total_cost
                            temp_run_de_time0 = time.time()

                        # print(total_sol)
                        lacam_end_time = time.time()
                        TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                              round((TA_end_time - TA_start_time) + (
                                                                      lacam_end_time - lacam_start_time), 2),
                                                              round(cout_total_cost, 2))
                        print("\033[33mAlgth_name\033[0m :", "\033[31mlacam*_8\033[0m\n"
                                                             "\033[33mTaskA_time\033[0m : {0}"
                                                             "\n\033[33mTotal_time\033[0m : {1}"
                                                             "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    # from multiprocessing import Process, Queue

                    # def solve_with_timeout(q, planner, **kwargs):
                    #     """在子进程中运行 solve 并返回结果"""
                    #     try:
                    #         solution = planner.solve(**kwargs)
                    #         q.put(solution)
                    #     except Exception as e:
                    #         q.put(e)

                    # def safe_solve(planner, **kwargs):
                    #     """带超时保护的 solve 调用"""
                    #     q = Queue()
                    #     p = Process(target=solve_with_timeout, args=(q, planner), kwargs=kwargs)
                    #     p.start()
                    #     # 超时时间 + 0.5 秒缓冲
                    #     time_limit_sec = (kwargs.get('time_limit_ms', 1000) / 1000) + 0.5
                    #     p.join(timeout=time_limit_sec)
                    #     if p.is_alive():
                    #         p.terminate()
                    #         p.join()
                    #         print("solve() timed out!")
                    #         return 'over_time'
                    #     result = q.get()
                    #     if isinstance(result, Exception):
                    #         print("solve() raised an exception:", result)
                    #         return 'error'
                    #     return result

                    # # ================================================
                    # # 原来的 lacam*_8 改造
                    # # ================================================
                    # if aaa == 0:
                    #     lacam_start_time = time.time()
                    #     cout_total_cost = 0  # 总成本
                    #     for i in range(0, len(AT[max_agent]) - 1):
                    #         start = [AT[agent][i] for agent in AT]
                    #         goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                    #         start_Config1.positions = start
                    #         goal_Config2.positions = goal
                    #         planner = LaCAM()

                    #         solution = safe_solve(
                    #             planner,
                    #             start_time=lacam_start_time,
                    #             grid=grid,
                    #             starts=start_Config1,
                    #             goals=goal_Config2,
                    #             seed=args.seed,
                    #             time_limit_ms=args.time_limit_ms,
                    #             flg_star=args.flg_star,
                    #             verbose=args.verbose
                    #         )

                    #         if solution in ['over_time', 'error']:
                    #             print(f"Step {i} solve failed or timed out, skipping...")
                    #             continue

                    #         total_sol = allocate.data_transform(i, solution, total_sol)
                    #         cout_total_cost += planner.total_cost

                    #     lacam_end_time = time.time()
                    #     TaskA_time, Total_time, Total_cost = (
                    #         round(TA_end_time - TA_start_time, 2),
                    #         round((TA_end_time - TA_start_time) + (lacam_end_time - lacam_start_time), 2),
                    #         round(cout_total_cost, 2)
                    #     )

                    #     print("\033[33mAlgth_name\033[0m :", "\033[31mlacam*_8\033[0m\n"
                    #                                         "\033[33mTaskA_time\033[0m : {0}"
                    #                                         "\n\033[33mTotal_time\033[0m : {1}"
                    #                                         "\n\033[33mTotal_cost\033[0m : {2}".format(
                    #                                             TaskA_time, Total_time, Total_cost
                    #                                         ))

                    #     data = (Total_time, Total_cost)
                    #     temp_list.append(data)
                    #     data_save_dict[f'{ins_num}'] = temp_list
                    "lacam0"
                    if aaa == 1 and not over_time_flg:
                        lacam0_start_time = time.time()
                        cout_total_cost0 = 0
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            planner = LaCAM0()
                            solution = planner.solve(
                                grid=grid,
                                starts=start_Config1,
                                goals=goal_Config2,
                                seed=args.seed,
                                time_limit_ms=args.time_limit_ms,
                                flg_star=args.flg_star,
                                verbose=args.verbose,
                            )
                            # 转换solution数据为各代理_路径
                            total_sol = allocate.data_transform(i, solution, total_sol)
                            # save result
                            save_configs_for_visualizer(solution, args.output_file)
                            # validate_mapf_solution(grid, starts, goals, solution)
                            cout_total_cost0 += planner.total_cost
                        lacam0_end_time = time.time()
                        TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                              round((TA_end_time - TA_start_time) + (
                                                                      lacam0_end_time - lacam0_start_time), 2),
                                                              round(cout_total_cost0, 2))

                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mlacam\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(round(TA_end_time - TA_start_time, 2),
                                     round((TA_end_time - TA_start_time) + (lacam0_end_time - lacam0_start_time), 2)
                                     , round(cout_total_cost0, 2)))
                        data = (Total_time, Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "PIBT"
                    if aaa == 2 and not over_time_flg:
                        pibt_start_time = time.time()
                        cout_total_cost1 = 0
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            temp_start = copy.deepcopy(start)
                            temp_goal = copy.deepcopy(goal)
                            start_Config1.positions = start
                            goal_Config2.positions = goal
                            pibt = PIBT(grid, start_Config1, goal_Config2, seed=args.seed)
                            solution, total_cost = pibt.run(max_timestep=args.time_limit_ms)
                            # 转换solution数据为各代理_路径
                            total_sol = allocate.data_transform(i, solution, total_sol)
                            # save result
                            save_configs_for_visualizer(solution, args.output_file)
                            cout_total_cost1 += total_cost

                        pibt_end_time = time.time()
                        TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                              round((TA_end_time - TA_start_time) + (
                                                                      pibt_end_time - pibt_start_time), 2),
                                                              round(cout_total_cost1, 2))
                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mPIBT\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "LNS2 ERROR"
                    if aaa == 3 and not over_time_flg:
                        LNS2_start_time = time.time()
                        cout_total_cost = 0
                        map_width = len(grid[0])
                        map_height = len(grid)
                        numNeighbour = 10
                        for i in range(0, len(AT[max_agent]) - 1):
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            print(start)
                            print(goal)
                            print(i)
                            # print(numNeighbour, map_width, map_height, grid,start,goal)
                            # print(start)
                            # print(goal)
                            # map=[[]]
                            # for num in range(len(grid)):
                            #     for _ in grid[num]:
                            #         map[num].append(_)
                            # map=[[not value for value in row] for row in grid]
                            # map = (~ map).tolist()
                            # print(map)
                            paths, num_replan = LNS2CBS(numNeighbour, map_width, map_height, map, start, goal)
                            cost = get_sum_of_cost(paths)
                            # print(cost)
                            cout_total_cost += cost

                        LNS2_end_time = time.time()
                        TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2),
                                                              round((TA_end_time - TA_start_time) + (
                                                                      LNS2_end_time - LNS2_start_time), 2),
                                                              round(cout_total_cost, 2))
                        print("\n\033[33mTaskA_time\033[0m : {0}"
                              "\n\033[33mTotal_time\033[0m : {1}"
                              "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "ML-EECBS_4"
                    if aaa == 4 and not over_time_flg:
                        ML_EECBS_4_start_time = time.time()
                        cout_total_cost = 0
                        dl = 30
                        over_time_flg_0 = False  # 当前算法是否超过30s
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            c_start = copy.deepcopy(start)
                            c_goal = copy.deepcopy(goal)
                            # 不需要对已到达的代理进行搜索
                            j = 0
                            while j < len(c_start):
                                if c_start[j] == c_goal[j]:
                                    del c_start[j], c_goal[j]
                                j += 1
                            map = copy.deepcopy(grid)
                            map = (~ map).tolist()
                            cbs = ML_EECBS_4(map, c_start, c_goal)
                            paths, cost = cbs.run(dl, ML_EECBS_4_start_time)
                            if paths == 'over_time':
                                over_time_flg_0 = True
                                break
                            else:
                                total_cost = get_sum_of_cost(paths)
                                cout_total_cost += total_cost

                        ML_EECBS_4_end_time = time.time()
                        if not over_time_flg_0:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2), round(
                                (TA_end_time - TA_start_time) + (ML_EECBS_4_end_time - ML_EECBS_4_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (
                                round(TA_end_time - TA_start_time, 2), -1, round(cout_total_cost, 2))

                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mML_EECBS_4\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "ML-EECBS_8"
                    if aaa == 5 and not over_time_flg:
                        ML_EECBS_start_time = time.time()
                        cout_total_cost = 0
                        dl = 600
                        over_time_flg_0 = False  # 当前算法是否超过30s
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            c_start = copy.deepcopy(start)
                            c_goal = copy.deepcopy(goal)
                            # 不需要对已到达的代理进行搜索
                            j = 0
                            while j < len(c_start):
                                if c_start[j] == c_goal[j]:
                                    del c_start[j], c_goal[j]
                                j += 1
                            map = copy.deepcopy(grid)
                            map = (~ map).tolist()
                            cbs = EECBSSolver(map, c_start, c_goal)
                            paths = cbs.find_solution(dl, ML_EECBS_start_time)
                            if paths == 'over_time':
                                over_time_flg_0 = True
                                break
                            else:
                                total_cost = get_sum_of_cost(paths)
                                cout_total_cost += total_cost

                        ML_EECBS_end_time = time.time()
                        if not over_time_flg_0:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2), round(
                                (TA_end_time - TA_start_time) + (ML_EECBS_end_time - ML_EECBS_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (
                                round(TA_end_time - TA_start_time, 2), -1, round(cout_total_cost, 2))

                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mML_EECBS_8\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "MA_CBS_4"
                    if aaa == 6 and not over_time_flg:
                        MA_CBS_start_time = time.time()
                        cout_total_cost = 0
                        dl = 30
                        over_time_flg_0 = False  # 当前算法是否超过30s
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            c_start = copy.deepcopy(start)
                            c_goal = copy.deepcopy(goal)
                            # 不需要对已到达的代理进行搜索
                            j = 0
                            while j < len(c_start):
                                if c_start[j] == c_goal[j]:
                                    del c_start[j], c_goal[j]
                                j += 1
                            map = copy.deepcopy(grid)

                            cbs = MACBS_four(dl, MA_CBS_start_time, map, c_start, c_goal)
                            paths, cost = cbs.get_result()
                            if paths == 'over_time':
                                over_time_flg_0 = True
                                break
                            else:
                                total_cost = cost
                                cout_total_cost += total_cost

                        MA_CBS_end_time = time.time()
                        if not over_time_flg_0:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2), round(
                                (TA_end_time - TA_start_time) + (MA_CBS_end_time - MA_CBS_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (
                                round(TA_end_time - TA_start_time, 2), -1, round(cout_total_cost, 2))

                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mMA_CBS_4\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "MA_CBS_8"
                    if aaa == 7 and not over_time_flg:
                        MA_CBS_8_start_time = time.time()
                        cout_total_cost = 0
                        dl = 50
                        over_time_flg_0 = False  # 当前算法是否超过30s
                        for i in range(0, len(AT[max_agent]) - 1):
                            # 取出各代理的第i个元素和第i+1个元素
                            start = [AT[agent][i] for agent in AT]
                            goal = [AT[agent][i + 1] for agent in AT if i + 1 < len(AT[agent])]
                            c_start = copy.deepcopy(start)
                            c_goal = copy.deepcopy(goal)
                            # 不需要对已到达的代理进行搜索
                            j = 0
                            while j < len(c_start):
                                if c_start[j] == c_goal[j]:
                                    del c_start[j], c_goal[j]
                                j += 1
                            map = copy.deepcopy(grid)

                            cbs = MACBS_eight(dl, MA_CBS_8_start_time, map, c_start, c_goal)
                            paths, cost = cbs.get_result()
                            if paths == 'over_time':
                                over_time_flg_0 = True
                                break
                            else:
                                total_cost = cost
                                cout_total_cost += total_cost
                            # print(paths)

                        MA_CBS_8_end_time = time.time()
                        if not over_time_flg_0:
                            TaskA_time, Total_time, Total_cost = (round(TA_end_time - TA_start_time, 2), round(
                                (TA_end_time - TA_start_time) + (MA_CBS_8_end_time - MA_CBS_8_start_time), 2),
                                                                  round(cout_total_cost, 2))
                        else:
                            TaskA_time, Total_time, Total_cost = (
                                round(TA_end_time - TA_start_time, 2), -1, round(cout_total_cost, 2))

                        print("\n\033[33mAlgth_name\033[0m :", "\033[31mMA_CBS_8\033[0m"
                                                               "\n\033[33mTaskA_time\033[0m : {0}"
                                                               "\n\033[33mTotal_time\033[0m : {1}"
                                                               "\n\033[33mTotal_cost\033[0m : {2}".
                              format(TaskA_time, Total_time, Total_cost))

                        data = (Total_time), (Total_cost)
                        temp_list.append(data)
                        data_save_dict['{0}'.format(ins_num)] = temp_list
                    "CBS"
                    if aaa == 8 and not over_time_flg:
                        "CBS"
                        cbs_start_time = time.time()
                        agents = agent_dict0
                        tasks = task_dict0
                        env = Environment(dimension, agents, tasks, obstacles, pairs)
                        cbs = CBS(env)

                        solution, cost_cbs = cbs.search()
                        total_sol = solution
                        if type(cost_cbs) == dict:
                            cout_total_cost = 0.00
                            print("未得出方案")
                        else:
                            cout_total_cost = cost_cbs
                        cbs_end_time = time.time()
                        print("\033[33mTaskA_time : {0}\nTotal_time : {1}\nTotal_cost : {2}\n\033[0m".
                              format(round(TA_end_time - TA_start_time, 2),
                                     round((TA_end_time - TA_start_time) + (cbs_end_time - cbs_start_time), 2)
                                     , round(cout_total_cost, 2)))

            "数据导入表中"
            if Isexcel == True:
                # 创建一个空的DataFrame列表
                dfs = []
                # 遍历字典，将数据添加到DataFrame列表中
                df = pd.DataFrame({
                    'scen': ['random-{0}'.format(index) for index in data_save_dict.keys()],
                    'LC4_T': [data_save_dict[index][0][0] for index in data_save_dict.keys()],
                    'LC4_C': [data_save_dict[index][0][1] for index in data_save_dict.keys()],
                    'LC8_T': [data_save_dict[index][1][0] for index in data_save_dict.keys()],
                    'LC8_C': [data_save_dict[index][1][1] for index in data_save_dict.keys()],
                    'ML4_T': [data_save_dict[index][2][0] for index in data_save_dict.keys()],
                    'ML4_C': [data_save_dict[index][2][1] for index in data_save_dict.keys()],
                    'ML8_T': [data_save_dict[index][3][0] for index in data_save_dict.keys()],
                    'ML8_C': [data_save_dict[index][3][1] for index in data_save_dict.keys()],
                    'MA4_T': [data_save_dict[index][4][0] for index in data_save_dict.keys()],
                    'MA4_C': [data_save_dict[index][4][1] for index in data_save_dict.keys()],
                    'MA8_T': [data_save_dict[index][5][0] for index in data_save_dict.keys()],
                    'MA8_C': [data_save_dict[index][5][1] for index in data_save_dict.keys()]
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
                    with pd.ExcelWriter(r'Datas3\{0}.xlsx'.format(set_map.replace(".map", ""))) as writer:
                        df1.to_excel(writer, sheet_name='20_100', index=False)
                        df2.to_excel(writer, sheet_name='40_100', index=False)
                        df3.to_excel(writer, sheet_name='60_100', index=False)
                        df4.to_excel(writer, sheet_name='80_100', index=False)
                        df5.to_excel(writer, sheet_name='100_100', index=False)
                        df6.to_excel(writer, sheet_name='120_100', index=False)
            "绘图"
            if total_sol != {}:
                copy_total_sol=allocate.transform_for_draw(pairs,total_sol,agent_dict,task_dict) #制作绘图数据
                # allocate.start_Hamiltonian()          # 展示哈密顿环
                allocate.draw_map(copy_total_sol, False, f"{set_scen}-ins{ins_num}")   # 动态MAPD
            "结束提示"
            winsound.Beep(880, 200)  # 880赫兹，持续200毫秒