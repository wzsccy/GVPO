from itertools import permutations


def generate_task_combinations(initial_comb):
    all_combinations = []

    # 获取初始任务组合的所有代理和任务
    agents = list(initial_comb.keys())
    tasks = {agent: initial_comb[agent] for agent in agents}

    # 生成任务交换组合
    for agent in agents:
        # 获取当前代理的所有任务
        current_tasks = tasks[agent]
        # 移除当前代理的任务，准备与其他代理交换组合
        tasks_for_swap = {k: v for k, v in tasks.items() if k != agent}

        # 遍历其他代理
        for target_agent, target_tasks in tasks_for_swap.items():
            # 对于当前代理的每个任务与目标代理的每个任务进行交换
            for task in current_tasks:
                for target_task in target_tasks:
                    # 创建新的组合
                    new_comb = {k: v.copy() for k, v in tasks.items()}
                    # 执行交换
                    new_comb[agent].remove(task)
                    new_comb[target_agent].remove(target_task)
                    new_comb[agent].append(target_task)
                    new_comb[target_agent].append(task)
                    all_combinations.append(new_comb)

    return all_combinations


# 初始组合
initial_combination = {
    'a0': ['t0', 't1'],
    'a1': ['t2', 't4'],
    'a2': ['t5', 't3']
}

# 生成所有组合
all_combinations = generate_task_combinations(initial_combination)

# 输出所有组合
for i, comb in enumerate(all_combinations, start=1):
    print(f"组合 {i}: {comb}")