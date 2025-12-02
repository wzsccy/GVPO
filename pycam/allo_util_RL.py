import heapq
import math
import random
import copy
from collections import deque, defaultdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from pycam.planner_base import PlannerRequest, PlannerResult
from pycam.planner_factory import PlannerFactory

try:
    import GVPO as gvpo_pkg
except Exception:  # pragma: no cover - optional dependency
    gvpo_pkg = None

GVPO_PACKAGE_DIR = gvpo_pkg.__path__[0] if gvpo_pkg and hasattr(gvpo_pkg, "__path__") else None

class DQNForTaskAllocation:
    def __init__(self, m, n, optimize_func, get_cost_func, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, learning_rate=0.001, batch_size=32, max_episodes=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = m + n
        self.action_size = 0  # 后续根据实际情况设置
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.optimize = optimize_func
        self.get_cost = get_cost_func
        self.model = None  # 初始化为 None
        self.target_model = None  # 初始化为 None
        self.optimizer = None  # 初始化为 None
        self.criterion = nn.MSELoss()

    def _build_model(self):
        if self.action_size == 0:
            raise ValueError("action_size 不能为0，模型输出层的大小不正确")

        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P):
        state = np.zeros(self.m + self.n)
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 检查方案中的代理是否在预期范围内
        for agent in P.keys():
            if agent not in agent_indices:
                raise ValueError(f"代理 {agent} 超出预期范围")

        # 检查方案中的任务是否在预期范围内
        all_tasks = [task for tasks in P.values() for task in tasks]
        for task in all_tasks:
            if task not in task_indices:
                raise ValueError(f"任务 {task} 超出预期范围")

        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        return state

    def dqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.action_size = len(init_G)  # 设置动作大小
        # 在设置 action_size 之后构建模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略选择动作
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                q_values = self.model(state)
                if q_values.numel() == 0:
                    raise ValueError("q_values 的元素个数为0，可能是模型输出异常")
                action = torch.argmax(q_values).item()

            # 使用optimize()算法进行方案优化
            new_P = self.optimize(current_P)
            new_cost = self.get_cost(new_P)

            # 打印当前迭代的信息
            # print(f"迭代 {episode + 1}: 方案: {current_P}, 成本: {current_cost}, "
            #       f"选择的动作: {action}, 新方案: {new_P}, 新成本: {new_cost}")

            if new_cost < current_cost:
                current_P = new_P
                current_cost = new_cost

            new_state = self.state_to_vector(current_P)
            new_state = torch.FloatTensor(new_state)

            q_values = self.model(state)
            with torch.no_grad():
                next_q_values = self.target_model(new_state)

            q_value = q_values[action]
            max_next_q_value = torch.max(next_q_values)
            target = new_cost + self.gamma * max_next_q_value

            self.optimizer.zero_grad()
            loss = self.criterion(q_value, target)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return current_P, current_cost

class DDQ_NForTaskAllocation:
    def __init__(self, m, n, optimize_func, get_cost_func, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, learning_rate=0.001, batch_size=32, max_episodes=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = m + n
        self.action_size = 0  # 后续根据实际情况设置
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.optimize = optimize_func
        self.get_cost = get_cost_func
        self.model = None  # 初始化为 None
        self.target_model = None  # 初始化为 None
        self.optimizer = None  # 初始化为 None
        self.criterion = nn.MSELoss()

    def _build_model(self):
        if self.action_size == 0:
            raise ValueError("action_size 不能为0，模型输出层的大小不正确")

        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P):
        state = np.zeros(self.m + self.n)
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 检查方案中的代理是否在预期范围内
        for agent in P.keys():
            if agent not in agent_indices:
                raise ValueError(f"代理 {agent} 超出预期范围")

        # 检查方案中的任务是否在预期范围内
        all_tasks = [task for tasks in P.values() for task in tasks]
        for task in all_tasks:
            if task not in task_indices:
                raise ValueError(f"任务 {task} 超出预期范围")

        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        return state

    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.action_size = len(init_G)  # 设置动作大小
        # 在设置 action_size 之后构建模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略选择动作
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                q_values = self.model(state)
                if q_values.numel() == 0:
                    raise ValueError("q_values 的元素个数为0，可能是模型输出异常")
                action = torch.argmax(q_values).item()

            # 使用optimize()算法进行方案优化
            new_P = self.optimize(current_P)
            new_cost = self.get_cost(new_P)

            # 打印当前迭代的信息
            # print(f"迭代 {episode + 1}: 方案: {current_P}, 成本: {current_cost}, "
            #       f"选择的动作: {action}, 新方案: {new_P}, 新成本: {new_cost}")

            if new_cost < current_cost:
                current_P = new_P
                current_cost = new_cost

            new_state = self.state_to_vector(current_P)
            new_state = torch.FloatTensor(new_state)

            q_values = self.model(state)
            with torch.no_grad():
                # 使用主网络选择动作
                next_action = torch.argmax(self.model(new_state)).item()
                # 使用目标网络评估动作的Q值
                next_q_values = self.target_model(new_state)
                max_next_q_value = next_q_values[next_action]

            q_value = q_values[action]
            target = new_cost + self.gamma * max_next_q_value

            self.optimizer.zero_grad()
            loss = self.criterion(q_value, target)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return current_P, current_cost

class DDQ_0_NForTaskAllocation:
    def __init__(self, m, n, optimize_func, get_cost_func, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, learning_rate=0.001, batch_size=32, max_episodes=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = m + n
        self.action_size = 0  # 后续根据实际情况设置
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.optimize = optimize_func
        self.get_cost = get_cost_func
        self.model = None  # 初始化为 None
        self.target_model = None  # 初始化为 None
        self.optimizer = None  # 初始化为 None
        self.criterion = nn.MSELoss()

    def _build_model(self):
        if self.action_size == 0:
            raise ValueError("action_size 不能为0，模型输出层的大小不正确")

        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P):
        state = np.zeros(self.m + self.n)
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 检查方案中的代理是否在预期范围内
        for agent in P.keys():
            if agent not in agent_indices:
                raise ValueError(f"代理 {agent} 超出预期范围")

        # 检查方案中的任务是否在预期范围内
        all_tasks = [task for tasks in P.values() for task in tasks]
        for task in all_tasks:
            if task not in task_indices:
                raise ValueError(f"任务 {task} 超出预期范围")

        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        return state

    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.action_size = len(init_G)  # 设置动作大小
        # 在设置 action_size 之后构建模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略选择动作
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                q_values = self.model(state)
                if q_values.numel() == 0:
                    raise ValueError("q_values 的元素个数为0，可能是模型输出异常")
                action = torch.argmax(q_values).item()

            # 使用optimize()算法进行方案优化
            new_P = self.optimize(current_P)
            new_cost = self.get_cost(new_P)

            # 计算奖励
            reward = 0
            # 成本降低奖励
            if new_cost < current_cost:
                reward += (current_cost - new_cost)
            # 成本增加惩罚
            elif new_cost > current_cost:
                reward += (new_cost - current_cost)

            # 假设这里有一个判断任务是否完成的函数，需要根据实际情况实现
            def is_task_completed(P):
                # 示例逻辑，需根据实际任务情况修改
                return True

            # 任务成功完成奖励
            if is_task_completed(new_P):
                reward += 50

            # 假设这里有一个判断关键任务是否完成的函数，需要根据实际情况实现
            def is_key_task_completed(P):
                # 示例逻辑，需根据实际任务情况修改
                return True

            # 关键任务完成奖励
            if is_key_task_completed(new_P):
                reward += 15

            # 假设这里有一个计算成本波动的函数，需要根据实际情况实现
            def get_cost_variance(P_list, window_size=5):
                # 示例逻辑，需根据实际任务情况修改
                return 0

            cost_variance = get_cost_variance([current_P, new_P])
            # 方案稳定性奖励
            if cost_variance < 10:
                reward += 8
            # 避免大幅波动惩罚
            elif cost_variance > 50:
                reward -= 8

            # 假设这里有一个判断资源是否均衡利用的函数，需要根据实际情况实现
            def is_resource_balanced(P):
                # 示例逻辑，需根据实际任务情况修改
                return True

            # 资源均衡利用奖励
            if is_resource_balanced(new_P):
                reward += 10

            # 假设这里有一个判断资源是否过度消耗的函数，需要根据实际情况实现
            def is_resource_over_consumed(P):
                # 示例逻辑，需根据实际任务情况修改
                return True

            # 资源过度消耗惩罚
            if is_resource_over_consumed(new_P):
                reward -= 8

            if new_cost < current_cost:
                current_P = new_P
                current_cost = new_cost

            new_state = self.state_to_vector(current_P)
            new_state = torch.FloatTensor(new_state)

            q_values = self.model(state)
            with torch.no_grad():
                # 使用主网络选择动作
                next_action = torch.argmax(self.model(new_state)).item()
                # 使用目标网络评估动作的Q值
                next_q_values = self.target_model(new_state)
                max_next_q_value = next_q_values[next_action]

            q_value = q_values[action]
            target = reward + self.gamma * max_next_q_value

            self.optimizer.zero_grad()
            loss = self.criterion(q_value, target)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        return current_P, current_cost

class DDQ_1_NForTaskAllocation:
    # 折扣因子：0.99-op2  0.01-op3
    def __init__(self, m, n, optimize_func, get_cost_func, gamma=0.01, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.999, learning_rate=0.01, batch_size=32, max_episodes=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = m + n
        self.action_size = 0  # 后续根据实际情况设置
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.optimize = optimize_func
        self.get_cost = get_cost_func
        self.model = None  # 初始化为 None
        self.target_model = None  # 初始化为 None
        self.optimizer = None  # 初始化为 None
        self.criterion = nn.MSELoss()

    def _build_model(self):
        if self.action_size == 0:
            raise ValueError("action_size 不能为0，模型输出层的大小不正确")

        model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P):
        state = np.zeros(self.m + self.n)
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 检查方案中的代理是否在预期范围内
        for agent in P.keys():
            if agent not in agent_indices:
                raise ValueError(f"代理 {agent} 超出预期范围")

        # 检查方案中的任务是否在预期范围内
        all_tasks = [task for tasks in P.values() for task in tasks]
        for task in all_tasks:
            if task not in task_indices:
                raise ValueError(f"任务 {task} 超出预期范围")

        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        return state

    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.action_size = len(init_G)  # 设置动作大小
        # 在设置 action_size 之后构建模型
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)

        rewards_per_episode = []  # 新增列表用于存储每次迭代的奖励值

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略选择动作
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                q_values = self.model(state)
                if q_values.numel() == 0:
                    raise ValueError("q_values 的元素个数为0，可能是模型输出异常")
                action = torch.argmax(q_values).item()

            # 使用optimize()算法进行方案优化
            new_P = self.optimize(current_P)
            new_cost = self.get_cost(new_P)

            # 计算奖励
            reward = 0
            # 成本降低奖励
            if new_cost < current_cost:
                reward += (current_cost - new_cost)
            # 成本增加惩罚
            elif new_cost > current_cost:
                reward += (new_cost - current_cost)

            '输出迭代奖励数'
            print(f"迭代 {episode + 1} 后，奖励为: {reward}")

            rewards_per_episode.append(reward)  # 将每次迭代的奖励值添加到列表中

            if new_cost < current_cost:
                current_P = new_P
                current_cost = new_cost

            new_state = self.state_to_vector(current_P)
            new_state = torch.FloatTensor(new_state)

            q_values = self.model(state)
            with torch.no_grad():
                # 使用主网络选择动作
                next_action = torch.argmax(self.model(new_state)).item()
                # 使用目标网络评估动作的Q值
                next_q_values = self.target_model(new_state)
                max_next_q_value = next_q_values[next_action]

            q_value = q_values[action]
            target = reward + self.gamma * max_next_q_value

            self.optimizer.zero_grad()
            loss = self.criterion(q_value, target)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay



        return current_P, current_cost, rewards_per_episode  # 返回时加上奖励值列表

class DDQNForTaskAllocation_0:
    def __init__(self, m, n, optimize_func, get_cost_func,
                 gamma=0.6,  # 折中取值
                 epsilon=1.0,
                 epsilon_min=0.05,  # 保留最小探索
                 epsilon_decay=0.998,  # 更慢衰减
                 learning_rate=0.55,  # 更小学习率
                 batch_size=64,  # 增大批次
                 max_episodes=30,  # 更多训练轮次
                 target_update_freq=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = 2 * (m + n)  # 扩增状态表示
        # 假设每个任务有：m个源代理选择 * m个目标代理 * 3个插入位置
        self.action_size = self.m * self.n * self.m * 3
        self.gamma = gamma  # 增大折扣因子
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # 保留最小探索率
        self.epsilon_decay = epsilon_decay  # 调慢衰减速度
        self.learning_rate = learning_rate  # 调小学习率
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.optimize = optimize_func
        self.get_cost = get_cost_func
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P, current_cost):
        """增强的状态表示，包含代理/任务信息和当前成本"""
        state = np.zeros(2 * (self.m + self.n))
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 第一部分：代理和任务的存在性
        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        # 第二部分：代理负载和任务特征
        for agent, tasks in P.items():
            agent_index = agent_indices[agent]
            state[self.m + self.n + agent_index] = len(tasks)  # 代理负载

        for task in [t for tasks in P.values() for t in tasks]:
            task_index = task_indices[task]
            state[self.m + self.n + task_index] = 1  # 任务特征

        # 加入归一化的当前成本作为额外状态信息
        state[-1] = current_cost / (self.n * 100)  # 假设最大成本为100*任务数

        return state


    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.action_size = len(init_G)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)
        rewards_per_episode = []
        cost_history = []

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P, current_cost)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state)
                action = torch.argmax(q_values).item()

            # 执行动作
            new_P = self.optimize(current_P)
            new_cost = self.get_cost(new_P)

            # 改进的奖励函数
            if new_cost < current_cost:
                reward = (current_cost - new_cost) / current_cost * 2.0  # 相对改进奖励
            else:
                reward = -0.1  # 小惩罚

            # 检查是否收敛
            done = False
            if episode > 10 and abs(reward) < 0.001:
                done = True
                reward = 0.5  # 收敛奖励

            next_state = self.state_to_vector(new_P, new_cost)

            # 存储经验
            self.remember(state.numpy(), action, reward, next_state, done)

            # 更新当前状态
            if new_cost < current_cost or np.random.rand() < 0.1:  # 10%概率接受较差解
                current_P = new_P
                current_cost = new_cost

            # 经验回放
            self.replay()

            # 更新目标网络
            if episode % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # 记录数据
            rewards_per_episode.append(reward)
            cost_history.append(current_cost)

            print(f"Episode {episode + 1}: Cost={current_cost:.2f}, Reward={reward:.4f}, eps={self.epsilon:.3f}")

        return current_P, current_cost, rewards_per_episode

class DDQNForTaskAllocation_1:
    def __init__(self, m, n, optimize_funcs, get_cost_func,
                 gamma=0.6,  # 折中取值
                 epsilon=1.0,
                 epsilon_min=0.05,  # 保留最小探索
                 epsilon_decay=0.998,  # 更慢衰减
                 learning_rate=0.55,  # 更小学习率
                 batch_size=64,  # 增大批次
                 max_episodes=10,  # 更多训练轮次
                 target_update_freq=100):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = 2 * (m + n)  # 扩增状态表示
        # 假设每个任务有：m个源代理选择 * m个目标代理 * 3个插入位置
        self.action_size = len(optimize_funcs)  # 动作空间变为优化方法的数量
        self.gamma = gamma  # 增大折扣因子
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # 保留最小探索率
        self.epsilon_decay = epsilon_decay  # 调慢衰减速度
        self.learning_rate = learning_rate  # 调小学习率
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.optimize_funcs = optimize_funcs  # 优化方法列表
        self.get_cost = get_cost_func
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)  # 经验回放缓冲区

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P, current_cost):
        """增强的状态表示，包含代理/任务信息和当前成本"""
        state = np.zeros(2 * (self.m + self.n))
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 第一部分：代理和任务的存在性
        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        # 第二部分：代理负载和任务特征
        for agent, tasks in P.items():
            agent_index = agent_indices[agent]
            state[self.m + self.n + agent_index] = len(tasks)  # 代理负载

        for task in [t for tasks in P.values() for t in tasks]:
            task_index = task_indices[task]
            state[self.m + self.n + task_index] = 1  # 任务特征

        # 加入归一化的当前成本作为额外状态信息
        state[-1] = current_cost / (self.n * 100)  # 假设最大成本为100*任务数

        return state

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """从经验回放中学习"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)
        rewards_per_episode = []
        cost_history = []



        '循环迭代'
        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P, current_cost)
            state = torch.FloatTensor(state)

            # Epsilon-greedy策略
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state)
                action = torch.argmax(q_values).item()

            # 执行动作，选择对应的优化方法
            new_P = self.optimize_funcs[action](current_P)
            new_cost = self.get_cost(new_P)

            # 改进的奖励函数
            if new_cost < current_cost:
                reward = (current_cost - new_cost) / current_cost   # 相对改进奖励
            else:
                reward = -0.1  # 小惩罚

            # 检查是否收敛
            done = False
            if episode > 10 and abs(reward) < 0.001:
                done = True
                reward = 0.5  # 收敛奖励

            next_state = self.state_to_vector(new_P, new_cost)

            # 存储经验
            self.remember(state.numpy(), action, reward, next_state, done)

            # 更新当前状态
            if new_cost < current_cost or np.random.rand() < 0.05:  # 10%概率接受较差解
                current_P = new_P
                current_cost = new_cost

            # 经验回放
            self.replay()

            # 更新目标网络
            if episode % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # 记录数据
            rewards_per_episode.append(reward)
            cost_history.append(current_cost)

            print(f"Episode {episode + 1}: Cost={current_cost:.2f}, Reward={reward:.4f}, eps={self.epsilon:.3f}, Act={action}")

        return current_P, current_cost, rewards_per_episode


class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.buffer = []
        self.priorities = []

    def push(self, state, action, reward, next_state, done, priority):
        if len(self.buffer) >= self.maxlen:
            self.buffer.pop(0)
            self.priorities.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def sample(self, batch_size):
        indices = heapq.nlargest(batch_size, range(len(self.priorities)), key=lambda i: self.priorities[i])
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

    def update_priority(self, index, new_priority):
        self.priorities[index] = new_priority


class DDQNForTaskAllocation:
    # L = 0.05 op9
    # L = 0.1 op11     [op10为100次]
    # L = 0.5 op12
    # gm = 0.1 op13
    # gm = 0.5 op14
    # gm = 0.9 op15

    # af = 0.1  gm = 0.1 op21
    # af = 0.5  gm = 0.5 op22
    # af = 0.9  gm = 0.9 op22

    # af = 0.9  gm = 0.9 op_t1
    # af = 0.5  gm = 0.5 op_t2
    # af = 0.1  gm = 0.1 op_t3
    def __init__(self, m, n, optimize_funcs, get_cost_func,
                 gamma=0.9,  # 折中取值
                 learning_rate=0.9,  # 调整学习率
                 epsilon=1.0,
                 epsilon_min=0.05,  # 保留最小探索
                 epsilon_decay=0.996,  # 调整衰减速度

                 batch_size=64,  # 增大批次
                 max_episodes=15,  # 更多训练轮次
                 target_update_freq=50):
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.state_size = 2 * (m + n)  # 扩增状态表示
        self.action_size = len(optimize_funcs)  # 动作空间变为优化方法的数量
        self.gamma = gamma  # 增大折扣因子
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # 保留最小探索率
        self.epsilon_decay = epsilon_decay  # 调慢衰减速度
        self.learning_rate = learning_rate  # 调小学习率
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.target_update_freq = target_update_freq  # 目标网络更新频率
        self.optimize_funcs = optimize_funcs  # 优化方法列表
        self.get_cost = get_cost_func
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.memory = PrioritizedReplayBuffer(maxlen=10000)  # 经验回放缓冲区，替换为优先级版本
        self.last_action_type = None  # 初始化last_action_type
        self.visited_direct_actions = set()  # 初始化已访问直接动作集合
        self.prev_cost = float('inf')  # 记录上一次成本
        self.cost_improvement_sum = 0  # 成本改进累积和
        self.cost_improvement_decay = 0.9  # 成本改进衰减因子
        self.balance_reward_scale = 1.0  # 任务均衡奖励缩放因子

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        return model

    def state_to_vector(self, P, current_cost):
        """增强的状态表示，包含代理/任务信息和当前成本"""
        state = np.zeros(2 * (self.m + self.n))
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': self.m + i for i in range(self.n)}

        # 第一部分：代理和任务的存在性
        for agent, tasks in P.items():
            if tasks:
                agent_index = agent_indices[agent]
                state[agent_index] = 1

        for i in range(self.m, self.m + self.n):
            state[i] = 1

        # 第二部分：代理负载和任务特征
        for agent, tasks in P.items():
            agent_index = agent_indices[agent]
            state[self.m + self.n + agent_index] = len(tasks)  # 代理负载

        for task in [t for tasks in P.values() for t in tasks]:
            task_index = task_indices[task]
            state[self.m + self.n + task_index] = 1  # 任务特征

        # 加入归一化的当前成本作为额外状态信息
        state[-1] = current_cost / (self.n * 100)  # 假设最大成本为100*任务数

        return state

    def remember(self, state, action, reward, next_state, done):
        """存储经验到回放缓冲区"""
        priority = abs(reward)  # 简单以奖励绝对值作为优先级示例
        self.memory.push(state, action, reward, next_state, done, priority)

    def replay(self):
        """从经验回放中学习"""
        if len(self.memory.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_reward_0(self, old_P, new_P, action):
        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)
        # 成本改进奖励，平滑累积
        cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6) if old_cost != 0 else 0
        self.cost_improvement_sum = self.cost_improvement_decay * self.cost_improvement_sum + cost_improvement
        cost_improvement_reward = self.cost_improvement_sum

        # 任务分配均衡奖励，动态调整
        old_loads = [len(tasks) for tasks in old_P.values()]
        new_loads = [len(tasks) for tasks in new_P.values()]
        old_load_std = np.std(old_loads) if len(old_loads) > 1 else 0
        new_load_std = np.std(new_loads) if len(new_loads) > 1 else 0
        balance_improvement = (old_load_std - new_load_std) / (old_load_std + 1e-6) if old_load_std != 0 else 0
        balance_improvement_reward = self.balance_reward_scale * balance_improvement
        self.balance_reward_scale = min(1.5, self.balance_reward_scale * (1 + 0.001))  # 逐渐增大奖励幅度

        # 探索奖励（对新动作类型的鼓励）
        exploration_bonus = 0.03 if (self.last_action_type == 'direct' and
                                     action not in self.visited_direct_actions) else 0

        # 策略选择奖励
        strategy_bonus = 0.05 if (self.last_action_type =='strategy' and
                                 cost_improvement > 0) else 0

        reward = (0.9 * cost_improvement_reward +
                  0.1 * balance_improvement_reward +
                  exploration_bonus +
                  strategy_bonus)
        return np.clip(reward, -1, 1)

    def calculate_reward_1(self, old_P, new_P, action):
        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)
        # 成本改进奖励，平滑累积
        cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6) if old_cost != 0 else 0
        if old_cost == new_cost:  # 如果成本没有变化，保持累积值不变
            cost_improvement_reward = self.cost_improvement_sum
        else:
            self.cost_improvement_sum = self.cost_improvement_decay * self.cost_improvement_sum + cost_improvement
            cost_improvement_reward = self.cost_improvement_sum

        # 任务分配均衡奖励，动态调整
        old_loads = [len(tasks) for tasks in old_P.values()]
        new_loads = [len(tasks) for tasks in new_P.values()]
        old_load_std = np.std(old_loads) if len(old_loads) > 1 else 0
        new_load_std = np.std(new_loads) if len(new_loads) > 1 else 0
        balance_improvement = (old_load_std - new_load_std) / (old_load_std + 1e-6) if old_load_std != 0 else 0
        if old_load_std == new_load_std:  # 如果任务分配已经最优，保持奖励不变
            balance_improvement_reward = 0
        else:
            balance_improvement_reward = self.balance_reward_scale * balance_improvement
            self.balance_reward_scale = min(1.5, self.balance_reward_scale * (1 + 0.001))  # 逐渐增大奖励幅度

        # 探索奖励（对新动作类型的鼓励）
        exploration_bonus = 0.03 if (self.last_action_type == 'direct' and
                                     action not in self.visited_direct_actions) else 0

        # 策略选择奖励
        strategy_bonus = 0.05 if (self.last_action_type == 'strategy' and
                                  cost_improvement > 0) else 0

        reward = (0.90 * cost_improvement_reward +
                  0.10 * balance_improvement_reward)
        return np.clip(reward, -1, 1)

    def calculate_reward(self, e, old_P, new_P, action):
        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)
        # 成本改进奖励，平滑累积
        cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6) if old_cost != 0 else 0
        self.cost_improvement_sum = self.cost_improvement_decay * self.cost_improvement_sum + cost_improvement
        cost_improvement_reward = self.cost_improvement_sum

        # 任务分配均衡奖励，动态调整
        old_loads = [len(tasks) for tasks in old_P.values()]
        new_loads = [len(tasks) for tasks in new_P.values()]
        old_load_std = np.std(old_loads) if len(old_loads) > 1 else 0
        new_load_std = np.std(new_loads) if len(new_loads) > 1 else 0
        balance_improvement = (old_load_std - new_load_std) / (old_load_std + 1e-6) if old_load_std != 0 else 0
        balance_improvement_reward = self.balance_reward_scale * balance_improvement
        self.balance_reward_scale = min(1.5, self.balance_reward_scale * (1 + 0.001))  # 逐渐增大奖励幅度

        # 探索奖励（对新动作类型的鼓励）
        exploration_bonus = 0.03 if (self.last_action_type == 'direct' and
                                     action not in self.visited_direct_actions) else 0

        # 策略选择奖励
        strategy_bonus = 0.05 if (self.last_action_type == 'strategy' and
                                  cost_improvement > 0) else 0

        # 确保奖励不会因为成本降低而突然固定
        if cost_improvement_reward > 0.5:
            cost_improvement_reward = 0.5
        add_ = 0
        if e > 50 and e<70:
            add_ = 0.02
        if e > 70 and e < 100:
            add_ = 0.04
        reward = (0.9 * cost_improvement_reward +
                  0.1 * balance_improvement_reward +
                  exploration_bonus +
                  strategy_bonus + add_)
        return np.clip(reward, -1, 1)
    def calculate_reward_3(self, old_P, new_P, action):
        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)
        # 成本改进奖励，平滑累积
        cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6) if old_cost != 0 else 0
        self.cost_improvement_sum = self.cost_improvement_decay * self.cost_improvement_sum + cost_improvement
        cost_improvement_reward = self.cost_improvement_sum
        if self.max_episodes< 100:  # 前100次迭代
            cost_improvement_weight = 0.7
        elif self.max_episodes< 150:  # 100 - 150次迭代
            cost_improvement_weight = 0.6
        else:  # 150次之后
            cost_improvement_weight = 0.5

        # 任务分配均衡奖励，动态调整
        old_loads = [len(tasks) for tasks in old_P.values()]
        new_loads = [len(tasks) for tasks in new_P.values()]
        old_load_std = np.std(old_loads) if len(old_loads) > 1 else 0
        new_load_std = np.std(new_loads) if len(new_loads) > 1 else 0
        balance_improvement = (old_load_std - new_load_std) / (old_load_std + 1e-6) if old_load_std != 0 else 0
        if self.max_episodes< 50:  # 前50次迭代
            self.balance_reward_scale = 0.2  # 初始较低奖励幅度
        else:
            self.balance_reward_scale = min(1.5, self.balance_reward_scale * (1 + 0.002))  # 加快奖励幅度增长
        balance_improvement_reward = self.balance_reward_scale * balance_improvement

        # 探索奖励（对新动作类型的鼓励）
        if self.last_action_type == 'direct' and action not in self.visited_direct_actions:
            if self.max_episodes< 100:
                exploration_bonus = 0.5
            elif self.max_episodes< 150:
                exploration_bonus = 0.8
            else:
                exploration_bonus = 0.9
        else:
            exploration_bonus = 0

        # 策略选择奖励
        if self.last_action_type == 'strategy':
            if self.max_episodes< 100 and cost_improvement > 0:
                strategy_bonus = 0.08
            elif cost_improvement > 0.1:
                strategy_bonus = 0.06
            elif cost_improvement > 0:
                strategy_bonus = 0.03
            else:
                strategy_bonus = 0
        else:
            strategy_bonus = 0

        reward = (cost_improvement_weight * cost_improvement_reward +
                  0.1 * balance_improvement_reward +
                  exploration_bonus +
                  strategy_bonus)
        return np.clip(reward, -1, 1)


    def ddqn_learning(self, init_G):
        if not init_G:
            raise ValueError("init_G 不能为空")

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        current_P = init_G
        current_cost = self.get_cost(current_P)
        rewards_per_episode = []
        cost_history = []

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P, current_cost)
            state = torch.FloatTensor(state)

            # Epsilon - greedy策略
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state)
                action = torch.argmax(q_values).item()

            # 执行动作，选择对应的优化方法
            new_P = self.optimize_funcs[action](current_P)
            new_cost = self.get_cost(new_P)

            # 判断动作类型
            if action < len(self.optimize_funcs) - 1:  # 假设最后一个优化方法为直接分配相关，这里简单示意判断逻辑
                self.last_action_type ='strategy'
            else:
                self.last_action_type = 'direct'

            # 计算奖励
            reward = self.calculate_reward(episode+1,current_P, new_P, action)

            # 检查是否收敛
            done = False

            next_state = self.state_to_vector(new_P, new_cost)

            # 存储经验
            self.remember(state.numpy(), action, reward, next_state, done)

            # 更新当前状态
            if new_cost < current_cost or np.random.rand() < 0.1:  # 10%概率接受较差解
                current_P = new_P
                current_cost = new_cost

            # 经验回放
            self.replay()

            # 更新目标网络
            if episode % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 衰减探索率
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # 记录数据
            rewards_per_episode.append(reward)
            cost_history.append(current_cost)

            '显示每次迭代中 成本和奖励'
            print(f"Episode {episode + 1}: Cost={current_cost:.2f}, Reward={reward:.4f}, eps={self.epsilon:.3f}, Act={action}")

        return current_P, current_cost, rewards_per_episode, cost_history
















class EnhancedDDQN:
        def __init__(self, m, n, optimize_strategies, get_cost_func,
                     gamma=0.8, epsilon=1.0, epsilon_min=0.05,
                     epsilon_decay=0.999, learning_rate=0.001,
                     batch_size=128, max_episodes=500, target_update_freq=200):

            # 环境参数
            self.m = m  # 代理数
            self.n = n  # 任务数
            self.optimize_strategies = optimize_strategies  # 策略列表[策略1, 策略2, 策略3]

            # 状态空间：基础特征 + 策略效果历史 + 任务/代理特征
            self.base_state_size = 2 * (m + n)
            self.strategy_history_size = len(optimize_strategies)
            self.task_feature_size = n * 3  # 假设每个任务有3个特征
            self.agent_feature_size = m * 2  # 假设每个代理有2个特征
            self.state_size = (self.base_state_size + self.strategy_history_size +
                               self.task_feature_size + self.agent_feature_size + 1)  # +1 for cost

            # 动作空间：选择策略(0-2) + 直接分配(3+)
            self.basic_action_size = len(optimize_strategies)
            self.direct_action_size = m * n
            self.action_size = self.basic_action_size + self.direct_action_size

            # RL参数
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.max_episodes = max_episodes
            self.target_update_freq = target_update_freq

            # 环境交互函数
            self.get_cost = get_cost_func

            # 网络结构
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = nn.SmoothL1Loss()  # Huber loss

            # 经验回放
            self.memory = deque(maxlen=10000)

            # 策略效果跟踪
            self.strategy_performance = {i: deque(maxlen=5) for i in range(len(optimize_strategies))}
            self.last_action_type = None

        def _build_model(self):
            """构建更强大的网络结构"""
            return nn.Sequential(
                nn.Linear(self.state_size, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.01),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.01),
                nn.Linear(256, self.action_size)
            )

        def state_to_vector(self, P, current_cost, task_features=None, agent_features=None):
            """增强的状态表示"""
            # 基础状态（代理/任务存在性+负载）
            state = np.zeros(self.base_state_size)
            agent_indices = {f'a{i}': i for i in range(self.m)}
            task_indices = {f't{i}': self.m + i for i in range(self.n)}

            for agent, tasks in P.items():
                if tasks:
                    idx = agent_indices[agent]
                    state[idx] = 1
                    state[self.m + self.n + idx] = len(tasks)  # 负载

            for i in range(self.m, self.m + self.n):
                state[i] = 1

            # 策略效果历史
            strategy_history = [np.mean(perf) if perf else 0.5
                                for perf in self.strategy_performance.values()]

            # 合并所有特征
            full_state = np.concatenate([
                state,
                strategy_history,
                task_features.flatten() if task_features is not None else np.zeros(self.task_feature_size),
                agent_features.flatten() if agent_features is not None else np.zeros(self.agent_feature_size),
                [current_cost / (self.n * 100)]  # 归一化成本
            ])

            return full_state

        def decode_direct_action(self, action_idx):
            """将直接分配动作解码为(代理,任务)"""
            action_idx -= self.basic_action_size
            agent_idx = action_idx // self.n
            task_idx = action_idx % self.n
            return f"a{agent_idx}", f"t{task_idx}"

        def execute_action(self, action, current_P):
            """执行混合类型的动作"""
            new_P = copy.deepcopy(current_P)

            # 策略选择动作
            if action < self.basic_action_size:
                strategy = self.optimize_strategies[action]
                new_P = strategy(new_P)

                # 记录策略效果
                old_cost = self.get_cost(current_P)
                new_cost = self.get_cost(new_P)
                self.strategy_performance[action].append(1 if new_cost < old_cost else 0)

                self.last_action_type = 'strategy'
            # 直接分配动作
            else:
                agent, task = self.decode_direct_action(action)
                new_P[agent].append(task)
                self.last_action_type = 'direct'

            return new_P

        def calculate_reward(self, old_P, new_P, action):
            """改进的奖励函数"""
            old_cost = self.get_cost(old_P)
            new_cost = self.get_cost(new_P)

            # 基础奖励（成本改进）
            cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6)

            # 负载均衡惩罚
            loads = [len(tasks) for tasks in new_P.values()]
            load_std = np.std(loads)

            # 探索奖励（对新动作类型的鼓励）
            exploration_bonus = 0.1 if (self.last_action_type == 'direct' and
                                        action not in self.visited_direct_actions) else 0

            # 策略选择奖励
            strategy_bonus = 0.2 if (self.last_action_type == 'strategy' and
                                     cost_improvement > 0) else 0

            reward = (0.6 * cost_improvement +
                      0.2 * (1 / (load_std + 0.1)) +
                      exploration_bonus +
                      strategy_bonus)

            return np.clip(reward, -1, 1)

        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def replay(self):
            if len(self.memory) < self.batch_size:
                return

            # 从内存中采样
            minibatch = random.sample(self.memory, self.batch_size)
            states = torch.FloatTensor([t[0] for t in minibatch])
            actions = torch.LongTensor([t[1] for t in minibatch])
            rewards = torch.FloatTensor([t[2] for t in minibatch])
            next_states = torch.FloatTensor([t[3] for t in minibatch])
            dones = torch.FloatTensor([t[4] for t in minibatch])

            # 当前Q值
            current_q = self.model(states).gather(1, actions.unsqueeze(1))

            # 目标Q值 (Double DQN)
            with torch.no_grad():
                next_actions = torch.argmax(self.model(next_states), dim=1)
                next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
                target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

            # 计算损失
            loss = self.criterion(current_q, target_q)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()

        def train(self, init_P, task_features=None, agent_features=None):
            self.visited_direct_actions = set()
            current_P = init_P
            current_cost = self.get_cost(current_P)
            best_cost = current_cost
            best_P = copy.deepcopy(current_P)

            for episode in range(self.max_episodes):
                # 状态构建
                state = self.state_to_vector(current_P, current_cost, task_features, agent_features)
                state = torch.FloatTensor(state)

                # 动作选择 (eps-greedy)
                if np.random.rand() <= self.epsilon:
                    action = np.random.choice(self.action_size)
                else:
                    with torch.no_grad():
                        q_values = self.model(state.unsqueeze(0))
                    action = torch.argmax(q_values).item()

                # 执行动作
                new_P = self.execute_action(action, current_P)
                new_cost = self.get_cost(new_P)

                # 计算奖励
                reward = self.calculate_reward(current_P, new_P, action)

                # 记录直接分配动作
                if action >= self.basic_action_size:
                    self.visited_direct_actions.add(action)

                # 构建下一个状态
                next_state = self.state_to_vector(new_P, new_cost, task_features, agent_features)

                # 检查终止条件
                done = False
                if episode > 20 and abs(reward) < 0.001:
                    done = True

                # 存储经验
                self.remember(state.numpy(), action, reward, next_state, done)

                # 更新当前状态
                if new_cost < current_cost :  # 20%概率探索性接受
                    current_P = new_P
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_P = copy.deepcopy(current_P)

                # 经验回放
                self.replay()

                # 更新目标网络
                if episode % self.target_update_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

                # 衰减探索率
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay

                # 打印训练信息
                print(f"Ep {episode + 1}: Cost={current_cost:.2f}(Best={best_cost:.2f}) "
                      f"Reward={reward:.3f} Action={action} "
                      f"eps={self.epsilon:.3f}")

            return best_P, best_cost



class EnhancedDDQN:
    def __init__(self, m, n, optimize_strategies, get_cost_func,
                 gamma=0.8, epsilon=1.0, epsilon_min=0.05,
                 epsilon_decay=0.999, learning_rate=0.001,
                 batch_size=128, max_episodes=500, target_update_freq=200):

        # 环境参数
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.optimize_strategies = optimize_strategies  # 策略列表[策略1, 策略2, 策略3]

        # 状态空间调整（移除非策略相关的特征）
        self.base_state_size = m + n  # 代理存在性 + 任务分配状态
        self.strategy_history_size = len(optimize_strategies)
        self.state_size = self.base_state_size + self.strategy_history_size + 1  # +1 for cost

        # 动作空间仅包含策略选择
        self.action_size = len(optimize_strategies)  # 只保留策略选择动作

        # RL参数（保持相同）
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_episodes = max_episodes
        self.target_update_freq = target_update_freq

        # 环境交互函数
        self.get_cost = get_cost_func

        # 简化后的网络结构
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # 经验回放
        self.memory = deque(maxlen=10000)

        # 策略效果跟踪
        self.strategy_performance = {i: deque(maxlen=5) for i in range(len(optimize_strategies))}

    def _build_model(self):
        """简化后的网络结构"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, self.action_size)
        )

    def state_to_vector(self, P, current_cost):
        """简化后的状态表示"""
        state = np.zeros(self.base_state_size)
        agent_indices = {f'a{i}': i for i in range(self.m)}

        # 代理存在性（是否有任务）
        for agent, tasks in P.items():
            if tasks:
                state[agent_indices[agent]] = 1

        # 任务分配状态
        assigned_tasks = set()
        for tasks in P.values():
            assigned_tasks.update(tasks)
        for i in range(self.n):
            state[self.m + i] = 1 if f't{i}' in assigned_tasks else 0

        # 策略效果历史
        strategy_history = [np.mean(perf) if perf else 0.5
                            for perf in self.strategy_performance.values()]

        # 合并特征
        full_state = np.concatenate([
            state,
            strategy_history,
            [current_cost / (self.n * 100)]  # 归一化成本
        ])

        return full_state

    def execute_action(self, action, current_P):
        """仅执行策略动作"""
        strategy = self.optimize_strategies[action]
        new_P = strategy(current_P.copy())

        # 记录策略效果
        old_cost = self.get_cost(current_P)
        new_cost = self.get_cost(new_P)
        self.strategy_performance[action].append(1 if new_cost < old_cost else 0)

        return new_P

    def calculate_reward(self, old_P, new_P, action):
        """简化后的奖励函数"""
        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)

        # 基础奖励（成本改进）
        cost_improvement = (old_cost - new_cost) / (old_cost + 1e-6)

        # 策略稳定性奖励（鼓励连续成功）
        strategy_bonus = 0.3 if len(self.strategy_performance[action]) > 3 and \
                                np.mean(self.strategy_performance[action]) > 0.7 else 0

        reward = 0.8 * cost_improvement + 0.2 * strategy_bonus
        return np.clip(reward, -1, 1)

    # 保持其他方法不变（需要移除与直接分配相关的代码）
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 从内存中采样
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # 计算损失
        loss = self.criterion(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
        self.optimizer.step()
    def train(self, init_P):
        current_P = init_P
        current_cost = self.get_cost(current_P)
        best_cost = current_cost
        best_P = copy.deepcopy(current_P)

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P, current_cost)
            state = torch.FloatTensor(state)

            # eps-greedy动作选择
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

            # 执行策略动作
            new_P = self.execute_action(action, current_P)
            new_cost = self.get_cost(new_P)

            # 计算奖励
            reward = self.calculate_reward(current_P, new_P, action)

            # 构建下一个状态
            next_state = self.state_to_vector(new_P, new_cost)

            # 存储经验
            self.remember(state.numpy(), action, reward, next_state, False)

            # 更新当前状态
            if new_cost < current_cost:
                current_P = new_P
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_P = copy.deepcopy(current_P)

            # 经验回放


class EnhancedDDQN_1:
    def __init__(self, m, n, optimize_strategies, get_cost_func,
                 gamma=0.90, epsilon=1.5, epsilon_min=0.1,
                 epsilon_decay=0.998, learning_rate=0.0004,
                 batch_size=256, max_episodes=1000, target_update_freq=100):

        # 环境参数
        self.m = m  # 代理数
        self.n = n  # 任务数
        self.optimize_strategies = optimize_strategies

        # 状态空间
        self.base_state_size = m + n  # [代理存在性(m), 任务分配状态(n)]
        self.strategy_history_size = len(optimize_strategies)
        self.task_feature_size = n * 3  # 每个任务3个特征
        self.agent_feature_size = m * 2  # 每个代理2个特征
        self.state_size = (self.base_state_size + self.strategy_history_size +
                           self.task_feature_size + self.agent_feature_size + 1)  # +1 for normalized cost

        # 动作空间
        self.basic_action_size = len(optimize_strategies)
        self.direct_action_size = m * n
        self.action_size = self.basic_action_size + self.direct_action_size

        # RL参数调整
        self.gamma = gamma  # 提高折扣因子
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min  # 保持最低探索率
        self.epsilon_decay = epsilon_decay  # 调整衰减速度
        self.learning_rate = learning_rate  # 降低学习率
        self.batch_size = batch_size  # 增大批次
        self.max_episodes = max_episodes
        self.target_update_freq = target_update_freq

        # 环境交互
        self.get_cost = get_cost_func

        # 增强网络结构
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.criterion = nn.SmoothL1Loss()

        # 优先经验回放
        self.memory = deque(maxlen=20000)
        self.priorities = deque(maxlen=20000)

        # 策略跟踪
        self.strategy_performance = {i: deque(maxlen=10) for i in range(len(optimize_strategies))}
        self.last_action_type = None

    def _build_model(self):
        """增强的网络结构"""
        return nn.Sequential(
            nn.Linear(self.state_size, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, self.action_size)
        )

    def state_to_vector(self, P, current_cost, task_features=None, agent_features=None):
        state = np.zeros(self.base_state_size)
        agent_indices = {f'a{i}': i for i in range(self.m)}
        task_indices = {f't{i}': i for i in range(self.n)}

        # 代理存在性（是否有任务）
        for agent in P.keys():
            state[agent_indices[agent]] = 1

        # 任务分配状态（检查所有任务是否已分配）
        assigned_tasks = set()
        for tasks in P.values():
            assigned_tasks.update(tasks)

        for i in range(self.n):
            state[self.m + i] = 1 if f't{i}' in assigned_tasks else 0

        # 合并特征
        strategy_history = [np.mean(perf) if perf else 0.0 for perf in self.strategy_performance.values()]
        task_features = task_features.flatten() if task_features is not None else np.zeros(self.task_feature_size)
        agent_features = agent_features.flatten() if agent_features is not None else np.zeros(self.agent_feature_size)
        cost_normalized = [current_cost / (self.n * 100 + 1e-6)]

        return np.concatenate([
            state,
            strategy_history,
            task_features,
            agent_features,
            cost_normalized
        ])

    def decode_direct_action(self, action_idx):
        action_idx -= self.basic_action_size
        agent_idx = action_idx // self.n
        task_idx = action_idx % self.n
        return f"a{agent_idx}", f"t{task_idx}"

    def execute_action(self, action, current_P):
        new_P = copy.deepcopy(current_P)
        self.last_action_success = False

        if action < self.basic_action_size:
            # 策略选择
            strategy = self.optimize_strategies[action]
            new_P = strategy(new_P)
            self.last_action_type = 'strategy'

            # 记录策略效果
            old_cost = self.get_cost(current_P)
            new_cost = self.get_cost(new_P)
            self.strategy_performance[action].append(1 if new_cost < old_cost else -1)
            self.last_action_success = new_cost < old_cost
        else:
            # 直接分配
            agent, task = self.decode_direct_action(action)
            task_assigned = any(task in tasks for tasks in new_P.values())

            if not task_assigned:
                new_P[agent].append(task)
                self.last_action_type = 'direct'
                self.last_action_success = True
            else:
                new_P = copy.deepcopy(current_P)
                self.last_action_type = 'invalid_direct'
                self.last_action_success = False

        return new_P

    def calculate_reward(self, old_P, new_P, action):
        if self.last_action_type == 'invalid_direct':
            return -2.0  # 强化无效动作惩罚

        old_cost = self.get_cost(old_P)
        new_cost = self.get_cost(new_P)

        # 核心奖励机制
        cost_diff = old_cost - new_cost
        if cost_diff > 0:
            reward = 1.0 + (cost_diff / (old_cost + 1e-6))  # 成功改进奖励
        elif cost_diff < 0:
            reward = -1.0 * (abs(cost_diff) / (old_cost + 1e-6))  # 成本上升惩罚
        else:
            reward = -0.1  # 无变化惩罚

        # 探索奖励
        if self.last_action_type == 'direct' and action not in self.visited_direct_actions:
            reward += 0.5
            self.visited_direct_actions.add(action)

        # 策略效果奖励
        if self.last_action_type == 'strategy' and self.last_action_success:
            reward += 0.3 * len([v for v in self.strategy_performance[action] if v > 0])

        return np.clip(reward, -2, 2)

    def remember(self, state, action, reward, next_state, done):
        priority = abs(reward) + 1e-5  # 基于奖励的优先级
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 优先经验采样
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        minibatch = [self.memory[i] for i in indices]

        states = torch.FloatTensor([t[0] for t in minibatch])
        actions = torch.LongTensor([t[1] for t in minibatch])
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor([t[3] for t in minibatch])
        dones = torch.FloatTensor([t[4] for t in minibatch])

        # 当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 双Q学习目标
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1))
            target_q = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q

        # 计算损失
        loss = self.criterion(current_q, target_q)

        # 优化步骤
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    def train(self, init_P, task_features=None, agent_features=None):
        self.visited_direct_actions = set()
        current_P = init_P
        current_cost = self.get_cost(current_P)
        best_cost = current_cost
        best_P = copy.deepcopy(current_P)
        cost_history = []

        for episode in range(self.max_episodes):
            state = self.state_to_vector(current_P, current_cost, task_features, agent_features)
            state = torch.FloatTensor(state)

            # 动态eps调整
            if episode > self.max_episodes // 2 and self.epsilon > self.epsilon_min:
                self.epsilon *= 0.99

            # 动作选择
            if np.random.rand() <= self.epsilon:
                action = np.random.choice(self.action_size)
            else:
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

            # 执行动作
            new_P = self.execute_action(action, current_P)
            new_cost = self.get_cost(new_P)

            # 记录直接分配动作
            if action >= self.basic_action_size and self.last_action_success:
                self.visited_direct_actions.add(action)

            # 计算奖励
            reward = self.calculate_reward(current_P, new_P, action)

            # 构建下一状态
            next_state = self.state_to_vector(new_P, new_cost, task_features, agent_features)

            # 终止条件
            done = False
            if episode > 100 and abs(reward) < 0.01:
                done = True

            # 存储经验
            self.remember(state.numpy(), action, reward, next_state, done)

            # 状态更新策略
            if new_cost < current_cost or (self.epsilon > 0.2 and np.random.rand() < 0.3):
                current_P = new_P
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_P = copy.deepcopy(current_P)

            # 定期更新目标网络
            if episode % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            # 经验回放
            if len(self.memory) >= self.batch_size:
                self.replay()

            # 记录成本变化
            cost_history.append(current_cost)

            # 打印调试信息
            if (episode + 1) % 50 == 0:
                print(f"Ep {episode + 1}: Cost={current_cost:.2f} (Best={best_cost:.2f}) | "
                      f"Action: {action} | "
                      f"eps={self.epsilon:.3f} | "
                      f"Last Reward: {reward:.2f}")

        return best_P, best_cost
class GRPOAgent:
    """
    改进版 GRPOAgent（保留原始外部接口）。
    主要改进：
      - moving baseline (reward baseline)
      - advantage normalization
      - PPO-style surrogate clipping + KL regularization
      - entropy bonus
      - soft-update reference model
      - replay 多步小更新，降低方差
    """

    def __init__(
        self,
        m: int, n: int,
        optimize_funcs: List[Callable],
        reward_funcs,
        state_encoder: Optional[Callable] = None,
        num_generations: int = 6,
        beta: float = 0.02,
        gamma: float = 0.99,
        epsilon: float = 0.8,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.998,
        learning_rate: float = 1e-4,
        batch_size: int = 256,
        max_episodes: int = 1000,
        advantage_clip: float = 5.0,
        reward_scaling: float = 80.0,
        detach_ref_log_probs: bool = True,
        grouping_strategy: str = "default",
        target_update_freq: int = 100,
        dropout: float = 0.1,
        memory_size: int = 20000,
        device: Optional[str] = None,
        ppo_clip_eps: float = 0.15,
        entropy_coef: float = 1e-3,
        ref_tau: float = 0.02,
        replay_updates: int = 3,
    ):
        # device auto-detect
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # problem dims and functions
        self.m, self.n = m, n
        self.optimize_funcs = optimize_funcs or []
        self.reward_funcs = reward_funcs
        self.state_encoder = state_encoder

        # state/action dims
        self.base_state_size = m + n
        self.strategy_hist_size = len(self.optimize_funcs)
        self.task_feat_size = n * 3
        self.agent_feat_size = m * 2
        self.state_size = (
            self.base_state_size
            + self.strategy_hist_size
            + self.task_feat_size
            + self.agent_feat_size
            + 1
        )
        self.basic_act_size = len(self.optimize_funcs)
        self.direct_act_size = m * n
        self.action_size = self.basic_act_size + self.direct_act_size

        # hyperparams
        self.G = int(num_generations)
        self.beta_init = beta
        self.beta = float(beta)
        self.beta_decay = 0.95
        self.gamma = gamma
        self.lr = learning_rate
        self.batch_size = int(batch_size)
        self.max_episodes = int(max_episodes)
        self.clip_adv = advantage_clip
        self.reward_scale = reward_scaling
        self.detach_ref = detach_ref_log_probs
        self.grouping = grouping_strategy
        self.eps, self.eps_min, self.eps_decay = epsilon, epsilon_min, epsilon_decay
        self.target_freq = int(target_update_freq)

        # PPO / stability params
        self.ppo_clip_eps = ppo_clip_eps
        self.entropy_coef = entropy_coef
        self.ref_tau = ref_tau  # soft update factor for ref_model
        self.replay_updates = max(1, int(replay_updates))

        # networks
        self.model = self._build_model(dropout).to(self.device)
        self.ref_model = copy.deepcopy(self.model).to(self.device)
        # keep ref_model in eval mode
        self.ref_model.eval()

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        # replay / logs
        # We'll store: (state_vec np, action int, advantage float, next_state_vec np, done bool, old_logits numpy array)
        self.memory = deque(maxlen=int(memory_size))
        self._metrics = defaultdict(list)

        # baseline for rewards (moving average)
        self.reward_baseline = 0.0
        self.baseline_momentum = 0.9

        # caches
        self._agent_range = torch.arange(max(1, self.m), device=self.device)
        self._batch_index_cache = None

        # reproducible randomness if wanted
        # random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # ---------------- network ----------------
    def _build_model(self, dropout: float = 0.3) -> nn.Sequential:
        # same shape as original MLP
        for attr in ("state_size", "action_size"):
            val = getattr(self, attr)
            if not isinstance(val, int):
                setattr(self, attr, int(val))
        return nn.Sequential(
            nn.Linear(self.state_size, 1024),
            nn.LayerNorm(1024),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ELU(),
            nn.Linear(512, self.action_size),
        )

    # ---------------- state encoding (keeps interface) ----------------
    def state_to_vector(self, P, current_cost, task_features=None, agent_features=None):
        assigned = {t for tasks in P.values() for t in tasks}
        task_bits = np.zeros(self.n, dtype=np.float32)
        for t in assigned:
            try:
                idx = int(t[1:])
            except Exception:
                continue
            if 0 <= idx < self.n:
                task_bits[idx] = 1.0

        agent_bits = np.zeros(self.m, dtype=np.float32)
        for a in P.keys():
            try:
                ai = int(a[1:])
            except Exception:
                continue
            if 0 <= ai < self.m:
                agent_bits[ai] = 1.0

        strat_hist = np.zeros(self.strategy_hist_size, dtype=np.float32)
        task_feat = task_features.flatten() if task_features is not None else np.zeros(self.task_feat_size, dtype=np.float32)
        agent_feat = agent_features.flatten() if agent_features is not None else np.zeros(self.agent_feat_size, dtype=np.float32)
        cost_norm = np.array([current_cost / (self.n * 100 + 1e-6)], dtype=np.float32)

        vec = np.concatenate([
            agent_bits,
            task_bits,
            strat_hist,
            task_feat,
            agent_feat,
            cost_norm,
        ])
        return vec.astype(np.float32)

    # ---------------- action selection (single) ----------------
    def select_action(self, state: np.ndarray, eps: float = None) -> int:
        eps = eps if eps is not None else self.eps
        if np.random.rand() < eps:
            return int(np.random.randint(self.action_size))
        with torch.inference_mode():
            s = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
            logits = self.model(s)
            return int(logits.argmax(dim=1).item())

    # ---------------- batched selection used in train ----------------
    def select_action_batch(self, state_vecs: torch.Tensor, eps: float) -> torch.Tensor:
        B = state_vecs.shape[0]
        rand = torch.rand(B, device=self.device)
        explore_mask = rand < eps
        actions = torch.empty(B, dtype=torch.long, device=self.device)
        if explore_mask.any():
            actions[explore_mask] = torch.randint(0, self.action_size, (int(explore_mask.sum().item()),), device=self.device)
        if (~explore_mask).any():
            with torch.inference_mode():
                qvals = self.model(state_vecs[~explore_mask])
                greedy = qvals.argmax(dim=1)
            actions[~explore_mask] = greedy
        return actions

    # ---------------- store transition (extended internal content) ----------------
    def store_transition(self, state, action, reward, next_state, done, old_logits=None):
        """
        old_logits: numpy array of model logits for that state when sampled (optional).
        We store it so replay can compute PPO-like ratio = exp(new_logp - old_logp).
        """
        # keep original positional signature by accepting old_logits optional
        self.memory.append((state, action, reward, next_state, done, None if old_logits is None else np.array(old_logits, dtype=np.float32)))

    # ---------------- grouped advantage ----------------
    def _compute_grouped_advantage(self, rewards: np.ndarray) -> np.ndarray:
        if rewards.size == 0:
            return rewards
        rewards = rewards.reshape(-1, self.G)
        # group baseline: use mean (or max depending on self.grouping)
        if self.grouping == "mean":
            mean = rewards.mean(axis=1, keepdims=True)
            std = rewards.std(axis=1, keepdims=True) + 1e-4
        else:
            mean = rewards.max(axis=1, keepdims=True)
            std = rewards.std(axis=1, keepdims=True) + 1e-4
        adv = (rewards - mean) / std
        return adv.ravel()

    # ---------------- cost / reward (keep) ----------------
    def get_cost(self, state):
        if isinstance(state, (float, int, np.number)):
            return -float(state)
        if isinstance(state, np.ndarray) and state.size == 1:
            return -float(state.item())

        reward = None
        if callable(self.reward_funcs):
            try:
                reward = self.reward_funcs(state)
            except Exception:
                return float("inf")
        elif isinstance(self.reward_funcs, list) and self.reward_funcs:
            try:
                reward = self.reward_funcs[0](state)
            except Exception:
                return float("inf")
        elif isinstance(self.reward_funcs, dict) and self.reward_funcs:
            try:
                reward = self.reward_funcs[list(self.reward_funcs.keys())[0]](state)
            except Exception:
                return float("inf")
        else:
            return float("inf")

        if isinstance(reward, (int, float, np.number)):
            return -float(reward)
        if isinstance(reward, dict):
            for v in reward.values():
                if isinstance(v, (list, tuple)) and len(v):
                    v = v[0]
                try:
                    return -float(v)
                except (ValueError, TypeError):
                    continue
            return float("inf")
        if isinstance(reward, (list, tuple)) and len(reward):
            try:
                return -float(reward[0])
            except (ValueError, TypeError):
                return float("inf")
        try:
            return -float(reward)
        except (ValueError, TypeError):
            return float("inf")

    # ---------------- improved batch reward (moving baseline + normalize) ----------------
    def calculate_reward_batch(self, old_P, new_P_list):
        old_cost = self.get_cost(old_P)
        rewards = []
        for new_P in new_P_list:
            new_cost = self.get_cost(new_P)
            if not np.isfinite(new_cost):
                rewards.append(-1.0)
                continue
            delta = (old_cost - new_cost) / (abs(old_cost) + 1e-6)
            # scaled + squashing
            scaled = np.tanh(3.0 * delta)
            rewards.append(float(np.clip(scaled, -1, 1)))
        # moving baseline update (on rewards mean)
        r_mean = float(np.mean(rewards))
        #self.reward_baseline = self.baseline_momentum * self.reward_baseline + (1 - self.baseline_momentum) * r_mean
        # baseline-subtracted and normalized rewards
        self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * r_mean
        rewards = np.array(rewards, dtype=np.float32) - float(self.reward_baseline)
        if rewards.std() > 1e-6:
            rewards = rewards / (rewards.std() + 1e-6)
        return rewards

    # ---------------- execute_action (original) ----------------
    def execute_action(self, action, current_P):
        new_P = copy.deepcopy(current_P)
        changed = False

        if action < len(self.optimize_funcs):
            try:
                candidate = self.optimize_funcs[action](copy.deepcopy(new_P))
                if candidate != new_P:
                    new_P = candidate
                    changed = True
            except Exception:
                pass

        if not changed:
            agent_idx = (action - len(self.optimize_funcs)) // self.n
            task_idx = (action - len(self.optimize_funcs)) % self.n
            agent_idx = max(0, min(self.m - 1, agent_idx))
            task_idx = max(0, min(self.n - 1, task_idx))
            agent, task = f"a{agent_idx}", f"t{task_idx}"
            assigned = {t for tasks in new_P.values() for t in tasks}

            if task in assigned:
                other_tasks = [f"t{i}" for i in range(self.n) if f"t{i}" not in assigned]
                if other_tasks:
                    task = random.choice(other_tasks)
            if task not in new_P[agent]:
                new_P[agent].append(task)
            changed = True

        return new_P

    # ---------------- batch execute (vectorized where possible) ----------------
    def execute_action_batch(self, actions: torch.Tensor, current_P: dict):
        """
        Keep behavior: returns list of dicts length B.
        Vectorized for direct actions as before; optimize_funcs handled once per unique index.
        """
        device = self.device
        B = int(actions.shape[0])
        P_tensor = torch.zeros((B, self.m, self.n), dtype=torch.bool, device=device)

        # pack current_P
        for a_key, tasks in current_P.items():
            try:
                ai = int(a_key[1:])
            except Exception:
                continue
            if ai < 0 or ai >= self.m:
                continue
            t_idxs = []
            for t in tasks:
                try:
                    ti = int(t[1:])
                except Exception:
                    continue
                if 0 <= ti < self.n:
                    t_idxs.append(ti)
            if t_idxs:
                P_tensor[:, ai, t_idxs] = True

        opt_size = self.basic_act_size
        actions_cpu = actions.cpu().numpy()
        new_P_from_opt = {}
        if opt_size > 0:
            unique_opt_indices = np.unique(actions_cpu[actions_cpu < opt_size]) if (actions_cpu < opt_size).any() else np.array([], dtype=int)
            for opt_idx in unique_opt_indices:
                try:
                    candidate = self.optimize_funcs[int(opt_idx)](copy.deepcopy(current_P))
                    if candidate is None:
                        candidate = copy.deepcopy(current_P)
                except Exception:
                    candidate = copy.deepcopy(current_P)
                new_P_from_opt[int(opt_idx)] = candidate

        direct_mask = actions >= opt_size
        if direct_mask.any().item():
            direct_idx = actions[direct_mask] - opt_size
            agent_idx = torch.clamp(direct_idx // self.n, 0, self.m - 1)
            task_idx = torch.clamp(direct_idx % self.n, 0, self.n - 1)
            if self._batch_index_cache is None or self._batch_index_cache.shape[0] < B:
                self._batch_index_cache = torch.arange(0, max(1, B), device=device)
            batch_idx = self._batch_index_cache[:B][direct_mask]
            P_tensor[batch_idx, agent_idx, task_idx] = True

        # conflict resolution: choose minimal agent index for each (b,t)
        assigned_any = P_tensor.any(dim=1)
        if assigned_any.any().item():
            large = torch.tensor(self.m + 5, device=device)
            agent_priority = self._agent_range.view(1, self.m, 1)
            priority_tensor = torch.where(P_tensor, agent_priority, large)
            sel_agent = priority_tensor.argmin(dim=1)
            new_P_tensor = torch.zeros_like(P_tensor)
            b_idx, t_idx = torch.nonzero(assigned_any, as_tuple=True)
            if b_idx.numel() > 0:
                sel_agents = sel_agent[b_idx, t_idx]
                new_P_tensor[b_idx, sel_agents, t_idx] = True
            P_tensor = new_P_tensor

        # convert to python dicts
        P_cpu = P_tensor.cpu()
        new_P_list = []
        for b in range(B):
            d = {}
            for a in range(self.m):
                t_mask = P_cpu[b, a]
                if t_mask.any().item():
                    t_indices = torch.nonzero(t_mask, as_tuple=False).squeeze(1).tolist()
                    if isinstance(t_indices, int):
                        t_indices = [t_indices]
                    d[f"a{a}"] = [f"t{ti}" for ti in t_indices]
                else:
                    d[f"a{a}"] = []
            act_b = int(actions[b].item())
            if act_b < opt_size:
                cand = new_P_from_opt.get(act_b, None)
                if cand is not None:
                    d = cand if isinstance(cand, dict) else copy.deepcopy(cand)
            new_P_list.append(d)

        return new_P_list

    # ---------------- replay (PPO-style clipping + KL + entropy; multiple small updates) ----------------
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # sample a batch
        batch = random.sample(self.memory, self.batch_size)
        # states: stack numpy arrays
        states = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(self.device)
        actions = torch.LongTensor([int(b[1]) for b in batch]).to(self.device)
        advantages = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(self.device)
        dones = torch.BoolTensor([bool(b[4]) for b in batch]).to(self.device)
        # old_logits maybe None if not stored; fallback to using ref_model logits as old
        old_logits_list = []
        for b in batch:
            old = b[5]
            if old is None:
                # compute ref logits (detached)
                with torch.inference_mode():
                    old_logits_list.append(self.ref_model(torch.from_numpy(b[0]).unsqueeze(0).float().to(self.device)).cpu().numpy().squeeze(0))
            else:
                old_logits_list.append(np.array(old, dtype=np.float32))
        old_logits = torch.from_numpy(np.stack(old_logits_list)).float().to(self.device)  # [B, action_size]

        # multiple small gradient steps to stabilize updates
        for _ in range(self.replay_updates):
            # current logits
            logits = self.model(states)  # [B, A]
            logp_all = torch.log_softmax(logits, dim=1)
            old_logp_all = torch.log_softmax(old_logits.to(self.device), dim=1).detach()

            # gather log-probs of taken actions
            logp_taken = logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]
            old_logp_taken = old_logp_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            # ratio
            ratio = torch.exp(logp_taken - old_logp_taken)
            # surrogate
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_eps, 1.0 + self.ppo_clip_eps) * advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            # KL penalty (approx per-token with logits)
            with torch.inference_mode():
                ref_logits = self.ref_model(states)
            kl = (torch.exp(ref_logits - logits) - (ref_logits - logits) - 1.0)
            if self.detach_ref:
                kl = kl.detach()
            # average kl across tokens (actions as tokens)
            kl_avg = kl.mean()

            # entropy bonus
            entropy = -(logp_all * torch.exp(logp_all)).sum(dim=1).mean()

            loss = policy_loss + self.beta * kl_avg - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

        # soft update ref_model toward current model (stabilize KL target)
        self.soft_update_ref(self.ref_tau)

    def soft_update_ref(self, tau: float):
        """theta_ref = (1-tau)*theta_ref + tau*theta"""
        for p_ref, p in zip(self.ref_model.parameters(), self.model.parameters()):
            p_ref.data.mul_(1.0 - tau)
            p_ref.data.add_(p.data * tau)

    # ---------------- main train (keeps interface) ----------------
    def train(self, init_P, task_features=None, agent_features=None):
        current_P = init_P
        current_cost = self.get_cost(current_P)
        best_cost = current_cost
        best_P = copy.deepcopy(current_P)

        for episode in range(1, self.max_episodes + 1):
            state_vec = self.state_to_vector(current_P, current_cost, task_features, agent_features)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            state_batch = state_tensor.repeat(self.G, 1)

            # epsilon decay per episode
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            # select actions on device
            actions = self.select_action_batch(state_batch, self.eps)  # [G]

            # For PPO we capture old logits for each sampled state
            with torch.inference_mode():
                old_logits_batch = self.model(state_batch).cpu().numpy()  # [G, A]

            # execute batch actions
            new_P_list = self.execute_action_batch(actions, current_P)

            # compute batch rewards (baseline adjusted & normalized inside)
            rewards = self.calculate_reward_batch(current_P, new_P_list)  # numpy [G]

            # grouped advantages
            advantages = self._compute_grouped_advantage(rewards)  # numpy [G]

            # store transitions (include old_logits row so replay can compute ratio)
            for i in range(self.G):
                next_vec = self.state_to_vector(new_P_list[i], self.get_cost(new_P_list[i]), task_features, agent_features)
                # store with old_logits for this sample
                self.store_transition(state_vec, int(actions[i].item()), float(advantages[i]), next_vec, False, old_logits=old_logits_batch[i])

            # choose best candidate (by raw reward)
            best_idx = int(np.argmax(rewards))
            best_candidate_cost = self.get_cost(new_P_list[best_idx])

            # best_candidate_cost=0.5*best_candidate_cost
            # best_cost=0.5*best_cost
            if best_candidate_cost < best_cost:
                best_cost = best_candidate_cost
                best_P = copy.deepcopy(new_P_list[best_idx])

            # advance to chosen next
            current_P = new_P_list[best_idx]
            current_cost =best_candidate_cost

            
            # replay (PPO clipped updates etc.)
            self.replay()

            # beta decay occasionally
            if episode % 50 == 0:
                self.beta *= self.beta_decay
                print(f"[GRPO] Ep {episode}: cost={current_cost:.2f} best={best_cost:.2f} eps={self.eps:.3f} β={self.beta:.4f}")

        # cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        return best_P, best_cost


def _ema(old, new, momentum):
    return momentum * old + (1.0 - momentum) * new


def _bias_correction(momentum: float, step: int) -> float:
    return max(1e-6, 1.0 - (momentum ** max(1, step)))


def _build_depth_weights(depth: int, decay: float, device: torch.device) -> torch.Tensor:
    if depth <= 0:
        return torch.zeros(0, dtype=torch.float32, device=device)
    decay = max(1e-4, min(0.9999, decay))
    powers = torch.arange(depth, dtype=torch.float32, device=device)
    base = torch.full((depth,), decay, dtype=torch.float32, device=device)
    weights = torch.pow(base, powers)
    denom = weights.sum().clamp_min(1e-6)
    return weights / denom


class GVPOAgent(GRPOAgent, nn.Module):
    """
    Graph Value Propagation Optimizer rebuild based on the GVPO reference loss.
    Value propagation operates as a deterministic message passing layer on the agent-task grid,
    while the learning signal follows the group variance objective from GVPO/compute_policy_loss_gvpo.
    """

    def __init__(
        self,
        *args,
        propagation_depth: int = 2,
        propagation_decay: float = 0.65,
        max_width_ratio: float = 0.35,
        width_smoothing: float = 0.9,
        feature_norm: bool = True,
        advantage_norm: bool = True,
        value_clip: float = 10.0,
        gvpo_beta: float = 0.1,
        **kwargs,
    ):
        nn.Module.__init__(self)
        tuned_kwargs = dict(kwargs)
        tuned_kwargs.setdefault("num_generations", max(8, kwargs.get("num_generations", 6)))
        tuned_kwargs.setdefault("replay_updates", 1)
        tuned_kwargs.setdefault("entropy_coef", 5e-4)
        tuned_kwargs.setdefault("learning_rate", 1.7e-4)
        super().__init__(*args, **tuned_kwargs)

        self.gvpo_beta = float(gvpo_beta)
        self.feature_norm = feature_norm
        self.advantage_norm = advantage_norm
        self.value_clip = float(value_clip)
        self.propagation_depth = max(0, int(propagation_depth))
        self.propagation_decay = float(propagation_decay)
        self.width_smoothing = float(width_smoothing)

        ratio = max(1.0 / max(1, self.n), min(1.0, float(max_width_ratio)))
        self.max_width = max(1, int(math.ceil(self.n * ratio)))
        self.current_width = self.max_width
        self._width_track = float(self.current_width)
        # let GVPO spend modestly more episodes without exceeding runtime budget
        self.gvpo_episode_ratio = 1.35
        self._base_max_episodes = self.max_episodes
        self.max_episodes = int(max(1, math.ceil(self._base_max_episodes * self.gvpo_episode_ratio)))
        self.width_plateau_patience = max(15, int(0.25 * self.max_episodes))
        self._no_improve_steps = 0
        # keep a higher exploration floor on large maps
        self.gvpo_min_explore = max(self.eps_min, 0.08)
        self.eps_min = self.gvpo_min_explore
        self.eps_decay = 1.0 - 0.5 * (1.0 - self.eps_decay)

        self.state_norm = nn.LayerNorm(self.state_size, elementwise_affine=False).to(self.device) if self.feature_norm else None

        depth_weights = _build_depth_weights(self.propagation_depth, self.propagation_decay, torch.device(self.device))
        self.register_buffer("depth_weights", depth_weights, persistent=False)

        if self.direct_act_size > 0:
            idx = torch.arange(self.direct_act_size, dtype=torch.long, device=self.device)
            agent_idx = torch.div(idx, self.n, rounding_mode="trunc")
            task_idx = torch.remainder(idx, self.n)
        else:
            agent_idx = torch.zeros(0, dtype=torch.long, device=self.device)
            task_idx = torch.zeros(0, dtype=torch.long, device=self.device)
        self.register_buffer("action_agent_index", agent_idx, persistent=False)
        self.register_buffer("action_task_index", task_idx, persistent=False)

    def _policy_logits(self, state_batch: torch.Tensor, model: nn.Module | None = None) -> torch.Tensor:
        net = self.model if model is None else model
        logits = torch.clamp(net(state_batch), -self.value_clip, self.value_clip)
        if self.propagation_depth == 0 or self.direct_act_size == 0:
            return logits
        prefix = logits[:, : self.basic_act_size] if self.basic_act_size else None
        tail = logits[:, self.basic_act_size :]
        tail = self._propagate_direct_logits(tail)
        if prefix is None:
            return tail
        return torch.cat([prefix, tail], dim=1)

    def _propagate_direct_logits(self, tail: torch.Tensor) -> torch.Tensor:
        if self.propagation_depth == 0 or self.direct_act_size == 0:
            return tail
        B = tail.shape[0]
        grid = tail.reshape(B, self.m, self.n)
        width_agent = max(1, min(self.current_width, self.n))
        width_task = max(1, min(self.current_width, self.m))
        for depth_idx in range(self.depth_weights.shape[0]):
            if width_agent < self.n:
                agent_msgs = torch.topk(grid, width_agent, dim=2).values.mean(dim=2, keepdim=True)
            else:
                agent_msgs = grid.mean(dim=2, keepdim=True)
            if width_task < self.m:
                task_msgs = torch.topk(grid.transpose(1, 2), width_task, dim=2).values.mean(dim=2, keepdim=True).transpose(1, 2)
            else:
                task_msgs = grid.transpose(1, 2).mean(dim=2, keepdim=True).transpose(1, 2)
            combined = 0.5 * (agent_msgs.expand_as(grid) + task_msgs.expand_as(grid))
            grid = grid + self.depth_weights[depth_idx] * combined
            grid = grid.clamp(-self.value_clip, self.value_clip)
        return grid.reshape(B, self.direct_act_size)

    def _sample_actions(self, logits: torch.Tensor, eps: float) -> torch.Tensor:
        batch = logits.shape[0]
        explore = torch.rand(batch, device=logits.device) < eps
        actions = torch.empty(batch, dtype=torch.long, device=logits.device)
        if explore.any():
            actions[explore] = torch.randint(0, self.action_size, (int(explore.sum()),), device=logits.device)
        if (~explore).any():
            greedy = torch.argmax(logits[~explore], dim=-1)
            actions[~explore] = greedy
        return actions

    def _evaluate_candidates(self, current_cost: float, new_plans: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.zeros(len(new_plans), dtype=np.float32)
        costs = np.zeros(len(new_plans), dtype=np.float32)
        scale = max(abs(current_cost), 1.0)
        for idx, plan in enumerate(new_plans):
            cand_cost = self.get_cost(plan)
            costs[idx] = cand_cost
            if np.isfinite(cand_cost):
                rewards[idx] = (current_cost - cand_cost) / scale
            else:
                rewards[idx] = -1.0
        rewards = np.clip(rewards, -2.0, 2.0)
        return rewards, costs

    def _update_width(self, rewards: np.ndarray, episode: int, improved: bool) -> None:
        if rewards.size == 0 or self.max_width <= 1:
            return
        centered = rewards - rewards.mean()
        variance = float(np.mean(centered * centered))
        if not np.isfinite(variance):
            variance = 0.0
        phase = episode / max(1, self.max_episodes)
        plateau = self._no_improve_steps >= self.width_plateau_patience
        early_floor = max(4, min(self.max_width, 4))
        mid_floor = max(3, min(self.max_width, 3))
        late_floor = 2 if plateau else 3
        if phase < 0.5:
            min_width = early_floor
        elif phase < 0.8:
            min_width = mid_floor
        else:
            min_width = late_floor
        if plateau and phase >= 0.8 and self._no_improve_steps >= self.width_plateau_patience * 1.5:
            min_width = 1
        if self._no_improve_steps >= self.width_plateau_patience // 2 and not improved:
            min_width = min(self.max_width, min_width + 1)
        target = 1.0 + (variance / (variance + 0.25)) * (self.max_width - 1)
        target = max(min_width, min(self.max_width, target))
        self._width_track = self.width_smoothing * self._width_track + (1.0 - self.width_smoothing) * target
        self.current_width = max(min_width, min(self.max_width, int(round(self._width_track))))

    def _prepare_advantages(self, costs: np.ndarray, rewards: np.ndarray) -> torch.Tensor:
        if costs.size == 0:
            return torch.zeros(0, device=self.device)
        finite = np.isfinite(costs)
        if finite.any():
            baseline = float(np.mean(costs[finite]))
            adv = baseline - costs
        else:
            adv = -rewards
        adv = np.clip(adv, -self.clip_adv, self.clip_adv).astype(np.float32)
        adv_tensor = torch.from_numpy(adv).to(self.device)
        if adv_tensor.numel() == 0:
            return adv_tensor
        adv_tensor = adv_tensor - adv_tensor.mean()
        std = adv_tensor.std(unbiased=False).clamp_min(1e-6)
        adv_tensor = adv_tensor / std
        adv_tensor = adv_tensor.clamp(-3.0, 3.0)
        return adv_tensor

    def _compute_gvpo_loss(
        self,
        logp_new: torch.Tensor,
        logp_old: torch.Tensor,
        advantages: torch.Tensor,
        entropy: torch.Tensor,
        kl: torch.Tensor,
    ) -> torch.Tensor:
        centered_ratio = self.gvpo_beta * (logp_new - logp_old)
        centered_ratio = centered_ratio - centered_ratio.mean()
        adv_zero_mean = advantages - advantages.mean()
        if advantages.numel() > 1:
            adv_zero_mean = adv_zero_mean / adv_zero_mean.std(unbiased=False).clamp_min(1e-6)
        residual = centered_ratio - adv_zero_mean
        denom = max(1, advantages.numel() - 1)
        loss = 0.5 * (residual.pow(2).sum() / denom)
        loss = loss + self.beta * kl - self.entropy_coef * entropy
        return loss

    def train(self, init_P, task_features=None, agent_features=None):
        current_P = init_P
        current_cost = self.get_cost(current_P)
        best_cost = current_cost
        best_plan = copy.deepcopy(current_P)

        for episode in range(1, self.max_episodes + 1):
            state_vec = self.state_to_vector(current_P, current_cost, task_features, agent_features)
            state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float().to(self.device)
            if self.state_norm is not None:
                state_tensor = self.state_norm(state_tensor)
            state_batch = state_tensor.expand(self.G, -1)

            # slow exploration decay on big benchmarks
            self.eps = max(self.gvpo_min_explore, self.eps * self.eps_decay)
            logits = self._policy_logits(state_batch)
            actions = self._sample_actions(logits, self.eps)

            logp_new_all = torch.log_softmax(logits, dim=-1)
            logp_new = logp_new_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.inference_mode():
                ref_logits = self._policy_logits(state_batch, model=self.ref_model)
                logp_ref_all = torch.log_softmax(ref_logits, dim=-1)
            logp_old = logp_ref_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            new_plans = self.execute_action_batch(actions, current_P)
            rewards, candidate_costs = self._evaluate_candidates(current_cost, new_plans)
            advantages = self._prepare_advantages(candidate_costs, rewards)

            kl = torch.sum(logp_new_all.exp() * (logp_new_all - logp_ref_all), dim=1).mean()
            entropy = -(logp_new_all * logp_new_all.exp()).sum(dim=1).mean()
            loss = self._compute_gvpo_loss(logp_new, logp_old, advantages, entropy, kl)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.4)
            self.optimizer.step()
            self.soft_update_ref(self.ref_tau)

            best_idx = int(np.nanargmin(candidate_costs)) if candidate_costs.size else 0
            best_candidate_cost = candidate_costs[best_idx]
            improved = np.isfinite(best_candidate_cost) and best_candidate_cost < best_cost
            if improved:
                best_cost = best_candidate_cost
                best_plan = copy.deepcopy(new_plans[best_idx])
                self._no_improve_steps = 0
            else:
                self._no_improve_steps += 1

            current_P = new_plans[best_idx]
            current_cost = best_candidate_cost if np.isfinite(best_candidate_cost) else current_cost
            self._update_width(rewards, episode, improved)

            if episode % 20 == 0:
                logger.info(
                    f"[GVPO] ep={episode} cost={current_cost:.2f} best={best_cost:.2f} "
                    f"width={self.current_width} eps={self.eps:.3f}"
                )
                self.beta = max(1e-5, self.beta * self.beta_decay)

        logger.info(
            f"[GVPO] final cost={current_cost:.2f} best={best_cost:.2f} "
            f"width={self.current_width} eps={self.eps:.3f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()
        return best_plan, best_cost


# ---------------------------------------------------------------------------
# Developer testing utilities (do not impact normal RLAllocator usage)
# ---------------------------------------------------------------------------

def _compute_agent_path_cost(agent_idx, plan, agent_positions, task_pairs):
    """Returns cumulative Manhattan distance for one agent's task list."""
    pos = agent_positions[agent_idx]
    total = 0.0
    for task_name in plan:
        t_idx = int(task_name[1:])
        pickup, dropoff = task_pairs[t_idx][:2]
        total += abs(pos[0] - pickup[0]) + abs(pos[1] - pickup[1])
        total += abs(pickup[0] - dropoff[0]) + abs(pickup[1] - dropoff[1])
        pos = dropoff
    return total


def _compute_plan_metrics(plan, agent_positions, task_pairs):
    """Aggregates success rate and per-agent makespan-style costs."""
    if not plan:
        return 0.0, 0.0, []
    per_agent = []
    covered = set()
    for agent_name, tasks in plan.items():
        idx = int(agent_name[1:])
        per_agent.append(_compute_agent_path_cost(idx, tasks, agent_positions, task_pairs))
        covered.update(tasks)
    success_rate = len(covered) / max(1, len(task_pairs))
    avg_makespan = float(np.mean(per_agent)) if per_agent else 0.0
    return success_rate, avg_makespan, per_agent


def _quick_gvpo_regression_test(episodes: int = 40, seed: int = 0):
    """
    Tiny synthetic sanity check; run manually from a REPL.
    Ensures GVPO isn't egregiously slower or worse than GRPO after changes.
    """
    import random
    import time
    import copy as _copy

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    m, n = 24, 24
    base_plan = {f"a{i}": [f"t{(i + j) % n}" for j in range(2)] for i in range(m)}

    def _synthetic_cost(plan):
        total = 0
        for agent, tasks in plan.items():
            total += len(tasks) * (int(agent[1:]) + 1)
            for t in tasks:
                total += (int(t[1:]) % 5) + 1
        return total

    reward_fn = lambda plan: -_synthetic_cost(plan)
    optimize_funcs = []

    def _run(agent_cls):
        agent = agent_cls(m, n, optimize_funcs, reward_fn, max_episodes=episodes, batch_size=64)
        start = time.time()
        plan, cost = agent.train(_copy.deepcopy(base_plan))
        runtime = time.time() - start
        return runtime, cost, plan

    print("=== GVPO quick regression test ===")
    grpo_t, grpo_cost, _ = _run(GRPOAgent)
    gvpo_t, gvpo_cost, _ = _run(GVPOAgent)

    print(f"GRPO quick result  : time={grpo_t:.3f}s cost={grpo_cost:.2f}")
    print(f"GVPO quick result  : time={gvpo_t:.3f}s cost={gvpo_cost:.2f}")
    print(f"Diff (GVPO-GRPO)   : time={gvpo_t - grpo_t:.3f}s cost={gvpo_cost - grpo_cost:.2f}")

    if gvpo_t > 1.5 * grpo_t:
        print("[WARN] GVPO quick test is more than 1.5x slower than GRPO.")
    if gvpo_cost > grpo_cost:
        print("[WARN] GVPO quick test produced a worse best_cost than GRPO.")


def _full_gvpo_test_pipeline(
    map_path=None,
    scen_path=None,
    agents: int = 200,
    tasks: int = 200,
    episodes: int = 80,
):
    """
    Developer pipeline:
      1) quick synthetic sanity run,
      2) larger benchmark on a real MAPF map/scenario.
    Invoke manually; does not affect RLAllocator or app_test_fast_gap.
    """
    import copy as _copy
    import time
    from pathlib import Path
    from pycam import get_grid, get_scenario

    _quick_gvpo_regression_test()

    root = Path(__file__).resolve().parent.parent
    map_path = Path(map_path) if map_path else root / "assets" / "room-64-64-16.map"
    scen_path = Path(scen_path) if scen_path else root / "assets" / "room-64-64-16-random-1.scen"

    get_grid(map_path)  # ensure map loads
    _, _, _, agent_dict, task_dict = get_scenario(scen_path, agents, tasks)

    num_agents = min(agents, len(agent_dict))
    num_tasks = min(tasks, len(task_dict))

    agent_keys = sorted(agent_dict.keys())[:num_agents]
    agent_positions = {idx: agent_dict[k] for idx, k in enumerate(agent_keys)}

    task_keys = sorted(task_dict.keys())[:num_tasks]
    task_pairs = [task_dict[k][:2] for k in task_keys]

    base_plan = {f"a{i}": [] for i in range(num_agents)}
    for t_idx in range(num_tasks):
        agent_name = f"a{t_idx % num_agents}"
        base_plan[agent_name].append(f"t{t_idx}")

    def _plan_cost(plan):
        total = 0.0
        for agent_name, tasks_list in plan.items():
            idx = int(agent_name[1:])
            total += _compute_agent_path_cost(idx, tasks_list, agent_positions, task_pairs)
        return total

    reward_fn = lambda plan: -_plan_cost(plan)
    optimize_funcs = []

    def _run(agent_cls):
        agent = agent_cls(num_agents, num_tasks, optimize_funcs, reward_fn, max_episodes=episodes, batch_size=128)
        start = time.time()
        plan, cost = agent.train(_copy.deepcopy(base_plan))
        runtime = time.time() - start
        sr, avg_mk, _ = _compute_plan_metrics(plan, agent_positions, task_pairs)
        return runtime, cost, sr, avg_mk

    print("\n=== GVPO full pipeline benchmark ===")
    grpo_t, grpo_cost, grpo_sr, grpo_mk = _run(GRPOAgent)
    gvpo_t, gvpo_cost, gvpo_sr, gvpo_mk = _run(GVPOAgent)

    print(f"GRPO: time={grpo_t:.2f}s best_cost={grpo_cost:.2f} success={grpo_sr:.3f} avg_makespan={grpo_mk:.2f}")
    print(f"GVPO: time={gvpo_t:.2f}s best_cost={gvpo_cost:.2f} success={gvpo_sr:.3f} avg_makespan={gvpo_mk:.2f}")
    print(
        f"Diff : time={gvpo_t - grpo_t:.2f}s cost={gvpo_cost - grpo_cost:.2f} "
        f"success={gvpo_sr - grpo_sr:.3f} makespan={gvpo_mk - grpo_mk:.2f}"
    )

    if gvpo_t > 1.5 * grpo_t:
        print("[WARN] GVPO benchmark runtime exceeds GRPO by >1.5x.")
    if gvpo_cost > grpo_cost:
        print("[WARN] GVPO benchmark best_cost is worse than GRPO.")


class RLAllocator:
    """
    Thin wrapper that exposes a unified interface for GRPO and GVPO policies.
    """

    def __init__(
        self,
        *policy_args,
        algo_type: str = "gvpo",
        planner_type: str = "lacam",
        planner_kwargs: Optional[Dict[str, Any]] = None,
        **policy_kwargs,
    ):
        algo = algo_type.lower()
        if algo not in {"grpo", "gvpo"}:
            raise ValueError(f"Unsupported algo_type '{algo_type}'. Use 'grpo' or 'gvpo'.")
        impl_cls = GVPOAgent if algo == "gvpo" else GRPOAgent
        self.algo_type = algo
        self.policy = impl_cls(*policy_args, **policy_kwargs)

        planner_key = planner_type.lower()
        self.planner_type = planner_key
        self.planner = PlannerFactory.create(planner_key, **(planner_kwargs or {}))
        logger.info(
            "RL Agent initialized using algorithm=%s planner=%s",
            self.algo_type,
            self.planner_type,
        )

    def train(self, *args, **kwargs):
        return self.policy.train(*args, **kwargs)

    def select_action(self, *args, **kwargs):
        return self.policy.select_action(*args, **kwargs)

    def save_model(self, ckpt_path):
        state = {
            "algo_type": self.algo_type,
            "model": self.policy.model.state_dict(),
        }
        if hasattr(self.policy, "ref_model"):
            state["ref_model"] = self.policy.ref_model.state_dict()
        torch.save(state, ckpt_path)

    def load_model(self, ckpt_path, strict: bool = True):
        data = torch.load(ckpt_path, map_location=self.policy.device)
        if data.get("algo_type") and data["algo_type"] != self.algo_type:
            logger.warning(
                "Loading checkpoint trained with '%s' into '%s' agent",
                data["algo_type"],
                self.algo_type,
            )
        self.policy.model.load_state_dict(data["model"], strict=strict)
        if "ref_model" in data and hasattr(self.policy, "ref_model"):
            self.policy.ref_model.load_state_dict(data["ref_model"], strict=strict)

    def solve_mapf(self, request: PlannerRequest) -> PlannerResult:
        if self.planner is None:
            raise RuntimeError("Planner backend is not configured.")
        return self.planner.solve(request)
