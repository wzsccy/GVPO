import numpy as np
import random
import copy


class DQN_Optimizer:
    def __init__(self, P, P_cost, TA_4, grid, dimension, obstacles):
        self.P = P
        self.P_cost = P_cost
        self.TA_4 = TA_4
        self.grid = grid
        self.dimension = dimension
        self.obstacles = obstacles
        self.num_iterations = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100

        # 初始化Q表和目标Q表
        self.Q = {}
        self.target_Q = {}
        for agent in self.P.keys():
            self.Q[agent] = {}
            self.target_Q[agent] = {}

        # 经验回放池
        self.experience_replay = []

        self.best_P = copy.deepcopy(self.P)
        self.best_cost = self.P_cost
        self.cost_history = []

    # 计算路径成本
    def calculate_cost(self, P):
        cost = 0
        for agent, path in P.items():
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return cost

    # 检查路径冲突
    def check_collision(self, P):
        positions = set()
        for agent, path in P.items():
            for step in path:
                if step in positions:
                    return True
                positions.add(step)
        return False

    # 检查是否经过特定点
    def check_specific_points(self, P):
        for agent, path in P.items():
            specific_points = self.TA_4[agent]
            for point in specific_points:
                if point not in path:
                    return False
        return True

    # 八方向搜索
    def eight_direction_search(self, current):
        x, y = current
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] and (
                    new_x, new_y) not in self.obstacles:
                valid_neighbors.append((new_x, new_y))
        return valid_neighbors

    # 状态表示函数
    def get_state(self, agent, path, current_index):
        current = path[current_index]
        target = self.TA_4[agent][-1]
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)
        path_length = len(path) - current_index - 1
        return (current, distance_to_target, path_length)

    def optimize(self):
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (
                            0.2 - self.min_exploration_rate) * np.exp(-self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 更细致的奖励设计
                    reward = -1
                    if i == len(path) - 2:
                        if self.check_collision(P_copy) or not self.check_specific_points(P_copy):
                            reward = -100
                        else:
                            new_cost = self.calculate_cost(P_copy)
                            if new_cost < self.P_cost:
                                reward = 100
                            else:
                                # 根据路径长度给予奖励
                                path_length = len(path)
                                reward -= path_length
                                # 根据是否接近目标点给予奖励
                                target = self.TA_4[agent][-1]
                                distance_to_target = np.sqrt(
                                    (target[0] - new_position[0]) ** 2 + (target[1] - new_position[1]) ** 2)
                                reward -= distance_to_target

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))

                    path[i + 1] = new_position

            if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                new_cost = self.calculate_cost(P_copy)
                if new_cost < self.best_cost:
                    self.best_P = copy.deepcopy(P_copy)
                    self.best_cost = new_cost

            self.cost_history.append(self.best_cost)

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        target = reward + self.discount_factor * max(
                            self.target_Q[agent][next_state[0]].values() if next_state[0] in self.target_Q[agent] else [0])
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target

            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break

        return self.best_P, self.best_cost





class DQN_1_Optimizer:
    def __init__(self, P, P_cost, TA_4, grid, dimension, obstacles):
        self.P = P
        self.P_cost = P_cost
        self.TA_4 = TA_4
        self.grid = grid
        self.dimension = dimension
        self.obstacles = obstacles
        self.num_iterations = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100

        # 初始化Q表和目标Q表
        self.Q = {}
        self.target_Q = {}
        for agent in self.P.keys():
            self.Q[agent] = {}
            self.target_Q[agent] = {}

        # 经验回放池
        self.experience_replay = []

        self.best_P = copy.deepcopy(self.P)
        self.best_cost = self.P_cost
        self.cost_history = []

    # 计算路径成本
    def calculate_cost(self, P):
        cost = 0
        for agent, path in P.items():
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return cost

    # 检查路径冲突
    def check_collision(self, P):
        positions = set()
        for agent, path in P.items():
            for step in path:
                if step in positions:
                    return True
                positions.add(step)
        return False

    # 检查是否经过特定点
    def check_specific_points(self, P):
        for agent, path in P.items():
            specific_points = self.TA_4[agent]
            for point in specific_points:
                if point not in path:
                    return False
        return True

    # 八方向搜索
    def eight_direction_search(self, current):
        x, y = current
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] and (
                    new_x, new_y) not in self.obstacles:
                valid_neighbors.append((new_x, new_y))
        return valid_neighbors

    # 状态表示函数
    def get_state(self, agent, path, current_index):
        current = path[current_index]
        target = self.TA_4[agent][-1]
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)
        path_length = len(path) - current_index - 1
        return (current, distance_to_target, path_length)

    def optimize(self):
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (
                            0.2 - self.min_exploration_rate) * np.exp(-self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 更细致的奖励设计
                    reward = -1
                    if i == len(path) - 2:
                        if self.check_collision(P_copy) or not self.check_specific_points(P_copy):
                            reward = -100
                        else:
                            new_cost = self.calculate_cost(P_copy)
                            if new_cost < self.P_cost:
                                reward = 100
                            else:
                                # 根据路径长度给予奖励
                                path_length = len(path)
                                reward -= path_length
                                # 根据是否接近目标点给予奖励
                                target = self.TA_4[agent][-1]
                                distance_to_target = np.sqrt(
                                    (target[0] - new_position[0]) ** 2 + (target[1] - new_position[1]) ** 2)
                                reward -= distance_to_target

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))

                    path[i + 1] = new_position

            if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                new_cost = self.calculate_cost(P_copy)
                if new_cost < self.best_cost:
                    self.best_P = copy.deepcopy(P_copy)
                    self.best_cost = new_cost

            self.cost_history.append(self.best_cost)

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        # DDQN的核心修改：使用主Q表选择动作，用目标Q表评估该动作的Q值
                        if next_state[0] in self.Q[agent]:
                            next_action = max(self.Q[agent][next_state[0]], key=self.Q[agent][next_state[0]].get)
                            target = reward + self.discount_factor * self.target_Q[agent][next_state[0]].get(
                                next_action, 0)
                        else:
                            target = reward
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target

            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break

        return self.best_P, self.best_cost



class DQN_2_Optimizer:
    def __init__(self, P, P_cost, TA_4, grid, dimension, obstacles):
        self.P = P
        self.P_cost = P_cost
        self.TA_4 = TA_4
        self.grid = grid
        self.dimension = dimension
        self.obstacles = obstacles
        self.num_iterations = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100

        # 初始化Q表和目标Q表
        self.Q = {}
        self.target_Q = {}
        for agent in self.P.keys():
            self.Q[agent] = {}
            self.target_Q[agent] = {}

        # 经验回放池
        self.experience_replay = []

        self.best_P = copy.deepcopy(self.P)
        self.best_cost = self.P_cost
        self.cost_history = []

    # 计算路径成本
    def calculate_cost(self, P):
        cost = 0
        for agent, path in P.items():
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return cost

    # 检查路径冲突
    def check_collision(self, P):
        positions = set()
        for agent, path in P.items():
            for step in path:
                if step in positions:
                    return True
                positions.add(step)
        return False

    # 检查是否经过特定点
    def check_specific_points(self, P):
        for agent, path in P.items():
            specific_points = self.TA_4[agent]
            for point in specific_points:
                if point not in path:
                    return False
        return True

    # 八方向搜索
    def eight_direction_search(self, current):
        x, y = current
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] and (
                    new_x, new_y) not in self.obstacles:
                valid_neighbors.append((new_x, new_y))
        return valid_neighbors

    # 状态表示函数
    def get_state(self, agent, path, current_index):
        current = path[current_index]
        target = self.TA_4[agent][-1]
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)
        path_length = len(path) - current_index - 1
        return (current, distance_to_target, path_length)

    def optimize_0(self):
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (
                            0.2 - self.min_exploration_rate) * np.exp(-self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 更细致的奖励设计
                    reward = -1
                    if i == len(path) - 2:
                        if self.check_collision(P_copy) or not self.check_specific_points(P_copy):
                            reward = -100
                        else:
                            new_cost = self.calculate_cost(P_copy)
                            if new_cost < self.P_cost:
                                reward = 100
                            else:
                                # 根据路径长度给予奖励
                                path_length = len(path)
                                reward -= path_length
                                # 根据是否接近目标点给予奖励
                                target = self.TA_4[agent][-1]
                                distance_to_target = np.sqrt(
                                    (target[0] - new_position[0]) ** 2 + (target[1] - new_position[1]) ** 2)
                                reward -= distance_to_target

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))

                    path[i + 1] = new_position

            if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                new_cost = self.calculate_cost(P_copy)
                if new_cost < self.best_cost:
                    self.best_P = copy.deepcopy(P_copy)
                    self.best_cost = new_cost

            self.cost_history.append(self.best_cost)

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        # DDQN的核心修改：使用主Q表选择动作，用目标Q表评估该动作的Q值
                        if next_state[0] in self.Q[agent]:
                            next_action = max(self.Q[agent][next_state[0]], key=self.Q[agent][next_state[0]].get)
                            # Check if next_state exists in target_Q to avoid KeyError
                            next_state_q = self.target_Q[agent].get(next_state[0], {})
                            target = reward + self.discount_factor * next_state_q.get(next_action, 0)
                        else:
                            target = reward
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target
            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break

        return self.best_P, self.best_cost

    def optimize(self):
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            total_reward = 0  # 初始化总奖励

            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (0.2 - self.min_exploration_rate) * np.exp(
                        -self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 更细致的奖励设计
                    reward = -1
                    if i == len(path) - 2:
                        if self.check_collision(P_copy) or not self.check_specific_points(P_copy):
                            reward = -100
                        else:
                            new_cost = self.calculate_cost(P_copy)
                            if new_cost < self.P_cost:
                                reward = 100
                    # 根据路径长度给予奖励
                    path_length = len(path)
                    reward -= path_length
                    # 根据是否接近目标点给予奖励
                    target = self.TA_4[agent][-1]
                    distance_to_target = np.sqrt(
                        (target[0] - new_position[0]) ** 2 + (target[1] - new_position[1]) ** 2)
                    reward -= distance_to_target

                    total_reward += reward  # 累加当前奖励

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))
                    path[i + 1] = new_position

                    if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                        new_cost = self.calculate_cost(P_copy)
                        if new_cost < self.best_cost:
                            self.best_P = copy.deepcopy(P_copy)
                            self.best_cost = new_cost
                            self.cost_history.append(self.best_cost)

            # 打印本次迭代的总奖励
            print(f"Iteration: {iteration}, Total Reward: {total_reward}")

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        # DDQN的核心修改：使用主Q表选择动作，用目标Q表评估该动作的Q值
                        if next_state[0] in self.Q[agent]:
                            next_action = max(self.Q[agent][next_state[0]], key=self.Q[agent][next_state[0]].get)
                            next_state_q = self.target_Q[agent].get(next_state[0], {})
                            target = reward + self.discount_factor * next_state_q.get(next_action, 0)
                        else:
                            target = reward
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target

            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break
        return self.best_P, self.best_cost



class DQN_3_Optimizer:
    def __init__(self, P, P_cost, TA_4, grid, dimension, obstacles):
        self.P = P
        self.P_cost = P_cost
        self.TA_4 = TA_4
        self.grid = grid
        self.dimension = dimension
        self.obstacles = obstacles
        self.num_iterations = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100

        # 初始化Q表和目标Q表
        self.Q = {}
        self.target_Q = {}
        for agent in self.P.keys():
            self.Q[agent] = {}
            self.target_Q[agent] = {}

        # 经验回放池
        self.experience_replay = []

        self.best_P = copy.deepcopy(self.P)
        self.best_cost = self.P_cost
        self.cost_history = []

    # 计算路径成本
    def calculate_cost(self, P):
        cost = 0
        for agent, path in P.items():
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return cost

    # 检查路径冲突
    def check_collision(self, P):
        positions = set()
        for agent, path in P.items():
            for step in path:
                if step in positions:
                    return True
                positions.add(step)
        return False

    # 检查是否经过特定点
    def check_specific_points(self, P):
        for agent, path in P.items():
            specific_points = self.TA_4[agent]
            for point in specific_points:
                if point not in path:
                    return False
        return True

    # 八方向搜索
    def eight_direction_search(self, current):
        x, y = current
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] and (
                    new_x, new_y) not in self.obstacles:
                valid_neighbors.append((new_x, new_y))
        return valid_neighbors

    # 状态表示函数
    def get_state(self, agent, path, current_index):
        current = path[current_index]
        target = self.TA_4[agent][-1]
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)
        path_length = len(path) - current_index - 1
        return (current, distance_to_target, path_length)

    def optimize(self):
        alpha = 0.9  # 假设衰减系数为0.9，可根据实际情况调整
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (
                            0.2 - self.min_exploration_rate) * np.exp(-self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 计算到目标点的距离
                    current_distance = state[1]
                    next_distance = next_state[1]
                    reward = 0
                    if current_distance > next_distance:
                        reward = -0.05  # Move closer to goal
                    elif current_distance < next_distance:
                        reward = -0.1  # Move away from goal
                    if state[0] == new_position:
                        reward = -0.175 * (alpha ** (iteration - 1))  # Stay
                    # 简单判断振荡，记录最近3次位置，若有重复则认为振荡
                    position_history = [state[0]]
                    if i > 0:
                        position_history.append(path[i - 1])
                    if i > 1:
                        position_history.append(path[i - 2])
                    if new_position in position_history:
                        reward = -0.3  # Oscillation
                    if self.check_collision(P_copy) or (new_position in self.obstacles):
                        reward = -2  # Collide with agents or obstacles
                    if new_position == self.TA_4[agent][-1]:
                        reward = 5  # Reach the goal position

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))

                    path[i + 1] = new_position

            if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                new_cost = self.calculate_cost(P_copy)
                if new_cost < self.best_cost:
                    self.best_P = copy.deepcopy(P_copy)
                    self.best_cost = new_cost

            self.cost_history.append(self.best_cost)

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        target = reward + self.discount_factor * max(
                            self.target_Q[agent][next_state[0]].values() if next_state[0] in self.target_Q[agent] else [0])
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target

            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break

        return self.best_P, self.best_cost

class DQNOptimizer:
    def __init__(self, P, P_cost, TA_4, grid, dimension, obstacles):
        self.P = P
        self.P_cost = P_cost
        self.TA_4 = TA_4
        self.grid = grid
        self.dimension = dimension
        self.obstacles = obstacles
        self.num_iterations = 100
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2
        self.min_exploration_rate = 0.01
        self.exploration_decay = 0.001
        self.batch_size = 32
        self.target_update_frequency = 100

        # 初始化Q表和目标Q表
        self.Q = {}
        self.target_Q = {}
        for agent in self.P.keys():
            self.Q[agent] = {}
            self.target_Q[agent] = {}

        # 经验回放池
        self.experience_replay = []

        self.best_P = copy.deepcopy(self.P)
        self.best_cost = self.P_cost
        self.cost_history = []

    # 计算路径成本
    def calculate_cost(self, P):
        cost = 0
        for agent, path in P.items():
            for i in range(len(path) - 1):
                x1, y1 = path[i]
                x2, y2 = path[i + 1]
                cost += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return cost

    # 检查路径冲突
    def check_collision(self, P):
        positions = set()
        for agent, path in P.items():
            for step in path:
                if step in positions:
                    return True
                positions.add(step)
        return False

    # 检查是否经过特定点
    def check_specific_points(self, P):
        for agent, path in P.items():
            specific_points = self.TA_4[agent]
            for point in specific_points:
                if point not in path:
                    return False
        return True

    # 八方向搜索
    def eight_direction_search(self, current):
        x, y = current
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        valid_neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(self.grid) and 0 <= new_y < len(self.grid[0]) and self.grid[new_x][new_y] and (
                    new_x, new_y) not in self.obstacles:
                valid_neighbors.append((new_x, new_y))
        return valid_neighbors

    # 状态表示函数
    def get_state(self, agent, path, current_index):
        current = path[current_index]
        target = self.TA_4[agent][-1]
        distance_to_target = np.sqrt((target[0] - current[0]) ** 2 + (target[1] - current[1]) ** 2)
        path_length = len(path) - current_index - 1
        return (current, distance_to_target, path_length)

    def optimize(self):
        alpha = 0.9  # 假设衰减系数为0.9，可根据实际情况调整
        for iteration in range(self.num_iterations):
            P_copy = copy.deepcopy(self.P)
            for agent in self.P.keys():
                path = P_copy[agent]
                for i in range(len(path) - 1):
                    state = self.get_state(agent, path, i)
                    if state[0] not in self.Q[agent]:
                        self.Q[agent][state[0]] = {}
                    neighbors = self.eight_direction_search(state[0])
                    for neighbor in neighbors:
                        if neighbor not in self.Q[agent][state[0]]:
                            self.Q[agent][state[0]][neighbor] = 0

                    # 动态调整探索率
                    self.exploration_rate = self.min_exploration_rate + (
                            0.2 - self.min_exploration_rate) * np.exp(-self.exploration_decay * iteration)

                    if random.random() < self.exploration_rate:
                        new_position = random.choice(neighbors)
                    else:
                        # 根据Q值选择动作
                        max_q = max(self.Q[agent][state[0]].values())
                        best_neighbors = [neighbor for neighbor, q in self.Q[agent][state[0]].items() if q == max_q]
                        new_position = random.choice(best_neighbors)

                    next_state = self.get_state(agent, path, i + 1)

                    # 计算到目标点的距离
                    current_distance = state[1]
                    next_distance = next_state[1]
                    reward = 0
                    if current_distance > next_distance:
                        reward = -0.05  # Move closer to goal
                    elif current_distance < next_distance:
                        reward = -0.1  # Move away from goal
                    if state[0] == new_position:
                        reward = -0.175 * (alpha ** (iteration - 1))  # Stay
                    # 简单判断振荡，记录最近3次位置，若有重复则认为振荡
                    position_history = [state[0]]
                    if i > 0:
                        position_history.append(path[i - 1])
                    if i > 1:
                        position_history.append(path[i - 2])
                    if new_position in position_history:
                        reward = -0.3  # Oscillation
                    if self.check_collision(P_copy) or (new_position in self.obstacles):
                        reward = -2  # Collide with agents or obstacles
                    if new_position == self.TA_4[agent][-1]:
                        reward = 5  # Reach the goal position

                    # 打印当前迭代后的奖励
                    # print(f"Iteration: {iteration}, Agent: {agent}, Current Step: {i}, Reward: {reward}")

                    # 存储经验到经验回放池
                    self.experience_replay.append((agent, state, new_position, reward, next_state))

                    path[i + 1] = new_position

            if not self.check_collision(P_copy) and self.check_specific_points(P_copy):
                new_cost = self.calculate_cost(P_copy)
                if new_cost < self.best_cost:
                    self.best_P = copy.deepcopy(P_copy)
                    self.best_cost = new_cost

            self.cost_history.append(self.best_cost)

            # 经验回放
            if len(self.experience_replay) > self.batch_size:
                batch = random.sample(self.experience_replay, self.batch_size)
                for agent, state, action, reward, next_state in batch:
                    if state[0] in self.Q[agent] and action in self.Q[agent][state[0]]:
                        target = reward + self.discount_factor * max(
                            self.target_Q[agent][next_state[0]].values() if next_state[0] in self.target_Q[agent] else [0])
                        self.Q[agent][state[0]][action] = (1 - self.learning_rate) * self.Q[agent][state[0]][action] + \
                                                          self.learning_rate * target

            # 更新目标网络
            if iteration % self.target_update_frequency == 0:
                self.target_Q = copy.deepcopy(self.Q)

            # 更好的终止条件：连续多次迭代成本没有显著降低
            if len(self.cost_history) > 100:
                avg_cost = sum(self.cost_history[-100:]) / 100
                if abs(self.best_cost - avg_cost) < 0.01:
                    break

        return self.best_P, self.best_cost