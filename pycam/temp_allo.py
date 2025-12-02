# 开发时间：2025/3/14 18:43
# 开发语言：Python
import copy
import math
import random
import time
import numpy as np
from scipy.optimize import linear_sum_assignment

class TAE():
    def __init__(self, w, m, n):
        self.w, self.m, self.n = w, m, n

    def Alg_4_1(self):
        start = time.time()
        self.random_part = self.get_random_partition(self.m , self.n)
        self.tol_part = self.get_optimized_partition_4_1(self.random_part)
        end = time.time()
        print("  time :",round(end-start,2))
        print(w,m,n,self.tol_part)
        return self.tol_part

    def get_optimized_partition_4_1(self,random_part):
        "随机分区"
        init_G = random_part
        "转移交换优化"
        temp_G = self.transfer(init_G,True)
        temp_G = self.exchage(temp_G,True)
        "集群优化"
        temp_G = self.transfer_outliers(temp_G, 1.5)
        temp_G = self.transfer(temp_G, True)
        temp_G = self.exchage(temp_G, True)
        # print(temp_G)
        # print("Alg.4.1_Cost :{0}".format(self.get_Gi_Sa_新版(temp_G)),end='')
        tol_G_seq = self.get_sequence(temp_G)
        return tol_G_seq

    "---------------------算法2工具函数-------------------"
    def aaa2(self, m, n):
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(m)]
        tasks = [f't{i}' for i in range(n)]
        le_list, rg_list, result_list, all_cost, all_comb, min_cost = [], [], [], [], [], 0
        le_list = copy.deepcopy(agents)
        rg_list = copy.deepcopy(tasks)

        while rg_list:
            for a_ in le_list:
                for t_ in rg_list:
                    cost = self.get_cost(a_, t_)
                    all_comb.append((a_, t_))
                    all_cost.append(cost)
            min_cost += min(all_cost)
            index = all_cost.index(min(all_cost))
            le_temp, rg_temp = all_comb[index][0], all_comb[index][1]
            all_cost.clear()
            all_comb.clear()

            if le_list in tasks:
                re_index = result_list.index(le_temp)
                result_list.insert(re_index, rg_temp)
                continue

            elif le_temp in tasks:
                result_list.append(rg_temp)
            elif le_temp in agents:
                result_list.append(le_temp)
                result_list.append(rg_temp)
            le_index = le_list.index(le_temp)
            le_list[le_index] = rg_temp
            rg_list.remove(rg_temp)
        if not rg_list and any(element in agents for element in le_list):
            extracted_elements = [element for element in le_list if element in agents]
            result_list.extend(extracted_elements)

        result_dict = {}
        key = bool
        # 遍历列表，将元素分组到字典中
        for i in range(len(result_list)):
            item = result_list[i]
            if item.startswith('a'):  # 检查元素是否以 'a' 开头
                # 如果元素以 'a' 开头，它将是新的键
                key = item
                # 初始化该键对应的值列表为空列表
                result_dict[key] = []
            else:
                # 如果元素不以 'a' 开头，它将被添加到最近的键对应的值列表中
                result_dict[key].append(item)
        return result_dict, min_cost
    # 获取随机分区
    def get_random_partition(self, m, n):
        av_task_num = math.ceil(n / m) if math.ceil(n / m) * (m - 1) < n else math.ceil(n / m) - 1  # 每个分区内的平均任务数
        init_part_dict = dict()  # 分区字典
        agents = [f'a{i}' for i in range(m)]  # ['a0', 'a1']
        tasks = [f't{i}' for i in range(n)]  # ['t0', 't1']
        tasks_use0 = copy.deepcopy(tasks)

        if m >= n:
            temp_list = agents
            if m != n:
                valid_agent_list = self.get_valid_agent(n, m, agents, tasks)
                if valid_agent_list == 'None':
                    raise Exception("\033[31m当前权重矩阵w 不存在对应row与line\033[0m")
                temp_list = valid_agent_list
            for i in range(len(temp_list)):
                dict_index = temp_list[i]
                dict_value = []
                dict_value.append(tasks[i])
                init_part_dict[dict_index] = dict_value
            # print(init_part_dict)
            return init_part_dict

        else:
            temp_list = []
            while tasks_use0:
                temp = tuple(tasks_use0[:av_task_num])
                temp_list.append(temp)
                del tasks_use0[:av_task_num]
                if len(temp_list) == m - 1:
                    temp = tuple(tasks_use0)
                    temp_list.append(temp)
                    tasks_use0.clear()
            for i in range(len(agents)):
                dict_index = agents[i]
                dict_value = list(temp_list[i])
                init_part_dict[dict_index] = dict_value
            # print(init_part_dict)
            return init_part_dict
    "---------------------转移工具函数-------------------"
    def transfer(self, G, isnew):
        G_copy = copy.deepcopy(G)
        if self.detect(G):  # 探测是否为所有代理分区仅执行一个任务
            return G
        Comb = G
        Comb_sup = {agent: tasks for agent, tasks in Comb.items() if len(tasks) <= 1}
        # 排除组合内仅完成一个任务的代理
        Comb = {agent: tasks for agent, tasks in Comb.items() if len(tasks) > 1}
        if len(Comb) == 1:  # 如果就一个含多任务的代理分区
            return G_copy

        init_G = Comb
        # 生成所有可能的任务转移组合
        for agent, tasks in Comb.items():
            for task in tasks:
                # 生成当前代理的任务组合
                current_tasks = tasks.copy()
                current_tasks.remove(task)
                # 遍历其他代理
                for target_agent in Comb.keys():
                    if target_agent != agent:
                        # 在目标代理的不同位置插入任务
                        for i in range(len(Comb[target_agent]) + 1):
                            new_comb = {k: v.copy() for k, v in Comb.items()}
                            new_comb[agent] = current_tasks
                            new_comb[target_agent].insert(i, task)
                            init_G = self.select_G(init_G, new_comb, isnew)

        init_G.update(Comb_sup)
        return init_G
    # 转移离群点
    def transfer_outliers(self, G, a):
        U_id0 = []
        U = []
        U_id1 = []  # 存储离群点转移的分区id
        # print("初始G:\n",G)
        for agent_name, g in G.items():
            N = len(g)
            for v in g:
                # print("{0}内的{1}参与检测".format(agent_name,v))
                if self.get_g_m_w(g, v, agent_name) > (2 / N * a * self.get_g_w(g)):
                    # print("{0}检测为离群点".format(v))
                    temp = self.get_v_opt_g(v, agent_name, G)
                    if temp != None:
                        U_id0.append(agent_name)
                        U.append(v)
                        U_id1.append(temp)
                    continue
        # print(U_id0,U,U_id1)

        for i in range(len(U)):
            i_0 = U_id0[i]
            t_id = U[i]
            i_1 = U_id1[i]
            G[i_0] = [element for element in G[i_0] if element != t_id]
            G[i_1].append(t_id)
        # print(G)
        return G
    "---------------------交换工具函数-------------------"
    def exchage(self, G, isnew):
        init_G = G
        exchange = []  # 存储所有交换组合
        # 获取初始任务组合的所有代理和任务
        agents = list(G.keys())
        tasks = {agent: G[agent] for agent in agents}

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
                        init_G = self.select_G(init_G, new_comb, isnew)

            # # print(exchange)
            # # 获取Sa值最小的G
            # init_G = G
            # min_G = init_G
            # for ech_G in exchange:
            #
            #     init_G = min_G

        return init_G
    "---------------------基础工具函数-------------------"
    # 获取分区权重
    def get_g_w(self, g):
        g_w = 0
        for i in range(1, len(g)):
            v_0, v_1 = g[i - 1], g[i]
            g_w += self.get_cost(v_0, v_1)
        return g_w
    # 获取某点在该分区内的贡献值(离群点：贡献值最大)
    def get_g_m_w(self, g, v, agent_name):
        g_m_w = 0
        for i in range(0, len(g)):
            if v == g[i]: continue
            v_0, v_1 = v, g[i]
            g_m_w += self.get_cost(v_0, v_1)
        if len(g) == 1:
            g_m_w += self.get_cost(agent_name, g[0])
        return g_m_w
    # 获取某点的非自身所在的最佳分区
    def get_v_opt_g(self, v, v_id, G):
        dict1 = dict()
        for agent_name, p in G.items():
            dict1[agent_name] = self.get_g_m_w(p, v, agent_name)
        # print("{0}在总分区内的贡献值集合{1}".format(v,dict1))
        # 使用min函数和lambda表达式找出值最小的键
        min_id = min(dict1, key=lambda k: dict1[k])
        if v_id != min_id: return min_id
        return None
    # 将数据转换为序列
    def get_sequence(self, tol_G):
        seq = []
        for a_id, task_list in tol_G.items():
            seq.append(a_id)
            for t_id in task_list:
                seq.append(t_id)
        return seq
    # 选择评价值较小的组合，有两种评价值
    def select_G(self, Comb_init, Comb_next, isnew):
        if isnew == False:
            if self.get_Gi_Sa_原版(Comb_init) <= self.get_Gi_Sa_原版(Comb_next):
                return Comb_init
            else:
                return Comb_next
        else:
            if self.get_Gi_Sa_新版(Comb_init) <= self.get_Gi_Sa_新版(Comb_next):
                return Comb_init
            else:
                return Comb_next
    # 返回G中MAX(Sa(g))
    def get_Gi_Sa_原版(self, G):
        # print("G",G)
        # 单任务分区
        def Sa_Formula_0(agent_name, task_0):
            return self.get_cost(agent_name, task_0)

        # 多任务分区
        def Sa_Formula_1(N, agent_name, task_0, task_1):
            result = 2 / (N - 1) * (
                        ((self.get_cost(agent_name, task_0) + self.get_cost(agent_name, task_1)) / 2) + self.get_cost(
                    task_0, task_1))
            return result

        temp_cout = 0
        sig_count = 0
        mag_count = 0
        Sa_g_list = []
        for agent_name, value_list in G.items():
            if len(value_list) == 0:
                temp_cout += 0
            elif len(value_list) == 1:
                task_0 = value_list[0]
                sig_count += Sa_Formula_0(agent_name, task_0)
            else:
                for i in range(1, len(value_list)):
                    N = len(value_list)
                    task_0, task_1 = value_list[i - 1], value_list[i]
                    Sa_g_list.append(Sa_Formula_1(N, agent_name, task_0, task_1))  # 存储Sa(g)值
        # print(Sa_g_list)
        if len(Sa_g_list) != 0:
            # print(Sa_g_list)
            MAX_SAG = max(Sa_g_list)
            return MAX_SAG
        return 0
    def get_Gi_Sa_新版(self, G):
        # 遍历列表，将元素分组到字典中
        result_list = []
        for id, value in G.items():
            result_list.append(id)
            for v in value:
                result_list.append(v)
        test_cost = 0
        for i in range(1, len(result_list)):
            test_cost += self.get_cost(result_list[i - 1], result_list[i])
        return test_cost
    # 获取有效的执行代理（排除效用低的代理）
    def get_valid_agent(self, n, m, agents, tasks):
        valid_agent = dict()  # 存储代理到所有任务的效用值
        for agent_name in agents:
            temp_list = []
            for tasks_name in tasks:
                cost = self.get_cost(agent_name, tasks_name)
                if cost == 'None': return 'None'
                temp_list.append(cost)
            valid_agent['{0}'.format(agent_name)] = temp_list

        # print(valid_agent)
        # 创建效用权重表(旧版)
        utility_table = [-round((1 / n) * (n - i), 2) for i in range(n)]

        # 方便查看排序后的valid_agent
        # for key, value in valid_agent.items():
        #     value.sort()
        #     print(value)

        # 根据效用表排序
        for key, value in valid_agent.items():
            value.sort()
            temp = [round(x * y, 2) for x, y in zip(value, self.get_utility_table(value))]
            # temp = [round(x * y, 2) for x, y in zip(value, utility_table)]
            valid_agent[key] = temp
        sorted_data = sorted(valid_agent.items(), key=lambda x: sum(x[1]), reverse=False)
        valid_agent = dict(sorted_data)

        # print(valid_agent)

        # 排除效用值高的代理
        valid_agent_list = list(valid_agent.items())
        if len(valid_agent_list) >= m - n:
            valid_agent_list = valid_agent_list[:-(m - n)]
        valid_agent = dict(valid_agent_list)

        return list(valid_agent.keys())
    # 获取效用表
    def get_utility_table(self, value):
        temp = sum(value)
        table = []
        for i in range(len(value)):
            a = round(value[i] / temp, 2)
            table.append(a)
        return table
    # 获取顶点间在权重矩阵的row,line
    def get_matrix_value(self, m, v_0, v_1):
        agent_num, a, b = m, v_0, v_1
        row, line = 0, 0
        if len(a) == 2 and len(b) == 2:  # 代理（0-9），任务（0-9）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]), int(b[1]) + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) + agent_num, int(b[1])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) + agent_num, int(b[1]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]), int(b[1])
        if len(a) == 2 and len(b) == 3:  # 代理（0-9），任务（10-99）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]), int(b[1]) * 10 + int(b[2]) + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) + agent_num, int(b[1]) * 10 + int(b[2])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) + agent_num, int(b[1]) * 10 + int(b[2]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]), int(b[1]) * 10 + int(b[2])
        if len(a) == 3 and len(b) == 2:  # 代理（10-99），任务（0-9）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1]) + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1])
        if len(a) == 3 and len(b) == 3:  # 代理（10-99），任务（10-99）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1]) + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1]) * 10 + int(b[2])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1]) * 10 + int(b[2]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1]) * 10 + int(b[2])
        # 代理（0-9），任务（100-999）
        if len(a) == 2 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])
        # 代理（10-99），任务（100-999）
        if len(a) == 3 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) * 10 + int(a[2]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(
                    b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]) * 10 + int(a[2]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])
        # # 代理（100-999），任务（100-999）
        if len(a) == 4 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]) * 100 + int(a[2]) * 10 + int(a[3]), int(b[1]) * 100 + int(b[2]) * 10 + int(
                    b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) * 100 + int(a[2]) * 10 + int(a[3]) + agent_num, int(b[1]) * 100 + int(
                    b[2]) * 10 + int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) * 100 + int(a[2]) * 10 + int(a[3]) + agent_num, int(b[1]) * 100 + int(
                    b[2]) * 10 + int(
                    b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]) * 100 + int(a[2]) * 10 + int(a[3]), int(b[1]) * 100 + int(b[2]) * 10 + int(
                    b[3])

        return row, line
    # 获取顶点间的路径成本
    def get_cost(self, v_0, v_1):
        cost = 0
        row, line = self.get_matrix_value(self.m, v_0, v_1)
        try:
            cost = self.w[row][line]
        except:
            # print("\033[31m当前权重矩阵w 不存在对应row与line\033[0m")
            return 'None'

        return cost
    # 探测是否为所有代理分区仅执行一个任务
    def detect(self, G) -> bool:
        for key, value in G.items():
            if len(value) == 1:
                continue
            else:
                return False
        return True
def generate_matrix(m, n):
    rows = m + n
    cols = m + n
    matrix = [[random.randint(1, 40) for _ in range(cols)] for _ in range(rows)]
    # 获取矩阵的行数和列数
    rows = len(matrix)
    # 将前两列的元素设为0
    for i in range(rows):
        for j in range(2):
            matrix[i][j] = 0
    for i in range(rows):
        matrix[i][i] = 0

    return matrix

def generate_matrix_temp(m, n):
    rows = m
    cols = n
    matrix = [[random.randint(1, 9) for _ in range(cols)] for _ in range(rows)]
    return matrix
if __name__ == "__main__":
    m, n = 2, 3
    w = generate_matrix(m, n)
    # print(w)
    T = TAE(w, m, n)
    T.Alg_4_1()

