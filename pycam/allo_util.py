# 开发时间：2024/8/30 14:52
# 开发语言：Python
'''---------------------------------------'''
import copy
import math
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from draw_qushi_pic import PlotListReciprocal
from pycam.allo_util_RL import DQNForTaskAllocation
from pycam.allo_util_RL import DDQNForTaskAllocation,EnhancedDDQN_1,GRPOAgent

"""
        ========转移优化========
        假设代理集合A ={ a0, a1, a2, a3 }, 任务集合T = { t0, t1, t2, t3, t4, t5, t6 }
        1、现得出初始任务分配组合 Comb_init = { 'a1': ['t0', 't1'], 'a0': ['t2'], 'a3': ['t4', 't5'], 'a2':['t3', 't6']}
        2、每一个组合都有评价函数Sa, 通过Sa(Comb)得出评价值Valuation
        3、通过转移操作，得出下一个组合Comb,最终保留所有的组合，再通过对初始组合和Comb的评价值进行比较，选出评价值较小的组合，作为下一次的初始组合，继续与转移操作得出的Comb比较
        4、转移操作具体如下：
            1、排除组合内仅完成一个任务的代理，得出{ 'a1': ['t0', 't1'],  'a3': ['t4', 't5'], 'a2':['t3', 't6'] }
            2、原代理分区中的任务转移到某个代理分区中时，原代理分区中要删除该任务，某个代理分区中要补充该任务
            3、任务转移到某个代理分区中的不同位置时，比如第1次转移：{ 'a1': [ 't1'],  'a3': ['t0', 't4', 't5'], 'a2':['t3', 't6'] } 
               第2次转移：{ 'a1': [ 't1'],  'a3': ['t4', 't0', 't5'], 'a2':['t3', 't6'] } 
               第3次转移：{ 'a1': [ 't1'],  'a3': ['t4', 't5', 't0'], 'a2':['t3', 't6'] } 这里的't0'是从a1代理分区转移到a3代理分区，但由于't0'插入的位置不同，会生成多个组合          

        ========交换优化========
        假设初始组合：{'a0': ['t0', 't1'], 'a1': ['t2', 't4', 't3'], 'a2': ['t5']}
        每个代理分区中的每个任务都需要与其他所有代理分区的每个任务进行交换
        每做一次交换，就生成一个组合
        最终输出所有组合

"""

class TAE():
    def __init__(self,w,m,n):
        self.w , self.m , self.n = w,m,n
        # result, cost = self.aaa2(self.m, self.n)
        # print(result,self.get_Gi_Sa(result))
        # self.random_part = self.get_random_partition(self.m , self.n)
        # self.tol_part = self.get_optimized_partition(result)
        # print(self.tol_part)

    def Alg_4_0(self):
        start = time.time()
        self.random_part = self.get_random_partition(self.m, self.n)
        self.tol_part = self.get_optimized_partition_4_0(self.random_part)
        end = time.time()
        # print("  time :",round(end-start,2))
        # print(self.tol_part)
        return self.tol_part
    def Alg_4_1(self):
        start = time.time()
        self.random_part = self.get_random_partition(self.m , self.n)
        self.tol_part = self.get_optimized_partition_4_1(self.random_part)
        end = time.time()
        print("  time :",round(end-start,2))
        return self.tol_part
    def Alg_5(self):
        start = time.time()
        result, cost = self.aaa2(self.m, self.n)
        self.tol_part = self.get_optimized_partition(result,1.5)
        # print(self.tol_part)
        end = time.time()
        # print("  time :",round(end-start,2))
        return self.tol_part

    def Alg_5_RL(self):
        start = time.time()
        result, cost = self.aaa2(self.m, self.n) # 初始解为贪婪解
        # result = self.get_random_partition(self.m, self.n)  # 初始解为随机解
        # print("初始解为{}".format(result))
        self.tol_part = self.get_optimized_partition_RL(result,1.5)
        end = time.time()
        # print("  time :",round(end-start,2))
        return self.tol_part


    """
    第一步：random_partition (随机分区函数)
        m >= n: 排除效用低的代理
        m < n : 
            1偶数：将tasks中的任务依次按av_task_num个以元组存储在temp_list
            2奇数：round(av_task_num),记录tasks_use0的剩余个数，如果小于round(av_task_num)，则将剩余任务存储在最后分区        
    """
    # 获取随机分区
    def get_random_partition(self,m,n):
        av_task_num = math.ceil(n/m) if math.ceil(n/m)*(m-1) < n else math.ceil(n/m)-1   # 每个分区内的平均任务数
        init_part_dict = dict() # 分区字典
        agents = [f'a{i}' for i in range(m)] # ['a0', 'a1']
        tasks = [f't{i}' for i in range(n)]  # ['t0', 't1']
        tasks_use0 = copy.deepcopy(tasks)

        if m >= n :
            temp_list = agents
            if m != n:
                valid_agent_list = self.get_valid_agent(n,m,agents,tasks)
                if valid_agent_list == 'None':
                    raise Exception("\033[31m当前权重矩阵w 不存在对应row与line\033[0m")
                temp_list = valid_agent_list
            for i in range(len(temp_list )):
                dict_index = temp_list[i]
                dict_value = []
                dict_value.append(tasks[i])
                init_part_dict[dict_index]=dict_value
            # print(init_part_dict)
            return init_part_dict

        else:
            temp_list = []
            while tasks_use0:
                temp = tuple(tasks_use0[:av_task_num])
                temp_list.append(temp)
                del tasks_use0[:av_task_num]
                if len(temp_list) == m-1:
                    temp = tuple(tasks_use0)
                    temp_list.append(temp)
                    tasks_use0.clear()
            for i in range(len(agents)):
                dict_index = agents[i]
                dict_value = list(temp_list[i])
                init_part_dict[dict_index] = dict_value
            # print(init_part_dict)
            return init_part_dict

    # 算法4_0：获取最优分配,通过转移优化、交换优化
    def get_optimized_partition_4_0(self, random_part):
        "随机分区"
        init_G = random_part
        "转移交换优化"
        temp_G = self.transfer(init_G,False)
        temp_G = self.exchage(temp_G,False)
        "集群优化"
        temp_G = self.transfer_outliers(temp_G,1.5)
        temp_G = self.transfer(temp_G,False)
        temp_G = self.exchage(temp_G,False)
        # print(temp_G)
        # print("Alg.4.0_Cost :{0}".format(self.get_Gi_Sa_新版(temp_G)),end='')
        tol_G_seq = self.get_sequence(temp_G)
        return tol_G_seq

    """
    
    第二步：
        Transfer and Exchange：
            1、遍历G总分区中，根据每次的转移组合得出G‘,再对比{ Ev(G), Ev(G') }，通过迭代最终获取评价值最小的总区G*
            2、交换同理
        Transfer_outliers:
            1、遍历每个分区的每个任务点，找出超出阈值的离群点集合
            2、将所有离群点转移到其最佳子图中去
    """
    # 算法4_1：获取最优分配,改进Sa、含顺序转移、考虑代理因素
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
        print("Alg.4.1_Cost :{0}".format(self.get_Gi_Sa_新版(temp_G)),end='')

        tol_G_seq = self.get_sequence(temp_G)
        return tol_G_seq
    # 算法5：
    # def get_optimized_partition_RL(self, greedy_part, a):
    #     # 1. 收集全部任务
    #     full_G = copy.deepcopy(greedy_part)
    #     all_tasks = [t for tasks in full_G.values() for t in tasks]
    #     m = self.m
    #     full_n = len(all_tasks)

    #     # 2. 留 20 % 空白
    #     num_leave = max(1, full_n // 5)
    #     assigned_tasks = set(random.sample(all_tasks, k=full_n - num_leave))
    #     hidden_tasks   = set(all_tasks) - assigned_tasks
    #     visible_tasks  = sorted(assigned_tasks)
    #     task_id_map    = {t: idx for idx, t in enumerate(visible_tasks)}
    #     idx_to_task    = {idx: t for t, idx in task_id_map.items()}

    #     print(f"[INIT] 总任务={full_n}  可见={len(visible_tasks)}  隐藏={len(hidden_tasks)}")

    #     # 3. 构造「仅含 80 % 任务」的初始方案（用连续索引）
    #     init_P = {f"a{i}": [] for i in range(m)}
    #     for t in assigned_tasks:
    #         idx = task_id_map[t]
    #         init_P[f"a{random.randint(0, m-1)}"].append(f"t{idx}")  # 新编号

    #     # 4. 覆盖环境参数，让 Agent 只看到 80 %
    #     old_n = self.n
    #     self.n = len(visible_tasks)          # 关键：缩小任务数
    #     # 若用 task_features，也要同步掩码
    #     if hasattr(self, 'task_features') and self.task_features is not None:
    #         self.task_features = self.task_features[np.isin(all_tasks, visible_tasks)]

    #     # 5. 实例化 GRPO 并训练
    #     optimize_funcs = [self.transfer_rl, self.exchage_rl]
    #     agent = GRPOAgent(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
    #     best_P_idx, best_cost = agent.train(init_P)  # 返回「索引版」方案

    #     # 6. 把索引映射回原始任务名
    #     best_P = {}
    #     for a, tasks in best_P_idx.items():
    #         best_P[a] = [idx_to_task[int(t[1:])] for t in tasks]

    #     # 7. 恢复原始任务数（防止影响后续调用）
    #     self.n = old_n

    #     return self.get_sequence(best_P)
    #     "贪婪分区"
    #     init_G = greedy_part
    #     first_G = copy.deepcopy(init_G)
    #     "集群优化"
    #     temp_G = self.transfer_outliers(first_G, a)
    #     if self.get_Gi_Sa_新版(temp_G) < self.get_Gi_Sa_新版(init_G):
    #         print("执行!!!!!!!!!!!!!!!")
    #         temp_G = self.transfer(temp_G,True)
    #         temp_G = self.exchage(temp_G, True)
    #         tol_G_seq = self.get_sequence(temp_G)
    #     else:
    #         tol_G_seq = self.get_sequence(init_G)
    #     # print("Alg.5.0_Cost :{0}\n".format(self.get_Gi_Sa_新版(temp_G)),end='')

    #     return tol_G_seq
    def get_optimized_partition(self, greedy_part, a):

        "贪婪分区"
        init_G = greedy_part
        first_G = copy.deepcopy(init_G)
        # print("初始方案{}\n成本{}",first_G,self.get_Gi_Sa_新版(first_G))
        first_G = self.sort_dict_by_agent(first_G)
        

        "实例化类并运行DQN算法进行优化"
        # dqn_agent = DQNForTaskAllocation(self.m, self.n, self.exchage_rl, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost = dqn_agent.dqn_learning(first_G)
        # 实例化类并运行DDQN算法进行优化
        optimize_funcs = [self.transfer_rl,self.exchage_rl]
        #DDQN的运行
        # dqn_agent = DDQNForTaskAllocation(self.m, self.n,optimize_funcs,self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost, reward_ls, cost_history = dqn_agent.ddqn_learning(first_G)
        #EnhancedDDQN的运行
        # dqn_agent = EnhancedDDQN_1(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost= dqn_agent.train(first_G)
        #GRPO的运行
        #print("[INFO] Start GRPO.train()", flush=True)
        t0 = time.time()
        dqn_agent = EnhancedDDQN_1(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        optimized_P, optimized_cost= dqn_agent.train(first_G)
        # dqn_agent = GRPOAgent(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost = dqn_agent.train(first_G)
        t1 = time.time()
        #print("[INFO] GRPO.train() returned", flush=True)
        print(f"[INFO] optimized_cost={optimized_cost}, time={t1-t0:.2f}s", flush=True)
    
        # optional: show small summary to ensure result used
        print("[INFO] sample of optimized_P (per-agent counts):",
            {k: len(v) for k, v in optimized_P.items()}, flush=True)
        # # 将列表写入文件
        # with open('drl_data\output_test03.txt', 'w') as file:
        #     for element in reward_ls:
        #         file.write(str(element) + '\n')
        # # plt = PlotListReciprocal(reward_ls)
        # print(f"最终优化方案: {optimized_P}, 成本: {optimized_cost}")
        #
        # with open("drl_cost\op_0.txt",'w') as file:
        #     for element in cost_history:
        #         file.write(str(element) + '\n')


        RL_G = copy.deepcopy(optimized_P)
        return self.get_sequence(RL_G)
    def get_optimized_partition_RL(self, greedy_part, a):

        "贪婪分区"
        init_G = greedy_part
        first_G = copy.deepcopy(init_G)
        # print("初始方案{}\n成本{}",first_G,self.get_Gi_Sa_新版(first_G))
        first_G = self.sort_dict_by_agent(first_G)
        

        "实例化类并运行DQN算法进行优化"
        # dqn_agent = DQNForTaskAllocation(self.m, self.n, self.exchage_rl, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost = dqn_agent.dqn_learning(first_G)
        # 实例化类并运行DDQN算法进行优化
        optimize_funcs = [self.transfer_rl,self.exchage_rl]
        #DDQN的运行
        # dqn_agent = DDQNForTaskAllocation(self.m, self.n,optimize_funcs,self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost, reward_ls, cost_history = dqn_agent.ddqn_learning(first_G)
        #EnhancedDDQN的运行
        # dqn_agent = EnhancedDDQN_1(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost= dqn_agent.train(first_G)
        #GRPO的运行
        #print("[INFO] Start GRPO.train()", flush=True)
        t0 = time.time()
        # dqn_agent = EnhancedDDQN_1(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        # optimized_P, optimized_cost= dqn_agent.train(first_G)
        dqn_agent = GRPOAgent(self.m, self.n, optimize_funcs, self.get_Gi_Sa_新版)
        optimized_P, optimized_cost = dqn_agent.train(first_G)
        t1 = time.time()
        #print("[INFO] GRPO.train() returned", flush=True)
        print(f"[INFO] optimized_cost={optimized_cost}, time={t1-t0:.2f}s", flush=True)
    
        # optional: show small summary to ensure result used
        print("[INFO] sample of optimized_P (per-agent counts):",
            {k: len(v) for k, v in optimized_P.items()}, flush=True)
        # # 将列表写入文件
        # with open('drl_data\output_test03.txt', 'w') as file:
        #     for element in reward_ls:
        #         file.write(str(element) + '\n')
        # # plt = PlotListReciprocal(reward_ls)
        # print(f"最终优化方案: {optimized_P}, 成本: {optimized_cost}")
        #
        # with open("drl_cost\op_0.txt",'w') as file:
        #     for element in cost_history:
        #         file.write(str(element) + '\n')


        RL_G = copy.deepcopy(optimized_P)
        return self.get_sequence(RL_G)


    "---------------------RL优化工具函数-------------------"
    # 转移优化
    def transfer_rl(self, G):
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
                            init_G = self.select_G(init_G, new_comb, True)

        init_G.update(Comb_sup)
        return init_G
    # 交换优化
    def exchage_rl(self,G):
        init_G = G
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
                        # 对比初始方案，得出成本更小的方案
                        init_G = self.select_G(init_G, new_comb, True)
        return init_G
    # 离群点优化
    def transfer_outliers_rl(self,G,a=0.5):

        init_G = copy.deepcopy(G)
        U_id0 = []
        U = []
        U_id1 = [] # 存储离群点转移的分区id
        # print("初始G:\n",G)
        for agent_name, g in G.items():
            N = len(g)
            for v in g:
                # print("{0}内的{1}参与检测".format(agent_name,v))
                if self.get_g_m_w(g,v,agent_name) > (2 / N * a * self.get_g_w(g)):
                    # print("{0}检测为离群点".format(v))
                    temp = self.get_v_opt_g(v,agent_name,G)
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
        init_G = self.select_G(init_G, G, True)
        return init_G
    "---------------------算法2工具函数-------------------"
    def aaa2(self, m, n):
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(m)]
        tasks = [f't{i}' for i in range(n)]
        le_list, rg_list, result_list, all_cost,all_comb,min_cost =[], [],[], [], [], 0
        # row,line=self.get_matrix_value()
        # cost=self.distance_matrix[row][line]
        # print(self.distance_matrix)
        le_list=copy.deepcopy(agents)
        rg_list=copy.deepcopy(tasks)

        while rg_list:
            for a_ in le_list:
                for t_ in rg_list:
                    cost = self.get_cost(a_, t_)
                    all_comb.append((a_,t_))
                    all_cost.append(cost)
            min_cost+=min(all_cost)
            index=all_cost.index(min(all_cost))
            le_temp, rg_temp = all_comb[index][0], all_comb[index][1]
            all_cost.clear()
            all_comb.clear()

            if le_list in tasks:
                re_index=result_list.index(le_temp)
                result_list.insert(re_index,rg_temp)
                continue

            elif le_temp in tasks:
                result_list.append(rg_temp)
            elif le_temp in agents:
                result_list.append(le_temp)
                result_list.append(rg_temp)
            le_index=le_list.index(le_temp)
            le_list[le_index]=rg_temp
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
        return result_dict,min_cost
    "---------------------转移工具函数-------------------"
    """
            关键点：无顺序、评价函数（不考虑代理情况）
            { 'a1': ['t0', 't1'],  'a3': ['t4', 't5'], 'a2':['t3', 't6'] }
            第1次转移：{ 'a1': ['t1'],  'a3': ['t0', 't4', 't5'], 'a2':['t3', 't6'] }
            第2次转移：{ 'a1': ['t1'],  'a3': [ 't4', 't5'], 'a2':['t0','t3', 't6'] }
            第3次转移：{ 'a1': ['t0'],  'a3': ['t1', 't4', 't5'], 'a2':[,'t3', 't6'] }
            第4次转移：{ 'a1': ['t0'],  'a3': ['t4', 't5'], 'a2':['t1', 't3', 't6'] }
            第5次转移：{ 'a1': ['t4','t1','t0'],  'a3': ['t5'], 'a2':[ 't3', 't6'] }
            第6次转移：{ 'a1': ['t1','t0'],  'a3': ['t5'], 'a2':['t4', 't3', 't6'] }
            第7次转移：{ 'a1': [ 't3','t1','t0'],  'a3': ['t4','t5'], 'a2':[ 't6'] }
            第8次转移：{ 'a1': [ 't1','t0'],  'a3': ['t3','t4','t5'], 'a2':[ 't6'] }
            第9次转移：{ 'a1': [ 't6', 't1','t0'],  'a3': ['t4','t5'], 'a2':['t3'] }
            第10次转移：{ 'a1': [ 't1','t0'],  'a3': ['t6','t4','t5'], 'a2':['t3'] }
            """

    def transfer(self, G, isnew):
        G_copy = copy.deepcopy(G)
        if self.detect(G):  # 探测是否为所有代理分区仅执行一个任务
            return G
        Comb = G
        Comb_sup = {agent: tasks for agent, tasks in Comb.items() if len(tasks) <= 1}
        # 排除组合内仅完成一个任务的代理
        Comb = {agent: tasks for agent, tasks in Comb.items() if len(tasks) > 1}
        if len(Comb) == 1: # 如果就一个含多任务的代理分区
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
                            init_G = self.select_G(init_G, new_comb,isnew)


        init_G.update(Comb_sup)
        return init_G
    # 转移离群点
    def transfer_outliers(self,G,a=1.5):
        U_id0 = []
        U = []
        U_id1 = [] # 存储离群点转移的分区id
        # print("初始G:\n",G)
        for agent_name, g in G.items():
            N = len(g)
            for v in g:
                # print("{0}内的{1}参与检测".format(agent_name,v))
                if self.get_g_m_w(g,v,agent_name) > (2 / N * a * self.get_g_w(g)):
                    # print("{0}检测为离群点".format(v))
                    temp = self.get_v_opt_g(v,agent_name,G)
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
    def exchage(self,G, isnew):
        init_G = G
        exchange = [] # 存储所有交换组合
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
    def get_row_line(self,fe):
        test_cost=0
        for i in range(1,len(fe)):
            row,line= self.get_matrix_value(self.m,fe[i-1],fe[i])
            test_cost +=self.w[row][line]
        return test_cost
    # 按照代理号进行从小到大排序{‘'a0': ['t8', 't2'],'a1': ['t9', 't6', 't14', 't17', 't12'], 'a2': [],.....}
    def sort_dict_by_agent(self,d):
        sorted_keys = sorted(d.keys(), key=lambda x: int(x[1:]))  # 按照代理号（即去掉'a'后的数字）从小到大排序
        new_dict = {k: d[k] for k in sorted_keys}  # 构建新的字典
        return new_dict
    # 获取分区权重
    def get_g_w(self,g):
        g_w = 0
        for i in range(1,len(g)):
            v_0, v_1 = g[i-1], g[i]
            g_w += self.get_cost(v_0, v_1)
        return g_w
    # 获取某点在该分区内的贡献值(离群点：贡献值最大)
    def get_g_m_w(self, g, v,agent_name):
        g_m_w = 0
        for i in range(0,len(g)):
            if v == g[i]:continue
            v_0, v_1 = v, g[i]
            g_m_w += self.get_cost(v_0, v_1)
        if len(g) == 1:
            g_m_w += self.get_cost(agent_name, g[0])
        return g_m_w
    # 获取某点的非自身所在的最佳分区
    """
    1、计算{'a1':ctb1,...}
    """
    def get_v_opt_g(self,v,v_id,G):
        dict1 = dict()
        for agent_name, p in G.items():
            dict1[agent_name] = self.get_g_m_w(p,v,agent_name)
        # print("{0}在总分区内的贡献值集合{1}".format(v,dict1))
        # 使用min函数和lambda表达式找出值最小的键
        min_id = min(dict1, key=lambda k: dict1[k])
        if v_id != min_id : return min_id
        return None
    # 将数据转换为序列
    def get_sequence(self,tol_G):
        seq = []
        for a_id,task_list in tol_G.items():
            seq.append(a_id)
            for t_id in task_list:
                seq.append(t_id)
        return seq
    # 选择评价值较小的组合，有两种评价值
    def select_G(self, Comb_init, Comb_next, isnew):
        if isnew == False:
            if self.get_Gi_Sa_原版(Comb_init) <=  self.get_Gi_Sa_原版(Comb_next):
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
            return self.get_cost(agent_name,task_0)
        # 多任务分区
        def Sa_Formula_1(N,agent_name, task_0, task_1):
            result = 2/(N-1) * ( ((self.get_cost(agent_name,task_0)+self.get_cost(agent_name,task_1))/2) + self.get_cost(task_0,task_1))
            return result


        temp_cout = 0
        sig_count = 0
        mag_count = 0
        Sa_g_list = []
        for agent_name, value_list in G.items():
            if len(value_list) == 0:
                temp_cout += 0
            elif len(value_list) == 1 :
                task_0 = value_list[0]
                sig_count += Sa_Formula_0(agent_name,task_0)
            else:
                for i in range(1,len(value_list)):
                    N = len(value_list)
                    task_0, task_1 = value_list[i-1], value_list[i]
                    Sa_g_list.append(Sa_Formula_1(N,agent_name,task_0,task_1))  # 存储Sa(g)值
        # print(Sa_g_list)
        if len(Sa_g_list) != 0 :
            # print(Sa_g_list)
            MAX_SAG = max(Sa_g_list)
            return MAX_SAG
        return 0
    def get_Gi_Sa_新版(self, G):
        # 遍历列表，将元素分组到字典中
        result_list = []
        for id,value in G.items():
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
    def get_utility_table(self,value):
        temp = sum(value)
        table = []
        for i in range(len(value)):
            a = round(value[i] / temp,2)
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
        for key,value in G.items():
            if len(value) == 1:
                continue
            else:
                return False
        return True

"""
TA + HA
"""
class THA():
    def __init__(self,w,m,n):
        self.w, self.m, self.n = w, m, n
        result_list, min_cost = self.TA(m, n)
        # print( result_list, min_cost )
        self.run()

    # Core code
    def run(self):
        a,b = self.TA(self.m, self.n)
        print(a)


    "------------------------ TA_func ------------------------"
    def TA(self, m, n):
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(m)]
        tasks = [f't{i}' for i in range(n)]
        le_list, rg_list, result_list, all_cost, all_comb, min_cost = [], [], [], [], [], 0
        # row,line=self.get_matrix_value()
        # cost=self.distance_matrix[row][line]
        # print(self.distance_matrix)
        le_list = copy.deepcopy(agents)
        rg_list = copy.deepcopy(tasks)
        while rg_list:
            le_agent_list = [item for item in le_list if not item.startswith('t')]
            if len(le_agent_list) == len(rg_list):
                print(le_agent_list, rg_list)
                result = self.HA(le_agent_list,rg_list)
                print(result)



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
        return result_list, min_cost
    "------------------------ HA_func ------------------------"
    def HA(self, agents, tasks):
        # 使用 scipy 的 linear_sum_assignment 函数实现匈牙利算法

        row_list, line_list = [], []
        for i in range(len(agents)):
            v_0 = agents[i]
            v_1 = tasks[i]
            row, line = self.get_matrix_value(self.m, v_0, v_1)
            row_list.append(row)
            line_list.append(line)

        print("111",row_list, line_list)
        # 创建新矩阵
        w =self.w[np.ix_(row_list, line_list)]

        row_ind, col_ind = linear_sum_assignment(w)
        print(row_ind, col_ind)
        # 创建分配结果
        assignment = {agents[i]: tasks[j] for i, j in zip(row_ind, col_ind)}

        return assignment


    "------------------------ util_func ----------------------"

    # 获取两点间的距离值
    def get_cost(self, v_0, v_1):
        row, line = self.get_matrix_value(self.m, v_0, v_1)
        try:
            cost = self.w[row][line]
        except:
            # print("\033[31m当前权重矩阵w 不存在对应row与line\033[0m")
            return 'None'
        return cost
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
def generate_matrix(m, n):
    rows = m + n
    cols = m + n
    random.seed(28)
    matrix = [[random.randint(1, 40) for _ in range(cols)] for _ in range(rows)]
    # 获取矩阵的行数和列数
    rows = len(matrix)
    # 将前两列的元素设为0
    for i in range(rows):
        for j in range(2):
            matrix[i][j] = 0
    for i in range(rows):
        matrix[i][i]=0

    return matrix
def generate_matrix_temp(m, n):
    rows = m
    cols = n
    matrix = [[random.randint(1, 9) for _ in range(cols)] for _ in range(rows)]
    return matrix
# 保存矩阵，输出矩阵
def data_w(matrix,text):
    if text=='write':
        # 将矩阵保存到文本文件
        with open('matrix.txt', 'w') as file:
            for row in matrix:
                # 将每一行转换为字符串，并用空格分隔每个元素
                row_str = ' '.join(map(str, row))
                # 写入文件，并添加换行符
                file.write(row_str + '\n')

    elif text=='read':
        # 从文本文件中读取矩阵
        with open('matrix.txt', 'r') as file:
            lines = file.readlines()

        # 将读取的字符串转换回矩阵形式
        matrix_from_file = []
        for line in lines:
            # 移除行尾的换行符，并将字符串分割成列表
            row = line.strip().split()
            # 将字符串元素转换为整数
            row = [int(num) for num in row]
            matrix_from_file.append(row)
        return matrix_from_file
if __name__ == "__main__":
    w0 = [[0, 0, 1, 4, 5],
         [0, 0, 4, 1, 3],
         [0, 0, 0, 1, 1],
         [0, 0, 1, 0, 6],
         [0, 0, 3, 2, 0]]

    m,n=10,20
    w = generate_matrix(m,n)
    # print(w)
    T = TAE(w, m, n)

    # T.Alg_4_0()
    # T.Alg_4_1()
    # T.Alg_5()
    T.Alg_5_RL()

