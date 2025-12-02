# 开发时间：2024/2/19 13:59
# 开发语言：Python
'''---------------------------------------'''
import math  #浮点绝对值
class AStar():
    def __init__(self, env): #传入一个env对象
        self.agent_dict = env.agent_dict  # 代理字典,测试结果可知 [{'name': 'agent0', 'start': [8, 0], 'task': [7, 3], 'goal': [10, 3]}, {'name': 'agent1', 'start': [5, 0], 'task': [2, 2], 'goal': [7, 2]}]
        # print(len(self.agent_dict))
        # print("此处为A*搜索下输出",self.agent_dict)
        self.admissible_heuristic1 = env.admissible_heuristic1  # h值
        self.admissible_heuristic2 = env.admissible_heuristic2  # h值
        self.is_at_task =env.is_at_task  # 检查代理是否到达任务位置的函数
        self.is_at_goal = env.is_at_goal  # 检查代理是否到达目标位置的函数
        self.get_neighbors = env.get_neighbors  # 获取代理某个状态的所有邻居节点
        self.path1,self.path2=[],[]

    '''重塑单代理规划的总路径'''
    def reconstruct_path1(self, came_from, current):
        self.path1 = [current]  # 总路径，存储当前节点
        while current in came_from.keys():  # 当前位置在came_from字典的键中
            current = came_from[current]  # 将当前位置更新为came_from字典中当前位置的值
            # print("前驱节点信息",current,type(current))
            self.path1.append(current)  # 将更新后的位置添加到total_path列表中
        r_time=self.agent_dict[self.agent_name]['release_time']

        if r_time >= len(self.path1):
            cha=r_time-len(self.path1)
            element=self.path1[0]
            self.path1=[element]*cha + self.path1
            # print("代理{0}释放时间生效".format(self.agent_name))
        return self.path1[::-1]
    def reconstruct_path2(self, came_from, current):
        self.path2 = [current]  # 总路径，存储当前节点
        while current in came_from.keys():  # 当前位置在came_fro
            # m字典的键中
            current = came_from[current]  # 将当前位置更新为came_from字典中当前位置的值
            # print("前驱节点信息",current,type(current))
            self.path2.append(current)  # 将更新后的位置添加到total_path列表中
        self.path2.pop()
        return  self.path2[::-1]
        # print("path2",self.path2)
    def search1(self, agent_name):
        """
        low level search
        """

        # 设定初始值
        initial_state = self.agent_dict[agent_name]["start"]  # 获取代理的起始状态
        self.agent_name=agent_name
        step_cost = 1  # 每个步骤的成本为1

        open_set = {initial_state}  # 存已访问
        closed_set = set()  # 存已访问中符合条件的节点
        came_from = {}  # 存储节点的前驱状态

        g_score = {}  # 存g值
        g_score[initial_state] = 0  # 初始g值
        f_score = {}  # 存储每个状态的f值
        f_score[initial_state] = 0 + self.admissible_heuristic1(initial_state, agent_name)

        while open_set:  # 当open_set集合不为空时
            # 创建一个临时字典，用于存储open_set中每个状态的fz值，temp_dict={state:f_score}
            # 如果某状态的f_score不存在，则使用无穷大float("inf")作为默认值
            temp_dict = {open_item:f_score.setdefault(open_item, float("inf")) for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)  # 获取估计距离f值 最小的节点状态

            if self.is_at_task(current, agent_name):
                return self.reconstruct_path1(came_from, current)

            # 从open_set中挑选f值最小的current放入closed_set中
            open_set -= {current}  # 将当前状态从open_set集合中移除
            closed_set |= {current}  # 将当前状态添加到closed_set集合中

            neighbor_list = self.get_neighbors(current)  # 获取当前状态的邻居点

            # 遍历邻居位置列表
            '''作用1：检测邻居节点是否已经被使用，放止重复
                作用2：将合理的邻居节点作为下一个移动节点
            '''
            for neighbor in neighbor_list:
                if neighbor in closed_set:  # 如果邻居位置在closed_set集合中（已访问），因为上下左右移动时可能有重复走过的节点
                    continue  # 跳过当前邻居位置

                # 计算从起点到每个状态的估计距离，其中键是状态，值是估计距离。

                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set:  # 如果邻居位置不在open_set集合中（未访问）
                    open_set |= {neighbor}  # 将邻居位置添加到open_set集合中
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):  # 如果邻居位置在open_set集合中，且估计距离不小于已知最短距离
                    continue  # 跳过当前邻居位置

                came_from[neighbor] = current  # 将当前状态设置为邻居位置的前驱状态

                g_score[neighbor] = tentative_g_score  # 将邻居位置的已知最短距离设置为估计距离
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic1(neighbor, agent_name)  # 更新


        return False
    def search2(self, agent_name):
        """
        low level search
        """


        # 设定初始值
        initial_state = self.agent_dict[agent_name]["task"]  # 获取代理的起始状态
        step_cost = 1  # 每个步骤的成本为1

        open_set = {initial_state}  # 存已访问
        closed_set = set()  # 存已访问中符合条件的节点
        came_from = {}  # 存储节点的前驱状态

        g_score = {}  # 存g值
        g_score[initial_state] = 0  # 初始g值
        f_score = {}  # 存储每个状态的f值
        f_score[initial_state] = 0 + self.admissible_heuristic2(initial_state, agent_name)

        while open_set:  # 当open_set集合不为空时
            # 创建一个临时字典，用于存储open_set中每个状态的fz值，temp_dict={state:f_score}
            # 如果某状态的f_score不存在，则使用无穷大float("inf")作为默认值
            temp_dict = {open_item:f_score.setdefault(open_item, float("inf")) for open_item in open_set}
            current = min(temp_dict, key=temp_dict.get)  # 获取估计距离f值 最小的节点状态

            if self.is_at_goal(current, agent_name):
                return self.reconstruct_path2(came_from, current)

            # 从open_set中挑选f值最小的current放入closed_set中
            open_set -= {current}  # 将当前状态从open_set集合中移除
            closed_set |= {current}  # 将当前状态添加到closed_set集合中

            neighbor_list = self.get_neighbors(current)  # 获取当前状态的邻居点

            # 遍历邻居位置列表
            '''作用1：检测邻居节点是否已经被使用，放止重复
                作用2：将合理的邻居节点作为下一个移动节点
            '''
            for neighbor in neighbor_list:
                if neighbor in closed_set:  # 如果邻居位置在closed_set集合中（已访问），因为上下左右移动时可能有重复走过的节点
                    continue  # 跳过当前邻居位置

                # 计算从起点到每个状态的估计距离，其中键是状态，值是估计距离。

                tentative_g_score = g_score.setdefault(current, float("inf")) + step_cost

                if neighbor not in open_set:  # 如果邻居位置不在open_set集合中（未访问）
                    open_set |= {neighbor}  # 将邻居位置添加到open_set集合中
                elif tentative_g_score >= g_score.setdefault(neighbor, float("inf")):  # 如果邻居位置在open_set集合中，且估计距离不小于已知最短距离
                    continue  # 跳过当前邻居位置

                came_from[neighbor] = current  # 将当前状态设置为邻居位置的前驱状态

                g_score[neighbor] = tentative_g_score  # 将邻居位置的已知最短距离设置为估计距离
                f_score[neighbor] = g_score[neighbor] + self.admissible_heuristic2(neighbor, agent_name)  # 更新


        return False

