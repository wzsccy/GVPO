from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from loguru import logger


from .dist_table_4 import DistTable
from .mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    get_neighbors,
    len_neighbors_swap,
    get_neighbors_4,
    is_valid_coord,
)

NO_AGENT: int = np.iinfo(np.int32).max # 符号常量：表示该位置没有被代理占用
NO_LOCATION: Coord = (np.iinfo(np.int32).max, np.iinfo(np.int32).max) # 符号常量：表示无位置
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # d_y, d_x



@dataclass
class LowLevelNode:
    who: list[int] = field(default_factory=lambda: [])
    where: list[Coord] = field(default_factory=lambda: [])
    depth: int = 0

    def get_child(self, who: int, where: Coord) -> LowLevelNode:
        return LowLevelNode(
            who=self.who + [who],
            where=self.where + [where],
            depth=self.depth + 1,
        )


@dataclass
class HighLevelNode:
    Q: Config
    order: list[int]
    parent: HighLevelNode | None = None # 如果父节点为None，就设置为None
    tree: deque[LowLevelNode] = field(default_factory=lambda: deque([LowLevelNode()]))
    g: int = 0
    h: int = 0
    f: int = g + h
    neighbors: set[HighLevelNode] = field(default_factory=lambda: set())

    def __eq__(self, other) -> bool:
        if isinstance(other, HighLevelNode):
            return self.Q == other.Q
        return False

    def __hash__(self) -> int:
        return self.Q.__hash__()


class LaCAM:
    def __init__(self) -> None:
        pass

    def solve(
        self,
        grid: Grid,
        starts: Config,
        goals: Config,
        start_time ,
        time_limit_ms,
        deadline: Deadline | None = None,
        flg_star: bool = True,
        seed: int = 0, # 猜测是设定seed个随机数,估计要限定
        verbose: int = 1, # 输出的详细程度
        fl00: int=0,
    ) -> Configs:
        # set problem
        self.num_agents: int = len(starts)
        self.grid: Grid = grid
        self.starts: Config = starts
        self.goals: Config = goals
        self.deadline: Deadline = (
            deadline if deadline is not None else Deadline(time_limit_ms)
        )
        self.start_time=start_time
        self.time_limit_ms=time_limit_ms
        self.seed=seed
        # set hyper parameters
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed) # 随机数
        self.verbose = verbose # 控制输出信息的多少
        self.fl00=fl00
        return self._solve()

    def _solve(self) :
        # self.info(1, "start solving MAPF")
        # 碰撞检测 构建两个网格数组
        # set cache, used for collision check
        self.occupied_from: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)
        self.occupied_to: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)

        # set distance tables  self.dist_tables[i].get(v_now_i)
        self.dist_tables: list[DistTable] = [
            DistTable(self.grid, goal) for goal in self.goals
        ]

        # set search scheme
        OPEN: deque[HighLevelNode] = deque([])
        EXPLORED: dict[Config, HighLevelNode] = {}
        N_goal: HighLevelNode | None = None

        # set initial node
        Q_init = self.starts
        N_init = HighLevelNode(
            Q=Q_init, order=self.get_order(Q_init), h=self.get_h_value(Q_init)
        )
        OPEN.appendleft(N_init)
        EXPLORED[N_init.Q] = N_init

        # and not self.deadline.is_expired
        # main loop 主循环
        while len(OPEN) > 0 :
            N: HighLevelNode = OPEN[0]
            # 检查是否找到目标节点
            end = time.time()
            if end - self.start_time >= self.time_limit_ms:
                return 'over_time'

            if N.Q == self.goals:
                self.total_cost=N.g
                return self.backtrack(N)
                # self.info(1, f"initial solution found, cost={round(N_goal.g, 3)}")
                # no refinement -> terminate 无改进则终止
                # if not self.flg_star:
                #     break

            # 如果找到目标节点，并且目标节点的代价g小于或等于当前节点代价f，则跳过当前节点
            if N_goal is not None and N_goal.g <= N.f:
                OPEN.popleft()
                continue

            # low-level search end
            if len(N.tree) == 0:
                OPEN.popleft()
                continue

            # low-level search
            C: LowLevelNode = N.tree.popleft()  # constraints
            if C.depth < self.num_agents:
                i = N.order[C.depth]
                v = N.Q[i]
                cands = [v] + get_neighbors_4(self.grid, v)
                # print(i,cands)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u)) # 将i的候选点插入树中

            "此处需要将之替换为PIBT算法来作为配置生成器"
            Q_to = self.CG_PIBT(N,C)
            # Q_to = self.get_new_config(N,C)
            # print("\033[34m 得出新配置：{0}\033[0m".format(Q_to))
            # print(Q_to)

            if Q_to is None:
                # invalid configuration
                continue
            elif Q_to in EXPLORED.keys():
                # known configuration
                # print(Q_to.positions)
                N_known = EXPLORED[Q_to]
                N.neighbors.add(N_known)
                OPEN.appendleft(N_known)  # typically helpful
                # rewrite, Dijkstra update
                D = deque([N])
                while len(D) > 0:
                    N_from = D.popleft()
                    for N_to in N_from.neighbors:
                        g = N_from.g + self.get_edge_cost(N_from.Q, N_to.Q)
                        if g < N_to.g:
                            if N_goal is not None and N_to is N_goal:
                                self.info(2, f"cost update: {N_goal.g:2f} -> {g:2f}")
                            N_to.g = g
                            N_to.f = N_to.g + N_to.h
                            N_to.parent = N_from
                            D.append(N_to)
                            if N_goal is not None and N_to.f < N_goal.g:
                                OPEN.appendleft(N_to)
            else:
                # new configuration
                N_new = HighLevelNode(
                    Q=Q_to,
                    parent=N,
                    order=self.get_order(Q_to),
                    g=N.g + self.get_edge_cost(N.Q, Q_to),
                    h=self.get_h_value(Q_init),
                )
                N.neighbors.add(N_new)
                OPEN.appendleft(N_new)
                EXPLORED[Q_to] = N_new


        self.total_cost=0
        # categorize result
        if N_goal is not None and len(OPEN) == 0:
            # self.info(1, f"optimal solution, cost={round(N_goal.g, 3)}")
            self.total_cost=round(N_goal.g, 3)
        elif N_goal is not None:
            # self.info(1, f"suboptimal solution, cost={round(N_goal.g, 3)}")
            self.total_cost = round(N_goal.g, 3)
        elif len(OPEN) == 0:
            pass
            # self.info(1, "detected unsolvable instance")
        else:
            self.info(1, "failure due to timeout")
        return self.backtrack(N_goal)

    "给定当前节点回溯到根节点，返回出完整的路径列表"
    @staticmethod
    def backtrack(_N: HighLevelNode | None) -> Configs:
        configs: Configs = []
        N = _N
        while N is not None:
            configs.append(N.Q)
            N = N.parent
        configs.reverse()
        return configs

    def get_edge_cost(self, Q_from: Config, Q_to: Config) -> int:
        # e.g., \sum_i | not (Q_from[i] == Q_to[k] == g_i) |
        cost = 0
        for i in range(self.num_agents):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                x2,y2,x1,y1=Q_to[i][0],Q_to[i][1],Q_from[i][0],Q_from[i][1]
                coste=math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # print(coste, Q_from[i], Q_to[i])
                cost += coste
        return cost

    "从距离表中获取当前配置与目标配置的预估距离"
    "[Config->int]"
    def get_h_value(self, Q: Config) -> int:
        # e.g., \sum_i dist(Q[i], g_i)
        cost = 0
        for agent_idx, loc in enumerate(Q):
            c = self.dist_tables[agent_idx].get(loc)
            if c is None:
                return np.iinfo(np.int32).max
            cost += c
        return 0

    "获取代理的顺序列表，矩目标点远的智能体优先移动"
    "[Config->list[int]]"
    def get_order(self, Q: Config) -> list[int]:
        # e.g., by descending order of dist(Q[i], g_i)
        order = list(range(self.num_agents)) #
        self.rng.shuffle(order)
        order.sort(key=lambda i: self.dist_tables[i].get(Q[i]), reverse=True)

        return order

    def get_new_config(self,N,C) -> Config | None:
        # setup next configuration
        Q_from = N.Q
        Q_to = Config([NO_LOCATION for _ in range(self.num_agents)])
        self.occupied_from: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)
        self.occupied_to: np.ndarray = np.full(self.grid.shape, NO_AGENT, dtype=int)
        for i in range(self.num_agents):
            self.occupied_from[Q_from[i]] = i

        # 添加约束
        for k in range(C.depth):
            agent_id = C.who[k]
            agent_loc = C.where[k]
            Q_to[agent_id] = agent_loc
            # vc 如果当前约束下的agent_id位置已有代理占据，就为顶点冲突
            if self.occupied_from[agent_loc] < self.num_agents: return  None
            self.occupied_to[agent_loc] = agent_id
            # ec
            if self.occupied_from[agent_loc] < self.num_agents:
                other_agent_id = self.occupied_from[agent_loc]
                if other_agent_id != agent_id and Q_to[other_agent_id] != NO_LOCATION and Q_to[other_agent_id] == Q_from[agent_id]:
                    return None
        # print("约束下的Q_to",Q_to.positions)
        # apply PIBT
        # print("N择定代理顺序", self.get_order(Q_from))
        for agent_id in N.order:
            if Q_to[agent_id] == NO_LOCATION :
                # print("\033[31m执行PIBT_SWAP  当前代理为{0}\033[0m".format(agent_id))
                success = self.PIBT_SWAP(agent_id, Q_from, Q_to)
                if success == None: return None

        return Q_to
    def CG_PIBT(self, N: HighLevelNode, C: LowLevelNode) -> Config | None:
        Q_from = N.Q
        Q_to = Config([NO_LOCATION for _ in range(self.num_agents)])
        flg_success = True  # 标记，生成配置是否成功
        # 添加约束
        # print(N.Q,N.tree)
        # print(C)
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]
        # print("Q_to约束已知配置={0}".format(Q_to.positions))
        for i in range(0,len(self.starts)):
            self.occupied_from[Q_from[i]] = i
        # print("当前占用\n",self.occupied_from)


        for i in self.get_order(Q_from):
            if Q_to[i] == NO_LOCATION:
                "set PIBT_swap"
                if self.PIBT_SWAP(i, Q_from, Q_to) == None:
                    flg_success = False  # 如果新位置无效，则设置 flg_success 为 False 并退出循环
                    break

            v_i_to: Coord = Q_to[i]  # 获取智能体 i 在新配置 Q_to 中的位置

            # 检查下一占用表顶点冲突
            if self.occupied_to[v_i_to] != NO_AGENT:
                flg_success = False  # 如果新位置已被占用，则设置 flg_success 为 False 并退出循环
                break
            # 检查边冲突
            j = self.occupied_from[v_i_to]
            if j != NO_AGENT and j != i and Q_to[j] == Q_from[i]:
                flg_success = False  # 如果存在边冲突，则设置 flg_success 为 False 并退出循环
                break
            self.occupied_to[v_i_to] = i  # 将智能体 i 在 occupied_to 缓存中标记为占用

        # print("下一占用\n", self.occupied_to)
        # 清理 occupied_from 和 occupied_to 缓存
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = NO_AGENT  # 清理 occupied_from 缓存
            v_i_next = Q_to[i]
            if v_i_next != NO_LOCATION:
                self.occupied_to[v_i_next] = NO_AGENT  # 清理 occupied_to 缓存
        return Q_to if flg_success else None  # 如果 flg_success 为 True，则返回新的配置 Q_to；否则返回 None


    def PIBT_SWAP(self,i,Q_from,Q_to):
        Cadit_v = [Q_from[i]] + get_neighbors_4(self.grid, Q_from[i])
        Cadit_v.sort(key=lambda point: self.dist_tables[i].get(point))
        # print("代理{0} {1}的候选顶点{2}".format(i, Q_from[i], Cadit_v))

        # J = self.swap_required_and_possible(i,Cadit_v[0],self.occupied_from,Q_from,self.dist_tables,Q_to)
        J=self.swap_0_0(i,Cadit_v,Q_from)
        if J is not None :
            # print("使用交换操作")
            Cadit_v.reverse()
            # print("此时代理{0},交换代理{1}，逆置C={2}".format(i,J,Cadit_v))
        for u in Cadit_v:   # u:tuple      Q_to.positions:list[tuple]
            # vc
            if u in Q_to.positions:
                continue
            # ec
            element = next((x for x in Q_from.positions if x == u and Q_to.positions[Q_from.positions.index(x)] == Q_from.positions[i]), None)
            if element != None:
                continue
            Q_to[i] = u
            # print("代理{0}，选择顶点{1}".format(i,Q_to[i]))
            j = next((x for x in list(range(self.num_agents)) if x != i and Q_from[x] == u and Q_to[x] == NO_LOCATION),
                     None)

            if j != None:
                # print("\033[34m存在一个j\033[0m","当前Q_from={0},Q_to={1}".format(Q_from, Q_to))
                # Q_to[i]=NO_LOCATION
                if self.PIBT_SWAP(j, Q_from, Q_to) == None:continue

            # print(Q_to[J] == NO_LOCATION)
            #如果逆置的下一个点 and Q_to[J] == NO_LOCATION
            if u == Cadit_v[0] and J != None  and Q_to[J] == NO_LOCATION:
                Q_to[J]=Q_from[i]
                self.occupied_to[Q_from[i]]=J
                # print("\033[32m交换成功后，执行J跟随\033[0m",Q_to[J])

            return True
        Q_to[i] = Q_from[i]
        self.occupied_to[Q_from[i]]= i

        return None

    def swap_required_and_possible(self,i,C_0,occupied_from,Q_from,h_dict,Q_to):

        first_node=C_0
        agent_i = i
        config_from = Q_from
        print(occupied_from)
        # 如果存在一个代理j在i的C_0上，并使代理j当前的位置度<=2
        if len(get_neighbors(self.grid,first_node )) <= 2 and occupied_from[first_node] != NO_AGENT :
            if Q_to[occupied_from[first_node]] == NO_LOCATION:
                agent_j: int = occupied_from[first_node]
                if agent_j == agent_i:
                    return None
                print("符合交换，交换代理为{0}".format(agent_j))
                # necessity of the swap
                is_required = self.check_if_swap_required(agent_i, agent_j, config_from, h_dict,occupied_from)
                print("是否需要",is_required)

                # possibility of the swap
                is_possible = self.check_if_swap_possible(agent_i, agent_j, config_from,occupied_from)
                print("是否可能",is_possible)
                if is_required and is_possible:
                    return agent_j

                # case_c
                # for u in ai.v_now.neighbor:
                #     ak = occupied_now.get(u.id, None)
                #     if ak is None or C_next[i][0] == ak.v_now:
                #         continue
                #     if is_swap_required(ak.id, ai.id, ai.v_now, C_next[i][0]) and is_swap_possible(C_next[i][0],
                #                                                                                    ai.v_now):
                #         return ak

        for u in get_neighbors(self.grid,Q_from[agent_i]):
            agent_k = occupied_from[u]
            if agent_k > self.num_agents:continue
            if C_0 == Q_from[agent_k]:continue
            if len(get_neighbors(self.grid, C_0)) <= 2 : continue
            print("---------------agent_k_________")
            # if is_swap_required(ak.id, ai.id, ai.v_now, C_next[i][0]) and is_swap_possible(C_next[i][0],
            is_required_k=self.check_if_swap_required(agent_k, agent_i,config_from, h_dict,occupied_from)
            is_possible_k=self.check_if_swap_possible(agent_k, agent_i, config_from,occupied_from)
            if is_required_k and is_possible_k:
                return agent_k

        return None

    def check_if_swap_required(self,agent_i, agent_j, config_from, h_dict,occupied_from)-> bool:

        prev_node_i = config_from[agent_i]
        prev_node_j = config_from[agent_j]
        print("i={0},j={1}".format(agent_i,agent_j))
        print("当前配置",prev_node_i,prev_node_j)
        while True:

            next_node_i = prev_node_j # i的下一配置是j当前所在位置
            blocked=get_neighbors(self.grid,prev_node_i)
            blocked.append(prev_node_i)
            next_node_j =self.get_next_node(agent_j,prev_node_j, blocked) # j的下一配置（除去i当前配置）

            print("初步next配置",next_node_i,next_node_j)
            if next_node_j is None: # 如果j无下一配置（不包含等待），则需要交换
                return True
            # print(get_neighbors(self.grid,next_node_j))
            print("【当前需要下位置度】i.d={0}  j.d={1}".format(len(get_neighbors(self.grid,next_node_i)),len(get_neighbors(self.grid,next_node_j))))



            # 情况2.1：如果i下一配置的位置度为1时，就需要交换
            if len(get_neighbors(self.grid,next_node_i)) == 1:
                return True

            # 情况2.2：如果当i到达gi时，恰好j朝向目标的最近邻居点为gi，就需要交换
            if next_node_i == self.goals[agent_i]:
                nei_nodes_j = [next_node_j] + get_neighbors(self.grid, next_node_j)
                # self.rng.shuffle(Cadit_v)
                nei_nodes_j.sort(key=lambda point: self.dist_tables[agent_j].get(point))
                # nei_nodes_j = get_sorted_nei_nodes(agent_j, config_from, h_dict)
                nearest_nei_to_goal_j = nei_nodes_j[0]
                # print("判断j朝i.g",nearest_nei_to_goal_j,self.goals[agent_i])
                if nearest_nei_to_goal_j == self.goals[agent_i]:
                    print("满足情况2.2")
                    return True

            # 情况1：j下一配置位置度大于2时，不需要交换
            if len(get_neighbors(self.grid, next_node_j)) > 2:
                return False


            prev_node_i = next_node_i
            prev_node_j = next_node_j

    def check_if_swap_possible(self,agent_i, agent_j, config_from,occupied_from):
        """
        This is done by reversing the emulation direction; that is,
        continuously moving j to i’s location while moving i to another vertex.
        It stops in two cases:
            (i) The swap is possible when i’s location has a degree of more than two.
            (ii) The swap is impossible when i is on a vertex with degree of one.
        :return:
        """
        prev_node_i = config_from[agent_i]
        prev_node_j = config_from[agent_j]
        print("反转Cur配置",prev_node_i,prev_node_j)
        while True:



            next_node_j = prev_node_i
            # next_node_i = get_next_node(prev_node_i, blocked=[prev_node_j])
            blocked = get_neighbors(self.grid, prev_node_j)
            blocked.append(prev_node_j)
            next_node_i = self.get_next_node(agent_i,prev_node_i, blocked)

            print("反转Next配置",next_node_i,next_node_j)

            if next_node_i is None:
                return False
            print("【反转】i.d={0}  j.d={1}".format(len(get_neighbors(self.grid,next_node_i)),len(get_neighbors(self.grid,next_node_j))))

            if len(get_neighbors(self.grid,prev_node_i))  > 2 :
                return True


            prev_node_i = next_node_i
            prev_node_j = next_node_j

    def get_next_node(self,agent_id,node, blocked):
        nei_nodes = get_neighbors(self.grid,node)
        # nei_nodes.remove(node)

        for n in blocked:
            if n in nei_nodes:
                nei_nodes.remove(n)

        if len(nei_nodes) == 0:
            return None

        # if len(nei_nodes)>1:
        #     nei_nodes.sort(key=lambda point: self.dist_tables[agent_id].get(point))
        return nei_nodes[0]

    def swap_0_0(self,i,Candit_v_i,Q_from):
        C_0 = Candit_v_i[0]     # i代理的下一位置点
        if Q_from[i] == C_0:    # 如果代理 i 想要留在当前位置，则不需要交换
            return None

        j = self.occupied_from[C_0]
        if j < self.num_agents and j != i:
            possible=self.possible(i,j,Q_from,Candit_v_i)
            required=self.required(i,j,Q_from,Candit_v_i)
            if possible and required:
                return j
        return None
    def possible(self,i,j,Q_from,Candit_v_i):
        try:
            Cadit_v_j = [Q_from[j]] + get_neighbors(self.grid, Q_from[j])
            Cadit_v_j.sort(key=lambda point: self.dist_tables[j].get(point))
            if len(Cadit_v_j) > 2:return False  # j位置度>2,不需要交换
            if len(Candit_v_i) == 1:return True # i位置度=1，需要交换
            if Q_from[i]==self.goals[i] and Cadit_v_j[0]== Q_from[i]: #
                return True
        except:
           return False
    def required(self,i,j,Q_from,Candit_v_i):
        if len(Candit_v_i) > 2:return True
        if len(Candit_v_i) == 1:return False

    def swap_possible_and_required(self,i,Candit_v_i,Q_from):
        C_0=Candit_v_i[0]
        # print(C_0)
        # # 如果代理 i 想要留在当前位置，则不需要交换
        if Q_from[i] == C_0:
            return None
        """
        1、代理i即将占据的下一个位置，正好被j代理占据，导致i的位置度<=2 就需要进行检测器模拟
        2、是否需要交换：
            代理j的下一个位置如果恰好是当前i的位置[测j的升序]
        
        """
        # 通常的交换情况，参操作 case-a, b
        j = self.occupied_from[C_0]  # j
        if j == i : return None
        if j < self.num_agents and j!=i:
            required=self.is_swap_required(i, j, Q_from[i], Q_from[j])
            possible=self.is_swap_possible(Q_from[j], Q_from[i])
            # print("是否需要交换：{0}，能否成功交换：{1}".format(required,possible))
            if required and possible:
                return j

        # 清除操作的情况，参考 case-c
        for u in get_neighbors(self.occupied_from,Q_from[i]):
            k = self.occupied_from[u]
            if k > self.num_agents or C_0 == Q_from[k]:continue
            if self.is_swap_required(k, i, Q_from[i], C_0) and self.is_swap_possible(C_0, Q_from[i]):
                return k

        return None
    def is_swap_required(self, pusher, puller, v_pusher, v_puller):
        "i, j, Q_from[i], Q_from[j]"
        "k, i, Q_from[i], C_0"
        tmp = None
        while self.dist_tables[pusher].get(v_puller) < self.dist_tables[pusher].get(v_pusher):
            n = len(get_neighbors(self.occupied_from,v_puller))
            # 移除不需要移动的代理
            for u in get_neighbors(self.occupied_from,v_puller):
                a = self.occupied_from[u]
                if u == v_puller or (len(get_neighbors(self.occupied_from,u)) == 1
                                     and a < self.num_agents and self.goals[a] == u):
                    n -= 1
                else:
                    tmp = u
            if n > 2:
                return False  # 无需交换
            if n <= 0:
                break
            v_pusher = v_puller
            v_puller = tmp
        # 根据距离判断
        return (self.dist_tables[puller].get(v_pusher) < self.dist_tables[puller].get(v_puller)) and \
            (self.dist_tables[pusher].get(v_pusher) == 0 or
             self.dist_tables[pusher].get(v_puller) < self.dist_tables[pusher].get(v_pusher))
    def is_swap_possible(self,  v_pusher, v_puller):
        "Q_from[j], Q_from[i]"
        tmp = None
        while v_puller != v_pusher:  # 避免循环
            n = len(get_neighbors(self.occupied_from,v_puller))  # 计算可以拉取的相邻顶点数量
            for u in get_neighbors(self.occupied_from, v_puller):
                a = self.occupied_from[u]
                if u == v_pusher or (len(get_neighbors(self.occupied_from,u)) == 1 and a < self.num_agents and u == self.goals[a]):
                    n -= 1  # 如果 u 不能被拉取，则减少计数
                else:
                    tmp = u  # 如果 u 可以被拉取，则记录下来
            if n > 2:return True  # 如果有超过两个可以拉取的顶点，则可以交换
            if n <= 0:return False
            v_pusher = v_puller
            v_puller = tmp
        return False
    def PIBT(self,i,Q_from,Q_to):
        # 获取代理i的候选节点，并存储在列表C中 { int -> list=[Coord,..] }
        Cadit_v = [Q_from[i]]+get_neighbors(self.grid, Q_from[i])
        # Cadit_v = [x for x in Cadit_v if is_valid_coord(self.grid, x)]
        # 对Cadit_v重新排序，按照i的候选顶点距离i的目标顶点升序排列
        Cadit_v.sort(key=lambda point: self.dist_tables[i].get(point))
        # print("代理{0} {1}的候选顶点{2}".format(i, Q_from[i], Cadit_v))
        # 先遍历C，选出与已知Q_to中不发生冲突的候选顶点u
        for u in Cadit_v:   # u:tuple      Q_to.positions:list[tuple]
            # 检查是否发生顶点冲突、交换冲突
            if u in Q_to.positions:
                # print("发生顶点冲突")
                continue
            element=next((x for x in Q_from.positions if x==u and Q_to.positions[Q_from.positions.index(x)]==Q_from.positions[i]),None)
            if element != None:
                continue
            Q_to[i] = u
            # print("代理{0}，选择顶点{1}".format(i,Q_to[i]))
            # 判断如果存在另一个j and Q_from[j]=u and Q_to[i]=NO_LOCATION ，则执行PIBT(j)
            j=next((x for x in list(range(self.num_agents)) if x != i and Q_from[x] == u and Q_to[x] == NO_LOCATION),None)
            # if j != i and j in list(range(self.num_agents)) and Q_from[j] == u and Q_to[j] == NO_LOCATION:
            if j != None:
                # print(j,Q_from[j],u,Q_to[j])
                # print("\033[34m存在一个j\033[0m")
                if self.PIBT(j,Q_from,Q_to) == None:continue
            return True
        Q_to[i] = Q_from[i]

        return None

    "随机获取下一个配置"
    def configuration_generaotr(
        self, N: HighLevelNode, C: LowLevelNode
    ) -> Config | None:
        """
           生成一个新的配置，基于给定的高层次节点 N 和低层次节点 C。
           :param N: HighLevelNode 类型的对象，表示当前的高层次节点。
           :param C: LowLevelNode 类型的对象，表示当前的低层次节点。
           :return: Config 类型的对象，表示新的配置，如果生成失败则返回 None。
           """
        # 创建一个新配置，Config(positions=[(2147483647, 2147483647), (2147483647, 2147483647)])
        Q_to = Config([NO_LOCATION for _ in range(self.num_agents)])


        # 设置Q_to约束  【确定哪些代理】
        for k in range(C.depth):
            Q_to[C.who[k]] = C.where[k]    # 例如"1b"作为对下个代理的约束，同时也对代理1添加配置
        # print(Q_to)
        # 生成配置
        flg_success = True # 标记，生成配置是否成功
        for i in range(self.num_agents):
            v_i_from = N.Q[i]                   # 获取代理 i 在当前配置 N.Q 中的位置
            self.occupied_from[v_i_from] = i    # 将代理 i 在 occupied_from 缓存中标记为占用

            # 当没有约束时，通过随机选择动作来设置下一个位置
            if Q_to[i] == NO_LOCATION:
                a = self.rng.choice(ACTIONS)                  # 使用随机数生成器选择一个动作
                v = (v_i_from[0] + a[0], v_i_from[1] + a[1])  # 计算智能体 i 根据动作移动后的新位置
                if is_valid_coord(self.grid, v):              # 检查新位置是否有效
                    Q_to[i] = v                               # 如果有效，则更新智能体 i 的位置
                else:
                    flg_success = False  # 如果新位置无效，则设置 flg_success 为 False 并退出循环
                    break

            v_i_to: Coord = Q_to[i]  # 获取智能体 i 在新配置 Q_to 中的位置

            # 检查顶点冲突
            if self.occupied_to[v_i_to] != NO_AGENT:
                flg_success = False  # 如果新位置已被占用，则设置 flg_success 为 False 并退出循环
                break
            # 检查边冲突
            j = self.occupied_from[v_i_to]
            if j != NO_AGENT and j != i and Q_to[j] == v_i_from:
                flg_success = False  # 如果存在边冲突，则设置 flg_success 为 False 并退出循环
                break
            self.occupied_to[v_i_to] = i  # 将智能体 i 在 occupied_to 缓存中标记为占用

        # 清理 occupied_from 和 occupied_to 缓存
        for i in range(self.num_agents):
            v_i_from = N.Q[i]
            self.occupied_from[v_i_from] = NO_AGENT  # 清理 occupied_from 缓存
            v_i_next = Q_to[i]
            if v_i_next != NO_LOCATION:
                self.occupied_to[v_i_next] = NO_AGENT  # 清理 occupied_to 缓存

        return Q_to if flg_success else None  # 如果 flg_success 为 True，则返回新的配置 Q_to；否则返回 None
    def info(self, level: int, msg: str) -> None:
        if self.verbose < level:
            return
        logger.debug(f"{int(self.deadline.elapsed):4d}ms  {msg}")
