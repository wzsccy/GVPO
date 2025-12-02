from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
import numpy as np
from loguru import logger
from .dist_table import DistTable
from .mapf_utils import (
    Config,
    Configs,
    Coord,
    Deadline,
    Grid,
    get_neighbors,


    is_valid_coord,
)

NO_AGENT: int = np.iinfo(np.int32).max # 符号常量：表示该位置没有被代理占用
NO_LOCATION: Coord = (np.iinfo(np.int32).max, np.iinfo(np.int32).max) # 符号常量：表示无位置



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
        time_limit_ms: int = 3000,
        deadline: Deadline | None = None,
        flg_star: bool = True,
        seed: int = 0, # 猜测是设定seed个随机数,估计要限定
        verbose: int = 1, # 输出的详细程度
        fl00: int = 0,
    ) -> Configs:
        # set problem
        self.num_agents: int = len(starts)
        self.grid: Grid = grid
        self.starts: Config = starts
        self.goals: Config = goals
        self.deadline: Deadline = (
            deadline if deadline is not None else Deadline(time_limit_ms)
        )
        self.seed=seed
        # set hyper parameters
        self.flg_star: bool = flg_star
        self.rng: np.random.Generator = np.random.default_rng(seed=seed) # 随机数
        self.verbose = verbose # 控制输出信息的多少
        self.fl00=fl00
        return self._solve()

    def _solve(self) -> Configs:
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
                cands = [v] + get_neighbors(self.grid, v)
                # print(i,cands)
                self.rng.shuffle(cands)
                for u in cands:
                    N.tree.append(C.get_child(i, u)) # 将i的候选点插入树中

            "此处需要将之替换为PIBT算法来作为配置生成器"
            Q_to = self.CG_PIBT(N,C)

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
            temp=00000
            # self.info(1, "detected unsolvable instance")
        else:
            self.info(1, "failure due to timeout")
        return self.backtrack(N_goal)

    def calculate_euclidean_distance_sum(self,points):
        total_distance = 0
        for i in range(len(points) - 1):
            # 计算相邻两点之间的欧式距离
            distance = math.sqrt((points[i + 1][0] - points[i][0]) ** 2 + (points[i + 1][1] - points[i][1]) ** 2)
            # 累加到总距离中
            total_distance += distance
        return total_distance

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
    def PIBT_SWAP(self,i,Q_from,Q_to):
        Cadit_v = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        # self.rng.shuffle(Cadit_v)
        Cadit_v.sort(key=lambda point: self.dist_tables[i].get(point))
        # print("代理{0} {1}的候选顶点{2}".format(i, Q_from[i], Cadit_v))
        "补充：1、需交换代理j; 2、若存在j则逆置C"
        J = self.swap_0_0(i, Cadit_v, Q_from)
        # J = self.swap_possible_and_required(i, Cadit_v, Q_from)
        if J != None and J != i :
            print("使用交换操作")
            Cadit_v=Cadit_v[::-1]       # 逆置
            # print("此时代理{0},交换代理{1}，逆置C={2}".format(i,J,Cadit_v))
        for u in Cadit_v:   # u:tuple      Q_to.positions:list[tuple]
            # 检查是否发生顶点冲突、交换冲突
            if u in Q_to.positions:
                continue
            element = next((x for x in Q_from.positions if x == u and Q_to.positions[Q_from.positions.index(x)] == Q_from.positions[i]), None)
            if element != None:
                continue
            Q_to[i] = u
            # print("代理{0}，选择顶点{1}".format(i,Q_to[i]))
            j = next((x for x in list(range(self.num_agents)) if x != i and Q_from[x] == u and Q_to[x] == NO_LOCATION),
                     None)
            if j != None:
                # print("\033[34m存在一个j\033[0m")
                if self.PIBT_SWAP(j, Q_from, Q_to) == None: continue

            # 如果逆置的下一个点
            if u == Cadit_v[0] and J != None and Q_to[J] == NO_LOCATION :
                Q_to[J]=Q_from[i]

            return True
        Q_to[i] = Q_from[i]

        return None
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
            if len(Cadit_v_j) > 3:return False  # j位置度>2,不需要交换
            if len(Candit_v_i) == 2:return True # i位置度=1，需要交换
            if Q_from[i]==self.goals[i] and Cadit_v_j[0]== Q_from[i]: #
                return True
        except:
           return False
    def required(self,i,j,Q_from,Candit_v_i):
        if len(Candit_v_i) > 3:return True
        if len(Candit_v_i) == 2:return False

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


    "获取下一个配置"
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
