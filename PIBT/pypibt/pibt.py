import math

import numpy as np

from .dist_table import DistTable
from .mapf_utils import Config, Configs, Coord, Grid, get_neighbors


class PIBT:
    def __init__(self, grid: Grid, starts: Config, goals: Config, seed: int = 0):
        self.grid = grid
        self.starts = starts
        self.goals = goals
        self.N = len(self.starts)
        self.total_cost=0

        # distance table
        self.dist_tables = [DistTable(grid, goal) for goal in goals]

        # cache
        self.NIL = self.N  # meaning \bot
        self.NIL_COORD: Coord = self.grid.shape  # meaning \bot
        self.occupied_now = np.full(grid.shape, self.NIL, dtype=int)
        self.occupied_nxt = np.full(grid.shape, self.NIL, dtype=int)

        # used for tie-breaking
        self.rng = np.random.default_rng(seed)

    def funcPIBT(self, Q_from: Config, Q_to: Config, i: int) -> bool:
        # true -> valid, false -> invalid

        # get candidate next vertices
        C = [Q_from[i]] + get_neighbors(self.grid, Q_from[i])
        self.rng.shuffle(C)  # tie-breaking, randomize 打乱相同优先级的顺序，可使不同运行下有不同路径
        C = sorted(C, key=lambda u: self.dist_tables[i].get(u))
        # vertex assignment
        for v in C:
            # avoid vertex collision
            if self.occupied_nxt[v] != self.NIL:
                continue
            j = self.occupied_now[v]

            # avoid edge collision
            if j != self.NIL and Q_to[j] == Q_from[i]:
                continue

            # reserve next location
            Q_to[i] = v
            self.occupied_nxt[v] = i

            # priority inheritance (j != i due to the second condition)
            if (
                j != self.NIL
                and (Q_to[j] == self.NIL_COORD)
                and (not self.funcPIBT(Q_from, Q_to, j))
            ):
                continue

            return True

        # failed to secure node
        Q_to[i] = Q_from[i]
        self.occupied_nxt[Q_from[i]] = i
        return False

    def step(self, Q_from: Config, priorities: list[float]) -> Config:
        # setup
        N = len(Q_from)
        Q_to: Config = []
        for i, v in enumerate(Q_from):
            Q_to.append(self.NIL_COORD)
            self.occupied_now[v] = i

        # perform PIBT
        A = sorted(list(range(N)), key=lambda i: priorities[i], reverse=True)
        for i in A:
            if Q_to[i] == self.NIL_COORD:
                self.funcPIBT(Q_from, Q_to, i)

        # cleanup
        for q_from, q_to in zip(Q_from, Q_to):
            self.occupied_now[q_from] = self.NIL
            self.occupied_nxt[q_to] = self.NIL

        # caculate coste
        cost=self.get_coste(Q_from,Q_to)
        return Q_to,cost

    def get_coste(self,Q_from,Q_to):
        # e.g., \sum_i | not (Q_from[i] == Q_to[k] == g_i) |
        cost = 0
        for i in range(len(self.starts)):
            if not (self.goals[i] == Q_from[i] == Q_to[i]):
                x2, y2, x1, y1 = Q_to[i][0], Q_to[i][1], Q_from[i][0], Q_from[i][1]
                coste = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                # print(coste, Q_from[i], Q_to[i])
                cost += coste

        return cost

    def run(self, max_timestep: int = 1000) -> Configs:
        # define priorities

        priorities: list[float] = []
        for i in range(self.N):
            priorities.append(self.dist_tables[i].get(self.starts[i]) / self.grid.size)

        # main loop, generate sequence of configurations
        configs = [self.starts]
        while len(configs) <= max_timestep:
            # obtain new configuration
            Q,cost = self.step(configs[-1], priorities)

            temp_config
            configs.append(Q)

            # update priorities & goal check
            flg_fin = True
            for i in range(self.N):
                if Q[i] != self.goals[i]:
                    flg_fin = False
                    priorities[i] += 1
                else:
                    priorities[i] -= np.floor(priorities[i])
            if flg_fin:
                break  # goal
            self.total_cost +=cost

        return configs,self.total_cost
