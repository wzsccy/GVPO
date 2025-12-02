# 开发时间：2024/8/21 14:46
# 开发语言：Python
'''---------------------------------------'''
import argparse
import glob
import os
import time
from pathlib import Path

from MA_CBS.visualize import Animation
from MA_CBS.MA_CBS import MACBSSolver_four,MACBSSolver_eight
from MA_CBS.single_agent_planner import get_sum_of_cost,get_sum_of_cost_eight

SOLVER = " MACBS "


class MACBS_four():
    def __init__(self,dl,start_time,my_map,starts,goals):
        self.start_time, self.my_map, self.starts, self.goals = start_time, my_map, starts, goals
        self.map = [[not value for value in row] for row in self.my_map]  # True-False
        self.paths, self.cost = self.run(dl,self.map, self.starts, self.goals, self.start_time)

    def run(self,dl,map,starts,goals,start_time):
        solver = MACBSSolver_four(map, starts, goals)
        paths = solver.find_solution(dl,start_time)

        if paths == 'over_time':
            return 'over_time',0

        cost = get_sum_of_cost(paths)
        return paths,cost

    def get_result(self):

        return self.paths,self.cost

class MACBS_eight():
    def __init__(self, dl,start_time,my_map, starts, goals):
        self.start_time,self.my_map, self.starts, self.goals = start_time,my_map, starts, goals
        self.map = [[not value for value in row] for row in self.my_map]  # True-False
        self.paths, self.cost = self.run(dl,self.map, self.starts, self.goals,self.start_time)

    def run(self, dl,map, starts, goals , start_time):
        solver = MACBSSolver_eight(map, starts, goals)
        paths = solver.find_solution(dl,start_time)

        if paths == 'over_time':
            return 'over_time',0
        cost = get_sum_of_cost_eight(paths)
        return paths, cost

    def get_result(self):
        return self.paths, self.cost


if __name__ == "__main__":
    map=[[True,True,True,True],
         [True,True,True,True],
         [True,True,True,True]]
    starts = [(1,0),(0,2)]
    goals = [(2,2),(1,1)]
    a1 = MACBS_eight(map,starts,goals)
    c,d=a1.get_result()
    print(c,d)
