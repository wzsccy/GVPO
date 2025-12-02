#!/usr/bin/python
import argparse
import glob
from pathlib import Path

from ML_EECBS_4.eecbs import EECBSSolver

# from visualize import Animation
from ML_EECBS_4.single_agent_planner import get_sum_of_cost

SOLVER = "EECBS"

class ML_EECBS_4():
    def __init__(self,my_map, starts, goals):
        self.my_map, self.starts, self.goals = my_map, starts, goals

    def run(self,dl,start_time):
        cbs = EECBSSolver(self.my_map, self.starts, self.goals)
        paths = cbs.find_solution(dl,start_time)
        cost = get_sum_of_cost(paths)
        return paths,cost


