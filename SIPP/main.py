from util.movingai_parser import *
from search_functions.AnytimeSIPP import AnytimeSIPP
from multiagent.MAPF import MAPF
from entity.Agent import Agent
from search_functions.SIPP import SIPP
from search_functions.WSIPP import WdSIPP
from search_functions.WSIPP import WrSIPP
from search_functions.FocalSIPP import FocalSIPP
from entity.Map import TimestampSearchMap
from search_functions.AStar import AStar
from util.draw import Draw
from IPython.display import Image
from testing.run_scens import run_scens
from entity.Statistics import Statistics
import argparse
import random


def manhattan_distance(i1, j1, i2, j2):
    di = abs(i2 - i1)
    dj = abs(j2 - j1)
    return di + dj


def optional_heuristic(node):
    return node.i


if __name__ == '__main__':
    random.seed(243)
    print(1111)
    parser = argparse.ArgumentParser(description='Multi-agent path finding using different algorithms')
    print(1111)
    parser.add_argument('map_filename', type=str,
                        help='.map file containing map description',default="1111")
    print(1111)
    parser.add_argument('scen_filename', type=str,
                        help='.map.scen file containing scenarios description',default="../data/arena.map.scen")
    parser.add_argument('agents_n', type=int,default=50,
                        help='number of agents for a single task')
    parser.add_argument('algorithm', type=str,default="wdsipp",
                        help='name of algorithm for path planning (astar/sipp/wdsipp/wrsipp/focal/anytime)')
    parser.add_argument('--weight', type=float, default=1.25,
                        help='floating point to use in suboptimal algorithms')
    parser.add_argument('--tasks_n', type=int, default=5,
                        help='number of random tasks to generate and run')
    parser.add_argument('--time_limit', type=int, default=10,
                        help='time limit for a single task in seconds')
    parser.add_argument('--gif_filename', type=str,default="p1",
                        help='path to save a .gif file representing one of the tasks in motion')
    print(1111)

    args = parser.parse_args()

    print(1111)



    algos = {'astar': AStar(manhattan_distance),
             'sipp': SIPP(manhattan_distance),
             'wdsipp': WdSIPP(manhattan_distance, args.weight),
             'wrsipp': WrSIPP(manhattan_distance, args.weight),
             'focal': FocalSIPP(manhattan_distance, optional_heuristic, args.weight),
             'anytime': AnytimeSIPP(manhattan_distance)}

    agents_cnt = args.agents_n
    search_function = algos[args.algorithm]

    stats = Statistics()
    print(args.map_filename,'111')
    run_scens(args.map_filename, args.scen_filename, agents_cnt, search_function, stats, args.tasks_n,
              args.gif_filename, args.time_limit)

    print(stats)
