# 开发时间：2024/2/19 14:10
# 开发语言：Python
'''---------------------------------------'''
import datetime
import pickle
import random
import time
import os
import numpy as np
import torch.cuda

"""CONFLICT BASED SEARCH - PYTHON CODE
AIFA MULTI AGENT PATH FINDING"""
from CBS_TA.allocate import Allocate
import math
import sys
sys.path.insert(0, '../') #优先导入项目内模块
import argparse
import yaml
from math import fabs #浮点绝对值
from itertools import combinations #导入“组合”
from copy import deepcopy
from CBS_TA.a_star import AStar    #import the code for a* algorithm

'''位置类'''
class Location(object):
    # 位置初始化
    def __init__(self, x=-1, y=-1):
        self.x = x  # 位置的x坐标
        self.y = y  # 位置的y坐标

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y  # 检查位置是否相等（x和y坐标均相等）

    # 查看坐标
    def __str__(self):
        return str((self.x, self.y))  # 以字符串形式表示位置（(x, y)）
'''状态类'''
class State(object):
    def __init__(self, time, location): # 初始化，传入时间步长和位置
        self.time = time
        self.location = location
    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location.x) + str(self.location.y))
    def is_equal_except_time(self, state):
        return self.location == state.location
    def __str__(self):
        return str((self.time, self.location.x, self.location.y))
'''冲突类'''
class Conflict(object):
    VERTEX = 1
    EDGE = 2
    def __init__(self):
        self.time = -1
        self.type = -1

        self.agent_1 = ''
        self.agent_2 = ''

        self.location_1 = Location()   # 实例化一个位置对象
        self.location_2 = Location()

    def __str__(self):
        return '(' + str(self.time) + ', ' + self.agent_1 + ', ' + self.agent_2 + \
             ', '+ str(self.location_1) + ', ' + str(self.location_2) + ')'
'''顶点约束类'''
class VertexConstraint(object):
    def __init__(self, time, location):
        self.time = time
        self.location = location

    def __eq__(self, other):
        return self.time == other.time and self.location == other.location
    def __hash__(self):
        return hash(str(self.time)+str(self.location))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location) + ')'
'''边约束类'''
class EdgeConstraint(object):
    def __init__(self, time, location_1, location_2):
        self.time = time
        self.location_1 = location_1
        self.location_2 = location_2
    def __eq__(self, other):
        return self.time == other.time and self.location_1 == other.location_1 \
            and self.location_2 == other.location_2
    def __hash__(self):
        return hash(str(self.time) + str(self.location_1) + str(self.location_2))
    def __str__(self):
        return '(' + str(self.time) + ', '+ str(self.location_1) +', '+ str(self.location_2) + ')'
'''约束类'''
class Constraints(object):
    def __init__(self):
        self.vertex_constraints = set()   #设置顶点约束、边约束为集合类型
        self.edge_constraints = set()

    def add_constraint(self, other):
        self.vertex_constraints |= other.vertex_constraints  #原约束与其他合并
        self.edge_constraints |= other.edge_constraints

    def __str__(self):
        return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
            "EC: " + str([str(ec) for ec in self.edge_constraints])
'''环境类'''
class Environment(object):

    def __init__(self, dimension, agents, tasks, obstacles):
        self.dimension = dimension
        self.obstacles = obstacles
        self.agents = agents
        self.tasks = tasks

        "只运行CBS时，该行代码可注释"
        self.data_transform()  # 数据转换（原Lacam场景数据转换为CBS适应数据）

        self.pairs = []  # 存储最优的分配序列
        self.task_dict={}
        self.make_task_dict()
        self.agent_dict = {}
        self.make_agent_dict() # 将agents信息存入agent_dict


        self.constraints = Constraints()
        self.constraint_dict = {}  # 建一个约束字典

        self.a_star = AStar(self)   #初始化，a_star = AStar(Environment())

    def data_transform(self):
        a_temp,t_temp=[],[]
        for i,value in self.agents.items():
            element={'name':i,'start':list(value)}
            a_temp.append(element)
        for i,value in self.tasks.items():
            element={'name':i,'release_time':value[2],'Loading':list(value[0]),'Unloading':list(value[1])}
            t_temp.append(element)
        self.agents,self.tasks=a_temp,t_temp



    '''创建存储所有代理的字典'''

    # 对每个代理的开始节点和目标节点的状态进行初始化,并将每个代理以{代理名：{start:开始节点状态,goal:目标节点状态}}存入agent_dict

    '''创建存储任务字典'''
    def make_task_dict(self):
        for task in self.tasks:
            release_time = task['release_time']
            Loading=State(0,Location(task['Loading'][0],task['Loading'][1]))
            Unloading=State(0,Location(task['Unloading'][0],task['Unloading'][1]))
            self.task_dict.update(
                {task['name']:{'release_time':release_time, 'Loading':Loading ,'Unloading':Unloading}}
            )

    def install_virtual_agents(self,pairs):
        for pair in pairs:
            if len(pair) == 2 :
                if pair[1] == -1:  # 代理不带任务
                    del self.agent_dict[pair[0]]
                else:
                    self.agent_dict[pair[0]]['task'] = self.task_dict[pair[1]]['Loading']
                    self.agent_dict[pair[0]]['goal'] = self.task_dict[pair[1]]['Unloading']
                    self.agent_dict[pair[0]]['release_time'] = self.task_dict[pair[1]]['release_time']

            elif len(pair) > 2 :
                self.agent_dict[pair[0]]['task'] = self.task_dict[pair[1]]['Loading']
                self.agent_dict[pair[0]]['goal'] = self.task_dict[pair[1]]['Unloading']
                self.agent_dict[pair[0]]['release_time'] = self.task_dict[pair[1]]['release_time']
                for i in range(1,len(pair)-1):
                    if i == 1:
                        self.agent_dict.update({pair[0] + self.cout: {'start': self.task_dict[pair[i]]['Unloading']}})  # 创建新代理起点
                        self.agent_dict[pair[0] + self.cout]['task'] = self.task_dict[pair[i+1]]['Loading']  # 为新代理起点装取货点和交货点
                        self.agent_dict[pair[0] + self.cout]['goal'] = self.task_dict[pair[i+1]]['Unloading']
                        self.agent_dict[pair[0] + self.cout]['release_time'] = self.task_dict[pair[i+1]]['release_time']
                    else:
                        self.agent_dict.update({pair[0] + 2 * (i-1) * self.cout: {'start': self.task_dict[pair[i]]['Unloading']}})  # 创建新代理起点
                        self.agent_dict[pair[0] + 2 * (i-1) * self.cout]['task'] = self.task_dict[pair[i + 1]]['Loading']  # 为新代理起点装取货点和交货点
                        self.agent_dict[pair[0] + 2 * (i-1) * self.cout]['goal'] = self.task_dict[pair[i + 1]]['Unloading']
                        self.agent_dict[pair[0] + 2 * (i-1) * self.cout]['release_time'] =self.task_dict[pair[i + 1]]['release_time']
    # 旧算法，任务数最大值为10
    def install_virtual_agents0(self,pairs):
        for pair in pairs:
            if len(pair) == 2:  # 代理完成一个任务
                if pair[1] == -1:
                    del self.agent_dict[pair[0]]
                else:
                    a, b = pair[0], pair[1]
                    self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                    self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                    self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']
            elif len(pair) == 3:  # 代理完成2个任务
                a, b, c = pair[0], pair[1], pair[2]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']
                # 为任务点虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

            elif len(pair) == 4:  # 代理完成3个任务
                a, b, c, d = pair[0], pair[1], pair[2], pair[3]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']
                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']
            elif len(pair) == 5:  # 代理完成4个任务
                a, b, c, d, e = pair[0], pair[1], pair[2], pair[3], pair[4]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']
            elif len(pair) == 6:  # 代理完成5个任务
                a, b, c, d, e, f = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

            elif len(pair) == 7:  # 代理完成6个任务
                a, b, c, d, e, f, g = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * +self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

                # 为任务点5虚拟转化为代理点
                self.agent_dict.update({a + 8 * self.cout: {'start': self.task_dict[f]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 8 * self.cout]['task'] = self.task_dict[g]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 8 * self.cout]['goal'] = self.task_dict[g]['Unloading']
                self.agent_dict[a + 8 * self.cout]['release_time'] = self.task_dict[g]['release_time']

            elif len(pair) == 8:  # 代理完成7个任务
                a, b, c, d, e, f, g, h = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[7]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

                # 为任务点5虚拟转化为代理点
                self.agent_dict.update({a + 8 * self.cout: {'start': self.task_dict[f]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 8 * self.cout]['task'] = self.task_dict[g]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 8 * self.cout]['goal'] = self.task_dict[g]['Unloading']
                self.agent_dict[a + 8 * self.cout]['release_time'] = self.task_dict[g]['release_time']

                # 为任务点6虚拟转化为代理点
                self.agent_dict.update({a + 10 * self.cout: {'start': self.task_dict[g]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 10 * self.cout]['task'] = self.task_dict[h]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 10 * self.cout]['goal'] = self.task_dict[h]['Unloading']
                self.agent_dict[a + 10 * self.cout]['release_time'] = self.task_dict[h]['release_time']

            elif len(pair) == 9:  # 代理完成8个任务
                a, b, c, d, e, f, g, h, i = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[7], \
                pair[8]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

                # 为任务点5虚拟转化为代理点
                self.agent_dict.update({a + 8 * self.cout: {'start': self.task_dict[f]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 8 * self.cout]['task'] = self.task_dict[g]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 8 * self.cout]['goal'] = self.task_dict[g]['Unloading']
                self.agent_dict[a + 8 * self.cout]['release_time'] = self.task_dict[g]['release_time']

                # 为任务点6虚拟转化为代理点
                self.agent_dict.update({a + 10 * self.cout: {'start': self.task_dict[g]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 10 * self.cout]['task'] = self.task_dict[h]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 10 * self.cout]['goal'] = self.task_dict[h]['Unloading']
                self.agent_dict[a + 10 * self.cout]['release_time'] = self.task_dict[h]['release_time']

                # 为任务点7虚拟转化为代理点
                self.agent_dict.update({a + 12 * self.cout: {'start': self.task_dict[h]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 12 * self.cout]['task'] = self.task_dict[i]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 12 * self.cout]['goal'] = self.task_dict[i]['Unloading']
                self.agent_dict[a + 12 * self.cout]['release_time'] = self.task_dict[i]['release_time']
            elif len(pair) == 10:  # 代理完成9个任务
                a, b, c, d, e, f, g, h, i, j = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[7], \
                pair[8], pair[9]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

                # 为任务点5虚拟转化为代理点
                self.agent_dict.update({a + 8 * self.cout: {'start': self.task_dict[f]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 8 * self.cout]['task'] = self.task_dict[g]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 8 * self.cout]['goal'] = self.task_dict[g]['Unloading']
                self.agent_dict[a + 8 * self.cout]['release_time'] = self.task_dict[g]['release_time']

                # 为任务点6虚拟转化为代理点
                self.agent_dict.update({a + 10 * self.cout: {'start': self.task_dict[g]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 10 * self.cout]['task'] = self.task_dict[h]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 10 * self.cout]['goal'] = self.task_dict[h]['Unloading']
                self.agent_dict[a + 10 * self.cout]['release_time'] = self.task_dict[h]['release_time']

                # 为任务点7虚拟转化为代理点
                self.agent_dict.update({a + 12 * self.cout: {'start': self.task_dict[h]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 12 * self.cout]['task'] = self.task_dict[i]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 12 * self.cout]['goal'] = self.task_dict[i]['Unloading']
                self.agent_dict[a + 12 * self.cout]['release_time'] = self.task_dict[i]['release_time']

                # 为任务点8虚拟转化为代理点
                self.agent_dict.update({a + 14 * self.cout: {'start': self.task_dict[i]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 14 * self.cout]['task'] = self.task_dict[j]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 14 * self.cout]['goal'] = self.task_dict[j]['Unloading']
                self.agent_dict[a + 14 * self.cout]['release_time'] = self.task_dict[j]['release_time']
            elif len(pair) == 11:  # 代理完成10个任务
                a, b, c, d, e, f, g, h, i, j, k = pair[0], pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[
                    7], pair[8], pair[9], pair[10]
                self.agent_dict[a]['task'] = self.task_dict[b]['Loading']
                self.agent_dict[a]['goal'] = self.task_dict[b]['Unloading']
                self.agent_dict[a]['release_time'] = self.task_dict[b]['release_time']

                # 为任务点1虚拟转化为代理点
                self.agent_dict.update({a + self.cout: {'start': self.task_dict[b]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + self.cout]['task'] = self.task_dict[c]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + self.cout]['goal'] = self.task_dict[c]['Unloading']
                self.agent_dict[a + self.cout]['release_time'] = self.task_dict[c]['release_time']

                # 为任务点2虚拟转化为代理点
                self.agent_dict.update({a + 2 * self.cout: {'start': self.task_dict[c]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 2 * self.cout]['task'] = self.task_dict[d]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 2 * self.cout]['goal'] = self.task_dict[d]['Unloading']
                self.agent_dict[a + 2 * self.cout]['release_time'] = self.task_dict[d]['release_time']

                # 为任务点3虚拟转化为代理点
                self.agent_dict.update({a + 4 * self.cout: {'start': self.task_dict[d]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 4 * self.cout]['task'] = self.task_dict[e]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 4 * self.cout]['goal'] = self.task_dict[e]['Unloading']
                self.agent_dict[a + 4 * self.cout]['release_time'] = self.task_dict[e]['release_time']

                # 为任务点4虚拟转化为代理点
                self.agent_dict.update({a + 6 * self.cout: {'start': self.task_dict[e]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 6 * self.cout]['task'] = self.task_dict[f]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 6 * self.cout]['goal'] = self.task_dict[f]['Unloading']
                self.agent_dict[a + 6 * self.cout]['release_time'] = self.task_dict[f]['release_time']

                # 为任务点5虚拟转化为代理点
                self.agent_dict.update({a + 8 * self.cout: {'start': self.task_dict[f]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 8 * self.cout]['task'] = self.task_dict[g]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 8 * self.cout]['goal'] = self.task_dict[g]['Unloading']
                self.agent_dict[a + 8 * self.cout]['release_time'] = self.task_dict[g]['release_time']

                # 为任务点6虚拟转化为代理点
                self.agent_dict.update({a + 10 * self.cout: {'start': self.task_dict[g]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 10 * self.cout]['task'] = self.task_dict[h]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 10 * self.cout]['goal'] = self.task_dict[h]['Unloading']
                self.agent_dict[a + 10 * self.cout]['release_time'] = self.task_dict[h]['release_time']

                # 为任务点7虚拟转化为代理点
                self.agent_dict.update({a + 12 * self.cout: {'start': self.task_dict[h]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 12 * self.cout]['task'] = self.task_dict[i]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 12 * self.cout]['goal'] = self.task_dict[i]['Unloading']
                self.agent_dict[a + 12 * self.cout]['release_time'] = self.task_dict[i]['release_time']

                # 为任务点8虚拟转化为代理点
                self.agent_dict.update({a + 14 * self.cout: {'start': self.task_dict[i]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 14 * self.cout]['task'] = self.task_dict[j]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 14 * self.cout]['goal'] = self.task_dict[j]['Unloading']
                self.agent_dict[a + 14 * self.cout]['release_time'] = self.task_dict[j]['release_time']

                # 为任务点9虚拟转化为代理点
                self.agent_dict.update({a + 16 * self.cout: {'start': self.task_dict[j]['Unloading']}})  # 创建新代理起点
                self.agent_dict[a + 16 * self.cout]['task'] = self.task_dict[k]['Loading']  # 为新代理起点装取货点和交货点
                self.agent_dict[a + 16 * self.cout]['goal'] = self.task_dict[k]['Unloading']
                self.agent_dict[a + 16 * self.cout]['release_time'] = self.task_dict[k]['release_time']
    def make_agent_dict(self):
        for agent in self.agents:
            start_state = State(0, Location(agent['start'][0], agent['start'][1]))
            self.agent_dict.update(
                {agent['name']: {'start': start_state}}
            )
        self.allocate = Allocate(self.agent_dict, self.task_dict, self.dimension, self.obstacles)
        pairs = self.allocate.allocating_task()
        self.pairs=pairs
        self.cout = len(self.agent_dict)  # 全局变量self.cout
        self.install_virtual_agents(self.pairs)
    '''获取当前状态的八个方向的所有合法邻居节点'''
    def get_neighbors(self, state):
        neighbors = []   # 存储该节点的所有合法邻居节点

        # Wait action
        n = State(state.time + 1, state.location) # 等待动作，要算时长
        if self.state_valid(n):
            neighbors.append(n)
        # Up action
        n = State(state.time + 1, Location(state.location.x, state.location.y+1))  # 向上移动后的节点
        if self.state_valid(n) and self.transition_valid(state, n): # 若节点既满足状态合法性、移动合法性，就存入列表中
            neighbors.append(n)
        # Down action
        n = State(state.time + 1, Location(state.location.x, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Left action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # Right action
        n = State(state.time + 1, Location(state.location.x+1, state.location.y))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)

        '''补左上、左下、右上、右下'''
        # lu action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y + 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # ld action
        n = State(state.time + 1, Location(state.location.x-1, state.location.y - 1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # ru action
        n = State(state.time + 1, Location(state.location.x +1, state.location.y+1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)
        # rd action
        n = State(state.time + 1, Location(state.location.x + 1, state.location.y-1))
        if self.state_valid(n) and self.transition_valid(state, n):
            neighbors.append(n)

        return neighbors
    '''获取第一个冲突'''
    def get_first_conflict(self, solution):
        max_t = max([len(plan) for plan in solution.values()])  # 得出所有路径规划中最大的时间
        """
        举例：
        solution.values = {
            'plan1': [(1, 2), (3, 4), (5, 6),(4,3)],
            'plan2': [(7, 8), (9, 10), (11, 12)],
            'plan3': [(13, 14), (15, 16), (17, 18)],
            }
        [len(plan) for plan in solution.values()]=[4,3,3]   # 列表推导式：遍历solution中所有的plan,并以列表存储所有plan的长度
        """
        result = Conflict()


        #在每个t时，对两两代理进行冲突判断
        for t in range(max_t):

            # 在t时，返回出顶点的冲突
            for agent_1, agent_2 in combinations(solution.keys(), 2): # 将所有代理都进行一次一一组合,
                """               
                solution = {
                    'agent1': 'plan1',
                    'agent2': 'plan2',
                    'agent3': 'plan3',
                    }
                combinations(solution.keys(), 2)=('agent1', 'agent2'), ('agent1', 'agent3'), ('agent2', 'agent3')              
                """
                state_1 = self.get_state(agent_1, solution, t)  #获取代理1在t时的状态(坐标)
                state_2 = self.get_state(agent_2, solution, t)
                if state_1.is_equal_except_time(state_2): # 若 两代理在当前时刻的状态相同,就是冲突
                    result.time = t
                    result.type = Conflict.VERTEX
                    result.location_1 = state_1.location
                    result.location_2 = state_2.location
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    return result

            # 在t时，返回出边的冲突
            for agent_1, agent_2 in combinations(solution.keys(), 2):
                state_1a = self.get_state(agent_1, solution, t)
                state_1b = self.get_state(agent_1, solution, t+1)

                state_2a = self.get_state(agent_2, solution, t)
                state_2b = self.get_state(agent_2, solution, t+1)

                if state_1a.is_equal_except_time(state_2b) and state_1b.is_equal_except_time(state_2a):
                    result.time = t
                    result.type = Conflict.EDGE
                    result.agent_1 = agent_1
                    result.agent_2 = agent_2
                    result.location_1 = state_1a.location
                    result.location_2 = state_1b.location
                    return result
        return False # 返回False，表明没有冲突
    '''通过冲突创建约束'''
    def create_constraints_from_conflict(self, conflict):
        constraint_dict = {}
        if conflict.type == Conflict.VERTEX:  # 若是顶点冲突
            v_constraint = VertexConstraint(conflict.time, conflict.location_1)#实例一个顶点约束对象
            constraint = Constraints()
            constraint.vertex_constraints |= {v_constraint}   # 把当前v约束合并到constraint的顶点约束集合中
            constraint_dict[conflict.agent_1] = constraint    # 把约束存到constraint_dict对应代理中
            constraint_dict[conflict.agent_2] = constraint


        elif conflict.type == Conflict.EDGE: # 若是边冲突
            constraint1 = Constraints()
            constraint2 = Constraints()

            e_constraint1 = EdgeConstraint(conflict.time, conflict.location_1, conflict.location_2)
            e_constraint2 = EdgeConstraint(conflict.time, conflict.location_2, conflict.location_1)

            constraint1.edge_constraints |= {e_constraint1}
            constraint2.edge_constraints |= {e_constraint2}

            constraint_dict[conflict.agent_1] = constraint1
            constraint_dict[conflict.agent_2] = constraint2

        return constraint_dict
    '''获取某代理在t时的状态'''
    def get_state(self, agent_name, solution, t):
        if t < len(solution[agent_name]):
            return solution[agent_name][t] #返回在t时的状态坐标
        else:
            return solution[agent_name][-1] # 返回方案中最后一个坐标,t为最后时刻
    '''判断状态合法性'''
    # 若满足条件：1、状态的x坐标、y坐标均在地图范围内；2、该状态的顶点约束不在原约束的顶点约束集合中；3、该状态的节点不是障碍物
    def state_valid(self, state):
        return state.location.x >= 0 and state.location.x < self.dimension[0] \
            and state.location.y >= 0 and state.location.y < self.dimension[1] \
            and VertexConstraint(state.time, state.location) not in self.constraints.vertex_constraints \
            and (state.location.x, state.location.y) not in self.obstacles

    '''判断移动合法性'''
    # 若满足条件：该边不在原约束的边约束集合中
    def transition_valid(self, state_1, state_2):
        return EdgeConstraint(state_1.time, state_1.location, state_2.location) not in self.constraints.edge_constraints

    '''获取起点到任务点的启发式估计值h'''
    def admissible_heuristic1(self, state, agent_name):

        temp=self.agent_dict[agent_name]['task']
        return fabs(state.location.x - temp.location.x) + fabs(state.location.y - temp.location.y)

    '''获取任务点到终点点的启发式估计值h'''
    def admissible_heuristic2(self, state, agent_name):

        temp=self.agent_dict[agent_name]['goal']
        return fabs(state.location.x - temp.location.x) + fabs(state.location.y - temp.location.y)


    '''判断当前状态是否处于目标状态'''
    def is_at_goal(self, state, agent_name):
        goal_state = self.agent_dict[agent_name]["goal"]
        return state.is_equal_except_time(goal_state)

    '''补充----判断当前状态是否处于任务点的状态'''
    def is_at_task(self, state, agent_name):
        task_state = self.agent_dict[agent_name]["task"]
        return state.is_equal_except_time(task_state)

    def sol_merge(self,sol,pairs):
        for pair in pairs:
            # 初始化一个列表来存储所有需要删除的索引
            indices_to_delete = []
            if len(pair)>2:
                # 从第一个任务开始，直到任务数量减一
                for i in range(len(pair) - 2):
                    # 获取虚拟代理点的索引号
                    if i == 0:
                        index = int(pair[0] + self.cout)
                    else:
                        index = int(pair[0] + i*2*self.cout)
                    # 删除sol方案中的虚拟代理点搜索路径中的第一个元素
                    sol[index].pop(0)
                    # 将虚拟代理点剩下的路径全部接到原代理点路径后面
                    sol[pair[0]].extend(sol[index])
                    # 将当前索引添加到要删除的索引列表中
                    indices_to_delete.append(index)
                # 删除所有标记为要删除的索引
                for index in indices_to_delete:
                    del sol[index]
        # for i in sol.keys():
        #     print("692----",i)
        return  sol
    def sol_merge1(self,sol,pairs):
        for pair in pairs:
            if len(pair)==3:        #代理完成2个任务
                index=int(pair[0]+self.cout) #获取虚拟代理点索引号
                sol[index].pop(0) #删除sol方案中的虚拟代理点搜索路径中的第一个元素
                sol[pair[0]].extend(sol[index])#将虚拟代理点剩下的路径全部接到原代理点路径后面
                del sol[index] # 删除sol方案中的虚拟代理点
            elif len(pair)==4:      #代理完成3个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])

                index2= int(pair[0] + 2*self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                del sol[index1],sol[index2]
            elif len(pair) == 5:  # 代理完成4个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                del sol[index1], sol[index2],sol[index3]
            elif len(pair) == 6:  # 代理完成5个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                del sol[index1], sol[index2],sol[index3],sol[index4]
            elif len(pair) == 7:  # 代理完成6个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                index5 = int(pair[0] + 8 * self.cout)
                sol[index5].pop(0)
                sol[pair[0]].extend(sol[index5])
                del sol[index1], sol[index2],sol[index3],sol[index4],sol[index5]
            elif len(pair) == 8:  # 代理完成7个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                index5 = int(pair[0] + 8 * self.cout)
                sol[index5].pop(0)
                sol[pair[0]].extend(sol[index5])
                index6 = int(pair[0] + 10 * self.cout)
                sol[index6].pop(0)
                sol[pair[0]].extend(sol[index6])
                del sol[index1], sol[index2],sol[index3],sol[index4],sol[index5],sol[index6]
            elif len(pair) == 9:  # 代理完成8个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                index5 = int(pair[0] + 8 * self.cout)
                sol[index5].pop(0)
                sol[pair[0]].extend(sol[index5])
                index6 = int(pair[0] + 10 * self.cout)
                sol[index6].pop(0)
                sol[pair[0]].extend(sol[index6])
                index7 = int(pair[0] + 12 * self.cout)
                sol[index7].pop(0)
                sol[pair[0]].extend(sol[index7])
                del sol[index1], sol[index2], sol[index3], sol[index4], sol[index5], sol[index6],sol[index7]

            elif len(pair) == 10:  # 代理完成9个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                index5 = int(pair[0] + 8 * self.cout)
                sol[index5].pop(0)
                sol[pair[0]].extend(sol[index5])
                index6 = int(pair[0] + 10 * self.cout)
                sol[index6].pop(0)
                sol[pair[0]].extend(sol[index6])
                index7 = int(pair[0] + 12 * self.cout)
                sol[index7].pop(0)
                sol[pair[0]].extend(sol[index7])
                index8 = int(pair[0] + 14 * self.cout)
                sol[index7].pop(0)
                sol[pair[0]].extend(sol[index8])
                del sol[index1], sol[index2], sol[index3], sol[index4], sol[index5], sol[index6], sol[index7],sol[index8]

            elif len(pair) == 11:  # 代理完成10个任务
                index1 = int(pair[0] + self.cout)
                sol[index1].pop(0)
                sol[pair[0]].extend(sol[index1])
                index2 = int(pair[0] + 2 * self.cout)
                sol[index2].pop(0)
                sol[pair[0]].extend(sol[index2])
                index3 = int(pair[0] + 4 * self.cout)
                sol[index3].pop(0)
                sol[pair[0]].extend(sol[index3])
                index4 = int(pair[0] + 6 * self.cout)
                sol[index4].pop(0)
                sol[pair[0]].extend(sol[index4])
                index5 = int(pair[0] + 8 * self.cout)
                sol[index5].pop(0)
                sol[pair[0]].extend(sol[index5])
                index6 = int(pair[0] + 10 * self.cout)
                sol[index6].pop(0)
                sol[pair[0]].extend(sol[index6])
                index7 = int(pair[0] + 12 * self.cout)
                sol[index7].pop(0)
                sol[pair[0]].extend(sol[index7])
                index8 = int(pair[0] + 14 * self.cout)
                sol[index8].pop(0)
                sol[pair[0]].extend(sol[index8])
                index9 = int(pair[0] + 16 * self.cout)
                sol[index9].pop(0)
                sol[pair[0]].extend(sol[index9])
                del sol[index1], sol[index2], sol[index3], sol[index4], sol[index5], sol[index6], sol[index7],sol[index8],sol[index9]
        print("852",sol)
        return sol
    '''通过A*算法，计算出所有代理的最优解'''
    def compute_solution(self):
        solution = {}
        for agent in self.agent_dict.keys():
            self.constraints = self.constraint_dict.setdefault(agent, Constraints()) #获取约束字典里key=agent的约束
            '''是需要去判断一下agent带了几组任务，目前就设3个任务组'''
            local_solution = self.a_star.search1(agent)+self.a_star.search2(agent)

            if not local_solution:  # 如果A*算法没有找到最优解，则返回false
                return False           # 返回false，表明最优解计算失败
            solution.update({agent:local_solution}) #将最优解添加到solution字典中

        '''补充：对每个代理补充好其他做任务时的路径'''
        # 检查代理数是否超出初始代理数，如果超出，则说明某个代理有多任务。最后将所做的所有任务拼接到该代理原路径之后
        sol = solution
        """
        变量概述：
            pairs=[(4, 6, 3), (0, 2, 8, 5), (1, 4, 0, 7), (3, 1), (2, 9)]
                pairs中的每个元组中首个元素都是代理索引号，元组中除代理外，其他皆是任务
                譬如：(4,6,3)表示代理4先完成任务6，再完成任务3
            sol={0:[(1,0),(2,0),(3,1)],1:[(),(),(),(),(),()],....}
                sol中不仅存储上述例子中的5个原代理，还有10个虚拟代理点（任务点）各自的搜索路径
        -------
        算法概述：
            为原代理按任务顺序一个一个接上虚拟代理(任务点)的搜索路径。
                譬如(4,6,3),原sol中代理4的路径是[(0,0),(0,1),(1,2)],虚拟代理6的路径是[(1,2),(2,3),(3,4)],虚拟代理3的路径是[(3,4),(4,4),(5,5)]]
                由算法得出代理4的路径[(0,0),(0,1),(1,2),,(2,3),(3,4),(4,4),(5,5)]
            依次为所有5个原代理添加所带虚拟代理(任务)的路径。
        -------
        更改需求：
            原算法问题：如果原代理所带任务数超出5任务数，例如任务数为6，就需要为原算法继续添加一个elif以满足任务数为6的情况
            新算法要求：1、可以满足无论多少任务数，算法皆可以实现原代理拼接虚拟代理的需求
                      2、压缩代码行数
        """
        sol=self.sol_merge(sol,self.pairs)
        solution=sol
        '''_____________________________________'''
        return solution


    '''计算方案成本'''
    def compute_solution_cost(self, solution):
        return sum([self.path_cost(path) for path in solution.values()])# 列表推导式

    def path_cost(self,path):
        total_cost=0
        try:
            for i in range(1,len(path)):
                a,b=path[i-1],path[i]
                cost1,cost2=0,0
                if fabs(a.location.x-b.location.x)+fabs(a.location.y-b.location.y)==2:
                    cost1=math.sqrt(2)
                else:
                    cost2=1
                total_cost=cost1+cost2+total_cost
        except:
            print("",end="")
        return total_cost

'''高层节点:solution{} ,    constraint_dict{} ,     cost'''
class HighLevelNode(object):
    def __init__(self):
        self.solution = {}
        self.constraint_dict = {}
        self.cost = 0

    def __eq__(self, other):
        # 我们首先检查另一个对象是否属于当前类的类型。
        # 如果是，我们继续执行相等性判断；否则，我们返回一个 NotImplemented 对象，表示当前的相等性判断未实现。
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other): # 确定两个HighLevelNode对象的比较方式，这里是以“成本”比较
        return self.cost < other.cost

'''CBS类'''
class CBS(object):
    def __init__(self, environment):
        self.env = environment  #环境对象
        self.open_set = set()   #存储待访问的节点
        self.closed_set = set() #存储可用的节点

    # 核心代码
    def search(self):
        start = HighLevelNode()
        # 初始化赋值start的三个属性
        # TODO: Initialize it in a better way
        start.constraint_dict = {}
        for agent in self.env.agent_dict.keys():
            start.constraint_dict[agent] = Constraints()
        start.solution = self.env.compute_solution()
        # print("879，检测start.solution",start.solution)
        if not start.solution:
            return {},{}
        start.cost = self.env.compute_solution_cost(start.solution)

        self.open_set |= {start}# 将start存入open_set集合中

        # 从open_set取出最小f值的节点p，并在open_set移除p，将p添加到closed_set
        while self.open_set:
            P = min(self.open_set)
            self.open_set -= {P}
            self.closed_set |= {P}

            # 若当前p点的方案没有冲突，则生成计划
            # 若方案有冲突，则由冲突生成新的约束，并用新约束生成新节点，再回到while循环，再去探索
            self.env.constraint_dict = P.constraint_dict
            conflict_dict = self.env.get_first_conflict(P.solution)
            if not conflict_dict:
                self.total_cost=self.env.compute_solution_cost(P.solution)
                return self.generate_plan(P.solution),self.total_cost

            constraint_dict = self.env.create_constraints_from_conflict(conflict_dict)

            for agent in constraint_dict.keys():
                new_node = deepcopy(P)
                new_node.constraint_dict[agent].add_constraint(constraint_dict[agent])
                self.env.constraint_dict = new_node.constraint_dict
                new_node.solution = self.env.compute_solution()
                if not new_node.solution:
                    continue
                new_node.cost = self.env.compute_solution_cost(new_node.solution)

                # TODO: ending condition
                if new_node not in self.closed_set:
                    self.open_set |= {new_node}

        return {},{}
    # 生成计划
    def generate_plan(self, solution):
        plan = {}
        for agent, path in solution.items():
            path_dict_list = [( state.location.x, state.location.y) for state in path]
            plan[agent] = path_dict_list
        maxlength = max([len(value) for value in plan.values()])
        for i in plan:
            if len(plan[i]) < maxlength:
                plan[i] += [plan[i][-1] for j in range(maxlength - len(plan[i]))]
        return plan
def generate_datas(agent_num,t_num,obs,dim,a_a_dis,rtime_max):
    a,t=agent_num,t_num
    obstacles=obs
    max=dim[0]-1
    a_a_dis=a_a_dis # 保证随机点间的距离差
    rtime_max=rtime_max

    # 存随机代理坐标
    agent_coord = []
    while len(agent_coord) <= a-1  :
        # 随机生成一个坐标点,# 检查新坐标点与已有坐标点的距离并且不在障碍物集中
        point = (random.randint(0, max), random.randint(0, max//2))
        if all(math.sqrt((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2) >=a_a_dis for p in agent_coord) and point not in obstacles:
            agent_coord.append(point)
        # point = (random.randint(max//2, max), random.randint(max//2, max))
        # if all(math.sqrt((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2) >=a_a_dis for p in agent_coord) and point not in obstacles:
        #     agent_coord.append(point)
    # 存随机任务坐标
    task_coord=[]
    a_a_dis =3  # 保证随机点间的距离差
    while len(task_coord) <= 2*t -1 :
        # 随机生成一个坐标点,# 检查新坐标点与已有坐标点的距离并且不在障碍物集中
        point = (random.randint(0, max), random.randint(0,max))
        if all(math.sqrt((point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2) >=a_a_dis for p in task_coord+agent_coord) and point not in obstacles and point not in agent_coord:
            task_coord.append(point)
    # 随机数据放入对应列表中
    agents,tasks,pos=[],[],[]
    pos=list(agent_coord+task_coord)
    for i in range(0,a):
        element={}
        element['name']=i
        element['start']=list(pos[i])
        agents.append(element)
    k=0
    for i in range(a,len(pos),2):
        element={}
        element['name']=k
        element['release_time']=random.randint(0,rtime_max)
        if i + 1 < len(pos):
            element['Loading']=list(pos[i])
            element['Unloading']=list(pos[i+1])
            tasks.append(element)
            k=k+1
    return agents,tasks

def main0(agent_num,task_num,C,rtime_max):
    start_time = time.time()#记录执行开始时间
    # 打开input.yaml文件，并将其内容加载为一个param对象（with：确保文件读取完成后自动关闭）
    # 异常处理 可能出现的错误：yaml.YAMLError
    with open('input.yaml', 'r',encoding='utf-8') as input_yaml:
        try:
            param = yaml.load(input_yaml, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    # 文件数据传入各参数
    dimension = param["map"]["dimensions"]
    obstacles = param["map"]["obstacles"]
    agents = param['agents']
    tasks = param['tasks']
    '''自动生成数据，如需手动注释掉下一行，并在input.yaml中添加即可'''
    agents,tasks=generate_datas(agent_num,task_num,obstacles,dimension,C,rtime_max)

    print("agents={0}\ntasks={1}".format(agents,tasks))


    # 创建环境
    env = Environment(dimension, agents, tasks, obstacles)
    # Searching
    cbs = CBS(env)
    solution,cost = cbs.search()

    #保存两个列表到文件
    with open('lists', 'a',encoding='utf-8') as file:
        file.write(str(agents))
        file.write(str(tasks)+'\n')

    print(solution)
    # # 从文件中加载两个列表
    # with open('lists', 'rb') as file:
    #     agents = pickle.load(file)
    #     tasks = pickle.load(file)

    # 如果方案生成失败，则打印
    if not solution:
        print(" Solution not found" )
        return
    # 先打开output.yaml，再把结果数据写入该文件中
    with open('output.yaml', 'r') as output_yaml:
        try:
            output = yaml.load(output_yaml, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    #写入数据
    output["schedule"] = solution
    output["cost"] = cbs.total_cost
    with open('output.yaml', 'w') as output_yaml:
        yaml.safe_dump(output, output_yaml)
    end_time = time.time()
    run_time = end_time - start_time
    now = datetime.datetime.now()
    # print("-------\n方案：{0}\n成本：{1} \n \033[34m{2}  {3}\033[0m".format(solution, cost,now, run_time))
    print("-------\n成本：{0} \n \033[34m{1}  {2}\033[0m".format(cost,now, run_time))

    # env.allocate.start_Hamiltonian()
    # env.allocate.draw_map(solution,0) #绘制图片

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main0(5,5,7,3)


