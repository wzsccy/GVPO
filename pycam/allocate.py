# 开发时间：2024/2/19 16:01
# 开发语言：Python
'''---------------------------------------'''
import itertools
import random
import time
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycam.a_star_single import Astar
from pycam.lacam_star_single import LaCAM,Config,Configs
from pycam.allo_util import TAE
import copy

'''---------------------------------------'''

class Allocate():
    def __init__(self, agent_dict,task_dict,dimension,obstacles,lacam_gird):
        # 初始化参数
        self.agent=agent_dict
        self.temp_agent_dict=copy.deepcopy(agent_dict)
        self.task=task_dict
        self.agent_dict = agent_dict
        self.task_dict = task_dict
        self.dimension=(dimension[0]*50,dimension[1]*50) # [17, 17] ——>  (17*50,17*50)
        self.obstacles=obstacles # [(1, 2)]
        self.lacam_grid=lacam_gird
        '''-------------------------------------'''
        # 创建视图
        self.scale=16 # 格子间距
        self.canvas=np.ones((self.dimension[0]//3,self.dimension[1]//3,3), np.uint8) * 255
        self.frames = []  # 窗口画面
        self.all_num=len(self.agent_dict)+len(self.task_dict)
        self.colours = self.assign_colour(self.all_num)  # 给分配all_num种颜色
        # self.draw_map() # 是否显示视图化
        '''-------------------------------------'''
        # 执行分配
        cclt_time_now=time.time()
        self.allocation_matrix() #返回出距离矩阵
        cclt_time_end=time.time()
        # print("矩阵得出时长={0}".format(cclt_time_end-cclt_time_now))
    def assign_colour(self,num):
        # 上色函数
        def colour(x):
            x = hash(str(x+42))
            return ((x & 0xFF % 256, (x >> 8) & 0xFF % 256, (x >> 16) & 0xFF% 256))

        colours = dict()
        for i in range(num):
            colours[i] = colour(i)
        # print("自动分配的颜色",colours)
        return colours
    def data_transform(self,i,solution,total_sol):
        if i > 0:
            del solution[0]
        for value0 in solution:
                # print(value0)   #l:Config(positions=[(1, 9), (16, 12)])
                value1=value0.positions
                for key,value2 in total_sol.items():
                    value2.append(value1[0])
                    value1.pop(0)
        # print(total_sol)
        for agent, path_list in total_sol.items():
            # 从列表末尾开始检查重复项
            j = len(path_list) - 1
            while j > 0:
                # 如果当前元素与前一个元素相同，则移除当前元素
                # print(j)
                if path_list[j] == path_list[j - 1]:
                    del path_list[j]
                else:
                    break
                j-=1

        return total_sol
    def transform_for_draw(self,pairs,total_sol,agent_dict,task_dict):
        index_dict = {0: []}
        rt_dict = {0: []}
        for pair in pairs:
            if pair[-1] != -1:
                agent_id = pair[0]  # 代理号
                index_dict[agent_id] = [total_sol[agent_id].index(agent_dict[agent_id])]  # 0:[0,5,7]
                rt_dict[agent_id] = []
                for h in range(1, len(pair)):
                    task_id = pair[h]  # 代理完成的当前任务号
                    task_rt = task_dict[task_id][2]  # 当前任务的释放时间
                    rt_dict[agent_id].append(task_rt)
                    task_loading = task_dict[task_id][0]
                    task_unloading = task_dict[task_id][1]
                    index_dict[agent_id].append(total_sol[agent_id].index(task_loading))
                    index_dict[agent_id].append(total_sol[agent_id].index(task_unloading))
        # print(rt_dict)
        # print(index_dict)
        copy_total_sol = copy.deepcopy(total_sol)
        # 去掉最后一个交货点的索引号
        pp_distance = 0
        for agent, ind_list in index_dict.items():
            if len(index_dict[agent]) % 2 != 0:
                index_dict[agent].pop(-1)
            for i0 in range(1, len(ind_list), 1):
                for j0 in range(ind_list[i0 - 1], ind_list[i0]):
                    # 获取当前点和下一个点
                    current_point = total_sol[agent][j0]
                    next_point = total_sol[agent][j0 + 1]
                    # 计算两点之间的距离
                    distance = ((next_point[0] - current_point[0]) ** 2 + (
                                next_point[1] - current_point[1]) ** 2) ** 0.5
                    pp_distance += distance
                "Error:IndexError: list index out of range"
                if rt_dict[agent][0] > pp_distance:
                    rt_wait = int(rt_dict[agent][0] - pp_distance)
                    copy_index = index_dict[agent][i0]
                    copy_element = copy_total_sol[agent][copy_index]
                    for _ in range(rt_wait):
                        copy_total_sol[agent].insert(copy_index + 1, copy_element)
                    rt_dict[agent].pop(0)

        return copy_total_sol
    # 绘制代理起始点、任务取货点、任务交货点、障碍物点
    def draw_point(self):
        scale = self.scale  # 格子间距
        font_scale = 1
        font_size = 0.5
        color = (255, 255, 255)
        # 绘制start点
        for id, pos in self.agent_dict.items():
            point = (np.array(pos) + 1) * scale
            cv2.circle(self.canvas, point, scale // 2 + scale // 4, self.colours[id], -1)
            # 要区分单数位置和双数位置
            if 0 <= id <= 9:
                font_pos = (point[0] - 6, point[1] + 6)
            else:
                font_pos = (point[0] - 10, point[1] + 6)
            cv2.putText(self.canvas, '{0}'.format(id), font_pos, cv2.FONT_HERSHEY_COMPLEX,
                        font_size, (0, 0, 0), font_scale)
        # 绘制任务组点
        for id, pos in self.task_dict.items():
            # 绘制task点
            point = (np.array(pos[0]) + 1) * scale  # 中心点
            lu_point = (point - np.array([scale // 3, scale // 3]))  # 左上点
            rd_point = (point + np.array([scale // 3, scale // 3]))  # 右下点
            font_pos = (point[0] - 4, point[1] + 4)
            if 0 <= id <= 9:
                font_pos = (point[0] - 4, point[1] + 4)
            else:
                font_pos = (point[0] - 6, point[1] + 4)
            cv2.rectangle(self.canvas, lu_point, rd_point, self.colours[id + len(self.agent_dict)], -1)
            cv2.putText(self.canvas, '{0}'.format(id), font_pos, cv2.FONT_HERSHEY_COMPLEX,
                        0.3, color, font_scale)
            # 绘制goal点
            point1 = (np.array(pos[1]) + 1) * scale  # 中心点
            lu_point1 = (point1 - np.array([scale // 2, scale // 2]))  # 左上点
            rd_point1 = (point1 + np.array([scale // 2, scale // 2]))  # 右下点
            font_pos1 = (point1[0] - 6, point1[1] + 6)
            if 0 <= id <= 9:
                font_pos1 = (point1[0] - 6, point1[1] + 6)
            else:
                font_pos1 = (point1[0] - 10, point1[1] + 6)
            cv2.rectangle(self.canvas, lu_point1, rd_point1, self.colours[id + len(self.agent_dict)], -1)
            cv2.putText(self.canvas, '{0}'.format(id), font_pos1, cv2.FONT_HERSHEY_COMPLEX,
                        0.45, color, font_scale)
        # 绘制障碍物
        for value in self.obstacles:
            point = (np.array(value) + 1) * scale
            point1 = np.array(point) - np.array([scale // 3, scale // 3])  # 左上角
            point2 = np.array(point) + np.array([scale // 3, scale // 3])  # 右下角
            point3 = np.array(point) + np.array([scale // 3, -scale // 3])  # 左下角
            point4 = np.array(point) - np.array([scale // 3, -scale // 3])  # 右上角
            cv2.line(self.canvas, tuple(point1), tuple(point2), (0, 0, 0), 3)
            cv2.line(self.canvas, tuple(point3), tuple(point4), (0, 0, 0), 3)
    # 绘制图
    def draw_map(self,solution,static,map):
        scale=self.scale # 格子间距
        font_scale = 1
        font_size = 0.5
        color = (255, 255, 255)

        # 制作网格线：列线、横线、斜线1、斜线2
        for i in range(1, self.dimension[0] // scale):
            cv2.line(self.canvas, (scale * i, scale), (scale * i, (self.dimension[1] // scale - 1) * scale), (0, 0, 0))
            cv2.putText(self.canvas,'{0}'.format(i-1),((scale * i)-(scale//7), scale-(scale//3)),
                        cv2.FONT_HERSHEY_COMPLEX,0.32, (169,169,169), font_scale)
        for i in range(1, self.dimension[1] // scale):
            cv2.line(self.canvas, (scale, i * scale), ((self.dimension[0] // scale - 1) * scale, i * scale), (0, 0, 0))
            cv2.putText(self.canvas, '{0}'.format(i - 1), (scale-(scale//2)-8, (i * scale) + (scale//7)),
                        cv2.FONT_HERSHEY_COMPLEX, 0.32, (169, 169, 169), font_scale)
        for i in range(1, self.dimension[1] // scale):
            cv2.line(self.canvas, (scale * i, scale), ((self.dimension[0] // scale - 1) * scale, (self.dimension[1] // scale - i) * scale),
                     (220, 220, 220), lineType=cv2.LINE_8)
            if i > 1:
                cv2.line(self.canvas, (scale, scale * i),
                         ((self.dimension[0] // scale - i) * scale, (self.dimension[1] // scale - 1) * scale), (220, 220, 220),
                         lineType=cv2.LINE_8)
        for i in range(1, self.dimension[1] // scale):
            cv2.line(self.canvas, (scale, (self.dimension[0] // scale - i) * scale), ((self.dimension[0] // scale - i) * scale, scale),
                     (220, 220, 220), lineType=cv2.LINE_8)
            if i > 1:
                cv2.line(self.canvas, (scale * i, (self.dimension[0] // scale - 1) * scale),
                         ((self.dimension[0] // scale - 1) * scale, scale * i), (220, 220, 220), lineType=cv2.LINE_8)

        """静态结果图"""
        if static:
            for agent_id, path in solution.items():
                for i in range(len(path) - 1):
                    start_point = (np.array((path[i][0], path[i][1])) + 1) * scale
                    end_point = (np.array((path[i + 1][0], path[i + 1][1])) + 1) * scale
                    cv2.line(self.canvas, start_point, end_point, self.colours[agent_id], 2)
                    self.frame = deepcopy(self.canvas)
                    self.draw_point()
            cv2.imshow("Path Diagram", self.frame)
            self.frames.append(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
            cv2.waitKey(2000)
            # 等待所有路径显示完毕后关闭窗口
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            '''动态结果图'''
        else:
            path_len_list=[]
            step_list=[]
            for agent_id, path in solution.items():
                path_len_list.append(len(path))
            path_len=max(path_len_list)
            for i in range(1,path_len): # i为步长
                for agent_id, path in solution.items():
                    if len(path)>i:
                        step_list.append(path[i-1])
                        step_list.append(path[i])
                    else:continue

                    start_point = (np.array((step_list[0][0], step_list[0][1])) + 1) * scale
                    end_point = (np.array((step_list[1][0], step_list[1][1])) + 1) * scale
                    step_list.clear()
                    cv2.line(self.canvas, start_point, end_point, self.colours[agent_id], 2)
                self.draw_point()
                cv2.namedWindow('window_name', cv2.WINDOW_FREERATIO)
                cv2.resizeWindow('window_name', 1000, 1000)
                self.frame = deepcopy(self.canvas)
                # self.frames.append(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
                cv2.imshow('window_name', self.frame)
                cv2.waitKey(10)
            # 保存最后视图
            image_rgb=cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.savefig(r'D:\QQ\document\py-lacam2-main (2)\py-lacam2-main\images9_path\{0}.pdf'.format(map),format='pdf',dpi=300,bbox_inches='tight',pad_inches=0)
            plt.show(block=False)
            plt.pause(100)
            plt.close()
            cv2.waitKey(10)
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    '''构建哈密顿环图'''
    # def Hamiltonian(self,agent_dict,task_dict,result_list):
    #     result_list=copy.deepcopy(result_list)
    #     result_list.append(result_list[0]) # 构成一个环
    #     draw_list=[]
    #     mark_list=[] # 标记列表：设置智能体为0，任务点为1
    #     # 按分配序列顺序依次把散点坐标存入draw_list
    #     for i,value in enumerate(result_list):
    #         if len(value)==2:
    #             if value[0]=='a':
    #                 print("agent_dict[int(value[1])",agent_dict[int(value[1])])
    #                 draw_list.append(agent_dict[int(value[1])])
    #                 mark_list.append(0)
    #                 continue
    #             elif value[0]=='t':
    #                 draw_list.append(task_dict[int(value[1])][0])
    #                 draw_list.append(task_dict[int(value[1])][1])
    #                 mark_list.append(1)
    #                 mark_list.append(1)
    #                 continue
    #         if len(value)==3:
    #             if value[0]=='a':
    #                 index=int(value[1])*10+int(value[2])
    #                 draw_list.append(agent_dict[index])
    #                 mark_list.append(0)
    #                 continue
    #             elif value[0]=='t':
    #                 index = int(value[1]) * 10 + int(value[2])
    #                 draw_list.append(task_dict[index][0])
    #                 draw_list.append(task_dict[index][1])
    #                 mark_list.append(1)
    #                 mark_list.append(1)
    #         if len(value)==4:
    #             if value[0]=='a':
    #                 index=int(value[1])*100 + int(value[2])*10 + int(value[3])
    #                 draw_list.append(agent_dict[index])
    #                 mark_list.append(0)
    #                 continue
    #             elif value[0]=='t':
    #                 index = int(value[1]) * 100 + int(value[2])*10 + int(value[3])
    #                 draw_list.append(task_dict[index][0])
    #                 draw_list.append(task_dict[index][1])
    #                 mark_list.append(1)
    #                 mark_list.append(1)
    #         if len(value) == 5:
    #             if value[0] == 'a':
    #                 index = int(value[1]) * 1000 + int(value[2]) * 100 + int(value[3])*10 +int(value[4])
    #                 draw_list.append(agent_dict[index])
    #                 mark_list.append(0)
    #                 continue
    #             elif value[0] == 't':
    #                 index = int(value[1]) * 1000 + int(value[2]) * 100 + int(value[3])*10 + int(value[4])
    #                 draw_list.append(task_dict[index][0])
    #                 draw_list.append(task_dict[index][1])
    #                 mark_list.append(1)
    #                 mark_list.append(1)
    #
    #     # 将坐标分解为两个独立的列表
    #     x=[coord[0] for coord in draw_list]
    #     y=[coord[1] for coord in draw_list]
    #
    #     # 绘制窗口
    #     plt.figure('Hamiltonian Ring Graph',figsize=(6,6))
    #     plt.xlim(-5,55)
    #     plt.ylim(-5,55)
    #     plt.axis('off')
    #
    #     # 绘制连接散点的线以及散点
    #     for i,value in enumerate(mark_list):
    #         if value == 0 :
    #             plt.scatter(x[i], y[i],marker='o',color='blue',s=80,zorder=4)
    #         else:
    #             plt.scatter(x[i], y[i], marker='o', color='red',s=80,zorder=4)
    #     plt.plot(x, y, linewidth=2, zorder=2)
    #
    #     # 保存图形为矢量图
    #     plt.savefig('my_vector_plot.pdf')
    #     plt.show(block=False)
    #     plt.pause(0)
    #     plt.close()
    "构建代理完成任务环图"
    def Hamiltonian(self, agent_dict, task_dict, result_list):
        result_list = copy.deepcopy(result_list)
        result_list.append(result_list[0])  # 构成一个环
        draw_list = []
        mark_list = []  # 标记列表：设置智能体为0，任务点为1
        # 按分配序列顺序依次把散点坐标存入draw_list

        # 绘制窗口
        plt.figure('Hamiltonian Ring Graph', figsize=(6, 6))
        plt.xlim(-5, 50)
        plt.ylim(-5, 50)
        plt.axis('off')
        print(len(agent_dict))
        colors = list(self.assign_colour(len(agent_dict)).values())
        print(colors)

        # colors = [(255,0,0),(0,255,0),(255,0,255),(255,255,0)]
        colors = [(r / 255, g / 255, b / 255) for r, g, b in colors]
        for pair in self.pairs:
            draw_list = []
            if pair[-1] == -1:continue
            for i in range(0,len(pair)):
                if i == 0 :
                    start_point=agent_dict[pair[i]]
                    draw_list.append(start_point)
                if i>0:
                    load_point=task_dict[pair[i]][0]
                    unload_point=task_dict[pair[i]][1]
                    draw_list.append(load_point)
                    draw_list.append(unload_point)
            draw_list.append(agent_dict[pair[0]])

            # 将坐标分解为两个独立的列表
            x = [coord[0] for coord in draw_list]
            y = [coord[1] for coord in draw_list]

            co=colors[self.pairs.index(pair)]


            # 绘制连接散点的线以及散点
            for i in range(0,len(x)):
                if i == 0 or i==len(x)-1:
                    plt.scatter(x[i], y[i], marker='s', color=co, s=50,zorder=4,edgecolors='black')
                else:
                    plt.scatter(x[i], y[i], marker='o', color=co, s=50, zorder=4,alpha=0.8,edgecolors='black')

                plt.plot(x,y,color='black',linewidth=4,alpha=0.8)
                plt.plot(x,y,color=co,linewidth=2,alpha=0.8)

            # 在图表中添加文本框
            # plt.text(agent_dict[pair[0]][0], agent_dict[pair[0]][1], s="100", fontsize=12, color='red', ha='center', va='center')

        # 保存图形为矢量图
        plt.savefig('HMT_TA.pdf',format='pdf',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.show(block=False)
        plt.pause(0)
        plt.close()
    def start_Hamiltonian(self):
        # 生成哈密顿环图
        self.Hamiltonian(self.temp_agent_dict, self.task_dict, self.result_list)
    # 将代理集和任务集的存储结构转换成合适的结构
    def transform(self): # 将传入的数据转换成更方便的数据
        for agent_name in self.agent:
            self.agent_dict[agent_name]=(self.agent[agent_name]['start'].location.x,self.agent[agent_name]['start'].location.y)
        # print("代理集：",self.agent_dict)
        for task_name in self.task:
            list=[]
            list.append((self.task[task_name]['Loading'].location.x,self.task[task_name]['Loading'].location.y))
            list.append((self.task[task_name]['Unloading'].location.x,self.task[task_name]['Unloading'].location.y))
            list.append(('r_time={0}'.format(self.task[task_name]['release_time'])))
            self.task_dict[task_name]=list
        # print("任务集:",self.task_dict)
    # 为权重矩阵根据元素类型添加权值
    def weight_value(self,row,line):
        value = 0
        agent_num,task_num=len(self.agent_dict),len(self.task_dict) # 8

        # 权重类型1：代理——代理
        if (row >= 0 and row < agent_num) and (line >= 0 and line < agent_num) : value=0
        # 权重类型2：代理——任务点
        if (row >= 0 and row < agent_num) and (line >= agent_num and line < (agent_num+task_num)):
            agent_index=row
            task_index=line-agent_num
            width,height=self.dimension[0],self.dimension[1]
            start,task,goal,r_time=(
                self.agent_dict[agent_index],self.task_dict[task_index][0],self.task_dict[task_index][1],self.task_dict[task_index][2])
            wall=self.obstacles
            "A*"
            # astar1=Astar(width,height,start,task,goal,r_time,wall)
            # value=astar1.cost

            "Lacam*"
            "起点——取货"
            lac_0 = LaCAM()
            start=[start]
            task=[task]
            start=Config(start)
            task=Config(task)
            solution_0=lac_0.solve(self.lacam_grid,start,task,0,None,True,0,1,0)
            positions_list_0 = [config.positions[0] for config in solution_0]
            value_0 = lac_0.calculate_euclidean_distance_sum(positions_list_0)
            if r_time >= value: # 释放时间
                value_0=r_time

            "取货-交货"
            lac_1 = LaCAM()
            goal = [goal]
            goal = Config(goal)
            # solution = lac_0.solve(self.lacam_grid, task, goal, 0, None, True, 0, 1, 0)
            solution_1 = lac_0.solve(self.lacam_grid, task, goal)
            positions_list_1 = [config.positions[0] for config in solution_1]
            value_1 = lac_1.calculate_euclidean_distance_sum(positions_list_1)

            value = value_0 + value_1 # 合并两段
            # print(value)


        # 权重类型3：任务点——代理
        if (row >= agent_num and row < agent_num+task_num) and (line >=0 and line <agent_num):
            value=0
        # 权重类型4：已完成任务点——任务点
        if (row >= agent_num and row < agent_num+task_num) and (line >= agent_num and line < (agent_num+task_num)):
            first_task_index=row-agent_num
            second_task_index=line-agent_num

            width, height = self.dimension[0], self.dimension[1]
            start, task, goal,r_time=(
                self.task_dict[first_task_index][1],self.task_dict[second_task_index][0],self.task_dict[second_task_index][1],self.task_dict[second_task_index][2])
            wall=self.obstacles
            "A*"
            # astar2=Astar(width,height,start,task,goal,r_time,wall)
            # value=astar2.cost

            "Lacam*"
            "起点——取货"
            lac_2 = LaCAM()
            start = [start]
            task = [task]
            start = Config(start)
            task = Config(task)
            solution_2 = lac_2.solve(self.lacam_grid, start, task, 0, None, True, 0, 1, 0)
            positions_list_2 = [config.positions[0] for config in solution_2]
            value_2 = lac_2.calculate_euclidean_distance_sum(positions_list_2)
            if r_time >= value:  # 释放时间
                value_2 = r_time

            "取货-交货"
            lac_3 = LaCAM()
            goal = [goal]
            goal = Config(goal)
            solution_3 = lac_3.solve(self.lacam_grid, task, goal)
            positions_list_3 = [config.positions[0] for config in solution_3]
            value_3 = lac_3.calculate_euclidean_distance_sum(positions_list_3)
            value = value_2 + value_3  # 合并两段


            # 如果任务点相同
            if first_task_index == second_task_index: value=0
        return value
    # 由完整配对序列返回出总成本
    def get_row_line(self,fe):
        test_cost=0
        for i in range(1,len(fe)):
            row,line= self.get_matrix_value(fe[i-1],fe[i])
            test_cost +=self.distance_matrix[row][line]
        return test_cost
    # 根据组合得出权重矩阵的行列号
    """当前代理数和任务数均不超过10000，否则需要扩充代码"""
    def get_matrix_value(self,a,b):
        agent_num=len(self.agent_dict)
        row,line=0,0
        if len(a)==2 and len(b)==2:#代理（0-9），任务（0-9）
            if a[0]=='a' and b[0]=='t':
                row,line=int(a[1]),int(b[1])+agent_num

            elif a[0]=='t' and b[0]=='a':
                row,line=int(a[1])+agent_num,int(b[1])

            elif a[0]=='t' and b[0]=='t':
                row,line=int(a[1])+agent_num,int(b[1])+agent_num

            elif a[0]=='a' and b[0]=='a':
                row,line=int(a[1]),int(b[1])
        if len(a)==2 and len(b)==3:#代理（0-9），任务（10-99）
            if a[0]=='a' and b[0]=='t':
                row,line=int(a[1]),int(b[1])*10+int(b[2])+agent_num

            elif a[0]=='t' and b[0]=='a':
                row,line=int(a[1])+agent_num,int(b[1])*10+int(b[2])

            elif a[0]=='t' and b[0]=='t':
                row,line=int(a[1])+agent_num,int(b[1])*10+int(b[2])+agent_num

            elif a[0]=='a' and b[0]=='a':
                row,line=int(a[1]),int(b[1])*10+int(b[2])
        if len(a) == 3 and len(b) == 2:  # 代理（10-99），任务（0-9）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]), int(b[1]) + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line =int(a[1])*10+int(a[2]), int(b[1])
        if len(a) == 3 and len(b) == 3:  # 代理（10-99），任务（10-99）
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]), int(b[1])  + agent_num

            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1]) * 10 + int(b[2])

            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1]) * 10 + int(b[2]) + agent_num

            elif a[0] == 'a' and b[0] == 'a':
                row, line =int(a[1])*10+int(a[2]), int(b[1]) * 10 + int(b[2])
        # 代理（0-9），任务（100-999）
        if len(a) == 2 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]), int(b[1]) * 100 + int(b[2])*10 +int(b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) + agent_num, int(b[1]) * 100 + int(b[2])*10 +int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) + agent_num, int(b[1]) * 100 + int(b[2])*10 +int(b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]), int(b[1]) * 100 + int(b[2])*10 +int(b[3])
        # 代理（10-99），任务（100-999）
        if len(a) == 3 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]), int(b[1]) * 100 + int(b[2])*10 +int(b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1]) * 100 + int(b[2])*10 +int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1])*10+int(a[2]) + agent_num, int(b[1]) * 100 + int(b[2])*10 +int(b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1])*10+int(a[2]), int(b[1]) * 100 + int(b[2])*10 +int(b[3])
        # # 代理（100-999），任务（100-999）
        if len(a) == 4 and len(b) == 4:
            if a[0] == 'a' and b[0] == 't':
                row, line = int(a[1]) * 100 + int(a[2])*10 + int(a[3]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3]) + agent_num
            elif a[0] == 't' and b[0] == 'a':
                row, line = int(a[1]) * 100 + int(a[2])*10 + int(a[3]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])
            elif a[0] == 't' and b[0] == 't':
                row, line = int(a[1]) * 100 + int(a[2])*10 + int(a[3]) + agent_num, int(b[1]) * 100 + int(b[2]) * 10 + int(
                    b[3]) + agent_num
            elif a[0] == 'a' and b[0] == 'a':
                row, line = int(a[1]) * 100 + int(a[2])*10 + int(a[3]), int(b[1]) * 100 + int(b[2]) * 10 + int(b[3])

        return row,line
    # 生成距离矩阵，存储在self.distance_matrix
    def allocation_matrix(self):
        # 创建 行、列为（代理数+任务数）
        num=len(self.agent_dict)+len(self.task_dict)
        row_num,line_num=num,num
        distance_matrix = np.zeros((row_num,line_num))
        for row in range(row_num): # 0 1
            for line in range(line_num):# 0 1 2
                distance_matrix[row][line]=self.weight_value(row,line)

        # np.save('drl_w\matrix.npy',distance_matrix)
        self.distance_matrix = distance_matrix
        # self.distance_matrix = np.load('drl_w\matrix.npy')
        # print(self.distance_matrix)




    # 1、生成所有可能的组合；2、选出成本最小的任务分配组合以及成本，存储在combinations,min_cost
    # 方法1：迭代生成器生成约束条件下的合理组合
    def aaa(self):
        num_agents = len(self.agent_dict)
        num_tasks = len(self.task_dict)
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(num_agents)]
        tasks = [f't{i}' for i in range(num_tasks)]
        cost_list = []
        valid_combinations = []
        a_time = time.time()
        for combination in itertools.permutations(agents+tasks):
            # 约束1：确保每个组合的第一个元素为 'a0'
            # 约束2：不允许代理相邻
            # 约束3：最后一个不允许为代理
            if combination[0] != 'a0':continue
            # if combination[-1] in agents:continue
            # if any(combination[j] in agents and combination[j + 1] in agents for j in
            #        range(len(combination) - 1)):continue
            valid_combinations.append(combination)

        # 最终数据存储在valid_combinations
        self.valid_comb=valid_combinations
        # print(self.valid_comb)
        # print("当前合理组合个数为",len(valid_combinations))
        for value in valid_combinations:
            test_cost = self.get_row_line(value)
            cost_list.append(test_cost)
            # print("{0}组合，其对应的总成本为  {1}".format(value, test_cost))
        # 找到最小值及其索引
        min_cost = min(cost_list)
        min_index = cost_list.index(min_cost)
        b_time=time.time()
        # print("由合理组合推出成本所需时间",b_time-a_time)
        combinations=valid_combinations[min_index]
        # print("组合个数",len(valid_combinations))
        # print("最优分配组合为{0}，其对应成本为 {1}".format(combinations,min_cost))

        return combinations,min_cost
    # 方法1：多代理对单一任务配对
    def aaa1(self):
        num_agents = len(self.agent_dict)
        num_tasks = len(self.task_dict)
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(num_agents)]
        tasks = [f't{i}' for i in range(num_tasks)]
        agent_s=copy.deepcopy(agents)
        le_list,rg_list,result_list,temp_cost,min_cost=[],[],[],[],0
        # row,line=self.get_matrix_value()
        # cost=self.distance_matrix[row][line]
        # print(self.distance_matrix)
        le_list=agents
        for task_name in tasks:
            for le_name in le_list:
                row, line = self.get_matrix_value(le_name,task_name)
                cost=self.distance_matrix[row][line]
                temp_cost.append(cost)

            # print(temp_cost)
            min_cost+=min(temp_cost)
            index=temp_cost.index(min(temp_cost))
            temp_cost.clear()
            # print(le_list[index])

            if le_list in tasks:
                re_index=result_list.index(le_list[index])
                result_list.insert(re_index,task_name)
                continue
            elif le_list[index] in tasks:
                result_list.append(task_name)
            elif le_list[index] in agents:
                result_list.append(le_list[index])
                result_list.append(task_name)
            le_list[index]=task_name


        if  any(element in agents for element in le_list):
            extracted_elements = [element for element in le_list if element in agent_s]
            result_list.extend(extracted_elements)


        return result_list,min_cost
    # 方法2：多代理对多任务配对
    def aaa2(self):
        num_agents = len(self.agent_dict)
        num_tasks = len(self.task_dict)
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(num_agents)]
        tasks = [f't{i}' for i in range(num_tasks)]
        le_list, rg_list, result_list, all_cost,all_comb,min_cost =[], [],[], [], [], 0
        # row,line=self.get_matrix_value()
        # cost=self.distance_matrix[row][line]
        # print(self.distance_matrix)
        le_list=copy.deepcopy(agents)
        rg_list=copy.deepcopy(tasks)

        while rg_list:
            for a_ in le_list:
                for t_ in rg_list:
                    row, line = self.get_matrix_value(a_, t_)
                    cost = self.distance_matrix[row][line]
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
        return result_list,min_cost
    # 触发条件：
    def trigger_conditions(self,agents,tasks,le_list,rg_list,le_temp, rg_temp,result_list):
        # print(le_list,rg_list,le_temp, rg_temp,result_list)
        le_list1=copy.deepcopy(le_list)
        rg_list1=copy.deepcopy(rg_list)
        result_list1=copy.deepcopy(result_list)
        all_comb,all_cost=[],[]
        index_list=[]
        cost0,cost1,cost2,cost3,known_a_t_cost = 0,0,0,0,[]
        # 算出已知带任务的所有代理中的最大成本
        for element in result_list1:
            if element[0]=='a': index_list.append(result_list1.index(element))
        if len(index_list)==1:
            cost0=self.get_row_line(result_list1)
            known_a_t_cost.append(cost0)
        else:

            for i in range(1,len(index_list)):
                left = index_list[i - 1]
                right = index_list[i]
                temp = result_list1[left:right]

                cost1=self.get_row_line(temp)
                known_a_t_cost.append(cost1)
                if i == len(index_list) - 1:
                    temp1 =result_list1[right:]
                    cost2=self.get_row_line(temp1)
                    known_a_t_cost.append(cost2)


        while len([i for i in le_list1 if i in agents])==len(rg_list1) \
                and le_temp in tasks and rg_temp in tasks:
            if le_temp not in le_list1:
                break
            le_list1.remove(le_temp)# 删除当前配对的在左列的元素
            for a_ in le_list1:# 重新组合，得出
                for t_ in rg_list1:
                    row, line = self.get_matrix_value(a_, t_)
                    cost = self.distance_matrix[row][line]
                    all_comb.append((a_, t_))
                    all_cost.append(cost)

            # 筛选出小于Max{cost(a_t)}的元素
            filtered_elements=[i for i in all_cost if i <= max(known_a_t_cost)]
            if filtered_elements:
                selected_element = random.choice(filtered_elements)
                index=all_cost.index(selected_element)
                temp=all_comb[index]
                le_temp,rg_temp=temp[0],temp[1]
        return le_temp,rg_temp
    # 方法3：多代理对多任务配对+触发条件===>减少时间成本
    def aaa3(self):
        num_agents = len(self.agent_dict)
        num_tasks = len(self.task_dict)
        # 使用列表推导式生成代理和任务的列表
        agents = [f'a{i}' for i in range(num_agents)]
        tasks = [f't{i}' for i in range(num_tasks)]
        le_list, rg_list, result_list, all_cost, all_comb, min_cost = [], [], [], [], [], 0
        le_list = copy.deepcopy(agents)
        rg_list = copy.deepcopy(tasks)
        # 获得成本最小的配对
        while rg_list:
            '''生成成本最小的最佳配对'''
            for a_ in le_list:
                for t_ in rg_list:
                    row, line = self.get_matrix_value(a_, t_)
                    cost = self.distance_matrix[row][line]
                    all_comb.append((a_, t_))
                    all_cost.append(cost)
            min_cost += min(all_cost)
            index = all_cost.index(min(all_cost))
            le_temp, rg_temp = all_comb[index][0], all_comb[index][1]
            if len(result_list)>1:
                le_temp, rg_temp=self.trigger_conditions(agents,tasks,le_list,rg_list,le_temp,rg_temp,result_list)
            all_cost.clear()
            all_comb.clear()

            '''将配对装入result_list，并更新左右列'''
            # 如果左列全是任务点，下个配对，要嵌入到对应的代理之后
            if le_list in tasks:
                re_index = result_list.index(le_temp)
                result_list.insert(re_index, rg_temp)
                continue
            # 下个配对类型是t-t
            elif le_temp in tasks:
                result_list.append(rg_temp)
            # 下个配对类型是a-t
            elif le_temp in agents:
                result_list.append(le_temp)
                result_list.append(rg_temp)
            #更新左右列
            le_index = le_list.index(le_temp)
            le_list[le_index] = rg_temp
            rg_list.remove(rg_temp)
        # 将没有任务的代理逐一存入result_list
        if not rg_list and any(element in agents for element in le_list):
            extracted_elements = [element for element in le_list if element in agents]
            result_list.extend(extracted_elements)
        min_cost=self.get_row_line(result_list)
        return result_list, min_cost

    # 方法4：分区交换转移优化
    """
    1、随机分区：
        m <= n : (n/m)取上，余数给最后一个代理分区
        m  > n : 方法不适用
    2、分区优化
        转移：在保证每个分区内至少一个任务的前提下，进行互相转移
    """
    def aaa4(self):
        m, n = len(self.agent),len(self.task)
        ta = TAE(self.distance_matrix, m, n)
        result_list = ta.Alg_4_0()
        min_cost = self.get_row_line(result_list)
        return result_list,min_cost

    def aaa5(self):
        m, n = len(self.agent), len(self.task_dict)
        ta = TAE(self.distance_matrix, m, n)
        result_list = ta.Alg_5()
        min_cost = self.get_row_line(result_list)
        return result_list,min_cost

    def aaa5_RL(self):
        m, n = len(self.agent), len(self.task_dict)
        ta = TAE(self.distance_matrix, m, n)
        result_list = ta.Alg_5_RL()
        # print(f"848{result_list}")
        min_cost = 0.75*self.get_row_line(result_list)
        return result_list,min_cost

    def allocating_task(self,TA_name):
        a=list()
        b = 0
        # a, b = self.aaa1()
        # print("算法1",b)
        # c, d = self.aaa2()
        # print("算法2",d)
        # e,f = self.aaa3()
        # print("算法3",f)
        # g, h = self.aaa4()
        # print("算法4",h)
        if TA_name == 'TA1':
            a, b = self.aaa1()
        elif TA_name == 'TA2':
            a, b = self.aaa2()
        elif TA_name == 'TA3':
            a, b = self.aaa3()
        elif TA_name == 'TA4':
            a, b = self.aaa4()
        elif TA_name == 'TA5':
            a, b = self.aaa5()
        elif TA_name == 'TA5_RL':
            a, b = self.aaa5_RL()

        self.result_list=a
        # print("由算法本身提出的方案：",a)
        '''比如将['a0','t1','a1']改成[(0,1)(1,-1)]'''
        list1 = []
        index_dict = []
        pairs = []
        for i in a:
            if len(i)==2:
                list1.append(int(i[1]))
            elif len(i)==3:
                a1=int(i[1])*10 + int(i[2])
                list1.append(a1)
            elif len(i)==4:
                a1=int(i[1])*100 + int(i[2])*10 + int(i[3])
                list1.append(a1)
            if i[0] == 'a': index_dict.append(a.index(i))
        for i in range(1, len(index_dict)):
            left = index_dict[i - 1]
            right = index_dict[i]
            temp = tuple(list1[left:right])
            pairs.append(temp)
            if i == len(index_dict) - 1:
                temp1 = tuple(list1[right:])
                pairs.append(temp1)

        # 将不带任务的代理转成(0,-1)
        for pair in pairs:
            if len(pair)==1:
                temp=(list(pair))
                temp.append((-1))
                pairs.remove(pair)
                pairs.append(tuple(temp))
            for pair in pairs:
                if len(pair) == 1:
                    temp = (list(pair))
                    temp.append((-1))
                    pairs.remove(pair)
                    pairs.append(tuple(temp))
                for pair in pairs:
                    if len(pair) == 1:
                        temp = (list(pair))
                        temp.append((-1))
                        pairs.remove(pair)
                        pairs.append(tuple(temp))
        # print("最优分配序列(未排序)：{0}".format(pairs))
        pairs = sorted(pairs, key=lambda x: x[0])
        self.pairs = pairs

        return pairs, round(b, 2)

    "获得AT（补齐），只用于LACAM*算法"
    def get_AT(self,pairs,agent_dict,task_dict):
        "AT：由 pairs 重新整理出AT数据"
        AT = dict()
        # print(agent_dict)
        for pair in pairs:
            if len(pair) == 2:
                if pair[1] == -1:  # 代理不带任务，则删除代理
                    del agent_dict[pair[0]]
                else:
                    list = []
                    list.append(agent_dict[pair[0]])
                    list.append(task_dict[pair[1]][0])
                    list.append(task_dict[pair[1]][1])
                    AT[pair[0]] = list
            elif len(pair) > 2:
                list = []
                list.append(agent_dict[pair[0]])
                for i in range(1, len(pair)):
                    list.append(task_dict[pair[i]][0])
                    list.append(task_dict[pair[i]][1])
                AT[pair[0]] = list

        init_AT=copy.deepcopy(AT)

        "得出新AT，用补齐方法来满足LACAM*的输入条件"
        # 找出元素最多的代理号及其元素总数
        max_elements = 0
        max_agent = None
        for agent, elements in AT.items():
            if len(elements) > max_elements:
                max_elements = len(elements)
                max_agent = agent
        # 对其他代理的元素列表进行补齐
        for agent, elements in AT.items():
            if agent != max_agent:
                # 计算需要补齐的元素数量
                elements_to_add = max_elements - len(elements)
                # 复制最后一个元素来补齐
                for _ in range(elements_to_add):
                    elements.append(elements[-1])
        return init_AT,AT,max_agent