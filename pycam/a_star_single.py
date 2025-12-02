# 开发时间：2024/2/27 19:09
# 开发语言：Python
"八方向A*接口使用前提"
'''
地图标记:star、task、goal:1   wall：2    neighbour：4   path_points：6
传入参数：width,height,start,task,goal,wall
输出数据：生成图像、控制台输出路径以及成本
'''

import math
from math import fabs
import numpy as np
from matplotlib import pyplot as plt
#优先栈
class prior_stack():
    def __init__(self):
        self.list=[]   # 节点存储器，node(坐标，f值，h值)
    def push(self,node):
        self.list.append(node)
        self.list.sort(key=lambda x: (x[1], x[2]), reverse=True) #栈内所有节点排序
    def pop(self):
        return self.list.pop()[0]  #弹出第一个节点
    def __len__(self):
        return len(self.list)
class Astar():
    def __init__(self,width,height,start,task,goal,r_time,wall):
        self.width,self.height,self.start,self.task,self.goal,self.r_time = width,height,start,task,goal,r_time
        self.obstacle=np.array(wall)
        self.map=np.zeros((height,width))
        self.ps1=prior_stack()
        self.ps1.push((start,0,0))
        self.ps2 = prior_stack()
        self.ps2.push((task, 0, 0))
        self.map[start]=1
        self.map[task]=1
        self.map[goal]=1


        self.cost=0 # 存储路径成本
        # 设置图中的墙为2
        for item in wall:
            self.map[item] = 2
            if wall[-1] == item: break
        '''搜索的最终结果存储在列表self.path中'''
        self.path=[]
        self.path1=[]   #存放最短路径
        self.path2 = []
        self.search_path1()  #对start-task执行搜索路径
        self.search_path2()  #对task-goal执行搜索路径



    def draw_map(self):
        fig = plt.pcolor(np.flipud(self.map))
        fig.set_edgecolor('w')
        plt.gca().set_aspect(1.0)  # 设置单元格的y/x=1，即正方单元格
        plt.title('A*  Algorithm')
        plt.xticks([])
        plt.yticks([])
        plt.show()  # 显示图表

    def draw_path1(self,parent):
        task = self.task  # 先获取最终的目标节点
        self.path1 = []
        while task:
            self.map[task] = 6  # 地图的值设置为6（颜色更深，更亮）
            # 将最短路径中 目标节点之前的节点依次逆着设置为4，最终最短路径的所有节点都再地图上设置为6
            self.path1.append(task)
            task = parent[task]
        #输出最短路径
        if self.r_time >= len(self.path1):
            cha=self.r_time-len(self.path1)
            element=self.path1[0]
            self.path1=[element]*cha + self.path1
        self.path1=self.path1[::-1]
        # print('输出最短路径:\n{0}'.format(self.path))
        # print("地图的数组显示：\n",self.map)
        self.map[self.start] = 1
        self.map[self.task] = 1
        # self.draw_map()   #更新地图
        return self.path1

    # 搜索的核心代码
    def search_path1(self):
        g_score = {}  # 字典存储g值
        parent = {}  # 字典存储父节点
        parent[self.start] = None  # 开始节点没有父节点，所以设置为NONE
        g_score[self.start] = 0

        while self.ps1:  # 若栈不为空，就一直死循环
            current = self.ps1.pop()  # 弹出栈内f值的最小的节点
            if current == self.task: break  # 当前节点已经是目标节点，就退出循环

            for node in self.get_neighbour(current):  # 取出当前节点的所有邻居节点
                new_g_score = g_score[current] + 1  # 邻居节点的g值=当前节点g值+1

                if node not in g_score or g_score[node] > new_g_score:
                    g_score[node] = new_g_score  # g值更新
                    h = abs(node[0] - self.task[0]) + abs(node[1] - self.task[1])  # 求曼哈顿距离的h值
                    self.ps1.push((node, new_g_score + h, h))  # 存(节点坐标，f,h)
                    self.map[node] = 4  # 所有邻居节点设置为4，方便绘制
                    parent[node] = current
            # self.draw_map() #显示每一步的探索
        self.draw_path1(parent)  # 绘制最后那条路径

    def draw_path2(self,parent):
        goal = self.goal  # 先获取最终的目标节点
        self.path2 = []
        while goal:
            self.map[goal] = 6
            # 将最短路径中 目标节点之前的节点依次逆着设置为4，最终最短路径的所有节点都再地图上设置为4
            self.path2.append(goal)
            goal = parent[goal]

        '''将前面被第二次搜索的路径 覆盖的第一段路径节点的地图值改为6（颜色）'''
        for path1 in self.path1:
            self.map[path1]=6
        #输出最短路径
        self.path2=self.path2[::-1]
        # print('输出最短路径:\n{0}'.format(self.path))
        # print("地图的数组显示：\n",self.map)
        self.map[self.start]= 1
        self.map[self.task] = 1
        self.map[self.goal] = 1
        self.path2.pop(0)
        self.path=self.path1+self.path2
        self.cost=self.path_cost(self.path)
        # print(self.path)
        # print(self.cost)
        '''显示图像的开关'''
        # self.draw_map()   #更新地图

        return self.path2

    # 搜索的核心代码
    def search_path2(self):
        g_score = {}  # 字典存储g值
        parent = {}  # 字典存储父节点
        parent[self.task] = None  # 开始节点没有父节点，所以设置为NONE
        g_score[self.task] = 0

        while self.ps2:  # 若栈不为空，就一直死循环
            current = self.ps2.pop()  # 弹出栈内f值的最小的节点
            if current == self.goal: break  # 当前节点已经是目标节点，就退出循环

            for node in self.get_neighbour(current):  # 取出当前节点的所有邻居节点
                new_g_score = g_score[current] + 1  # 邻居节点的g值=当前节点g值+1

                if node not in g_score or g_score[node] > new_g_score:
                    g_score[node] = new_g_score  # g值更新
                    h = abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])  # 求曼哈顿距离的h值
                    self.ps2.push((node, new_g_score + h, h))  # 存(节点坐标，f,h)
                    self.map[node] = 4  # 所有邻居节点设置为3，方便绘制
                    parent[node] = current
            # self.draw_map() #显示每一步的探索
        self.draw_path2(parent)  # 绘制最后那条路径

    def get_neighbour(self,node):
        list=[]  #存储合法的邻居节点
        '''注释部分是四方向代码（个人认为写的很不错）'''
        # for i in [-1,1]:
        #     if node[0]+i>=0 and node[0]+i<self.width:# 若左右邻居不超出横边界
        #         new_node=node[0]+i,node[1]
        #         if self.check(new_node):list.append(new_node)
        #
        #     if node[1]+i>=0 and node[1]+i<self.height:# 若上下邻居不超出纵边界
        #         new_node=node[0],node[1]+i
        #         if self.check(new_node):list.append(new_node)
        # 上下左右，左上，左下，右上，右下
        if node[1]+1 < self.height: # 上
            new_node=node[0],node[1]+1
            if self.check(new_node):list.append(new_node)
        if node[1]-1 >= 0:  #下
            new_node=node[0],node[1]-1
            if self.check(new_node):list.append(new_node)
        if node[0]+1 < self.width:# 右
            new_node = node[0]+1, node[1]
            if self.check(new_node): list.append(new_node)
        if node[0]-1 >= 0:# 左
            new_node = node[0]-1, node[1]
            if self.check(new_node): list.append(new_node)

        if node[0]-1 >=0 and node[1]+1 < self.height: # 左上
            new_node = node[0]-1, node[1]+1
            if self.check(new_node): list.append(new_node)
        if node[0]-1 >=0 and node[1]-1 >=0: # 左下
            new_node = node[0]-1, node[1]-1
            if self.check(new_node): list.append(new_node)
        if node[0]+1 < self.width and node[1]+1 < self.height: # 右上
            new_node = node[0]+1, node[1]+1
            if self.check(new_node): list.append(new_node)
        if node[0]+1 < self.width and node[1]-1 >=0: # 右下
            new_node = node[0]+1, node[1]-1
            if self.check(new_node): list.append(new_node)
        return list

    # 判断该节点是否遇上障碍物
    def check(self, node):
        b = np.all(self.obstacle == node, axis=1)  # 若邻居节点是障碍物，布尔型数组b就为True
        return not np.any(b)  # 则此处就返回False

    def path_cost(self,path):
        total_cost=0
        try:
            for i in range(1,len(path)):
                a,b=path[i-1],path[i]
                cost1,cost2=0,0
                if fabs(a[0]-b[0])+fabs(a[1]-b[1])==2:
                    cost1=math.sqrt(2)
                else:
                    cost2=1
                total_cost=cost1+cost2+total_cost
        except:
            print("",end="")
        return total_cost
# if __name__ == '__main__':
#     wall=(6,7),(7,7),(8,7),(9,7),(10,7),(11,7)
#     a=Astar(12,12,(0,0),(5,6),(11,11),wall)
