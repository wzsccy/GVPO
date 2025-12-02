'''-----------------------------------------------------------'''
from collections import deque # 导入一个双端队列
from dataclasses import dataclass, field
import numpy as np
from .mapf_utils import Coord, Grid, get_neighbors, is_valid_coord
'''-----------------------------------------------------------'''

"这个类的目的是提供一个简单的方式来计算图中顶点之间的最短路径。它使用广度优先搜索算法来遍历图，"
"并使用一个距离矩阵来记录每个顶点到目标顶点的最短距离。如果在搜索过程中没有找到目标顶点，则返回 NIL 值。"
""" 
input:grid,goal
method:BFS
output:.get(v_now_i) 获取当前代理位置到目标点的最短路径(如果输入顶点错误就会输出地图最大个数)
"""

'调用方法'
@dataclass
class DistTable:
    grid: Grid
    goal: Coord
    # field(init=False):指创建实例时，不设置这些类属性
    Q: deque = field(init=False) # 存储待处理的顶点
    table: np.ndarray = field(init=False)  # distance matrix（顶点到目标顶点的最短距离）
    NIL: int = field(init=False)

    def __post_init__(self):
        self.NIL = self.grid.size   # 存储二维数组的元素总个数
        self.Q = deque([self.goal]) # 双端对列Q
        #  创建一个与网格形状相同的 numpy 数组，所有元素初始化为 NIL
        self.table = np.full(self.grid.shape, self.NIL, dtype=int)
        self.table[self.goal] = 0 # 将表中的目标顶点值初始为0

    def get(self, target: Coord) -> int:
        # check valid input
        if not is_valid_coord(self.grid, target):
            return self.grid.size

        # distance has been known
        if self.table[target] < self.table.size:
            return self.table[target]

        # BFS with lazy evaluation
        while len(self.Q) > 0:
            u = self.Q.popleft()
            d = int(self.table[u])
            for v in get_neighbors(self.grid, u):
                if d + 1 < self.table[v]:
                    self.table[v] = d + 1
                    self.Q.append(v)
            if u == target:
                return d

        return self.NIL
