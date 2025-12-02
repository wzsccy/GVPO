import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

# TypeAlias是一种类型别名的声明方式，它允许你为现有类型创建一个新名字,譬如Grid是一个新类型名字，但其本质是一个多维数组
Grid: TypeAlias = np.ndarray
Coord: TypeAlias = tuple[int, int]  # y, x

@dataclass
class Config:
    positions: list[Coord] = field(default_factory=lambda: []) #一个存坐标的列表，在
    def __getitem__(self, k: int) -> Coord:
        return self.positions[k]
    def __setitem__(self, k: int, coord: Coord) -> None:
        self.positions[k] = coord
    def __len__(self) -> int:
        return len(self.positions)
    def __hash__(self) -> int:
        return hash(tuple(self.positions))
    def append(self, coord: Coord) -> None:
        self.positions.append(coord)

Configs: TypeAlias = list[Config]


@dataclass
class Deadline:
    time_limit_ms: int
    def __post_init__(self) -> None:
        self.start_time = time.time()
    @property
    # 返回运行时间
    def elapsed(self) -> float:
        return (time.time() - self.start_time) * 1000
    @property
    # 判断运行时间是否超出截止时间
    def is_expired(self) -> bool:
        return self.elapsed > self.time_limit_ms


"从map文件中获取地图文件"
def get_grid(map_file: str | Path):
    width, height,dimension = 0, 0,[]
    obstacales = []

    with open(map_file, "r") as f:
        # retrieve map size
        for row in f:
            # get width
            res = re.match(r"width\s(\d+)", row)
            if res:
                width = int(res.group(1))

            # get height
            res = re.match(r"height\s(\d+)", row)
            if res:
                height = int(res.group(1))

            if width > 0 and height > 0:
                break
        dimension.append(height)
        dimension.append(width)
        # retrieve map
        grid = np.zeros((height, width), dtype=bool)
        y = 0
        for row in f:
            row = row.strip()
            if len(row) == width and row != "map":
                grid[y] = [s == "." for s in row]
                y += 1
        for i in range(len(grid)):  # i是行号
            for j in range(len(grid[i])):  # j是列号
                if not grid[i][j]:  # 如果元素值为False
                    obstacales.append((i, j))  # 将坐标添加到列表中

    # simple error check
    assert y == height, f"map format seems strange, check {map_file}"

    # grid[y, x] -> True: available, False: obstacle
    return grid,dimension,obstacales

"从scen文件中获取starts,goals 【后续扩展为starts,tasks,goals】"

def get_scenario00(scen_file: str | Path,N):

    with open(scen_file, "r") as f:
        starts, goals = Config(), Config()

        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s,y_s, x_g,y_g= [int(res.group(k)) for k in range(1, 5)]

                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))
                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break


    return starts, goals
def get_scenario(scen_file: str | Path, Num_A: int,Num_T: int):
    All_data_list,flag=[],True
    agent_dict,task_dict=dict(),dict()

    if (Num_A + Num_T * 2) % 2 == 0:
        N = int((Num_A + Num_T * 2)/2)
    else:
        N=int(1+(Num_A + Num_T * 2)/2)
        flag=False
    with open(scen_file, "r") as f:
        starts, goals = Config(), Config()

        for row in f:
            res = re.match(
                r"\d+\t.+\.map\t\d+\t\d+\t(\d+)\t(\d+)\t(\d+)\t(\d+)\t.+", row
            )
            if res:
                x_s,y_s, x_g,y_g= [int(res.group(k)) for k in range(1, 5)]

                starts.append((y_s, x_s))  # align with grid
                goals.append((y_g, x_g))
                All_data_list.append((y_s, x_s))
                All_data_list.append(((y_g, x_g)))

                # check the number of agents
                if (N is not None) and len(starts) >= N:
                    break


    # print(All_data_list)
    if flag==False:
        All_data_list = All_data_list[0:-1]

    agent_list=All_data_list[0:Num_A]
    task_list=All_data_list[Num_A:]
    # set agent_dict,task_dict
    for i,v in enumerate(agent_list):
        agent_dict[i]=v
        # print(i,v)
    j=0
    for i in range(0,len(task_list),2): # 假定i=0
        list=[]
        list.append(task_list[i])
        list.append(task_list[i+1])
        # print(i+1)
        random.seed(1228)
        list.append(random.randint(0, 10))
        task_dict[j]=list
        j+=1
    # print(agent_dict,task_dict)
    return starts, goals , All_data_list,agent_dict,task_dict

"判断节点坐标合法性，是否超出边界，是否是障碍"
def is_valid_coord(grid: Grid, coord: Coord) -> bool:
    y, x = coord
    if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1] or not grid[coord]:
        return False
    return True


def len_neighbors_swap(grid: Grid, coord: Coord, grid_self: Grid ) :
    neigh: list[Coord] = []
    y, x = coord
    NO_AGENTS=2147483647

    # check valid input
    if not is_valid_coord(grid_self, coord):
        return 0

    y, x = coord
    if x > 0 and grid_self[y, x - 1] and grid[y, x - 1] == NO_AGENTS:
        neigh.append((y, x - 1))  # 左
    if x < grid.shape[1] - 1 and grid[y, x + 1] and grid[y, x + 1] == NO_AGENTS:
        neigh.append((y, x + 1))  # 右
    if y > 0 and grid[y - 1, x] and grid[y - 1, x ] == NO_AGENTS:
        neigh.append((y - 1, x))  # 上
    if y < grid.shape[0] - 1 and grid[y + 1, x] and grid[y + 1, x] == NO_AGENTS:
        neigh.append((y + 1, x))  # 下

    return len(neigh)

def get_neighbors_4(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []
    y, x = coord

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord
    if x > 0 and grid[y, x - 1]:
            neigh.append((y, x - 1))    # 左
    if x < grid.shape[1] - 1 and grid[y, x + 1]:
            neigh.append((y, x + 1))    # 右
    if y > 0 and grid[y - 1, x]:
            neigh.append((y - 1, x))    # 上
    if y < grid.shape[0] - 1 and grid[y + 1, x]:
            neigh.append((y + 1, x))    # 下

    return neigh
"获取节点的邻居节点，【后续扩展为八方向】"
def get_neighbors(grid: Grid, coord: Coord) -> list[Coord]:
    # coord: y, x
    neigh: list[Coord] = []
    y, x = coord

    # check valid input
    if not is_valid_coord(grid, coord):
        return neigh

    y, x = coord
    if x > 0 and grid[y, x - 1]:
            neigh.append((y, x - 1))    # 左
    if x < grid.shape[1] - 1 and grid[y, x + 1]:
            neigh.append((y, x + 1))    # 右
    if y > 0 and grid[y - 1, x]:
            neigh.append((y - 1, x))    # 上
    if y < grid.shape[0] - 1 and grid[y + 1, x]:
            neigh.append((y + 1, x))    # 下

    if x>0 and y>0 and grid[y - 1, x-1]:
            neigh.append((y - 1, x-1))  # 左上
    if x>0 and y < grid.shape[0] - 1 and grid[y + 1, x-1]:
            neigh.append((y + 1, x-1))  # 左下
    if x < grid.shape[1] - 1 and y>0 and grid[y-1,x+1]:
            neigh.append((y-1,x+1))     # 右上
    if x < grid.shape[1] - 1 and y<grid.shape[0]-1 and grid[y+1,x+1]:
            neigh.append((y+1,x+1))     # 右下
    return neigh

"把所有路径配置保存到输出文件中"
def save_configs_for_visualizer(configs: Configs, filename: str | Path) -> None:
    output_dirname = Path(filename).parent
    if not output_dirname.exists():
        output_dirname.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for t, config in enumerate(configs):
            row = f"{t}:" + "".join([f"({x},{y})," for (y, x) in config]) + "\n"
            f.write(row)

"验证方案是否有效的一部分，如果无效则输出哪个方面的问题，比如起点不一致"
def validate_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> None:
    # 验证起点位置
    assert all(
        [u == v for (u, v) in zip(starts, solution[0])]
    ), "invalid solution, check starts"

    # 验证目标位置
    assert all(
        [u == v for (u, v) in zip(goals, solution[-1])]
    ), "invalid solution, check goals"

    T = len(solution)
    N = len(starts)

    for t in range(T):
        for i in range(N):
            v_i_now = solution[t][i]
            v_i_pre = solution[max(t - 1, 0)][i]

            # check continuity
            assert v_i_now in [v_i_pre] + get_neighbors(
                grid, v_i_pre
            ), "invalid solution, check connectivity"

            # check collision
            for j in range(i + 1, N):
                v_j_now = solution[t][j]
                v_j_pre = solution[max(t - 1, 0)][j]
                assert not (v_i_now == v_j_now), "invalid solution, vertex collision"
                assert not (
                    v_i_now == v_j_pre and v_i_pre == v_j_now
                ), "invalid solution, edge collision"

"完整的方案验证有效方法，抛出异常"
def is_valid_mapf_solution(
    grid: Grid,
    starts: Config,
    goals: Config,
    solution: Configs,
) -> bool:
    try:
        validate_mapf_solution(grid, starts, goals, solution)
        return True
    except Exception as e:
        print(e)
        return False

"累计损失：计算每个配置中没有到达目标配置的代理数总和"
def get_sum_of_loss(configs: Configs) -> int:
    cost = 0
    for t in range(1, len(configs)):
        cost += sum(
            [
                not (v_from == v_to == goal)
                for (v_from, v_to, goal) in zip(configs[t - 1], configs[t], configs[-1])
            ]
        )
        print("203",cost)
    return cost
