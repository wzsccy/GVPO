# 开发时间：2024/4/3 14:42
# 开发语言：Python
'''---------------------------------------'''
def hamiltonian_cycle_util(graph, path, pos, visited):
    # 如果所有顶点都被访问，并且最后一个顶点与起始顶点相同
    if pos == len(graph):
        if graph[pos][0] == 1:
            return True
        else:
            return False

    # 遍历所有顶点
    for v in range(1, len(graph)):
        # 如果当前顶点未被访问，并且与上一个顶点相邻
        if visited[v] == False and graph[pos][v] == 1:
            # 标记当前顶点为已访问
            visited[v] = True
            path[pos] = v

            # 递归调用，尝试下一个顶点
            if hamiltonian_cycle_util(graph, path, pos + 1, visited):
                return True

            # 如果下一个顶点没有找到哈密顿环，回溯
            visited[v] = False
            path[pos] = -1

    return False

def hamiltonian_cycle(graph):
    path = [-1] * len(graph)
    visited = [False] * len(graph)

    # 从第一个顶点开始
    path[0] = 0
    visited[0] = True

    # 如果找到哈密顿环，返回True
    if hamiltonian_cycle_util(graph, path, 1, visited):
        return path
    else:
        return None

# 示例图，使用邻接矩阵表示
graph = [
    [0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
]

# 找到哈密顿环
hamiltonian_path = hamiltonian_cycle(graph)
if hamiltonian_path:
    print("找到哈密顿环:", hamiltonian_path)
else:
    print("没有找到哈密顿环。")
