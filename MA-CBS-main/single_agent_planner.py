import heapq
import copy
import math


def move(loc, dir):# 4_DIR
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)] #left, down, right, up, stay
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def move_eight(loc, dir):# 8_DIR
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0), (1,1),(-1,-1),(1,-1),(-1,1),(0, 0),] #left, down, right, up, stay
    return loc[0] + directions[dir][0], loc[1] + directions[dir][1]

def get_sum_of_cost(paths): # 4_DIR
    rst = 0
    for path in paths:
        rst += len(path) - 1
    return rst

def get_sum_of_cost_eight(paths): # 8_DIR
    total_cost=0
    if paths is None:
        return -1
    for path in paths:
        for i in range(1,len(path)):
            x1,y1=path[i-1][0],path[i-1][1]
            x2,y2=path[i][0],path[i][1]
            coste = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            total_cost += coste
    return total_cost

def compute_heuristics(my_map, goal):
    # Use Dijkstra to build a shortest-path tree rooted at the goal location
    open_list = []
    closed_list = dict()
    root = {'loc': goal, 'cost': 0}
    heapq.heappush(open_list, (root['cost'], goal, root))
    closed_list[goal] = root
    while len(open_list) > 0:
        (cost, loc, curr) = heapq.heappop(open_list)
        for dir in range(4):
            child_loc = move(loc, dir)
            child_cost = cost + 1
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            child = {'loc': child_loc, 'cost': child_cost}
            if child_loc in closed_list:
                existing_node = closed_list[child_loc]
                if existing_node['cost'] > child_cost:
                    closed_list[child_loc] = child
                    # open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
                    heapq.heappush(open_list, (child_cost, child_loc, child))
            else:
                closed_list[child_loc] = child
                heapq.heappush(open_list, (child_cost, child_loc, child))

    # build the heuristics table
    h_values = dict()
    for loc, node in closed_list.items():
        h_values[loc] = node['cost']
    return h_values


def remove_dup(paths):
    for i in range(len(paths)):
        if len(paths[i])>1:
            count = 0
            length = len(paths[i])-1
            while paths[i][length-count] == paths[i][length-count-1] and length-count-1 >= 0:
                count += 1
            paths[i] = paths[i][0:length+1-count]
    
    return paths



def build_constraint_table(constraints, agent):
    constraint_table = []
    max_time_step = -1
    for con in constraints:
        if con['agent'] == agent:
            if con['timestep'] > max_time_step and con['positive'] ==  False:
                max_time_step = con['timestep']
            constraint_table.append(con)
    return constraint_table, max_time_step


def MA_build_constraint_table(constraints, agent):
    constraint_table = []
    max_time_step = -1
    for con in constraints:
        if agent in con['meta-agent'] or con['agent'] == agent:
            if con['timestep'] > max_time_step and con['positive'] ==  False:
                max_time_step = con['timestep']
            constraint_table.append(con)
    return constraint_table, max_time_step



def get_location(path, time):
    if time < 0:
        return path[0]
    elif time < len(path):
        return path[time]
    else:
        return path[-1]  # wait at the goal location



def get_path(goal_node):
    path = []
    curr = goal_node
    while curr is not None:
        path.append(curr['loc'])
        curr = curr['parent']
    path.reverse()
    #path = remove_dup(path)
    return path


def get_paths(goal_node):
    paths = []
    for i in range(len(goal_node['loc'])):
        paths.append([])
    curr = goal_node
    while curr is not None:
        for i in range(len(goal_node['loc'])):
            paths[i].append(curr['loc'][i])
        curr = curr['parent']
    for i in range(len(goal_node['loc'])):
        paths[i].reverse()
    paths = remove_dup(paths)
    return paths


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
    for con in constraint_table:                
        if len(con['loc']) == 1:
            if con['loc'][0] == next_loc and con['timestep'] == next_time and con['positive'] == False:
                return True
        else:
            if con['loc'][1] == next_loc and con['timestep'] == next_time and con['loc'][0] == curr_loc and con['positive'] == False:
                return True
            
        if len(con['loc']) == 1:
            if con['loc'][0] != next_loc and con['timestep'] == next_time and con['positive'] == True:
                return True
        else:
            if con['loc'][1] != next_loc and con['timestep'] == next_time and con['loc'][0] == curr_loc and con['positive'] == True:
                return True
            
    return False


def push_node(open_list, node):
    heapq.heappush(open_list, (node['g_val'] + node['h_val'] + node['F_next'] ,node['h_val']+node['g_val'], [node['loc'],node['timestep']], node))


def pop_node(open_list):
    _, _, _, curr = heapq.heappop(open_list)
    return curr


def compare_nodes(n1, n2):
    """Return true is n1 is better than n2."""
    return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']

def create_OSF_table(curr,a,my_map,constraint_table,agent,h_values,goal_loc):
    curr_agent = agent[a]
    for i in range(5):
        d_g_val = 0
        d_h_val = 0
        child_loc = move(curr['loc'][a], i)
        
        if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
            or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
            continue
        if my_map[child_loc[0]][child_loc[1]]:
            continue
        if is_constrained(curr['loc'][a], child_loc, curr['timestep'] + 1, constraint_table):
            continue

        bank = curr['bank'][a]

        if child_loc == curr['loc'][a] and child_loc == goal_loc[curr_agent]: #no change at goal location
            bank = bank + 1
            d_g_val = 0
        else:
            d_g_val = bank + 1
            bank = 0
            
        d_h_val = h_values[curr_agent][child_loc] - h_values[curr_agent][curr['loc'][a]]
        d_f_val = d_g_val + d_h_val

        curr['OSF_table'][a].append((d_f_val,i,d_g_val,d_h_val,bank))

    curr['OSF_table'][a].sort()

        

        

def OSF_OP_FIND(op_list,i,sum,curr,OP_list,next_OSV):
    op_list_origin = copy.deepcopy(op_list)
    sum_origin = sum

    #print(op_list_origin)
    #print(sum_origin)
    
    for OSF in curr['OSF_table'][i]: #0 = d_f_val, 1 = op, 2 = d_g_val, 3 = d_h_val, 4 = new_bank_val
        sum = sum_origin + OSF[0]
        child_loc = move(curr['loc'][i], OSF[1])
        op_list['loc'] = op_list_origin['loc'] + [child_loc]
        op_list['g_val'] = op_list_origin['g_val'] + OSF[2]
        op_list['h_val'] = op_list_origin['h_val'] + OSF[3]
        op_list['bank'][i] = OSF[4]

        collision = False
        if i > 0:
            if child_loc in op_list_origin['loc']:
                continue

            for k in range(len(op_list['loc'])):
                for l in range(len(op_list['loc'])):
                    if l > k:
                        if curr['loc'][l] == op_list['loc'][k] and curr['loc'][k] == op_list['loc'][l]:
                            collision = True

        if collision == True:
            continue

        #print(op_list)

        if sum > curr['F_next']:
            sum_temp = sum
            for k in range(i+1,len(curr['OSF_table'])):
                sum_temp += curr['OSF_table'][k][0][0]
            if next_OSV == -1:
                next_OSV = sum_temp
                return next_OSV
            elif sum_temp <= next_OSV:
                next_OSV = sum_temp
                return next_OSV
            else:
                return next_OSV


        if i == len(curr['OSF_table']) - 1:
            if sum ==  curr['F_next']:
                OP_list.append(copy.deepcopy(op_list))
            continue
        
        next_OSV = OSF_OP_FIND(op_list,i+1,sum,curr,OP_list,next_OSV)
    return next_OSV

def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    constraint_table, max_time = build_constraint_table(constraints, agent)
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep' : 0, 'F_next' : 0}
    push_node(open_list, root)
    closed_list[(root['loc'],root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goal_loc and curr['timestep'] >= max_time:
            return get_path(curr)
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep' : curr['timestep'] + 1,
                    'F_next' : 0}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'],child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'],child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'],child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions

def a_star_eight(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """

    constraint_table, max_time = build_constraint_table(constraints, agent)
    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep' : 0, 'F_next' : 0}
    push_node(open_list, root)
    closed_list[(root['loc'],root['timestep'])] = root
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goal_loc and curr['timestep'] >= max_time:
            return get_path(curr)
        for dir in range(9):
            child_loc = move_eight(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraint_table):
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep' : curr['timestep'] + 1,
                    'F_next' : 0}
            if (child['loc'], child['timestep']) in closed_list:
                existing_node = closed_list[(child['loc'],child['timestep'])]
                if compare_nodes(child, existing_node):
                    closed_list[(child['loc'],child['timestep'])] = child
                    push_node(open_list, child)
            else:
                closed_list[(child['loc'],child['timestep'])] = child
                push_node(open_list, child)

    return None  # Failed to find solutions

def bounded_a_star(my_map, start_loc, goal_loc, h_values, agent, boundary):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        boundary    - specific time step to the goal state
    """

    open_list = []
    closed_list = dict()
    earliest_goal_timestep = 0
    h_value = h_values[start_loc]
    root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep' : 0, 'OSF' : 0}
    push_node(open_list, root)
    closed_list[(root['loc'],root['timestep'])] = root
    MDD = []
    count = 1
    while len(open_list) > 0:
        curr = pop_node(open_list)
        if curr['loc'] == goal_loc and curr['timestep'] == boundary:
            MDD.append(get_path(curr))
            continue
        for dir in range(5):
            child_loc = move(curr['loc'], dir)
            if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
               or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
               continue
            if my_map[child_loc[0]][child_loc[1]]:
                continue
            if curr['timestep'] >= boundary:
                continue
            child = {'loc': child_loc,
                    'g_val': curr['g_val'] + 1,
                    'h_val': h_values[child_loc],
                    'parent': curr,
                    'timestep' : curr['timestep'] + 1,
                    'OSF' : 0}
            if (child['loc'], child['timestep']) in closed_list:
                #existing_node = closed_list[(child['loc'],child['timestep'])]
                #if compare_nodes(child, existing_node):
                #    closed_list[(child['loc'],child['timestep'])] = child
                child['h_val'] = child['h_val'] + count
                count = count + 1
                push_node(open_list, child)
            else:
                closed_list[(child['loc'],child['timestep'])] = child
                push_node(open_list, child)
    return MDD

def EPEa_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
    """ my_map      - binary obstacle map
        start_loc   - start position
        goal_loc    - goal position
        agent       - the agent that is being re-planned
        constraints - constraints defining where robot should or cannot go at each timestep
    """
    open_list = []
    closed_list = []
    constraint_table = []
    max_time = []
    cost_bank_initial = []

    root = {'loc': [], 'g_val': 0, 'h_val': 0, 'parent': None, 'timestep' : 0, 'bank' : [], 'F_next' : 0, 'OSF_table': []}

    #print(constraints)
    
    for i in range(len(agent)):
        constraint_table.append([])
        cost_bank_initial.append(0)
        max_time.append(-1)
        
    for i in range(len(agent)):
        root['loc'].append(start_loc[agent[i]])
        constraint_table[i], max_time[i] = MA_build_constraint_table(constraints, agent[i])
        #constraint_table[i] = MA_build_constraint_table(constraints, agent[i]) + []
        root['h_val'] = root['h_val'] + h_values[agent[i]][start_loc[agent[i]]]

    root['bank'] = cost_bank_initial #cost bank for staying at goal location
    closed_list.append([root['loc'],root['timestep']])

    push_node(open_list, root)

    

    while len(open_list) > 0:
        curr = pop_node(open_list)
        next_OSV = -1
        #print('--------')
        #print(curr['g_val'] + curr['h_val'] + curr['F_next'])
        
            
        solution = True
        for i in range(len(agent)): #check every agent is located at goal location
            if curr['loc'][i] != goal_loc[agent[i]] or curr['timestep'] <= max_time[i]:
                solution = False
        if solution == True:
            return get_paths(curr)

        discard = False
        if curr['OSF_table'] == []:
            for i in range(len(agent)):
                if discard == True:
                    continue
                curr['OSF_table'].append([])
                create_OSF_table(curr,i,my_map,constraint_table[i],agent,h_values,goal_loc)
                if curr['OSF_table'][i] == []:
                    discard = True
        
        #print(curr['loc'])
        #print(curr['OSF_table'])

        if discard == True:
            continue
        #print('-------------------------------------------------')
        #print(curr)
        
        OP_list = []
        op_list = {'loc' : [], 'g_val' : curr['g_val'], 'h_val' : curr['h_val'], 'bank' : curr['bank']+[]}
        
        next_OSV = OSF_OP_FIND(op_list,0,0,curr,OP_list,next_OSV)
        #print(next_OSV)
        #print(OP_list)
        #print('-------------------------------------------------')
        for i in range(len(OP_list)):      
            child = {'loc': copy.deepcopy(OP_list[i]['loc']),
                    'g_val': copy.deepcopy(OP_list[i]['g_val']),
                    'h_val': copy.deepcopy(OP_list[i]['h_val']),
                    'parent': curr,
                    'timestep' : curr['timestep'] + 1,
                    'bank' : copy.deepcopy(OP_list[i]['bank']),
                    'F_next' : 0,
                    'OSF_table' : []}

            if [child['loc'],child['timestep']] not in closed_list:
                closed_list.append([child['loc'],child['timestep']])
                push_node(open_list, child)

        if next_OSV != -1:
            curr['F_next'] = next_OSV
            push_node(open_list, curr)
        

    return None  # Failed to find solutions
