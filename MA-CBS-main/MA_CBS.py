import time as timer
import heapq
import random
from single_agent_planner import compute_heuristics, a_star,a_star_eight, get_location, get_sum_of_cost, EPEa_star,get_sum_of_cost_eight
# from EPEA import EPEASolver
import copy
import psutil


def should_merge(agent_1, agent_2 , conflict_matrix, list_of_meta_agent, B):
    # Check if 2 agents have more conflicts than B number
    ## If checking two meta agents, then check for sum of all conflict
    index_1 = None
    index_2 = None
    sum_conflict = 0
    for meta in list_of_meta_agent:
        if agent_1 in meta:
            index_1 = list_of_meta_agent.index(meta)
        if agent_2 in meta:
            index_2 = list_of_meta_agent.index(meta)
    if (index_1 is not None) and (index_2 is None):
        for i in list_of_meta_agent[index_1]:
            sum_conflict += conflict_matrix[i][agent_2]
    elif (index_1 is None) and (index_2 is not None):
        for i in list_of_meta_agent[index_2]:
            sum_conflict += conflict_matrix[i][agent_1]
    elif (index_1 is not None) and (index_2 is not None):
        for i in list_of_meta_agent[index_1]:
            for j in list_of_meta_agent[index_2]:
                sum_conflict += conflict_matrix[i][j]
    elif (index_1 is None) and (index_2 is None):
        sum_conflict = conflict_matrix[agent_1][agent_2]

    if sum_conflict >= B:
        return True
    return False



def merge(agent_1, agent_2, collisions, list_of_meta_agent, starts, goals):
    index_meta_1 = None
    index_meta_2 = None
    meta_agent = None
    for meta in list_of_meta_agent:
        if agent_1 in meta:
            index_meta_1 = list_of_meta_agent.index(meta)
        if agent_2 in meta:
            index_meta_2 = list_of_meta_agent.index(meta)
            
    if (index_meta_1 is None) and (index_meta_2 is None):   # merge 2 single agents
        meta_agent = [agent_1, agent_2]
        meta_agent.sort()
        list_of_meta_agent.append(meta_agent)
    elif (index_meta_1 is not None) and (index_meta_2 is None): # agent 1 in meta agents and agent 2 is not in meta
        list_of_meta_agent[index_meta_1].append(agent_2)
        list_of_meta_agent[index_meta_1].sort()
        meta_agent = list_of_meta_agent[index_meta_1]
    elif (index_meta_1 is None) and (index_meta_2 is not None): #agent 2 in meta agents and agent 1 is not in meta
        list_of_meta_agent[index_meta_2].append(agent_1)
        list_of_meta_agent[index_meta_2].sort()
        meta_agent = list_of_meta_agent[index_meta_2]
    elif (index_meta_1 is not None) and (index_meta_2 is not None):       #merge 2 meta agents
        list_of_meta_agent[index_meta_1] = list_of_meta_agent[index_meta_1] + list_of_meta_agent[index_meta_2]
        list_of_meta_agent[index_meta_1].sort()
        meta_agent = list_of_meta_agent[index_meta_1]
        del list_of_meta_agent[index_meta_2]

    return meta_agent

def detect_collision(path1, path2):
    if len(path1)>len(path2):
        length = len(path1)
    else:
        length = len(path2)

    for i in range(length):
        if get_location(path1,i) == get_location(path2,i):
            return [i, [get_location(path1,i)]]
        elif get_location(path1,i) == get_location(path2,i+1) and get_location(path1,i+1) == get_location(path2,i):
            return [i+1, [get_location(path2,i), get_location(path1,i)]]
        
    return None
            
        


def detect_collisions(paths):
    collisions = []
    for i in range(len(paths)):
        for j in range(len(paths)):
            if j > i:
                timestep_loc = detect_collision(paths[i],paths[j])
                if timestep_loc != None:
                    collisions.append({'a1': i, 'a2': j, 'loc': timestep_loc[1], 'timestep': timestep_loc[0]})
    return collisions



def MA_splitting(collision, list_of_meta_agent):
    agent_1 = collision['a1']
    agent_2 = collision['a2']
    index_1 = None
    index_2 = None
    for meta in list_of_meta_agent:
        if agent_1 in meta:
            index_1 = list_of_meta_agent.index(meta)
        if agent_2 in meta:
            index_2 = list_of_meta_agent.index(meta)

    constraints = []
    if len(collision['loc']) == 2:
        if index_1 is not None: #agent 1 is in meta agent
            #for agent in list_of_meta_agent[index_1]:
            #    constraints.append({'meta-agent': list_of_meta_agent[index_1] ,'agent': agent,'opponent' :  collision['a2'],'loc': [collision['loc'][1],collision['loc'][0]], 'timestep': collision['timestep'], 'positive':False})
            constraints.append({'meta-agent': list_of_meta_agent[index_1] ,'agent': collision['a1'],'opponent' :  collision['a2'],'loc': [collision['loc'][1],collision['loc'][0]], 'timestep': collision['timestep'], 'positive':False})
        else:   #agent 1 is a signle agent
            constraints.append({'meta-agent': [] ,'agent': collision['a1'],'opponent' :  collision['a2'], 'loc': [collision['loc'][1],collision['loc'][0]], 'timestep': collision['timestep'], 'positive':False})

        if index_2 is not None:
            #for agent in list_of_meta_agent[index_2]:
            #    constraints.append({'meta-agent': list_of_meta_agent[index_2], 'agent': agent,'opponent' :  collision['a1'], 'loc': [collision['loc'][0],collision['loc'][1]], 'timestep': collision['timestep'], 'positive':False})
            constraints.append({'meta-agent': list_of_meta_agent[index_2], 'agent': collision['a2'],'opponent' :  collision['a1'], 'loc': [collision['loc'][0],collision['loc'][1]], 'timestep': collision['timestep'], 'positive':False})
        else:
            constraints.append({'meta-agent': [], 'agent': collision['a2'],'opponent' :  collision['a1'], 'loc': [collision['loc'][0],collision['loc'][1]], 'timestep': collision['timestep'], 'positive':False})
    else:
        if index_1 is not None: #agent 1 is in meta agent
            #for agent in list_of_meta_agent[index_1]:
            #    constraints.append({'meta-agent': list_of_meta_agent[index_1], 'agent': agent,'opponent' :  collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})
            constraints.append({'meta-agent': list_of_meta_agent[index_1], 'agent': collision['a1'],'opponent' :  collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})
        else:
            constraints.append({'meta-agent': [], 'agent': collision['a1'],'opponent' :  collision['a2'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})

        if index_2 is not None:
            #for agent in list_of_meta_agent[index_2]:
            #    constraints.append({'meta-agent': list_of_meta_agent[index_2],'agent': agent,'opponent' :  collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})
            constraints.append({'meta-agent': list_of_meta_agent[index_2],'agent': collision['a2'],'opponent' :  collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})
        else:
            constraints.append({'meta-agent': [],'agent': collision['a2'],'opponent' :  collision['a1'], 'loc': collision['loc'], 'timestep': collision['timestep'], 'positive':False})

    return constraints



def paths_violate_constraint(constraint, paths):
    assert constraint['positive'] is True
    rst = []
    for i in range(len(paths)):
        if i == constraint['agent']:
            continue
        curr = get_location(paths[i], constraint['timestep'])
        prev = get_location(paths[i], constraint['timestep'] - 1)
        if len(constraint['loc']) == 1:  # vertex constraint
            if constraint['loc'][0] == curr:
                rst.append(i)
        else:  # edge constraint
            if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
                    or constraint['loc'] == [curr, prev]:
                rst.append(i)
    return rst

class MACBSSolver_four(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        ########### FINAL
        conflict_matrix = [[0 for x in range(self.num_of_agents)] for y in range(self.num_of_agents)]      # Create a matrix to keep track agent pairs conflicts

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'matrix': conflict_matrix,
                'list_meta': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        upper_bound = 0
        
        #print(root)
        map_size = 0
        #print(self.my_map[0])
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[i])):
                if self.my_map[i][j] == False:
                    map_size = map_size+1
        upper_bound = map_size
        upper_bound = upper_bound * self.num_of_agents
        
        '''
        for i in range(self.num_of_agents):
            upper_bound = upper_bound + self.heuristics[i][self.starts[i]]
        print(upper_bound)
        '''
        #use upper bound or heuristic(shortest path from goal(0) to start loc(min-cost))
        
        ################
        #B = self.num_of_agents
        self.B = 1000
        ###############

        while len(self.open_list) != 0:
            curr = self.pop_node()
            # print(curr['list_meta'])
            
            if len(curr['collisions']) == 0:
                # print(curr['constraints'])
                # self.print_results(curr)
                return curr['paths']
            collision = curr['collisions'][0]
            

            # Check if merge 2 agents or not by using conflict matrix
            # Increment the conflict number of 2 agents
            agent_1 = collision['a1']
            agent_2 = collision['a2']
            curr['matrix'][agent_1][agent_2] +=1
            curr['matrix'][agent_2][agent_1] +=1
            
            #     ### Merge two agents
            #     ### Update constraint
            if (should_merge(agent_1, agent_2 , curr['matrix'], curr['list_meta'], self.B)):

                meta_agent = merge(agent_1, agent_2, curr['constraints'], curr['list_meta'], self.starts, self.goals)
                #print("meta agent is :", meta_agent, starts_loc, goals_loc)

                meta_paths = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                          meta_agent, curr['constraints'])


                if meta_paths != None:
                    #print("TTTTHIS IS NEW PATHSSSSSS: ",meta_paths)
                    index = 0
                    for agent in meta_agent:
                        curr['paths'][agent] = meta_paths[index] + []
                        index+=1
                    curr['cost'] = get_sum_of_cost(curr['paths'])
                    curr['collisions'] = detect_collisions(curr['paths'])
                    #print(meta_agent)
                    #print(curr['collisions'])
                    if upper_bound >= curr['cost']:#len(child['paths'][agent]):
                        self.push_node(curr)
                continue

            #     ### Call EPEA* search for new_agent

            constraints = MA_splitting(collision, curr['list_meta'])

            for constraint in constraints:

                copy_matrix = copy.deepcopy(curr['matrix'])
                copy_list_meta = copy.deepcopy(curr['list_meta'])
                copy_constraints = copy.deepcopy(curr['constraints'])
                copy_constraints.append(constraint)
                child = {'cost': 0,
                    'constraints': copy_constraints,
                    'paths': curr['paths'] + [],
                    'collisions': [],
                    'matrix': copy_matrix,
                    'list_meta': copy_list_meta}

            
                if len(constraint['meta-agent']) > 0:   ##If it is meta agent then run EPEA*
                    meta_paths = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                          constraint['meta-agent'], child['constraints'])

                    if meta_paths != None:
                        index = 0
                        for agent in constraint['meta-agent']:
                            child['paths'][agent] = meta_paths[index] + []
                            index+=1
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                        if upper_bound >= child['cost']:#len(child['paths'][agent]):
                            self.push_node(child)
                            
                else:   #single agent
                    agent = constraint['agent']
                    path = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                            [agent], child['constraints'])
                    if path != None:
                        child['paths'][agent] = path[0] + []
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                        if upper_bound >= child['cost']:#len(child['paths'][agent]):
                            self.push_node(child)
        raise BaseException('No solutions')

        # self.print_results(root)
        return root['paths']


    def print_results(self, node):
        #print("Expanded Nodes:")
        #print(self.open_list)
        #print(self.heuristics)
        #print(node['paths'])
        print("\n Found a solution! \n")
        CPU_time = (timer.time() - self.start_time) * 1000
        print('B = ' + str(self.B))
        print("CPU time (ms):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
        values = psutil.virtual_memory()
        used_memory= values.used >> 20
        print(values, used_memory)


class MACBSSolver_eight(object):
    """The high-level search of CBS."""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """

        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def push_node(self, node):
        heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
        # print("Generate node {}".format(self.num_of_generated))
        self.num_of_generated += 1

    def pop_node(self):
        _, _, id, node = heapq.heappop(self.open_list)
        # print("Expand node {}".format(id))
        self.num_of_expanded += 1
        return node

    def find_solution(self):
        """ Finds paths for all agents from their start locations to their goal locations

        disjoint    - use disjoint splitting or not
        """

        self.start_time = timer.time()

        ########### FINAL
        conflict_matrix = [[0 for x in range(self.num_of_agents)] for y in
                           range(self.num_of_agents)]  # Create a matrix to keep track agent pairs conflicts

        # Generate the root node
        # constraints   - list of constraints
        # paths         - list of paths, one for each agent
        #               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions     - list of collisions in paths
        root = {'cost': 0,
                'constraints': [],
                'paths': [],
                'collisions': [],
                'matrix': conflict_matrix,
                'list_meta': []}
        for i in range(self.num_of_agents):  # Find initial path for each agent
            path = a_star_eight(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
                          i, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)

        root['cost'] = get_sum_of_cost_eight(root['paths'])
        root['collisions'] = detect_collisions(root['paths'])
        self.push_node(root)

        upper_bound = 0

        # print(root)
        map_size = 0
        # print(self.my_map[0])
        for i in range(len(self.my_map)):
            for j in range(len(self.my_map[i])):
                if self.my_map[i][j] == False:
                    map_size = map_size + 1
        upper_bound = map_size
        upper_bound = upper_bound * self.num_of_agents

        '''
        for i in range(self.num_of_agents):
            upper_bound = upper_bound + self.heuristics[i][self.starts[i]]
        print(upper_bound)
        '''
        # use upper bound or heuristic(shortest path from goal(0) to start loc(min-cost))

        ################
        # B = self.num_of_agents
        self.B = 1000
        ###############

        while len(self.open_list) != 0:
            curr = self.pop_node()
            # print(curr['list_meta'])

            if len(curr['collisions']) == 0:
                # print(curr['constraints'])
                # self.print_results(curr)
                return curr['paths']
            collision = curr['collisions'][0]

            # Check if merge 2 agents or not by using conflict matrix
            # Increment the conflict number of 2 agents
            agent_1 = collision['a1']
            agent_2 = collision['a2']
            curr['matrix'][agent_1][agent_2] += 1
            curr['matrix'][agent_2][agent_1] += 1

            #     ### Merge two agents
            #     ### Update constraint
            if (should_merge(agent_1, agent_2, curr['matrix'], curr['list_meta'], self.B)):

                meta_agent = merge(agent_1, agent_2, curr['constraints'], curr['list_meta'], self.starts, self.goals)
                # print("meta agent is :", meta_agent, starts_loc, goals_loc)

                meta_paths = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                                       meta_agent, curr['constraints'])

                if meta_paths != None:
                    # print("TTTTHIS IS NEW PATHSSSSSS: ",meta_paths)
                    index = 0
                    for agent in meta_agent:
                        curr['paths'][agent] = meta_paths[index] + []
                        index += 1
                    curr['cost'] = get_sum_of_cost(curr['paths'])
                    curr['collisions'] = detect_collisions(curr['paths'])
                    # print(meta_agent)
                    # print(curr['collisions'])
                    if upper_bound >= curr['cost']:  # len(child['paths'][agent]):
                        self.push_node(curr)
                continue

            #     ### Call EPEA* search for new_agent

            constraints = MA_splitting(collision, curr['list_meta'])

            for constraint in constraints:

                copy_matrix = copy.deepcopy(curr['matrix'])
                copy_list_meta = copy.deepcopy(curr['list_meta'])
                copy_constraints = copy.deepcopy(curr['constraints'])
                copy_constraints.append(constraint)
                child = {'cost': 0,
                         'constraints': copy_constraints,
                         'paths': curr['paths'] + [],
                         'collisions': [],
                         'matrix': copy_matrix,
                         'list_meta': copy_list_meta}

                if len(constraint['meta-agent']) > 0:  ##If it is meta agent then run EPEA*
                    meta_paths = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                                           constraint['meta-agent'], child['constraints'])

                    if meta_paths != None:
                        index = 0
                        for agent in constraint['meta-agent']:
                            child['paths'][agent] = meta_paths[index] + []
                            index += 1
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                        if upper_bound >= child['cost']:  # len(child['paths'][agent]):
                            self.push_node(child)

                else:  # single agent
                    agent = constraint['agent']
                    path = EPEa_star(self.my_map, self.starts, self.goals, self.heuristics,
                                     [agent], child['constraints'])
                    if path != None:
                        child['paths'][agent] = path[0] + []
                        child['collisions'] = detect_collisions(child['paths'])
                        child['cost'] = get_sum_of_cost(child['paths'])
                        if upper_bound >= child['cost']:  # len(child['paths'][agent]):
                            self.push_node(child)
        raise BaseException('No solutions')

        # self.print_results(root)
        return root['paths']

    def print_results(self, node):
        # print("Expanded Nodes:")
        # print(self.open_list)
        # print(self.heuristics)
        # print(node['paths'])
        print("\n Found a solution! \n")
        CPU_time = (timer.time() - self.start_time) * 1000
        print('B = ' + str(self.B))
        print("CPU time (ms):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))
        values = psutil.virtual_memory()
        used_memory = values.used >> 20
        print(values, used_memory)