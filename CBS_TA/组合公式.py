import concurrent.futures
import itertools
import math
import time
from threading import Thread

import numpy

'''公式计算'''
def gs(m,n):
    result=0
    for i in range(1,n):
        a=math.comb(n,i)*math.perm(n-i,n-i)*math.perm(i,i)
        result =result+a
    result=result+(m*math.perm(n,n))
    return result

def jiechen(m,n):
    return math.factorial(m+n)

''''''
def fact(a,b):
    st=time.time()

    agents = [f'a{i}' for i in range(5)]
    tasks = [f't{i}' for i in range(6)]
    valid_combinations = []
    index_list = []
    gs_num=gs(a,b)
    a_num=int(gs_num/4)
    print()
    temp_tuple=tuple(itertools.permutations(agents + tasks))
    print(len(temp_tuple))
    for i in range(0,a_num):
        temp_list=[]
        combination=temp_tuple[i]
        temp_list.append(temp_tuple[i])
        if combination[0] != 'a0': continue
        if combination[-1] in agents: continue
        if any(combination[j] in agents and combination[j + 1] in agents for j in
               range(len(combination) - 1)): continue
        for i in agents:
            index_list.append(combination.index(i) )
        is_sorted = all(index_list[i] <= index_list[i + 1] for i in range(len(index_list) - 1))
        index_list.clear()
        if not is_sorted:continue
        # 如果组合满足所有条件，添加到 valid_combinations 列表
        valid_combinations.append(combination)


    tt=time.time()
    runt=tt-st
    return runt,len(valid_combinations)


if __name__=="__main__":
    a=jiechen(5,6)
    runt,len=fact(5,6)
    print(a,runt,len)