# 开发时间：2025/3/21 19:44
# 开发语言：Python
'''---------------------------------------'''
A = {'a3': ['t1'], 'a7': ['t18'], 'a0': ['t8', 't2'], 'a5': ['t3'], 'a1': ['t9', 't6', 't14', 't17', 't12'],
     'a6': ['t19', 't4', 't11', 't13', 't16', 't15'], 'a8': ['t5', 't0', 't7'], 'a9': ['t10'], 'a2': [], 'a4': []}

def sort_dict_by_agent(d):
    sorted_keys = sorted(d.keys(), key=lambda x: int(x[1:]))  # 按照代理号（即去掉'a'后的数字）从小到大排序
    new_dict = {k: d[k] for k in sorted_keys}  # 构建新的字典
    return new_dict

sorted_A = sort_dict_by_agent(A)
print(sorted_A)