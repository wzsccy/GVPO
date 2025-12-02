import matplotlib.pyplot as plt
import pandas as pd

# 示例数据，这里假设有三个列表，你可以根据实际情况修改
data1 = []
data2 = []
data3 = []

with open("drl_data/output_test01.txt", "r") as file:
    for line in file:
        data1.append(float(line))

with open("drl_data/output_test02.txt", "r") as file:
    for line in file:
        data2.append(float(line))

with open("drl_data/output_test03.txt", "r") as file:
    for line in file:
        data3.append(float(line))

# 设置滚动平均的窗口大小，即平滑程度
rolling_intv = 1
# 使用pandas的rolling方法对数据进行平滑处理
smooth_data1 = pd.Series(data1).rolling(window=rolling_intv, min_periods=1).mean().tolist()
smooth_data2 = pd.Series(data2).rolling(window=rolling_intv, min_periods=1).mean().tolist()
smooth_data3 = pd.Series(data3).rolling(window=rolling_intv, min_periods=1).mean().tolist()

# 生成x轴数据，这里假设x轴是从1开始的连续整数，个数与数据长度相同
x = list(range(1, len(smooth_data1) + 1))

# 绘制折线图
plt.plot(x, smooth_data1, label='α=0.9 γ=0.9')
plt.plot(x, smooth_data2, label='α=0.5 γ=0.5')
plt.plot(x, smooth_data3, label='α=0.1 γ=0.1')

# plt.plot(x, data1, label='α=0.1 γ=0.1')
# plt.plot(x, data2, label='α=0.5 γ=0.5')
# plt.plot(x, data3, label='α=0.9 γ=0.9')
# 添加标题和坐标轴标签
# plt.title('Three Lines Plot')
plt.xlabel('Episodes')
plt.ylabel('Reward')

# 添加图例
plt.legend(loc='lower right')

# 保存图片，这里指定保存为png格式，你可以根据需要修改文件名和格式
plt.savefig('3折线图.pdf')

# 显示图形
plt.show()