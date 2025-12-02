import math
import random

import matplotlib.pyplot as plt


class PlotListReciprocal:
    def __init__(self, input_list,file_path):
        self.input_list = input_list
        self.calculate_points()
        self.plot_and_save(file_path)

    def calculate_points(self):
        """
        计算x点和y点
        """
        y_points = [ element for element in self.input_list]

        x_points = [index + 1 for index in range(len(self.input_list))]
        return x_points, y_points

    def plot_and_save(self, file_path="plot_result.png"):
        """
        绘制图像并保存
        """
        x_points, y_points = self.calculate_points()

        plt.plot(x_points, y_points)  # 去掉了 marker='o' 参数
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        # plt.title('列表元素索引与元素倒数的关系图')
        plt.grid(True)

        plt.savefig(file_path)
        plt.close()

if __name__ == '__main__':
    read_list =[]
    with open('drl_data\output_test03.txt', 'r') as file:
        for line in file:
            read_list.append(float(line.strip()))

    # # 数据处理
    # for i in read_list:
    #     print(type(i))
    # data_list = []
    # random.seed(2)
    # for elm in read_list:
    #     num_float = random.random()
    #     i = elm + num_float
    #     data_list.append(i)
    #
    #
    # print(data_list)
    plt = PlotListReciprocal(read_list,'drl_pic/output_test03.pdf')