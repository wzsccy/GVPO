import copy
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib as mpl
import numpy as np
import os


class Draw():
    def __init__(self,img_path,wh,save_fig_name,size,deadline):
        self.img_path = img_path
        self.img_name = img_path.split('\\')[-1][:-4]
        self.size=size
        self.deadline=deadline
        self.save_fig_name = save_fig_name
        self.wh=wh

    def draw_one_graph(self,x,y1,y2,y3,img_path,save_fig_name):
        self.x = x
        self.y1 = [element / self.size for element in y1]
        self.y2 = [element / self.size for element in y2]
        self.y3 = [element / self.size for element in y3]
        # 创建折线图
        fig, ax = plt.subplots()
        # 画出三个不同颜色样式的折线
        line1, = ax.plot(x, y1, label='Lacam*', color='red', linestyle='-', linewidth=2)
        line2, = ax.plot(x, y2, label='PIBT', color='green', linestyle='--', linewidth=2)
        line3, = ax.plot(x, y3, label='CBS', color='blue', linestyle='-.', linewidth=2)
        # 设置图例标题靠左加粗
        plt.title('Soc\n{0} ---|A|=5'.format(self.img_name), loc='left', fontweight='bold')
        ax.legend(handles=[line1, line2, line3], loc='upper left')

        # 设置x、y轴标题靠右加粗
        # plt.xlabel('Number of tasks', x=1.0, ha='right',fontweight='bold')
        ax.set_xlabel('Number of tasks',fontsize=14, color='black', style='italic',fontweight='bold')
        ax.set_ylabel('Soc', fontsize=14, color='black', style='italic',fontweight='bold')

        # 设置x轴和y轴的范围
        ax.set_ylim(0, 1)

        # 添加y轴的虚线
        y_ticks = ax.get_yticks()   # 获取Y轴的刻度
        for tick in y_ticks:        # 为Y轴上除0以外所有的刻度添加虚线
            if tick != 0 or tick != 1.0:
                ax.axhline(y=tick, linestyle='--', color='gray', alpha=0.5)

        # 在折线图的右边插入一个设定的已知的矢量图
        # 假设矢量图的路径是 'path_to_vector_image.svg'
        vector_img_path =self.img_path  # 替换为你的矢量图路径
        axins = fig.add_axes([0.75, 0.75, 0.2, 0.2])  # 创建一个新轴，位置在主图的右边
        img = mpl.image.imread(self.img_path)
        axins.imshow(img, aspect='auto')
        axins.axis('off')  # 关闭新轴的坐标轴

        # 保存
        plt.savefig("images"+"\{0}".format(self.save_fig_name), format='svg', dpi=300, bbox_inches='tight', pad_inches=0)

        # 显示图表
        plt.show()
    def draw_four_graph(self,x,SOC_y,Run_Time,Success_Rate,r_uper,r_lower,s_uper,s_lower):
        a,b,c=SOC_y,Run_Time,Success_Rate # 数据传入
        a = [[None if x == 0 else x for x in a[0]],
             [None if x == 0 else x for x in a[1]],
             [None if x == 0 else x for x in a[2]],
             [None if x == 0 else x for x in a[3]]]
        b = [[self.deadline if x == 0 or x > self.deadline else x for x in b[0]],
             [self.deadline if x == 0 or x > self.deadline else x for x in b[1]],
             [self.deadline if x == 0 or x > self.deadline else x for x in b[2]],
             [self.deadline if x == 0 or x > self.deadline else x for x in b[3]],]

        r_uper=[[self.deadline if x == 0 or x > self.deadline else x for x in r_uper[0]],
                [self.deadline if x == 0 or x > self.deadline else x for x in r_uper[1]],
                [self.deadline if x == 0 or x > self.deadline else x for x in r_uper[2]],
                [self.deadline if x == 0 or x > self.deadline else x for x in r_uper[3]]]
        r_lower=[[self.deadline if x == 0 or x > self.deadline else x for x in r_lower[0]],
                 [self.deadline if x == 0 or x > self.deadline else x for x in r_lower[1]],
                 [self.deadline if x == 0 or x > self.deadline else x for x in r_lower[2]],
                 [self.deadline if x == 0 or x > self.deadline else x for x in r_lower[3]]]

        plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
        colors = ['#66CC66','#4e6cb0','#ba4f56'] # PIBT EECBS LACAM*
        line_style=['-.','--','-']# PIBT EECBS LACAM*
        fig, axs = plt.subplots(3, 1, figsize=(6,9)) # 创建一个包含三个子图的窗口，每个子图的排列方式为从上到下

        "第一折线图"
        axs[0].plot(x, a[2], label='PIBT', color=colors[0],marker='*', markersize=15,linestyle=line_style[0], linewidth=2)
        axs[0].plot(x, a[3], label='EECBS', color=colors[1],marker='v', markersize=10,linestyle=line_style[1], linewidth=2)
        # axs[0].plot(x, a[1], label='LACAM', color='#4169E1' , marker='H',markersize=10,linestyle='--', linewidth=1)
        axs[0].plot(x, a[0], label='LACAM*', color=colors[2], marker='d',markersize=8,linestyle=line_style[2], linewidth=2)
        axs[0].set_title('\n{0}\n{1} (|V|={2})  |T|=100  Runtime limit={3}\n'.format(self.img_name,self.wh,self.size,self.deadline), loc='left', fontweight='bold')
        axs[0].legend(loc='lower left') # 添加图例
        axs[0].grid(True)  # 为第一个子图添加网格线

        # 计算振幅的波动范围，例如上下波动10%
        y_upper = s_uper
        y_lower = s_lower
        x_c_0 = copy.deepcopy(x)
        x_c_1 = copy.deepcopy(x)
        x_c_2 = copy.deepcopy(x)
        x_c_3 = copy.deepcopy(x)
        x_c = [x_c_0,x_c_1,x_c_2,x_c_3]

        # 操作振幅
        for i in range(4):
            zero_indices = []
            for index, value in enumerate(y_upper[i]):
                if value==0.0 or value == 0  :
                    zero_indices.append(index)
            if zero_indices:
                for index in reversed(zero_indices):
                    del x_c[i][index],y_upper[i][index], y_lower[i][index]

        # 在主要折线图的上下添加淡色区域
        axs[0].fill_between(x_c_0, y_lower[0], y_upper[0],  color='#00FA9A', alpha=0.1)
        # axs[0].fill_between(x_c_1, y_lower[1], y_upper[1], color='#4169E1', alpha=0.1)
        axs[0].fill_between(x_c_2, y_lower[2], y_upper[2], color='#F4A460', alpha=0.3)
        axs[0].fill_between(x_c_3, y_lower[3], y_upper[3], color='blue', alpha=0.1)
        # axs[0].set_xlabel('Number of agents',loc='right',fontsize=10, color='black', style='italic',fontweight='bold')
        axs[0].set_ylabel('Soc',loc='center',fontsize=10, color='black', style='normal',fontweight='bold')
        # axs[0].set_xlim(0, 10)
        # axs[0].set_ylim(0, 15000)
        # 右上插图
        axins = fig.add_axes([0.84, 0.91, 0.12, 0.08])  # 创建一个新轴，位置在主图的右边
        img = mpl.image.imread(self.img_path)
        axins.imshow(img, aspect='auto')
        axins.axis('off')  # 关闭新轴的坐标轴


        "第二折线图"
        # axs[1].set_title('Run_Time',loc='left', fontweight='bold')
        axs[1].plot(x, b[2], label='PIBT', color=colors[0],  marker='*', markersize=15,linestyle=line_style[0],linewidth=2)
        axs[1].plot(x, b[3], label='EECBS', color=colors[1], marker='v', markersize=10,linestyle=line_style[1],linewidth=2)
        # axs[1].plot(x, b[1], label='LACAM', color='#4169E1', markerfacecolor='white', marker='H',markersize=10, linestyle='--', linewidth=1)
        axs[1].plot(x, b[0], label='LACAM*', color=colors[2],  marker='d', markersize=8,linestyle=line_style[2],linewidth=2)
        # 轴范围
        # axs[1].set_ylim(min(b[0])-1,max(b[0])+1)
        # 计算振幅的波动范围，例如上下波动10%
        y_upper = r_uper
        y_lower = r_lower

        # 在主要折线图的上下添加淡色区域
        axs[1].fill_between(x, y_lower[2], y_upper[2], color='#F4A460', alpha=0.1)
        # axs[1].fill_between(x, y_lower[1], y_upper[1], color='#4169E1', alpha=0.1)
        axs[1].fill_between(x, y_lower[0], y_upper[0], color='#00FA9A', alpha=0.1)
        axs[1].fill_between(x, y_lower[3], y_upper[3], color='blue', alpha=0.1)
        axs[1].set_ylabel('Run_Time(Sec)', loc='center', fontsize=10, color='black', style='normal', fontweight='bold')
        axs[1].grid(True)

        "第三折线图"
        axs[2].plot(x, c[2], label='PIBT', color=colors[0], marker='*',markersize=15, linestyle=line_style[0],linewidth=2)
        axs[2].plot(x, c[3], label='EECBS', color=colors[1], marker='v',markersize=10, linestyle=line_style[1],linewidth=2)
        # axs[2].plot(x, c[1], label='LACAM', color='#4169E1', markerfacecolor='white', marker='H', markersize=10,linestyle='--',linewidth=1)
        axs[2].plot(x, c[0], label='LACAM*', color=colors[2],  marker='d', markersize=8,linestyle=line_style[2], linewidth=2)
        # axs[2].set_title('Success_Rate',loc='left', fontweight='bold')
        axs[2].set_xlabel('Number of agents', loc='right', fontsize=10, color='black', style='normal',fontweight='bold')
        axs[2].set_ylabel('Success_Rate', loc='center', fontsize=10, color='black', style='normal', fontweight='bold')
        axs[2].set_ylim(-0.1,1.1)
        axs[2].grid(True)

        # 调整子图之间的间距
        fig.tight_layout()

        # 保存
        fig.savefig("images" + "\{0}.pdf".format(self.save_fig_name), format='pdf', dpi=300, bbox_inches='tight',pad_inches=0)
        # plt.savefig("tttt1111.svg", format='svg', dpi=300, bbox_inches='tight',pad_inches=0)
        # 显示图表
        plt.show()

    def draw_six_graph(self, x, SOC_y, Run_Time, Success_Rate, r_uper, r_lower, s_uper, s_lower):
        a, b, c = SOC_y, Run_Time, Success_Rate  # 数据传入
        a = [[None if x == 0 else x for x in row] for row in a]
        b = [[self.deadline if x == 0 or x > self.deadline else x for x in row] for row in b]
        r_uper = [[self.deadline if x == 0 or x > self.deadline  else x for x in row] for row in r_uper]
        r_lower = [[self.deadline if x == 0 or x > self.deadline  else x for x in row] for row in r_lower]

        plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
        colors = [ '#66CC66','#ba4f56', '#4e6cb0','#CFB53B',"#00CED1","#215E21"]  # PIBT EECBS LACAM*
        # Alg_name = ['LACAM*_ITA','ELACAM*_ITA','ML_EECBS_ITA','EML_EECBS_ITA','MA_CBS_ITA','EMA_CBS_ITA']
        Alg_name = ['ELaCAM*_IHTA+DDQN','ELaCAM*_IHTA+IGRPO','EML_EECBS_HTA','EML_EECBS_IHTA','EMA_CBS_HTA','EMA_CBS_IHTA']
        line_style = [':', '--', '-','-.']  # PIBT EECBS LACAM*
        fig, axs = plt.subplots(3, 1, figsize=(6, 9))  # 创建一个包含三个子图的窗口，每个子图的排列方式为从上到下

        "第一折线图"
        axs[0].plot(x, a[5], label=Alg_name[5], color=colors[5], marker='^', markersize=10, linestyle=line_style[1],linewidth=2)
        axs[0].plot(x, a[4], label=Alg_name[4], color=colors[4], marker='o', markersize=10, linestyle=line_style[0],linewidth=2)
        axs[0].plot(x, a[3], label=Alg_name[3], color=colors[3], marker='*', markersize=15, linestyle=line_style[1],linewidth=2)
        axs[0].plot(x, a[2], label=Alg_name[2], color=colors[2], marker='v', markersize=10, linestyle=line_style[3],linewidth=2)
        axs[0].plot(x, a[1], label=Alg_name[1], color=colors[1] , marker='H',markersize=10,linestyle=line_style[2], linewidth=1.5)
        axs[0].plot(x, a[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],linewidth=2)
        axs[0].set_title(
            '\n{0}\n{1} (|V|={2})  |T|=200  Runtime limit={3}\n'.format(self.img_name, self.wh, self.size, self.deadline),
            loc='left', fontweight='bold')
        # axs[0].legend(loc='lower left')  # 添加图例
        axs[0].grid(True)  # 为第一个子图添加网格线

        # 计算振幅的波动范围，例如上下波动10%
        y_upper,y_lower  = s_uper,s_lower
        x_c_0 = copy.deepcopy(x)
        x_c_1 = copy.deepcopy(x)
        x_c_2 = copy.deepcopy(x)
        x_c_3 = copy.deepcopy(x)
        x_c_4 = copy.deepcopy(x)
        x_c_5 = copy.deepcopy(x)
        x_c = [x_c_0, x_c_1, x_c_2, x_c_3, x_c_4, x_c_5]

        # 操作振幅
        for i in range(6):
            zero_indices = []
            for index, value in enumerate(y_upper[i]):
                if value == 0.0 or value == 0:
                    zero_indices.append(index)
            if zero_indices:
                for index in reversed(zero_indices):
                    del x_c[i][index], y_upper[i][index], y_lower[i][index]

        # 在主要折线图的上下添加淡色区域
        axs[0].fill_between(x_c_5, y_lower[5], y_upper[5], color=colors[5], alpha=0.2)
        axs[0].fill_between(x_c_4, y_lower[4], y_upper[4], color=colors[4], alpha=0.3)
        axs[0].fill_between(x_c_3, y_lower[3], y_upper[3], color=colors[3], alpha=0.2)
        axs[0].fill_between(x_c_2, y_lower[2], y_upper[2], color=colors[2], alpha=0.3)
        axs[0].fill_between(x_c_1, y_lower[1], y_upper[1], color=colors[1], alpha=0.2)
        axs[0].fill_between(x_c_0, y_lower[0], y_upper[0], color=colors[0], alpha=0.3)
        # axs[0].set_xlabel('Number of Agents',loc='center',fontsize=10, color='black', style='italic',fontweight='bold')
        axs[0].set_ylabel('Soc', loc='center', fontsize=15, color='black', style='normal', fontweight='bold')
        # axs[0].set_xlim(0, 10)
        # axs[0].set_ylim(0, 15000)
        # 右上插图
        axins = fig.add_axes([0.84, 0.91, 0.12, 0.08])  # 创建一个新轴，位置在主图的右边
        img = mpl.image.imread(self.img_path)
        axins.imshow(img, aspect='auto')
        axins.axis('off')  # 关闭新轴的坐标轴

        "第二折线图"
        # axs[1].set_title('Run_Time',loc='left', fontweight='bold')
        axs[1].plot(x, b[5], label=Alg_name[5], color=colors[5], marker='^', markersize=10, linestyle=line_style[1],
                    linewidth=2)
        axs[1].plot(x, b[4], label=Alg_name[4], color=colors[4], marker='o', markersize=10, linestyle=line_style[0],
                    linewidth=2)
        axs[1].plot(x, b[3], label=Alg_name[3], color=colors[3], marker='*', markersize=15, linestyle=line_style[1],
                    linewidth=2)
        axs[1].plot(x, b[2], label=Alg_name[2], color=colors[2], marker='v', markersize=10, linestyle=line_style[3],
                    linewidth=2)
        axs[1].plot(x, b[1], label= Alg_name[1], color=colors[1], marker='H', markersize=10, linestyle=line_style[2],
                    linewidth=1.5)
        axs[1].plot(x, b[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],linewidth=2)
        # 轴范围
        # axs[1].set_ylim(min(b[0])-1,max(b[0])+1)
        # 计算振幅的波动范围，例如上下波动10%
        y_upper = r_uper
        y_lower = r_lower

        # 在主要折线图的上下添加淡色区域
        axs[1].fill_between(x, y_lower[5], y_upper[5], color=colors[5], alpha=0.2)
        axs[1].fill_between(x, y_lower[4], y_upper[4], color=colors[4], alpha=0.3)
        axs[1].fill_between(x, y_lower[3], y_upper[3], color=colors[3], alpha=0.2)
        axs[1].fill_between(x, y_lower[2], y_upper[2], color=colors[2], alpha=0.3)
        axs[1].fill_between(x, y_lower[1], y_upper[1], color=colors[1], alpha=0.2)
        axs[1].fill_between(x, y_lower[0], y_upper[0], color=colors[0], alpha=0.3)

        # axs[1].fill_between(x, y_lower[2], y_upper[2], color='#F4A460', alpha=0.1)
        # # axs[1].fill_between(x, y_lower[1], y_upper[1], color='#4169E1', alpha=0.1)
        # axs[1].fill_between(x, y_lower[0], y_upper[0], color='#00FA9A', alpha=0.1)
        # axs[1].fill_between(x, y_lower[3], y_upper[3], color='blue', alpha=0.1)

        axs[1].set_ylabel('Run_Time(Sec)', loc='center', fontsize=15, color='black', style='normal', fontweight='bold')
        axs[1].grid(True)

        "第三折线图"
        axs[2].plot(x, c[5], label=Alg_name[5], color=colors[5], marker='^', markersize=10, linestyle=line_style[1],
                    linewidth=2)
        axs[2].plot(x, c[4], label=Alg_name[4], color=colors[4], marker='o', markersize=10, linestyle=line_style[0],
                    linewidth=2)
        axs[2].plot(x, c[3], label=Alg_name[3], color=colors[3], marker='*', markersize=15, linestyle=line_style[1],
                    linewidth=2)
        axs[2].plot(x, c[2], label=Alg_name[2], color=colors[2], marker='v', markersize=10, linestyle=line_style[3],
                    linewidth=2)
        axs[2].plot(x, c[1], label=Alg_name[1], color=colors[1], marker='H', markersize=10, linestyle=line_style[2],
                    linewidth=1.5)
        axs[2].plot(x, c[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],linewidth=2)

        # axs[2].set_title('Success_Rate',loc='left', fontweight='bold')
        axs[2].set_xlabel('Number of Agents', loc='center', fontsize=15, color='black', style='normal',
                          fontweight='bold')
        axs[2].set_ylabel('Success_Rate', loc='center', fontsize=15, color='black', style='normal', fontweight='bold')
        axs[2].set_ylim(-0.1, 1.1)
        axs[2].grid(True)
        axs[2].legend(loc='center right',bbox_to_anchor=(1, 0.75), frameon=True, fontsize=9, framealpha=0.5)  # 将图例放在左侧中间位置
        # 调整子图之间的间距
        fig.tight_layout()

        # 保存
        fig.savefig("images14" + "\{0}.pdf".format(self.save_fig_name), format='pdf', dpi=300, bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig("tttt1111.svg", format='svg', dpi=300, bbox_inches='tight',pad_inches=0)
        # 显示图表
        plt.show()
    def draw_RL_graph(self, x, SOC_y, Run_Time, Success_Rate):
        a, b, c = SOC_y, Run_Time, Success_Rate  # 数据传入
        a = [[None if x == 0 else x for x in row] for row in a]
        b = [[self.deadline if x == 0 or x > self.deadline else x for x in row] for row in b]

        plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
        colors = [ '#66CC66','#ba4f56','#191970']  # PIBT EECBS LACAM*
        # Alg_name = ['LACAM*_ITA','ELACAM*_ITA','ML_EECBS_ITA','EML_EECBS_ITA','MA_CBS_ITA','EMA_CBS_ITA']
        Alg_name = ['ELaCAM*_IHTA','ELaCAM*_IHTA+DRL']
        line_style = [':', '--', '-', '-.']  # PIBT EECBS LACAM*
        fig, axs = plt.subplots(1, 1, figsize=(7, 6))  # 创建一个包含三个子图的窗口，每个子图的排列方式为从上到下

        "第一折线图"
        axs.plot(x, a[1], label=Alg_name[1], color=colors[1], marker='H', markersize=10, linestyle=line_style[2],
                    linewidth=1.5)
        axs.plot(x, a[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],
                    linewidth=2)
        axs.set_title(
            '\n{0}\n{1} (|V|={2})  |T|=200  Runtime limit={3}\n'.format(self.img_name, self.wh, self.size,
                                                                        self.deadline),
            loc='left', fontweight='bold')
        # axs[0].legend(loc='lower left')  # 添加图例
        axs.set_ylabel('Soc', loc='center', fontsize=10, color='black', style='normal', fontweight='bold')
        axs.set_xlabel('Number of agents', loc='right', fontsize=10, color='black', style='normal',
                          fontweight='bold')
        axs.grid(True)  # 为第一个子图添加网格线

        # 右上插图
        axins = fig.add_axes([0.84, 0.91, 0.12, 0.08])  # 创建一个新轴，位置在主图的右边
        img = mpl.image.imread(self.img_path)
        axins.imshow(img, aspect='auto')
        axins.axis('off')  # 关闭新轴的坐标轴

        # "第二折线图"
        # axs[1].plot(x, b[1], label=Alg_name[1], color=colors[1], marker='H', markersize=10, linestyle=line_style[2],
        #             linewidth=1.5)
        # axs[1].plot(x, b[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],
        #             linewidth=2)
        #
        # axs[1].set_ylabel('Run_Time(Sec)', loc='center', fontsize=10, color='black', style='normal', fontweight='bold')
        # axs[1].grid(True)
        #
        # "第三折线图"
        #
        # axs[2].plot(x, c[1], label=Alg_name[1], color=colors[1], marker='H', markersize=10, linestyle=line_style[2],
        #             linewidth=1.5)
        # axs[2].plot(x, c[0], label=Alg_name[0], color=colors[0], marker='d', markersize=8, linestyle=line_style[2],
        #             linewidth=2)
        #
        #
        # axs[2].set_xlabel('Number of agents', loc='right', fontsize=10, color='black', style='normal',
        #                   fontweight='bold')
        # axs[2].set_ylabel('Success_Rate', loc='center', fontsize=10, color='black', style='normal', fontweight='bold')
        # axs[2].set_ylim(-0.1, 1.1)
        # axs[2].grid(True)
        # axs[2].legend(loc='lower left')  # 添加图例
        # # 调整子图之间的间距
        # fig.tight_layout()

        axs.grid(True)
        axs.legend(loc='lower left')  # 添加图例
        # 保存
        fig.savefig("images14" + "\{0}-rl.pdf".format(self.save_fig_name), format='pdf', dpi=300, bbox_inches='tight',
                    pad_inches=0)
        # plt.savefig("tttt1111.svg", format='svg', dpi=300, bbox_inches='tight',pad_inches=0)
        # 显示图表
        plt.show()
if __name__ == "__main__":
    pass