# 开发时间：2024/7/19 16:09
# 开发语言：Python
'''---------------------------------------'''
import os

import pandas as pd

from Draw_chart import Draw as dr

class T_P():
    def __init__(self,way,wh,size,deadline):
        self.way=way # empty-16-16
        self.size=size # 256
        self.deadline=deadline
        self.wh=wh
        self.x = [20, 40, 60, 80, 100, 120]
        # self.get_picture_0()
        self.get_picture_1()
        #self.get_picture_2()

    def get_picture_0(self):
        def dft( x):
            return df[x].tolist()
        df = pd.read_excel(r'Datas14\{0}.xlsx'.format(self.way), sheet_name='Sheet1')
        # 假设Excel表中有多列数据，我们想要将它们分别导入到列表中
        lacam1_r, lacam0_r, pibt_r, eecbs_r = "lacam1_r", "lacam0_r", "pibt_r", "eecbs_r"
        lacam1_s, lacam0_s, pibt_s, eecbs_s = "lacam1_s", "lacam0_s", "pibt_s", "eecbs_s"
        lacam1_r_uper, lacam0_r_uper, PIBT_r_uper, eecbs_r_uper = "lacam1_r_uper", "lacam0_r_uper", "PIBT_r_uper", "eecbs_r_uper"
        lacam1_r_lower, lacam0_r_lower, PIBT_r_lower, eecbs_r_lower = "lacam1_r_lower", "lacam0_r_lower", "PIBT_r_lower", "eecbs_r_lower"
        lacam1_s_uper, lacam0_s_uper, PIBT_s_uper, eecbs_s_uper = "lacam1_s_uper", "lacam0_s_uper", "PIBT_s_uper", "eecbs_s_uper"
        lacam1_s_lower, lacam0_s_lower, PIBT_s_lower, eecbs_s_lower = "lacam1_s_lower", "lacam0_s_lower", "PIBT_s_lower", "eecbs_s_lower"
        lacam1_sr, lacam0_sr, pibt_sr, eecbs_sr = "lacam1_sr", "lacam0_sr", "pibt_sr", "eecbs_sr"

        # 将每列数据转换为列表
        Run_Time = [dft(lacam1_r), dft(lacam0_r), dft(pibt_r), dft(eecbs_r)]
        SOC_y = [dft(lacam1_s), dft(lacam0_s), dft(pibt_s), dft(eecbs_s)]
        Success_Rate = [dft(lacam1_sr), dft(lacam0_sr), dft(pibt_sr), dft(eecbs_sr)]
        r_uper = [dft(lacam1_r_uper), dft(lacam0_r_uper), dft(PIBT_r_uper), dft(eecbs_r_uper)]
        r_lower = [dft(lacam1_r_lower), dft(lacam0_r_lower), dft(PIBT_r_lower), dft(eecbs_r_lower)]
        s_uper = [dft(lacam1_s_uper), dft(lacam0_s_uper), dft(PIBT_s_uper), dft(eecbs_s_uper)]
        s_lower = [dft(lacam1_s_lower), dft(lacam0_s_lower), dft(PIBT_s_lower), dft(eecbs_s_lower)]

        draw = dr(r"images\{0}.png".format(self.way), self.wh,'(Graph){0}'.format(self.way), self.size, self.deadline)
        draw.draw_four_graph(self.x, SOC_y, Run_Time, Success_Rate, r_uper, r_lower, s_uper, s_lower)

    def get_picture_1(self):
        def dft(list):
            temp = []
            for i in list:
                temp.append(df[i].tolist())
            return temp
        df = pd.read_excel(r'Datas14\{0}.xlsx'.format(self.way), sheet_name='Sheet1')
        # 假设Excel表中有多列数据，我们想要将它们分别导入到列表中
        T_A = ["LC1_T_A","LC2_T_A","ML1_T_A","ML2_T_A","MA1_T_A","MA2_T_A"]
        C_A = ["LC1_C_A","LC2_C_A","ML1_C_A","ML2_C_A","MA1_C_A","MA2_C_A"]
        T_MX = ["LC1_T_MX","LC2_T_MX","ML1_T_MX","ML2_T_MX","MA1_T_MX","MA2_T_MX"]
        T_MN = ["LC1_T_MN","LC2_T_MN","ML1_T_MN","ML2_T_MN","MA1_T_MN","MA2_T_MN"]
        C_MX = ["LC1_C_MX","LC2_C_MX","ML1_C_MX","ML2_C_MX","MA1_C_MX","MA2_C_MX"]
        C_MN = ["LC1_C_MN","LC2_C_MN","ML1_C_MN","ML2_C_MN","MA1_C_MN","MA2_C_MN"]
        S = ["LC1_S","LC2_S","ML1_S","ML2_S","MA1_S","MA2_S"]


        # 将每列数据转换为列表
        Run_Time = dft(T_A)
        SOC_y = dft(C_A)
        r_uper = dft(T_MX)
        r_lower = dft(T_MN)
        s_uper = dft(C_MX)
        s_lower = dft(C_MN)
        Success_Rate = dft(S)


        draw = dr(r"images14\{0}.png".format(self.way), self.wh,'(Graph){0}'.format(self.way), self.size, self.deadline)
        draw.draw_six_graph(self.x, SOC_y, Run_Time, Success_Rate, r_uper, r_lower, s_uper, s_lower)


    # 原版与RL版对比
    def get_picture_2(self):
        def dft(list):
            temp = []
            for i in list:
                temp.append(df[i].tolist())
            return temp
        df = pd.read_excel(r'Datas14\{0}.xlsx'.format(self.way), sheet_name='Sheet1')
        # 假设Excel表中有多列数据，我们想要将它们分别导入到列表中
        T_A = ["LC1_T_A","LC2_T_A"]
        C_A = ["LC1_C_A","LC2_C_A"]
        S = ["LC1_S","LC2_S"]


        # 将每列数据转换为列表
        Run_Time = dft(T_A)
        SOC_y = dft(C_A)
        Success_Rate = dft(S)


        draw = dr(r"images14\{0}.png".format(self.way), self.wh,'(Graph){0}'.format(self.way), self.size, self.deadline)
        # draw.draw_six_graph(self.x, SOC_y, Run_Time, Success_Rate, r_uper, r_lower, s_uper, s_lower)
        draw.draw_RL_graph(self.x, SOC_y, Run_Time, Success_Rate)


if __name__ == "__main__":
    # pass
    # "min"
    # # T0 = T_P('empty-16-16','16x16', 256, 200)
    T1 = T_P('empty-32-32_','32x32',1024,1800)
    # # T2 = T_P('random-32-32-10','32x32',922,200)
    # T3 = T_P('room-32-32-4','32x32',682,300)
    # "mid"
    #####T4 = T_P('empty-48-48_','48x48',2304,2000)
    # # T5 = T_P('room-64-64-8','64x64',3232,700)
    # T6 = T_P('room-64-64-16','64x64',3646,700)
    ######T7 = T_P('random-64-64-10_','64x64',3687,2900)
    #####T8 = T_P('random-64-64-20_','64x64',3270,2900)
    #####T9 = T_P('den312d_','',2445,2200)

    "max"
    #####T8 = T_P('ost003d_','',13214,15000)
    # T9 = T_P('lt_gallowstemplar_n','',10021,3000)
    # 绝对路径
    # path = r'E:\大学包\03硕士学包\SCIA实验室包\03实验项目\py-lacam2-main\Datas7\room-64-64-16.xlsx'

    # 判断目录是否存在
    # if os.path.isdir(path):
    #     print(f"目录 {path} 存在。")
    # else:
    #     print(f"目录 {path} 不存在。")
    # T10 = T_P('warehouse-20-40-10-2-2','',38756,8000)


