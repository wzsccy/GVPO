import time
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.exceptions import InvalidFileException
from CBS import main
import threading

# 设置截止时间
deadline = time.time() + 60 * 5  # 假设截止时间是当前时间加上10分钟

# 创建一个新的 Excel 工作簿
wb = Workbook()
ws = wb.active
ws.title = "运行时间记录"

# 写入表头
ws.append(["运行次数", "运行时间（秒）", "是否成功", "错误类型", "成功率"])

# 指定要运行的程序
def run_program():
    # 这里放置你的程序代码
    # 例如：time.sleep(1)       # 假设程序运行 1 秒
    main(10, 15, 7
         ,20)

# 运行程序 100 次
success_count = 0
error_types = []
for i in range(1, 101):
    try:
        # 记录开始时间
        start_time = time.time()
        # 运行程序
        run_program()
        # 记录结束时间
        end_time = time.time()
        # 计算运行时间
        run_time = end_time - start_time
        # # 如果运行时间非空，则减去一秒
        # if run_time > 0:
        #     run_time -= 1
        # 写入运行时间
        ws.append([i, run_time, "True", "", ""])
        success_count += 1
    except Exception as e:
        # 记录错误类型
        ws.append([i, "Error", "False", str(type(e).__name__), ""])
        # 打印错误类型
        print(f"运行 {i} 次时发生错误: {type(e).__name__}")
        error_types.append(str(type(e).__name__))
    # 检查是否到达截止时间
    # if time.time() > deadline:
    #     print("截止时间已到，程序运行结束。")
    #     break

# 计算成功率
success_rate = success_count / i
print(f"成功率为{success_rate:.2%}")
# 写入成功率
ws.append(["", "", "", "", f"{success_rate:.2%}"])

# 保存工作簿
try:
    wb.save("运行时间记录.xlsx")
except InvalidFileException as e:
    print(f"保存 Excel 文件时发生错误: {e}")

# 打印错误类型
print("程序运行完毕，结果已保存到文件中。")
if error_types:
    print("以下是一些错误类型：")
    for error_type in error_types:
        print(error_type)
