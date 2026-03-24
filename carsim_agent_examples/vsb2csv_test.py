import os
import subprocess
import pandas as pd

# --- 配置区 ---
# 1. erdconverter.exe 的路径 (通常在 CarSim 安装目录的 Programs 文件夹下)
ERD_CONVERTER_PATH = r"D:\Program Files\Carsim2020\Programs\erdconverter.exe"
# 2. 仿真生成的 .vsb 或 .vs 文件路径
VSB_RESULT_PATH = r"C:\Users\Public\Documents\CarSim2020.0_Data\Results\Carsim_Agent_Run_20260306_110046\LastRun.vsb"
# 3. 输出 CSV 的临时路径
TEMP_CSV_PATH = r"./temp_full_results.csv"
# 4. 你想要提取的变量列表 (必须与 CarSim 中的 Export 变量名一致)
TARGET_VARIABLES = ["Time", "VX", "Steer_SW", "Fx_L2","Fy_L2", "Fx_R2","Fy_R2","Alpha_R2","Kappa_R2"]

def convert_and_filter_results():
    """
    使用 erdconverter 将 vsb 转换为 csv，并筛选指定变量
    """
    print(f"开始处理结果文件: {VSB_RESULT_PATH}")

    if not os.path.exists(VSB_RESULT_PATH):
        print("错误: 找不到结果文件！")
        return

    # 步骤 1: 调用命令行工具进行格式转换
    # 参数说明: -csv 表示输出 CSV 格式
    command = [ERD_CONVERTER_PATH, VSB_RESULT_PATH, "-csv", "-o", TEMP_CSV_PATH]
    
    try:
        print("正在运行 erdconverter.exe...")
        subprocess.run(command, check=True, capture_output=True, text=True, shell=False,timeout=30)
    except subprocess.TimeoutExpired:
        print("转换超时，可能是因为 GUI 仍在等待交互。")
        return
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e.stderr.decode('gbk')}")
        return

    # 步骤 2: 使用 Pandas 读取生成的 CSV 并筛选变量
    if os.path.exists(TEMP_CSV_PATH):
        print(f"正在读取临时文件并筛选变量: {TARGET_VARIABLES}")
        
        # 注意：CarSim 输出的 CSV 可能包含一些头信息行，通常从第 1 行（索引 0）开始是列名
        # 如果读取出错，尝试调整 skiprows 参数
        df = pd.read_csv(TEMP_CSV_PATH)
        
        # 检查变量是否存在于结果中
        available_vars = [v for v in TARGET_VARIABLES if v in df.columns]
        missing_vars = [v for v in TARGET_VARIABLES if v not in df.columns]
        
        if missing_vars:
            print(f"警告: 以下变量在结果文件中未找到: {missing_vars}")
        
        # 筛选数据
        filtered_df = df[available_vars]
        
        # 步骤 3: 保存最终的精简 CSV
        final_csv_name = VSB_RESULT_PATH.replace(".vsb", "_filtered.csv").replace(".vs", "_filtered.csv")
        filtered_df.to_csv(final_csv_name, index=False)
        
        print(f"成功！精简结果已保存至: {final_csv_name}")
        
        # 清理临时大文件
        # os.remove(TEMP_CSV_PATH)
    else:
        print("错误: 转换工具未生成输出文件。")

if __name__ == "__main__":
    # 运行转换
    convert_and_filter_results()