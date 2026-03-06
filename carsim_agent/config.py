import os

# --- 配置区 (请根据实际安装路径确认) ---
CARSIM_PROGRAM_PATH = r"D:\Program Files\Carsim2020\Programs"
CARSIM_SOLVER_PATH = os.path.join(CARSIM_PROGRAM_PATH, r"VS_SolverWrapper_CLI_64\VS_SolverWrapper_CLI_64.exe")
CARSIM_DATA_DIR = r"C:\Users\Public\Documents\CarSim2020.0_Data"
# 2. 指向内核 DLL 文件
CARSIM_DLL_PATH = os.path.join(CARSIM_PROGRAM_PATH, r"solvers\carsim_64.dll")
# 运行目录设在 Data 目录的 Runs 文件夹下，确保相对路径有效
CARSIM_RUNS_DIR = os.path.join(CARSIM_DATA_DIR, "Runs")
# 运行目录设在Data目录的Results文件夹下，确保相对路径有效
CARSIM_RESULTS_DIR = os.path.join(CARSIM_DATA_DIR, "Results")

# 转换工具路径
ERD_CONVERTER_PATH = os.path.join(CARSIM_PROGRAM_PATH, "erdconverter.exe")


