import os
import subprocess
import re
import time

# --- 配置区 (请根据实际安装路径确认) ---
CARSIM_PROGRAM_PATH = r"D:\Program Files\Carsim2020\Programs"
CARSIM_SOLVER_PATH = os.path.join(CARSIM_PROGRAM_PATH, r"VS_SolverWrapper_CLI_64\VS_SolverWrapper_CLI_64.exe")
CARSIM_DATA_DIR = r"C:\Users\Public\Documents\CarSim2020.0_Data"
# 2. 指向内核 DLL 文件
CARSIM_DLL_PATH = os.path.join(CARSIM_PROGRAM_PATH, r"solvers\carsim_64.dll")
# 运行目录设在 Data 目录的 Runs 文件夹下，确保相对路径有效
CARSIM_RUNS_DIR = os.path.join(CARSIM_DATA_DIR, "Runs")

# 建议：将基础模板文件也放入Data目录下或者使用绝对路径
CURE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_ALL_PAR = os.path.join(CURE_DIR, "Template_ABS_Base.par")  

# 项目名设置为名称+时间戳
Project_Name= f"Carsim_Agent_Test_{time.strftime('%Y%m%d_%H%M%S', time.localtime())}"
# Project_Name = "Carsim_Agent_Test_20260227_142340"

class CarSimTester:
    def __init__(self):
        # 运行目录设置在 Data 目录的 Runs 文件夹下，确保相对路径有效
        self.RUN_PATH = os.path.join(CARSIM_RUNS_DIR, Project_Name)
        if not os.path.exists(self.RUN_PATH):
            os.makedirs(self.RUN_PATH)
        self.run_par = os.path.join(self.RUN_PATH, "Modified_Run.par")
        self.sim_file = os.path.join(self.RUN_PATH, "CarSim_Agent.sim")
        self.log_file = os.path.join(self.RUN_PATH, "LastRun_log.txt")
        self.echo_file = os.path.join(self.RUN_PATH, "LastRun_echo.par")
        self.erd_file = os.path.join(self.RUN_PATH, "LastRun.vs")
        self.end_par = os.path.join(self.RUN_PATH, "LastRun_end.par")

    def create_modified_parsfile(self, mods: dict):
        """步骤 1: 创建覆盖层 Parsfile"""
        print(f"[Step 1] 正在生成修改后的 Parsfile...")
        
        with open(self.run_par, 'w', encoding='ascii') as f:
            f.write("PARSFILE\n")
            # 引入基准模板
            f.write(f'PARSFILE {os.path.abspath(BASE_ALL_PAR)}\n')
            f.write("! --- Agent Modifications ---\n")
            for k, v in mods.items():
                f.write(f"{k} {v}\n")
            f.write("END\n")
        return

    def create_simfile(self):
        """步骤 2: 创建控制文件 simfile.sim"""
        print(f"[Step 2] 正在生成 simfile...")
        
        # 辅助函数：确保路径以反斜杠结尾
        def ensure_backslash(path):
            p = path.replace("/", "\\")
            return p if p.endswith("\\") else p + "\\"

        prog_dir = ensure_backslash(CARSIM_PROGRAM_PATH)
        data_dir = ensure_backslash(CARSIM_DATA_DIR)
        res_dir = ensure_backslash(os.path.join(CARSIM_PROGRAM_PATH, "Resources"))
 
        with open(self.sim_file, 'w', encoding='ascii') as f:
            f.write("SIMFILE\n\n")
            # f.write(f"SET_MACRO $(ROOT_PROJECT_NAME)$ {Project_Name}\n")
            # f.write(f"SET_MACRO $(WORK_DIR)$ {self.WORKSPACE}\n")
            f.write(f"SET_MACRO $(OUTPUT_FILE_PREFIX)$ {os.path.join(self.RUN_PATH, 'LastRun')}\n")
            f.write("FILEBASE $(OUTPUT_FILE_PREFIX)$\n")
            f.write(f"INPUT {self.run_par}\n")
            # f.write(f"INPUTARCHIVE {os.path.join(self.RUN_PATH, 'Modified_Run_all.par')}\n")
            f.write(f"ECHO {self.echo_file}\n")
            f.write(f"FINAL {self.end_par}\n")
            f.write(f"LOGFILE {self.log_file}\n")
            f.write(f"ERDFILE {self.erd_file}\n")
            # 修复 3：设置正确的程序目录、数据目录和资源目录
            f.write(f"PROGDIR {prog_dir}\n")
            f.write(f"DATADIR {data_dir}\n")
            # f.write(f"GUI_REFRESH_V CarSim_RefreshEvent_24812\n")
            f.write(f"RESOURCEDIR {res_dir}\n")
            
            f.write("PRODUCT_ID CarSim\n")
            f.write("PRODUCT_VER 2020.0\n")
            f.write("VEHICLE_CODE i_i\n")
            f.write("EXT_MODEL_STEP 0.0010000\n")
            f.write("PORTS_IMP 0\n")
            f.write("PORTS_EXP 0\n\n")
            
            f.write(f"DLLFILE {CARSIM_DLL_PATH}\n")
            f.write("END\n")
        return 

    def run_solver(self):
        """步骤 3: 运行求解器"""
        print(f"[Step 3] 正在调用求解器...")
        try:
            # 命令格式: solver.exe -s simfile.sim
            command = [CARSIM_SOLVER_PATH, 
                        "-sim", self.sim_file,
                        "-progdir", CARSIM_PROGRAM_PATH,
                        "-datadir", CARSIM_DATA_DIR]
            # 修复 2：添加 CarSim 必要的环境变量
            env = os.environ.copy()
            # 关键修复：必须将程序目录加入 PATH，否则 DLL 可能因缺少依赖无法加载
            env["PATH"] = CARSIM_PROGRAM_PATH + os.pathsep + env.get("PATH", "")
            
            # 运行求解器：取消 capture_output=True，直接输出所有日志（方便排查）
            result = subprocess.run(command,
                                    cwd=CARSIM_DATA_DIR,  # 修正工作目录
                                    env=env,             # 传入环境变量
                                    text=True, 
                                    check=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print("求解器运行成功")
            print(f"求解器输出: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"求解器报错返回: {e.returncode}")
            print(f"错误输出: {e.stderr}")
            if os.path.exists(self.log_file):
                print(f"查看日志文件获取更多细节: {self.log_file}")
            return False

    def verify_results(self, expected: dict):
        """步骤 4: 验证 Echo 文件中的参数"""
        print(f"[Step 4] 正在验证参数修改结果...")
        # 重新构建 echo_file 路径，确保使用正确的时间戳文件夹
        self.echo_file = os.path.join(self.RUN_PATH, "LastRun_echo.par")
        
        if not os.path.exists(self.echo_file):
            print(f"错误: 未找到 Echo 文件！路径：{self.echo_file}")
            return False

        with open(self.echo_file, 'r') as f:
            content = f.read()

        all_passed = True
        for key, val in expected.items():
            # 搜索形如: SPEED 150 ! (或其他分隔符)
            pattern = rf"^{key}\s+([\d\.-]+)"
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                actual_val = float(match.group(1))
                if abs(actual_val - val) < 1e-4:
                    print(f"✅ 参数 {key}: 预期 {val}, 实际 {actual_val} - 通过")
                else:
                    print(f"❌ 参数 {key}: 预期 {val}, 实际 {actual_val} - 失败")
                    all_passed = False
            else:
                print(f"❓ 参数 {key}: 未在 Echo 文件中找到！")
                all_passed = False
        return all_passed

# --- 运行验证 ---
if __name__ == "__main__":
    tester = CarSimTester()
    
    # 定义我们想要修改的参数 (基于你上传的 LastRun_echo.par 关键字)
    test_mods = {
        "SPEED": 150.0,      # 将车速从 120 改为 150 (VX -> SPEED)
        "TSTOP": 10.0,    # 仿真停止时间改为 10s
        "OPT_STOP": 1,     # 确保停止条件开启
        "OPT_ECHO_ALL_PARS": 1 # 强制 Echo 所有参数
    }

    tester.create_modified_parsfile(test_mods)
    tester.create_simfile()
    
    if tester.run_solver():
        success = tester.verify_results(test_mods)
        if success:
            print("\n结论: CarSim 修改器通过所有验证项！")
        else:
            print("\n结论: 验证失败，请检查参数关键字或 Solver 权限。")