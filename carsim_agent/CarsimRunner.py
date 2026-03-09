import os
import time
import shutil
import subprocess
import re
from typing import Dict, Any
from .config import CARSIM_PROGRAM_PATH, CARSIM_SOLVER_PATH, CARSIM_DATA_DIR, CARSIM_DLL_PATH, CARSIM_RESULTS_DIR
from .utils.logger import log_info, log_warning, log_error

class CarsimSimulationRunner:
    def __init__(self, template_dir: str, task_id: str = None, results_base_dir: str = None):
        self.template_dir = template_dir
        self.timestamp = task_id if task_id else time.strftime('%Y%m%d_%H%M%S_%f', time.localtime())
        self.project_name = f"Carsim_Agent_Run_{self.timestamp}"
        
        # 使用传入的根目录或默认 Results 目录
        base_dir = results_base_dir if results_base_dir else CARSIM_RESULTS_DIR
        self.results_path = os.path.join(base_dir, self.project_name)
        
        # 文件路径定义
        self.run_par = os.path.join(self.results_path, "Modified_Run.par")
        self.sim_file = os.path.join(self.results_path, "CarSim_Agent.sim")
        self.log_file = os.path.join(self.results_path, "LastRun_log.txt")
        self.echo_file = os.path.join(self.results_path, "LastRun_echo.par")
        self.erd_file = os.path.join(self.results_path, "LastRun.vs")
        self.end_par = os.path.join(self.results_path, "LastRun_end.par")
        
    def setup_workspace(self):
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)
            log_info(f"Created workspace: {self.results_path}")
            
    def create_parsfile(self, mods: Dict[str, Any]):
        base_all_par = os.path.join(self.template_dir, "Run_all.par")
        if not os.path.exists(base_all_par):
            log_error(f"Template par file not found: {base_all_par}")
            raise FileNotFoundError(f"Template par file not found: {base_all_par}")
            
        # 复制并引用
        base_local = os.path.join(self.results_path, "Run_all.par")
        shutil.copyfile(base_all_par, base_local)
        
        with open(self.run_par, 'w', encoding='utf-8') as f:
            f.write("PARSFILE\n")
            f.write(f'PARSFILE {base_local}\n')
            f.write("! --- Agent Modifications ---\n")
            for k, v in mods.items():
                f.write(f"{k} {v}\n")
            f.write("END\n")
        log_info(f"Created parsfile: {self.run_par}")
            
    def create_simfile(self):
        def ensure_backslash(path):
            p = path.replace("/", "\\")
            return p if p.endswith("\\") else p + "\\"
            
        prog_dir = ensure_backslash(CARSIM_PROGRAM_PATH)
        data_dir = ensure_backslash(CARSIM_DATA_DIR)
        res_dir = ensure_backslash(os.path.join(CARSIM_PROGRAM_PATH, "Resources"))
        
        with open(self.sim_file, 'w', encoding='utf-8') as f:
            f.write("SIMFILE\n\n")
            f.write(f"SET_MACRO $(OUTPUT_FILE_PREFIX)$ {os.path.join(self.results_path, 'LastRun')}\n")
            f.write("FILEBASE $(OUTPUT_FILE_PREFIX)$\n")
            f.write(f"INPUT {self.run_par}\n")
            f.write(f"ECHO {self.echo_file}\n")
            f.write(f"FINAL {self.end_par}\n")
            f.write(f"LOGFILE {self.log_file}\n")
            f.write(f"ERDFILE {self.erd_file}\n")
            f.write(f"PROGDIR {prog_dir}\n")
            f.write(f"DATADIR {data_dir}\n")
            f.write(f"RESOURCEDIR {res_dir}\n")
            f.write("PRODUCT_ID CarSim\n")
            f.write("PRODUCT_VER 2020.0\n")
            f.write("VEHICLE_CODE i_i\n")
            f.write(f"DLLFILE {CARSIM_DLL_PATH}\n")
            f.write("END\n")
        log_info(f"Created simfile: {self.sim_file}")

    def run_solver(self) -> bool:
        log_info(f"Launching solver for project: {self.project_name}")
        command = [CARSIM_SOLVER_PATH, 
                   "-sim", self.sim_file,
                   "-progdir", CARSIM_PROGRAM_PATH,
                   "-datadir", CARSIM_DATA_DIR]
        
        env = os.environ.copy()
        env["PATH"] = CARSIM_PROGRAM_PATH + os.pathsep + env.get("PATH", "")
        
        try:
            result = subprocess.run(command, cwd=CARSIM_DATA_DIR, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            log_info("Solver completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            log_error(f"Solver failed with code {e.returncode}")
            log_error(f"Solver error output: {e.stderr.decode('utf-8', errors='ignore')}")
            return False
