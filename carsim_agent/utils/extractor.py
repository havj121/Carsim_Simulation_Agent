import os
import subprocess
import pandas as pd
from ..config import ERD_CONVERTER_PATH
from .logger import log_info, log_error, log_warning

class VSBExtractor:
    """
    CarSim VSB/VS 结果文件提取器
    """
    def __init__(self, erd_converter_path: str = ERD_CONVERTER_PATH):
        self.converter_path = erd_converter_path
        self.default_target_vars = ["Time", "Vx", "Steer_SW", "Fx_L2", "Fy_L2", "Fx_R2", "Fy_R2", "Alpha_R2", "Kappa_R2"]

    def extract_to_csv(self, file_path: str, output_csv: str = None, target_vars: list = None, method: str = "csv") -> str:
        """
        将仿真结果转换为 CSV 并提取指定变量
        :param file_path: vsb/vs 或 csv 文件路径
        :param output_csv: 输出 CSV 路径，若为 None 则在原路径基础上修改后缀
        :param target_vars: 需要提取的变量列表，若为 None 则使用默认列表
        :param method: 提取方式，"csv" 为直接读取 CSV，"vsb" 为通过 erdconverter 转换 vsb/vs
        :return: 最终生成的 CSV 文件路径，失败返回 None
        """
        if target_vars is None:
            # 若未指定变量，使用默认列表
            target_vars = self.default_target_vars
            log_info(f"Target variables not specified, using default: {target_vars}")

        log_info(f"🔍 Starting data extraction from: {file_path} using method: {method}")

        if not os.path.exists(file_path):
            log_error(f"File not found: {file_path}")
            return None

        temp_csv = None

        if method == "csv":
            # 方式一：直接读取 CSV
            if not file_path.lower().endswith(".csv"):
                log_error(f"Method is 'csv' but file is not a CSV: {file_path}")
                return None
            temp_csv = file_path
        elif method == "vsb":
            # 方式二：转换 vsb/vs
            if not os.path.exists(self.converter_path):
                log_error(f"ERD Converter not found at: {self.converter_path}")
                return None

            # 1. 准备临时 CSV 路径
            temp_filename = "temp_extraction.csv"
            temp_csv = os.path.abspath(temp_filename)
            
            # 2. 调用 erdconverter 转换为完整 CSV
            command = [self.converter_path, file_path, "-csv", "-o", temp_csv]
            log_info(f"Running erdconverter: {command}")
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True, shell=False, timeout=30)
                if result.returncode == 0:
                    log_info("ERD conversion completed successfully.")
                    if not os.path.exists(temp_csv):
                        log_error("ERD converter did not generate output file.")
                        return None  
                else:
                    log_error(f"ERD conversion failed with return code {result.returncode}: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                log_error("ERD conversion timed out after 30 seconds！可能是因为 GUI 仍在等待交互")
                return None
            except subprocess.CalledProcessError as e:
                log_error(f"ERD conversion failed with return code {e.returncode}: {e.stderr or e.stdout}")
                return None
        else:
            log_error(f"Unknown extraction method: {method}")
            return None

        # 3. 读取 CSV 并筛选变量
        try:
            df = pd.read_csv(temp_csv)
            available_vars = [v for v in target_vars if v in df.columns]
            missing_vars = [v for v in target_vars if v not in df.columns]
            
            if missing_vars:
                log_warning(f"Variables not found in result: {missing_vars}")
            
            if not available_vars:
                log_error("No target variables found in the result file.")
                if method == "vsb" and os.path.exists(temp_csv):
                    os.remove(temp_csv)
                return None

            filtered_df = df[available_vars]
            
            # 4. 保存最终结果
            if output_csv is None:
                output_csv = file_path.replace(".vsb", "_extracted.csv").replace(".vs", "_extracted.csv").replace(".csv", "_extracted.csv")
            
            filtered_df.to_csv(output_csv, index=False)
            log_info(f"✅ Data successfully extracted to: {output_csv}")
            
            # 5. 清理临时文件 (仅当使用转换方式时)
            if method == "vsb" and os.path.exists(temp_csv):
                os.remove(temp_csv)
                
            return output_csv

        except Exception as e:
            log_error(f"Error during CSV filtering: {str(e)}")
            if method == "vsb" and temp_csv and os.path.exists(temp_csv):
                os.remove(temp_csv)
            return None
