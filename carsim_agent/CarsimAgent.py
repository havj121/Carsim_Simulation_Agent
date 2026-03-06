import os
import json
import traceback
from typing import Dict, Any, List
from .CarsimLibrary import CarsimLibrary
from .llm import CarsimLLMInterface
from .CarsimRunner import CarsimSimulationRunner
from .utils.logger import log_info, log_warning, log_error, setup_logger
from .utils.extractor import VSBExtractor

class CarsimAgent:
    class Option:
        """
        管理 CarsimAgent 执行过程的设置
        """
        def __init__(self):
            # 运行模式: "prompt" 或 "mods" (默认为 "mods")
            self.run_mode = "mods"
            # 是否提取数据
            self.extract_data = True
            # 默认提取变量列表
            self.default_extract_vars = ["Time", "VX", "Steer_SW", "Fx_L2", "Fy_L2", "Fx_R2", "Fy_R2", "Alpha_R2", "Kappa_R2"]
            # 数据提取方式: "csv" (读取 LastRun.csv) 或 "vsb" (转换 vsb/vs)
            self.extraction_method = "csv"

    def __init__(self, template_path: str, lib_json_path: str = None):
        self.template_path = os.path.abspath(template_path)
        self.library = CarsimLibrary()
        self.template_library = CarsimLibrary()
        self.llm = CarsimLLMInterface()
        self.extractor = VSBExtractor()
        self.options = self.Option()
        
        # 1. 首先加载当前模板的基准参数 (用于后续验证)
        echo_path = os.path.join(self.template_path, "LastRun_echo.par")
        if os.path.exists(echo_path):
            self.template_library.load_from_echo(echo_path)
            log_info(f"Successfully loaded base template library from: {echo_path}")
        else:
            log_error(f"Template echo file not found at {echo_path}. Cannot verify modifications correctly.")
            # 如果没有 echo，可能无法进行精确验证，但我们仍尝试继续
            
        # 2. 加载用于 LLM 上下文的参考库
        if lib_json_path is None:
            log_info("No library JSON path provided, attempting to load default Carsim_Config_Lib.json")
            # 优先从包内查找
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            default_lib_json_path = os.path.join(pkg_dir, "Carsim_Config_Lib.json")
            
            if not os.path.exists(default_lib_json_path):
                log_warning(f"Default library file not found: {default_lib_json_path}")
                # 如果包内没有，且 template_library 已经加载，则复用它
                if os.path.exists(echo_path):
                    log_info("Constructing Carsim_Config_Lib.json from template echo file.")
                    self.library = self.template_library
                    # 保存到默认路径
                    self.library.save_to_json(default_lib_json_path)
                    log_info(f"Successfully saved Carsim_Config_Lib.json to: {default_lib_json_path}")
                else:
                    raise FileNotFoundError(f"No library JSON found and no template echo found at {echo_path}")
            else:
                self.library.load_from_json(default_lib_json_path)
                log_info(f"Successfully loaded library from default JSON: {default_lib_json_path}")
             
        elif os.path.exists(lib_json_path):
            self.library.load_from_json(os.path.abspath(lib_json_path))
            log_info(f"Successfully loaded library from provided JSON: {lib_json_path}")
        else:
            log_error(f"Provided lib_json_path does not exist: {lib_json_path}")
            raise FileNotFoundError(f"Library file not found at {lib_json_path}")

    def run(self, prompt: str = None, mods_json_path: str = None, extract_data: bool = None, extract_vars: list = None):
        """
        运行仿真并提取数据。
        :param prompt: 自然语言指令 (仅在 run_mode="prompt" 时使用，或未提供 mods_json_path 时作为备选)
        :param mods_json_path: 直接修改的 JSON 文件路径 (仅在 run_mode="mods" 时使用，或未提供 prompt 时作为备选)
        :param extract_data: 是否进行数据提取，若为 None 则使用 options.extract_data
        :param extract_vars: 需要从结果中提取的变量列表，若为 None 则使用 options.default_extract_vars
        """
        # 在每个任务开始前打印分割线
        log_info("\n" + "="*60)
        log_info(f"🆕 NEW TASK STARTED AT: {os.path.basename(self.template_path)}")
        log_info("="*60)
        
        # 确定实际使用的 extract_data 和 extract_vars
        actual_extract_data = extract_data if extract_data is not None else self.options.extract_data
        actual_extract_vars = extract_vars if extract_vars is not None else self.options.default_extract_vars

        mods = {}
        # 根据 options.run_mode 选择执行方法
        if self.options.run_mode == "mods":
            if mods_json_path:
                log_info(f"🚀 Starting agent task with JSON file (mods mode): '{mods_json_path}'")
                if not os.path.exists(mods_json_path):
                    log_error(f"Provided mods_json_path does not exist: {mods_json_path}")
                    return
                try:
                    with open(mods_json_path, 'r', encoding='utf-8') as f:
                        mods = json.load(f)
                    log_info(f"Loaded {len(mods)} modifications from JSON.")
                except Exception as e:
                    log_error(f"Failed to load mods from JSON: {e}")
                    return
            else:
                log_error("run_mode is 'mods' but mods_json_path not provided.")
                return
        elif self.options.run_mode == "prompt":
            if prompt:
                log_info(f"🚀 Starting agent task with prompt (prompt mode): '{prompt}'")
                log_info("Analyzing intent with LLM...")
                context = self.library.get_parameter_context(limit=1000)
                mods = self.llm.generate_mods(prompt, context)
                log_info(f"Proposed modifications: {json.dumps(mods, indent=2)}")
            else:
                log_error("run_mode is 'prompt' but prompt not provided.")
                return
        else:
            log_error(f"Unknown run_mode: {self.options.run_mode}")
            return

        # 执行仿真工作流
        self._execute_simulation_workflow(mods, actual_extract_data, actual_extract_vars)

    def _execute_simulation_workflow(self, mods: Dict[str, Any], extract_data: bool = False, extract_vars: list = None):
        """
        核心仿真流水线逻辑：验证、生成配置、运行、数据提取。
        """
        # 1. 验证配置是否正确
        if not self.verify_modifications(mods):
            error_msg = "Parameter verification FAILED. Some parameters are missing in the template or invalid."
            log_error(error_msg)
            raise ValueError(error_msg)
        
        # 2. 初始化 Runner
        log_info(f"Initializing simulation runner using template path: {self.template_path}")
        runner = CarsimSimulationRunner(self.template_path)
        runner.setup_workspace()
            
        # 3. 创建配置文件
        runner.create_parsfile(mods)
        runner.create_simfile()
        
        # 4. 运行仿真
        try:
            if runner.run_solver():
                log_info("Simulation completed successfully.")
                
                # 5. 提取数据 (仅当仿真成功且 extract_data 为 True 时执行)
                if extract_data:
                    log_info(f"--- Step 5: Extracting Data (Method: {self.options.extraction_method}) ---")
                    
                    file_to_extract = None
                    if self.options.extraction_method == "csv":
                        # 方式一：读取 LastRun.csv
                        file_to_extract = os.path.join(runner.results_path, "LastRun.csv")
                        if not os.path.exists(file_to_extract):
                            log_error(f"Extraction method is 'csv' but LastRun.csv NOT FOUND at: {file_to_extract}")
                            raise FileNotFoundError(f"LastRun.csv not found for extraction.")
                    elif self.options.extraction_method == "vsb":
                        # 方式二：转换 vsb/vs
                        file_to_extract = runner.erd_file.replace(".vs", ".vsb")
                        if not os.path.exists(file_to_extract):
                            file_to_extract = runner.erd_file
                    else:
                        log_error(f"Unknown extraction method: {self.options.extraction_method}")
                        return

                    csv_path = self.extractor.extract_to_csv(
                        file_to_extract, 
                        target_vars=extract_vars, 
                        method=self.options.extraction_method
                    )
                    
                    if csv_path:
                        log_info(f"Task result extracted to: {csv_path}")
                    else:
                        log_warning("Data extraction failed or skipped.")
                else:
                    log_info("Data extraction skipped (extract_data=False).")
                    
                log_info("Task finished.")
            else:
                log_error("Simulation failed.")
        except Exception as e:
            error_msg = f"Simulation runner encountered an error: {e}\n{traceback.format_exc()}"
            log_error(error_msg)

    def verify_modifications(self, mods: Dict[str, Any]) -> bool:
        """
        验证调整参数：将预期修改与模板目录下的 LastRun_echo.par 文件中的原始参数进行对比。
        如果参数在模板中不存在，则验证失败。
        """
        log_info("Verifying parameter modifications against base template echo...")
        passed = True
        for key, target_val in mods.items():
            if key == "OPT_ECHO_ALL_PARS":
                continue
                
            # 使用特定于模板的参数库进行对比——LastRun_echo.par
            param_info = self.template_library.get_parameter(key)
            if param_info:
                original_val = param_info.get("value")
                unit = param_info.get("unit", "")
                log_info(f"✅ Parameter '{key}': Template value = {original_val} {unit}, New value = {target_val} {unit}")
                # 检查是否有实际变化
                try:
                    if abs(float(original_val) - float(target_val)) < 1e-4:
                        log_warning(f"⚠️ Parameter '{key}' value is identical to template value.")
                except (ValueError, TypeError):
                    pass
            else:
                log_error(f"❌ Parameter '{key}' NOT FOUND in template echo ({self.template_path})! Cannot verify or modify unknown parameters.")
                passed = False
        
        return passed
