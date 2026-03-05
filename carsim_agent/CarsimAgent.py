import os
import json
import traceback
from typing import Dict, Any, List
from .CarsimLibrary import CarsimLibrary
from .llm import CarsimLLMInterface
from .CarsimRunner import CarsimSimulationRunner
from .utils.logger import log_info, log_warning, log_error

class CarsimAgent:
    def __init__(self, template_path: str, lib_json_path: str = None):
        self.template_path = os.path.abspath(template_path)
        self.library = CarsimLibrary()
        self.template_library = CarsimLibrary()
        
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
            log_info("No library JSON path provided, attempting to load default carsim_param_lib.json")
            # 优先从包内查找
            pkg_dir = os.path.dirname(os.path.abspath(__file__))
            default_lib_json_path = os.path.join(pkg_dir, "carsim_param_lib.json")
            
            if not os.path.exists(default_lib_json_path):
                log_warning(f"Default library file not found: {default_lib_json_path}")
                # 如果包内没有，且 template_library 已经加载，则复用它
                if os.path.exists(echo_path):
                    self.library = self.template_library
                    log_info("Using template library as LLM context.")
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

        self.llm = CarsimLLMInterface()

    def run(self, prompt: str = None, mods_json_path: str = None):
        """
        运行仿真。支持两种模式：
        1. prompt 模式：通过自然语言输入，调用 LLM 生成修改项。
        2. mods_json_path 模式：直接提供 JSON 修改文件路径。
        """
        mods = {}
        if mods_json_path:
            log_info(f"🚀 Starting agent task with JSON file: '{mods_json_path}'")
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
        elif prompt:
            log_info(f"🚀 Starting agent task with prompt: '{prompt}'")
            # 1. 解析意图 -> 参数修改
            log_info("Analyzing intent with LLM...")
            context = self.library.get_parameter_context(limit=1000)
            mods = self.llm.generate_mods(prompt, context)
            log_info(f"Proposed modifications: {json.dumps(mods, indent=2)}")
        else:
            log_error("Neither prompt nor mods_json_path provided.")
            return

        # 执行仿真工作流
        self._execute_simulation_workflow(mods)

    def _execute_simulation_workflow(self, mods: Dict[str, Any]):
        """
        核心仿真流水线逻辑：验证、生成配置、运行。
        """
        # 2. 初始化 Runner
        log_info(f"Initializing simulation runner using template path: {self.template_path}")
        runner = CarsimSimulationRunner(self.template_path)
        runner.setup_workspace()
        
        # 3. 验证配置是否正确
        if not self.verify_modifications(mods):
            error_msg = "Parameter verification FAILED. Some parameters are missing in the template or invalid."
            log_error(error_msg)
            # 抛出异常以停止运行并显示错误
            raise ValueError(error_msg)
            
        # 4. 创建配置文件
        runner.create_parsfile(mods)
        runner.create_simfile()
        
        # 5. 运行仿真
        try:
            if runner.run_solver():
                log_info("Simulation completed successfully.")            
            else:
                log_error("Simulation failed.")
        except Exception as e:
            error_msg = f"Simulation runner encountered an error: {e}\n{traceback.format_exc()}"
            log_error(error_msg)
        
        log_info("Task finished.")

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
                
            # 使用特定于模板的参数库进行对比
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
