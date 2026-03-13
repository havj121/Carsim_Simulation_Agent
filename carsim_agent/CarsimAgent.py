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
            self.default_extract_vars = ["Time", "Vx", "Steer_SW", "Fx_L2", "Fy_L2", "Fx_R2", "Fy_R2", "Alpha_R2", "Kappa_R2"]
            # 数据提取方式: "csv" (读取 LastRun.csv) 或 "vsb" (转换 vsb/vs)
            self.extraction_method = "csv"
            # 是否保留每个任务的详细结果文件夹 (默认为 True)
            self.keep_detailed_results = True

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

    def run(self, prompt: str = None, mods_json_path: str = None, extract_data: bool = None, extract_vars: list = None, task_id: str = None, results_base_dir: str = None):
        """
        运行仿真并提取数据。
        :param prompt: 自然语言指令 (仅在 run_mode="prompt" 时使用，或未提供 mods_json_path 时作为备选)
        :param mods_json_path: 直接修改的 JSON 文件路径 (仅在 run_mode="mods" 时使用，或未提供 prompt 时作为备选)
        :param extract_data: 是否进行数据提取，若为 None 则使用 options.extract_data
        :param extract_vars: 需要从结果中提取的变量列表，若为 None 则使用 options.default_extract_vars
        :param task_id: 任务 ID，用于区分并发运行的文件夹
        :param results_base_dir: 结果存储的基准目录，若为 None 则使用全局配置
        """
        # 确定实际使用的 extract_data 和 extract_vars
        actual_extract_data = extract_data if extract_data is not None else self.options.extract_data
        actual_extract_vars = extract_vars if extract_vars is not None else self.options.default_extract_vars

        mods = {}
        # 根据 options.run_mode 选择执行方法
        if self.options.run_mode == "mods":
            if mods_json_path:
                if not os.path.exists(mods_json_path):
                    log_error(f"Provided mods_json_path does not exist: {mods_json_path}")
                    return
                try:
                    with open(mods_json_path, 'r', encoding='utf-8') as f:
                        mods = json.load(f)
                except Exception as e:
                    log_error(f"Failed to load mods from JSON: {e}")
                    return
            else:
                log_error("run_mode is 'mods' but mods_json_path not provided.")
                return
        elif self.options.run_mode == "prompt":
            if prompt:
                # LLM 分析时先打印到控制台，不触发日志文件生成
                print(f"🚀 Starting agent task with prompt (prompt mode): '{prompt}'")
                print("Analyzing intent with LLM...")
                context = self.library.get_parameter_context(limit=1000)
                mods = self.llm.generate_mods(prompt, context)
            else:
                log_error("run_mode is 'prompt' but prompt not provided.")
                return
        else:
            log_error(f"Unknown run_mode: {self.options.run_mode}")
            return

        # 执行仿真工作流
        return self._execute_simulation_workflow(mods, actual_extract_data, actual_extract_vars, task_id=task_id, results_base_dir=results_base_dir)

    def run_batch(self, batch_configs: List[Dict[str, Any]], processes: int = 4, results_dir: str = None):
        """
        批量运行仿真任务。
        :param batch_configs: 配置列表，每个元素为一个 mods 字典
        :param processes: 并行进程数
        :param results_dir: 批量结果存储目录
        :return: 批量运行结果列表
        """
        from multiprocessing import Pool
        from datetime import datetime
        from .config import CARSIM_RESULTS_DIR

        if results_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = os.path.join(CARSIM_RESULTS_DIR, f"Batch_Run_{timestamp}")
        
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
            
        # 设置统一的日志文件
        from .utils.logger import setup_logger
        batch_log_filename = "batch_run.log"
        setup_logger(log_dir=results_dir, log_filename=batch_log_filename, mode='w')
        
        log_info(f"🚀 Starting batch simulation with {len(batch_configs)} tasks using {processes} processes...")
        log_info(f"Results and consolidated log will be stored in: {results_dir}")

        # 准备任务参数
        tasks = []
        for i, mods in enumerate(batch_configs):
            task_id = f"Task_{i}"
            # 传递统一的日志路径
            tasks.append((self.template_path, mods, self.options, task_id, results_dir, batch_log_filename))

        with Pool(processes=processes) as pool:
            results = pool.map(_run_single_task_wrapper, tasks)

        # 保存汇总摘要
        summary_file = os.path.join(results_dir, "batch_results_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
        log_info(f"\n✅ Batch simulation finished. Summary saved to: {summary_file}")
        return results

    def get_simulation_data(self, batch_configs: List[Dict[str, Any]], target_vars: List[str] = None, processes: int = 4):
        """
        运行批量仿真并提取指定数据。
        """
        if target_vars:
            self.options.default_extract_vars = target_vars
        
        self.options.run_mode = "mods"
        self.options.extract_data = True
        self.options.keep_detailed_results = True
        self.options.extraction_method = "csv"
        
        results = self.run_batch(batch_configs, processes=processes)
        
        # 显式确保数据已提取
        for res in results:
            if res.get("status") == "success":
                res_path = res.get("results_path")
                csv_path = os.path.join(res_path, "LastRun.csv")
                if os.path.exists(csv_path):
                    self.extractor.extract_to_csv(csv_path, target_vars=self.options.default_extract_vars, method="csv")
        
        return results

    @staticmethod
    def load_historical_data(batch_run_dir: str, required_vars: List[str] = None):
        """
        从指定的仿真目录加载历史数据。
        优先读取 batch_results_summary.json，若数据缺失则尝试读取各任务的 LastRun.csv。
        """
        summary_path = os.path.join(batch_run_dir, "batch_results_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Summary file not found at {summary_path}")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            batch_results = json.load(f)
        
        # 如果没有提供 required_vars，则使用一个基础的默认列表
        if required_vars is None:
            required_vars = ["Time", "Vx", "Alpha_L1", "Fz_L1", "Fy_L1"]

        for i, res in enumerate(batch_results):
            data_missing = False
            if not res.get("extracted_data"):
                data_missing = True
            else:
                import pandas as pd
                df_temp = pd.DataFrame(res["extracted_data"])
                if not all(col in df_temp.columns for col in required_vars):
                    data_missing = True
            
            if data_missing:
                task_id = res.get("task_id", f"Task_{i}")
                possible_task_dirs = [
                    os.path.join(batch_run_dir, f"Carsim_Agent_Run_{task_id}"),
                    os.path.join(batch_run_dir, task_id),
                    os.path.join(batch_run_dir, f"Carsim_Agent_Run_Task_{i}")
                ]
                
                csv_found = False
                for task_dir in possible_task_dirs:
                    if os.path.exists(task_dir):
                        csv_path = os.path.join(task_dir, "LastRun.csv")
                        if os.path.exists(csv_path):
                            log_info(f"🔄 Data missing in JSON for {task_id}, loading from CSV: {csv_path}")
                            import pandas as pd
                            df_csv = pd.read_csv(csv_path)
                            res["extracted_data"] = df_csv.to_dict(orient='list')
                            csv_found = True
                            break
                
                if not csv_found:
                    log_warning(f"⚠️ Warning: Task directory or LastRun.csv for {task_id} not found.")

        return batch_results

    def _execute_simulation_workflow(self, mods: Dict[str, Any], extract_data: bool = False, extract_vars: list = None, task_id: str = None, results_base_dir: str = None):
        """
        核心仿真流水线逻辑：验证、生成配置、运行、数据提取。
        """
        # 定义任务结果
        task_result = {
            "config": mods,
            "status": "failed",
            "extracted_data": None,
            "results_path": None,
            "log_path": None
        }
        
        # 1. 验证配置是否正确
        if not self.verify_modifications(mods):
            error_msg = "Parameter verification FAILED. Some parameters are missing in the template or invalid."
            print(f"ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        # 2. 初始化 Runner
        # 初始化时使用 print，确保在日志文件建立前不触发默认初始化
        print(f"Initializing simulation runner using template path: {self.template_path}")
        runner = CarsimSimulationRunner(self.template_path, task_id=task_id, results_base_dir=results_base_dir)
        runner.setup_workspace()
        task_result["results_path"] = runner.results_path

        # 2.1 设置日志文件 (仅在此处进行显式初始化)
        # 如果是单次运行 (results_base_dir 为 None)，则在项目目录下创建日志
        # 如果是批量运行 (results_base_dir 已由 run_batch 提供)，则沿用 run_batch 设置的统一日志，不再重新初始化
        if results_base_dir is None:
            task_log_dir = os.path.join(runner.results_path, "logs")
            from .utils.logger import setup_logger, get_current_log_path
            setup_logger(log_dir=task_log_dir)
            log_info(f"Logger initialized in: {task_log_dir}")
            task_result["log_path"] = get_current_log_path()
        else:
            from .utils.logger import get_current_log_path
            task_result["log_path"] = get_current_log_path()
            log_info(f"Using unified batch logger: {task_result['log_path']}")

        # 在每个任务正式开始记录后打印分割线
        log_info("\n" + "="*60)
        log_info(f"🆕 NEW TASK STARTED AT: {os.path.basename(self.template_path)}")
        log_info("="*60)
            
        # 3. 创建配置文件
        runner.create_parsfile(mods)
        runner.create_simfile()
        
        # 4. 运行仿真
        try:
            if runner.run_solver():
                log_info("Simulation completed successfully.")
                task_result["status"] = "success"
                
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
                        return task_result

                    csv_path = self.extractor.extract_to_csv(
                        file_to_extract, 
                        target_vars=extract_vars, 
                        method=self.options.extraction_method
                    )
                    
                    if csv_path:
                        log_info(f"Task result extracted to: {csv_path}")
                        # 读取 CSV 数据并存入 task_result
                        try:
                            import pandas as pd
                            # 使用 chunksize 或强制读取后关闭，确保不锁定
                            with open(csv_path, 'r', encoding='utf-8') as f:
                                df = pd.read_csv(f)
                                task_result["extracted_data"] = df.to_dict(orient='records')
                            
                            # 如果是批量运行，不再保留独立的 extracted.csv
                            if results_base_dir is not None:
                                try:
                                    os.remove(csv_path)
                                    log_info(f"Removed temporary extracted CSV: {csv_path}")
                                except Exception as e:
                                    log_warning(f"Failed to remove temporary CSV {csv_path}: {e}")
                        except Exception as e:
                            log_error(f"Failed to load extracted CSV into memory: {e}")
                    else:
                        log_warning("Data extraction failed or skipped.")
                else:
                    log_info("Data extraction skipped (extract_data=False).")
                
                # 6. 清理结果 (如果 keep_detailed_results 为 False)
                if not self.options.keep_detailed_results:
                    log_info(f"Cleaning up detailed results folder: {runner.results_path}")
                    
                    # 批量运行时，不单独移动 CSV 和 LOG，因为数据已包含在返回的 JSON 中，日志已统一记录
                    # 为了能够删除文件夹，必须先关闭当前任务特定的日志处理器
                    from .utils.logger import close_logger
                    close_logger()
                    
                    import shutil
                    import time
                    # 尝试多次删除，防止 Windows 锁定
                    max_retries = 3
                    for i in range(max_retries):
                        try:
                            shutil.rmtree(runner.results_path)
                            log_info(f"Detailed results folder deleted.")
                            break
                        except Exception as e:
                            if i < max_retries - 1:
                                log_warning(f"Failed to delete results folder (attempt {i+1}): {e}. Retrying...")
                                time.sleep(1)
                            else:
                                log_error(f"Failed to delete results folder after {max_retries} attempts: {e}")
                    
                log_info("Task finished.")
            else:
                log_error("Simulation failed.")
                task_result["status"] = "failed"
        except Exception as e:
            error_msg = f"Simulation runner encountered an error: {e}\n{traceback.format_exc()}"
            log_error(error_msg)
            task_result["status"] = "error"
            task_result["error"] = str(e)
            
        return task_result

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

def _run_single_task_wrapper(args):
    """
    用于多进程调用的包装函数。
    """
    template_path, mods, options, task_id, results_dir, log_filename = args
    try:
        # 在子进程中重新设置 logger，以 'a' 模式追加到统一的日志文件
        from .utils.logger import setup_logger
        setup_logger(log_dir=results_dir, log_filename=log_filename, mode='a')
        
        agent = CarsimAgent(template_path=template_path)
        agent.options = options
        
        # 运行仿真
        result = agent._execute_simulation_workflow(mods, extract_data=agent.options.extract_data, task_id=task_id, results_base_dir=results_dir)
        
        # 将任务 ID 显式加入结果
        result["task_id"] = task_id
        return result
    except Exception as e:
        return {"task_id": task_id, "status": "error", "error": str(e)}
