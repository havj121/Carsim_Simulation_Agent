import logging
import os
from datetime import datetime

class CarsimLogger:
    _instance = None

    def __new__(cls, log_dir="logs", log_filename="carsim_agent.log"):
        if cls._instance is None:
            cls._instance = super(CarsimLogger, cls).__new__(cls)
            cls._instance._initialize_logger(log_dir, log_filename)
        return cls._instance

    def _initialize_logger(self, log_dir, log_filename):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_path = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger("CarsimAgent")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Add handler to logger
        if not self.logger.handlers:
            self.logger.addHandler(fh)

    def log(self, message, level="INFO"):
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        elif level.upper() == "DEBUG":
            self.logger.debug(message)
        else:
            self.logger.info(message)

# Create a default logger instance
logger_instance = CarsimLogger()

def log_info(message):
    logger_instance.log(message, "INFO")
    print(message)

def log_warning(message):
    logger_instance.log(message, "WARNING")
    print(f"WARNING: {message}")

def log_error(message):
    logger_instance.log(message, "ERROR")
    print(f"ERROR: {message}")
