import logging
import os
from datetime import datetime

class CarsimLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CarsimLogger, cls).__new__(cls)
            cls._instance.logger = None
            cls._instance.log_dir = None
        return cls._instance

    def setup(self, log_dir=None, log_filename=None):
        """Initialize or re-initialize the logger with a specific file."""
        # Use absolute path for log_dir to ensure it works from any CWD
        if log_dir is None:
            # Default to 'logs' in the project root (where the package is)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_file_dir))
            log_dir = os.path.join(project_root, "logs")
        else:
            if not os.path.isabs(log_dir):
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_file_dir))
                log_dir = os.path.abspath(os.path.join(project_root, log_dir))

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if log_filename is None:
            now = datetime.now()
            # Include microseconds to ensure unique filenames even when called rapidly
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
            log_filename = f"carsim_agent_{timestamp}.log"
            
        log_path = os.path.join(log_dir, log_filename)
        
        self.logger = logging.getLogger("CarsimAgent")
        self.logger.setLevel(logging.INFO)
        
        # Remove old handlers to ensure we only log to the new file
        if self.logger.handlers:
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
        
        # Create file handler
        try:
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            # Add handler to logger
            self.logger.addHandler(fh)
        except Exception as e:
            print(f"Error setting up log file: {e}")

    def log(self, message, level="INFO"):
        if self.logger is None:
            # Fallback to a default setup if not already initialized
            self.setup()
            
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

def setup_logger(log_dir=None, log_filename=None):
    """External function to trigger new log file generation."""
    logger_instance.setup(log_dir, log_filename)

def log_info(message):
    logger_instance.log(message, "INFO")
    print(message)

def log_warning(message):
    logger_instance.log(message, "WARNING")
    print(f"WARNING: {message}")

def log_error(message):
    logger_instance.log(message, "ERROR")
    print(f"ERROR: {message}")
