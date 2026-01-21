import logging
import sys
from pathlib import Path
import config

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers (Windows-safe)"""
    logger = logging.getLogger(name)  # Fixed: lowercase 'g'
    logger.setLevel(config.LOG_LEVEL)
    
    if logger.handlers:
        return logger
    
    # Convert string to Path object
    log_dir = Path(config.LOG_DIR)
    log_file = log_dir / f"{name}.log"
    
    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Fix console encoding for Windows
    if sys.platform == 'win32':
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        except:
            pass  # If this fails, continue without UTF-8 fix
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
