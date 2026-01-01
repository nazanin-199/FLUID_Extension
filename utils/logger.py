import logging
import sys
from pathlib import Path
from typing import Optional


class IFLUIDLogger:
    """Centralized logging for IFLUID pipeline."""
    
    _instance: Optional['IFLUIDLogger'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger('IFLUID')
            self.logger.setLevel(logging.INFO)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def add_file_handler(self, log_dir: str) -> None:
        """Add file handler for persistent logs."""
        log_path = Path(log_dir) / 'ifluid.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def info(self, msg: str) -> None:
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        self.logger.debug(msg)
