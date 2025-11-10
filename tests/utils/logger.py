"""Custom logging utility for test framework."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
import colorlog

from config.settings import LOGS_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


class TestLogger:
    """Custom logger for test framework with color support and file rotation."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str = "TestFramework") -> logging.Logger:
        """
        Get or create a logger instance.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, LOG_LEVEL))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        color_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)8s] [%(name)s] %(message)s%(reset)s",
            datefmt=LOG_DATE_FORMAT,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def log_test_start(cls, test_name: str):
        """Log test start."""
        logger = cls.get_logger()
        logger.info("=" * 80)
        logger.info(f"TEST STARTED: {test_name}")
        logger.info("=" * 80)
    
    @classmethod
    def log_test_end(cls, test_name: str, status: str):
        """Log test end."""
        logger = cls.get_logger()
        logger.info("=" * 80)
        logger.info(f"TEST {status.upper()}: {test_name}")
        logger.info("=" * 80)
    
    @classmethod
    def log_step(cls, step_description: str):
        """Log test step."""
        logger = cls.get_logger()
        logger.info(f"STEP: {step_description}")
    
    @classmethod
    def log_assertion(cls, assertion: str, result: bool):
        """Log assertion result."""
        logger = cls.get_logger()
        status = "PASS" if result else "FAIL"
        log_method = logger.info if result else logger.error
        log_method(f"ASSERTION [{status}]: {assertion}")


# Create default logger instance
logger = TestLogger.get_logger()
