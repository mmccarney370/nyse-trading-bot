# utils/log_setup.py
# Logging configuration for the NYSE trading bot
# Renamed from logging.py (Feb 28 2026) to avoid shadowing stdlib logging module

import logging
import sys
from pathlib import Path
from config import CONFIG

# Global flag to ensure setup only runs once
_setup_done = False

def setup_logging():
    """Configure logging for the entire bot — safe to call multiple times."""
    global _setup_done
    if _setup_done:
        return  # already configured — skip

    log_level = getattr(logging, CONFIG.get('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler('nyse_bot.log', mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from some libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Logging setup complete — console + file logging active")
    
    _setup_done = True  # Mark as done
    return logger
