# utils/log_setup.py
# Logging configuration for the NYSE trading bot
# Renamed from logging.py (Feb 28 2026) to avoid shadowing stdlib logging module

import logging
import logging.handlers
import sys
from pathlib import Path
# FIX #73: Wrap config import in try/except so log_setup can be imported independently
try:
    from config import CONFIG
except ImportError:
    CONFIG = {'LOG_LEVEL': 'INFO'}

# Global flag to ensure setup only runs once
_setup_done = False

def setup_logging():
    """Configure logging for the entire bot — safe to call multiple times."""
    global _setup_done
    if _setup_done:
        return logging.getLogger(__name__)  # L12 FIX: Consistent return type

    log_level = getattr(logging, CONFIG.get('LOG_LEVEL', 'INFO').upper(), logging.INFO)
    
    # FIX #42: Use absolute path based on project root instead of relative path
    _project_root = Path(__file__).resolve().parent.parent
    _logs_dir = _project_root / "logs"
    _logs_dir.mkdir(exist_ok=True)
    
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
    
    # File handler (rotating: 10 MB max, 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        str(_logs_dir / 'nyse_bot.log'), mode='a', encoding='utf-8',
        maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # M20 FIX: Comprehensive list of noisy library loggers
    for noisy_lib in [
        'httpx', 'httpcore', 'urllib3', 'websockets', 'asyncio',
        'aiohttp', 'matplotlib', 'PIL', 'filelock', 'fsspec',
        'pytorch_lightning', 'lightning', 'lightning.pytorch',
        'sklearn', 'pgmpy', 'numba',
    ]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete — console + file logging active")  # L11 FIX: removed emoji
    
    _setup_done = True  # Mark as done
    return logger
