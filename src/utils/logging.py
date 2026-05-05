from loguru import logger
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", ".logs"))
LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    LOG_DIR / "app.log",
    level=LOG_LEVEL,
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

def log(level: str, message: str, **kwargs):
    getattr(logger, level.lower())(message, **kwargs)

def debug(message: str, **kwargs):
    logger.debug(message, **kwargs)

def info(message: str, **kwargs):
    logger.info(message, **kwargs)

def warning(message: str, **kwargs):
    logger.warning(message, **kwargs)

def error(message: str, **kwargs):
    logger.error(message, **kwargs)

def critical(message: str, **kwargs):
    logger.critical(message, **kwargs)
