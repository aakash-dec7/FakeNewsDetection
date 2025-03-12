import os
import logging
from logging.handlers import RotatingFileHandler

# Logging Configuration Constants
LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(filename)s - %(funcName)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logger(name="app_logger"):
    """
    Configures the application logger with a rotating file handler and console handler.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate logs

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Ensure log directory exists
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)  # Change to INFO for production
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = configure_logger()
