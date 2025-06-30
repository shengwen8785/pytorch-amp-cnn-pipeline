import os
import logging
from pathlib import Path


def initialize_logger(file_name="__main__", log_name="app.log", log_dir="./logs", log_level=logging.INFO, console_output=True):
    """
    Initialize logging configuration：

    Args:
        file_name (str): the name of the file that calls this function, default: "__main__".
        log_name (str): logger's name, it should be unique for each application.
        log_dir (str): the saving path of a log file, default: "logs".
        log_level (int): set the logging display level, default: INFO。
        console_output (bool): whether to display messages on standard output (console).

    Returns:
        logger (logging.Logger): initialized logger.
    """
    root_path = str(Path(__file__).parent.parent)

    # Ensure 'log_dir' existed
    log_dir_path = os.path.join(root_path, log_dir)
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # Form log file path
    log_file_path = os.path.join(log_dir_path, log_name)

    # Initialize logger
    logger = logging.getLogger(file_name)
    logger.setLevel(log_level)

    # Prevent from conducting several handlers to cause duplicate messages.
    if not logger.handlers:

        # Set a file handler
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding="utf-8")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        if console_output:
            # Set a standard output handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

    return logger


def display_message(logger: logging.Logger, rank: int, info: str = "", level: str = "INFO"):
    """
    Only display messages on rank 0.

    Args:
        logger (logging.Logger): logger。
        rank (int): to determine whether to display messages on rank 0, such as, rank 0 means global rank 0.
        info (str): the message to be displayed and recorded in the log file.
        level (str): logging level, such as "INFO", "WARNING", "ERROR".
    """
    if rank == 0:
        if level.upper() == "INFO":
            logger.info(info)
        elif level.upper() == "WARNING":
            logger.warning(info)
        elif level.upper() == "ERROR":
            logger.error(info)
        elif level.upper() == "DEBUG":
            logger.debug(info)
        elif level.upper() == "CRITICAL":
            logger.critical(info)
        else:
            raise ValueError(f"Unsupported log level: {level}")


if __name__ == "__main__":
    logger = initialize_logger(file_name=__file__)
    logger.info("This is a test log.")