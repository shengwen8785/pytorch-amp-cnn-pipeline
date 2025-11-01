import os
import logging
from pathlib import Path

_logger = None  # Define a global logger variable

def get_logger(file_name="__main__", log_name="app.log", log_dir="./logs", log_level=logging.INFO, console_output=True, wandb_project=None, wandb_name=None, current_time=None):
    """
    Get the global logger (create if not already created).

    Args:
        file_name (str): the name of the file that calls this function, default: "__main__".
        log_name (str): logger's name, it should be unique for each application.
        log_dir (str): the saving path of a log file, default: "logs".
        log_level (int): set the logging display level, default: INFO。
        console_output (bool): whether to display messages on standard output (console).
        wandb_project (str): W&B project name for dynamic log naming.
        wandb_name (str): W&B run name for dynamic log naming.
        current_time (str): current time for dynamic log naming.

    Returns:
        logger (logging.Logger): initialized logger.
    """
    global _logger
    if _logger is None:
        # Initialize logger only once
        root_path = str(Path(__file__).parent.parent)

        # Define dynamic file name for logs
        if wandb_project and wandb_name:
            log_dir = f"{log_dir}/{wandb_project}/{wandb_name}"
            dynamic_log_name = f"{wandb_project}_{wandb_name}.log"
        else:
            log_dir = log_dir
            dynamic_log_name = log_name

        if current_time is not None:
            log_dir = f"{log_dir}/{current_time}"

        # Ensure 'log_dir' existed
        log_dir_path = os.path.join(root_path, log_dir)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path, exist_ok=True)

        # Form log file path
        log_file_path = os.path.join(log_dir_path, dynamic_log_name)

        # Initialize logger
        logger = logging.getLogger(file_name)
        logger.setLevel(log_level)

        # Prevent from conducting several handlers to cause duplicate messages.
        if not logger.handlers:

            # Set the log file format
            formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s - %(message)s")

            # Set a file handler
            file_handler = logging.FileHandler(log_file_path, mode='a', encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if console_output:
                # Set a standard output handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

        _logger = logger  # Cache the logger object

    return _logger


if __name__ == "__main__":
    from datetime import datetime
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(__name__)
    logger = get_logger(file_name=__name__, wandb_project="test", wandb_name="test_log", current_time=current_time)
    logger.info("This is a test log.")