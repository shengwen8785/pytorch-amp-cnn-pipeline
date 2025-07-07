from torchvision.models import get_model

from utils.log_utils import get_logger
from utils.torch_utils import torch_distributed_zero_first, is_main_process


def initialize_models(configs:dict):
    """
    Load model architecture and initialing weights.
    Args:
        configs: configuration dictionary.

    Returns:
        model: model architecture.

    """
    logger = get_logger(file_name=__name__)
    if is_main_process():
        logger.info(f"Loading model architecture and initialing weights: {configs['model']}")

    with torch_distributed_zero_first():
        model = get_model(configs['model'], weights=configs['weights'], num_classes=configs['classes'])

    if is_main_process():
        logger.info(f"Model architecture:\n{model}")

    return model
