from torchvision.models import get_model

from utils.log_utils import get_logger
from utils.torch_utils import torch_distributed_zero_first, is_main_process


def initialize_models(configs:dict, num_gpus:int = 1):
    """
    Load model architecture and initialing weights.
    Args:
        configs: configuration dictionary.
        num_gpus: the number of available GPUs.

    Returns:
        model: model architecture.

    """
    logger = get_logger(file_name=__name__)
    if is_main_process():
        logger.info(f"Loading model architecture and initialing weights: {configs['model']}")

    with torch_distributed_zero_first(num_gpus):
        model = get_model(configs['model'], weights=configs['weights'], num_classes=configs['classes'])

    if is_main_process():
        logger.info(f"Model architecture:\n{model}")

    return model
