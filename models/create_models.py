from torchvision.models import get_model

from utils.log_utils import initialize_logger, display_message
from utils.torch_utils import torch_distributed_zero_first

# Initialize logging
logger = initialize_logger(__name__)

def initialize_models(local_rank:int, configs:dict):
    """
    Load model architecture and initialing weights.
    Args:
        local_rank: GPU index.
        configs: configuration dictionary.

    Returns:
        model: model architecture.

    """
    display_message(logger, local_rank, f"Loading model architecture and initialing weights: {configs['model']}")
    with torch_distributed_zero_first(local_rank):
        model = get_model(configs['model'], weights=configs['weights'], num_classes=configs['classes'])
    display_message(logger, local_rank, f"Model architecture:/n{model}")

    return model
