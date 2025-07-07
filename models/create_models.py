from torchvision.models import get_model

from utils.log_utils import get_logger, display_message
from utils.torch_utils import torch_distributed_zero_first

# Get global logger (no need to pass dynamic names here)
logger = get_logger(__name__)

def initialize_models(configs:dict):
    """
    Load model architecture and initialing weights.
    Args:
        configs: configuration dictionary.

    Returns:
        model: model architecture.

    """
    display_message(logger, f"Loading model architecture and initialing weights: {configs['model']}")
    with torch_distributed_zero_first():
        model = get_model(configs['model'], weights=configs['weights'], num_classes=configs['classes'])
    display_message(logger, f"Model architecture:/n{model}")

    return model
