from torchvision import transforms
from torchvision.datasets import ImageNet, Imagenette

from utils.log_utils import get_logger
from utils.torch_utils import torch_distributed_zero_first

# Get global logger (no need to pass dynamic names here)
logger = get_logger(__name__)

def read_imagenet_dataset(image_size:int, data_dir: str, local_rank: int):
    """
    Load ImageNet dataset (Only the primary process will load data).
    Args:
        image_size: the size of the input image.
        data_dir: the root directory of ImageNet dataset.
        local_rank: current process index. (-1 means unlaunch DDP mode)

    Returns:
        train_dataset, val_dataset: ImageNet dataset.
    """
    # Preprocessing of training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Use context to make sure that only the main process will read data.
    with torch_distributed_zero_first():
        logger.info(f"Loading ImageNet dataset from {data_dir}.")
        train_dataset = ImageNet(data_dir, split='train', transform=train_transforms)
        val_dataset = ImageNet(data_dir, split='val', transform=val_transforms)

    return train_dataset, val_dataset

def read_imagenette_dataset(image_size:int, data_dir: str, size:str):
    """
    Download and load Imagenette dataset (Only the primary process will load data).
    Args:
        image_size: the size of the input image.
        size: the image size of Imagenette dataset. Supports "full"(default), "160px", "320px".
        data_dir: the root directory of ImageNet dataset.

    Returns:
        train_dataset, val_dataset: ImageNet dataset.
    """
    # Preprocessing of training and validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Use context to make sure that only the main process will read data.
    with torch_distributed_zero_first():
        logger.info(f"Downloading and loading Imagenette dataset from {data_dir}.")
        train_dataset = Imagenette(data_dir, split='train', size=size, download=True, transform=train_transforms)
        val_dataset = Imagenette(data_dir, split='val', size=size, download=False, transform=val_transforms)

    return train_dataset, val_dataset