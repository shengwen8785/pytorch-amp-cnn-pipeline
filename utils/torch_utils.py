import os
import torch
import torch.distributed as dist
from torch.backends import cudnn
from contextlib import contextmanager
from datetime import timedelta

from utils.log_utils import get_logger

def is_main_process():
    """
    Determine whether the current process is the main process.

    Returns:
        bool: True if the current process is the main process, False otherwise.
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def initialize_device():
    """
    Automatically detect the environment and choose a device.
    At the same time, determine whether to launch DDP mode based on GPU number.

    Returns:
        device: torch.device('cuda') or torch.device('cpu')
        num_gpus: the number of available GPUs.
    """
    logger = get_logger(file_name=__name__)
    # Check available GPU number
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # Launch DDP mode
        local_rank = int(os.getenv("LOCAL_RANK", -1))
        assert torch.cuda.device_count() > local_rank, 'Insufficient CUDA devices for DDP setup.'

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        device = torch.device("cuda", local_rank)

        if is_main_process():
            logger.info(f"Using Distributed Data Parallel (DDP) on {num_gpus} GPUs.")

    elif num_gpus == 1:
        # Run on a single GPU
        device = torch.device("cuda:0")
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(0)}.")

    else:
        # No GPU resource, use CPU
        num_gpus = 0
        device = torch.device("cpu")
        logger.warning("Using CPU as no GPU is available.")

    # Faster training is possible due to fixed input size and consistent architecture.
    cudnn.benchmark = True

    return device, num_gpus


def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    logger = get_logger(file_name=__name__)
    logger.info("Distributed training process has been terminated.")


@contextmanager
def torch_distributed_zero_first():
    """
    Let distributed processes wait for the rank = = 0 process to complete the necessary operations.

    """
    if not is_main_process():
        dist.barrier()  # Block non-main processes and wait for the main process
    yield
    if is_main_process():
        dist.barrier()  # Let other processes continue conducting after the main process completed.


if __name__ == "__main__":
    initialize_device()