import os
import torch
import torch.distributed as dist
from torch.backends import cudnn
from contextlib import contextmanager
from datetime import timedelta

from log_utils import get_logger, display_message

logger = get_logger(file_name=__file__)

def initialize_device():
    """
    Automatically detect the environment and choose a device.
    At the same time, determine whether to launch DDP mode based on GPU number.

    Returns:
        device: torch.device('cuda') or torch.device('cpu')
    """
    # Check available GPU number
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # Launch DDP mode
        local_rank = int(os.getenv("LOCAL_RANK", -1))
        assert torch.cuda.device_count() > local_rank, 'Insufficient CUDA devices for DDP setup.'

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
        device = torch.device("cuda", local_rank)
        display_message(logger, local_rank, f"Using Distributed Data Parallel (DDP) on {num_gpus} GPUs.")

    elif num_gpus == 1:
        # Run on a single GPU
        device = torch.device("cuda:0")
        display_message(logger, 0, f"Using single GPU: {torch.cuda.get_device_name(0)}.")

    else:
        # No GPU resource, use CPU
        device = torch.device("cpu")
        display_message(logger, 0, "Using CPU as no GPU is available.", level="WARNING")

    # Faster training is possible due to fixed input size and consistent architecture.
    cudnn.benchmark = True

    return device


def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()
    display_message(logger, 0, "Distributed training process has been terminated.")


@contextmanager
def torch_distributed_zero_first(rank: int):
    """
    Let distributed processes wait for the rank = = 0 process to complete the necessary operations.
    Args:
        rank: GPU index in whole distributed training.
    """
    if rank not in [-1, 0]:
        dist.barrier()  # Block non-main processes and wait for the main process
    yield
    if rank == 0:
        dist.barrier()  # Let other processes continue conducting after the main process completed.


if __name__ == "__main__":
    initialize_device()