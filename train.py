import os
import wandb
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

from datasets.create_datasets import read_imagenette_dataset
from models.create_models import initialize_models
from utils.file_utils import load_yaml, create_dirs
from utils.log_utils import get_logger
from utils.torch_utils import initialize_device, is_main_process, cleanup
from utils.wandb_utils import initialize_wandb


def train_on_epoch_with_amp(epoch, epochs, train_loader, model, optimizer, scheduler, criterion, scaler, device):
    """
    Train on one epoch with AMP.

    References:
        Pytorch official documentation: https://docs.pytorch.org/docs/stable/notes/amp_examples.html

    Args:
        epoch: the current epoch number.
        epochs: the total number of epochs to train.
        train_loader: DataLoader, the training data loader.
        model: nn.Module, the model to be trained.
        optimizer: Optimizer, the optimizer to be used.
        scheduler: _LRScheduler, the learning rate scheduler.
        criterion: nn.Module, the loss function to be used.
        scaler: GradScaler, the scaler used for automatic mixed precision training.
        device: torch.device, the device to be used for training.

    Returns:
        avg_train_loss: the average training loss over the current epoch.
        avg_train_acc: the average training accuracy over the current epoch.
    """
    model.train()  # Inform layers such as BatchNorm, Dropout

    if is_main_process():
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"(Train) Epoch {epoch}/{epochs - 1}", position=0)
    else:
        pbar = train_loader

    # Determine 'device_type' of autocast.
    if device.type == 'cuda':
        device_type = 'cuda'
    else:
        device_type = 'cpu'

    # Train on one epoch
    train_loss, train_acc = 0.0, 0.0
    for batch_index, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Runs the forward pass under autocast.
        with autocast(device_type=device_type):
            output = model(images)
            loss = criterion(output, labels)

        # Exits autocast before backward().
        # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
        # Backward ops run in the same dtype autocast chose for corresponding forward ops.
        scaler.scale(loss).backward()  # Backward passes under autocast are not recommended.

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for the next iteration.
        scaler.update()

        # Calculate the training accuracy of the current epoch.
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == labels).item() / len(labels)

        # Accumulate the training loss and accuracy.
        train_loss += loss.item()
        train_acc += acc

        # Only update the progress bar on the main process.
        if is_main_process():
            pbar.set_description(f"(Train) Epoch {epoch + 1}/{epochs} | Loss: {loss:.4g} | Acc: {acc:.4g}")

    # Update the learning rate if needed
    scheduler.step()

    # Calculate the average loss and accuracy.
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    return avg_train_loss, avg_train_acc


def train_on_epoch(epoch, epochs, train_loader, model, optimizer, scheduler, criterion, device):
    """
    Train on one epoch without amp.

    Args:
        epoch: the current epoch number.
        epochs: the total number of epochs to train.
        train_loader: DataLoader, the training data loader.
        model: nn.Module, the model to be trained.
        optimizer: Optimizer, the optimizer to be used.
        scheduler: _LRScheduler, the learning rate scheduler.
        criterion: nn.Module, the loss function to be used.
        device: torch.device, the device to be used for training.

    Returns:
        avg_train_loss: the average training loss over the current epoch.
        avg_train_acc: the average training accuracy over the current epoch.
    """
    model.train()  # Inform layers such as BatchNorm, Dropout

    if is_main_process():
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"(Train) Epoch {epoch}/{epochs - 1}", position=0)
    else:
        pbar = train_loader

    # Train on one epoch
    train_loss, train_acc = 0.0, 0.0
    for batch_index, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # Zero gradients for every batch
        optimizer.zero_grad()

        # Forward pass
        output = model(images)
        preds = torch.argmax(output, dim=1)

        # Compute the loss and its gradients
        loss = criterion(output, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Calculate the training accuracy of the current epoch.
        acc = torch.sum(preds == labels).item() / len(labels)

        # Accumulate the training loss and accuracy.
        train_loss += loss.item()
        train_acc += acc

        # Only update the progress bar on the main process.
        if is_main_process():
            pbar.set_description(f"(Train) Epoch {epoch + 1}/{epochs} | Loss: {loss:.4g} | Acc: {acc:.4g}")

    # Update the learning rate if needed
    scheduler.step()

    # Calculate the average loss and accuracy.
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)

    return avg_train_loss, avg_train_acc


def val_on_epoch(epoch, epochs, val_loader, model, criterion, device):
    """
    Validate on one epoch.

    Args:
        epoch: the current epoch number.
        epochs: the total number of epochs to train.
        val_loader: DataLoader, the validation data loader.
        model: nn.Module, the model to be validated.
        criterion: nn.Module, the loss function to be used.
        device: torch.device, the device to be used for training.

    Returns:
        avg_val_loss: the average validation loss over the current epoch.
        avg_val_acc: the average validation accuracy over the current epoch.

    """
    model.eval()

    if is_main_process():
        pbar = tqdm(val_loader, total=len(val_loader), desc=f"(Validate) Epoch {epoch + 1}/{epochs}", position=0)
    else:
        pbar = val_loader

    # Validate on one epoch
    val_loss, val_acc = 0.0, 0.0
    for batch_index, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)

            # Calculate the validation accuracy of the current epoch.
            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == labels).item() / len(labels)

            # Accumulate the training loss and accuracy.
            val_loss += loss.item()
            val_acc += acc

            # Only update the progress bar on the main process.
            if is_main_process():
                pbar.set_description(f"(Validation) Epoch {epoch + 1}/{epochs} | Loss: {loss:.4g} | Acc: {acc:.4g}")

    # Calculate the average loss and accuracy.
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)

    return avg_val_loss, avg_val_acc


def parser_args():
    """
    Parse command line arguments for training configuration.

    Returns:
        argparse.Namespace: Parsed command-line arguments containing:
            - Path arguments (config, wandb_project, wandb_name)
            - Environment settings (amp, wandb)
    """
    parser = argparse.ArgumentParser(description="PyTorch CNNs pipeline with AMP training.")
    parser.add_argument("--config", type=str, default="config/imagenet2012.yaml", help="Path to the YAML config file.")
    parser.add_argument("--weights", type=str, default="weights", help="Path to the weights file.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision training.")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases for logging.")
    parser.add_argument("--wandb_project", type=str, default="pytorch-amp-exp", help="Name of the project.")
    parser.add_argument("--wandb_name", type=str, help="Name of the run.")

    return parser.parse_args()


def main():
    # Get environment variables
    local_rank = int(os.getenv("LOCAL_RANK", -1))

    # Parse command line arguments
    args = parser_args()

    # Dynamically initialize logger based on args
    logger = get_logger(
        file_name=__name__,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )

    # Set up the running environment
    device, num_gpus = initialize_device()

    # Load the training configuration
    configs = load_yaml(args.config)

    # Initialize the Weights & Biases
    if args.wandb and is_main_process():
        initialize_wandb(args.wandb_project, args.wandb_name, configs)

    # Create the 'train_dataset' and the 'valid_dataset'
    image_size = configs['image_size']
    train_dataset, val_dataset = read_imagenette_dataset(image_size, configs['path'], configs['size'])

    # Use 'DistributedSampler' to ensure reasonable data distribution
    train_sampler = DistributedSampler(train_dataset) if num_gpus > 1 else None
    val_sampler = DistributedSampler(val_dataset) if num_gpus > 1 else None

    # Create the 'train_loader' and the 'valid_loader'
    batch_size, num_workers = configs['batch_size'], configs['workers']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=configs['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=configs['pin_memory'])

    # Load model architecture and initialize weights
    model = initialize_models(configs)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if num_gpus > 1 else model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) if num_gpus > 1 else model
    model.to(device)

    # Optimizer configuration
    lr, momentum, weight_decay = configs['lr'], configs['momentum'], configs['weight_decay']
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Scheduler Configuration
    mode, patience = configs['mode'], configs['patience']
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, patience=patience)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Hyper-parameters
    epochs, batch_size = configs['epochs'], configs['batch_size']

    # Training loop
    scaler = GradScaler()
    best_val_acc = 0.0
    for epoch in range(epochs):

        if num_gpus > 1:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        if args.amp:  # DDP mode
            if is_main_process():
                logger.info(f"Using Automatic Mixed Precision (AMP) training.")
            avg_train_loss, avg_train_acc = train_on_epoch_with_amp(epoch, epochs, train_loader, model, optimizer, scheduler, criterion, scaler, device)
            avg_val_loss, avg_val_acc = val_on_epoch(epoch, epochs, val_loader, model, criterion, device)

        else:
            logger.info(f"Using Full Precision (FP32) training.")
            avg_train_loss, avg_train_acc = train_on_epoch( epoch, epochs, train_loader, model, optimizer, scheduler, criterion, device)
            avg_val_loss, avg_val_acc = val_on_epoch(epoch, epochs, val_loader, model, criterion, device)

        # Record the values during the training process.
        if is_main_process():
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'learning_rate': current_lr
            })
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4g} | Train Acc: {avg_train_acc:.4g}")
            logger.info(f"Epoch {epoch + 1}/{epochs} | Val Loss: {avg_val_loss:.4g} | Val Acc: {avg_val_acc:.4g}")

        # Save the weights for every 5 epochs.
        if epoch%5 == 0 and is_main_process():
            save_path = f"{args.weights}/{args.wandb_project}/{args.wandb_name}/model_{epoch}.pth"
            create_dirs(save_path)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Model saved at epoch {epoch}.")

        # Save the best weights
        if avg_val_acc > best_val_acc and is_main_process():
            best_val_acc = avg_val_acc
            save_path = f"{args.weights}/{args.wandb_project}/{args.wandb_name}/best_model_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Best model saved at epoch {epoch}.")

    cleanup()
    wandb.finish()

if __name__ == '__main__':
    main()