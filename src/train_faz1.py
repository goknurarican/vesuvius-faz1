"""
Vesuvius Faz 1 - Training Script
3D U-Net ile binary surface segmentation training
"""
print(">>> train_faz1.py imported", flush=True)

import os
import argparse
import yaml
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Local imports
from dataset import create_dataloaders
from model_unet3d import UNet3D
from losses import BCEDiceLoss
from utils import (
    set_seed, get_device, count_parameters,
    dice_metric, iou_metric, f1_metric,
    MetricTracker, Logger, AverageMeter,
    save_checkpoint, load_checkpoint,
    print_metrics, format_time
)


def train_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        config: dict
) -> dict:
    """
    Bir epoch training yap

    Returns:
        Ortalama metrikler
    """
    model.train()

    # Metric trackers
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        # Data to device
        ct = batch['ct'].to(device)
        mask = batch['mask'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(ct)
        logits = outputs['mask']

        # Calculate loss
        loss_dict = criterion(logits, mask)
        loss = loss_dict['total']

        # Backward pass
        loss.backward()

        # Gradient clipping (opsiyonel)
        if config.get('gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])

        # Optimizer step
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            dice_score = dice_metric(logits, mask)

        # Update meters
        batch_size = ct.size(0)
        loss_meter.update(loss.item(), batch_size)
        dice_meter.update(dice_score, batch_size)

        # Update progress bar
        pbar.set_postfix({
            'Loss': f"{loss_meter.avg:.4f}",
            'Dice': f"{dice_meter.avg:.4f}"
        })

        # Log intermediate results (opsiyonel)
        if batch_idx % config.get('log_interval', 50) == 0 and batch_idx > 0:
            print(f"\nBatch {batch_idx}/{len(dataloader)}: "
                  f"Loss={loss_meter.avg:.4f}, Dice={dice_meter.avg:.4f}")

    # Return average metrics
    metrics = {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg
    }

    return metrics


def validate_epoch(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device,
        epoch: int
) -> dict:
    """
    Validation epoch

    Returns:
        Ortalama metrikler
    """
    model.eval()

    # Metric trackers
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            # Data to device
            ct = batch['ct'].to(device)
            mask = batch['mask'].to(device)

            # Forward pass
            outputs = model(ct)
            logits = outputs['mask']

            # Calculate loss
            loss_dict = criterion(logits, mask)
            loss = loss_dict['total']

            # Calculate metrics
            dice_score = dice_metric(logits, mask)
            iou_score = iou_metric(logits, mask)

            # Update meters
            batch_size = ct.size(0)
            loss_meter.update(loss.item(), batch_size)
            dice_meter.update(dice_score, batch_size)
            iou_meter.update(iou_score, batch_size)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss_meter.avg:.4f}",
                'Dice': f"{dice_meter.avg:.4f}",
                'IoU': f"{iou_meter.avg:.4f}"
            })

    # Calculate additional metrics
    metrics = {
        'loss': loss_meter.avg,
        'dice': dice_meter.avg,
        'iou': iou_meter.avg
    }

    return metrics


def main(config_path: str):
    """
    Ana training fonksiyonu

    Args:
        config_path: YAML config dosyası yolu
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 50)
    print("Vesuvius Faz 1 - 3D Surface Segmentation Training")
    print("=" * 50)

    # Set seed for reproducibility
    set_seed(config.get('seed', 42))

    # Device selection
    device = get_device(config.get('device'))
    print(f"Using device: {device}")

    # Create output directory
    output_dir = config.get('output_dir', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Logger
    logger = Logger(output_dir, config.get('experiment_name', 'faz1'))
    logger.log_config(config)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    print("\nCreating model...")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=config['model']['base_channels'],
        num_levels=config['model']['num_levels'],
        bilinear=config['model'].get('bilinear', False)
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss function
    criterion = BCEDiceLoss(
        bce_weight=config.get('bce_weight', 0.5),
        dice_weight=config.get('dice_weight', 0.5)
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # Learning rate scheduler
    scheduler_type = config.get('scheduler', 'reduce')
    if scheduler_type == 'reduce':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # maximize dice score
            factor=0.5,
            patience=config.get('scheduler_patience', 5),
            min_lr=1e-6
        )
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )
    else:
        scheduler = None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0

    if config.get('resume_checkpoint'):
        print(f"\nResuming from checkpoint: {config['resume_checkpoint']}")
        model, optimizer, start_epoch, metrics = load_checkpoint(
            model, optimizer, config['resume_checkpoint'], device
        )
        best_dice = metrics.get('dice', 0)

    # Training loop
    print("\nStarting training...")
    print("=" * 50)

    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, device, epoch, config
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion,
            device, epoch
        )

        # Update scheduler
        if scheduler:
            if scheduler_type == 'reduce':
                scheduler.step(val_metrics['dice'])
            else:
                scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start

        print("\n" + "=" * 50)
        print_metrics(epoch, 'train', train_metrics)
        print_metrics(epoch, 'val', val_metrics, epoch_time)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.2e}")
        print("=" * 50)

        # Log to file
        logger.log_metrics(epoch, train_metrics, 'train')
        logger.log_metrics(epoch, val_metrics, 'val')

        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
            print(f"New best Dice score: {best_dice:.4f}")

        checkpoint_path = os.path.join(
            output_dir, 'checkpoints',
            f"checkpoint_epoch_{epoch:03d}.pth"
        )

        save_checkpoint(
            model, optimizer, epoch,
            val_metrics['loss'], val_metrics,
            checkpoint_path, is_best
        )

        # Save last checkpoint
        last_checkpoint_path = os.path.join(
            output_dir, 'checkpoints', 'last_checkpoint.pth'
        )
        save_checkpoint(
            model, optimizer, epoch,
            val_metrics['loss'], val_metrics,
            last_checkpoint_path, False
        )

        # Early stopping check
        if config.get('early_stopping_patience'):
            if epoch - start_epoch > config['early_stopping_patience']:
                # Check if we haven't improved
                checkpoint_files = sorted([
                    f for f in os.listdir(os.path.join(output_dir, 'checkpoints'))
                    if f.startswith('checkpoint_epoch_')
                ])

                if len(checkpoint_files) > config['early_stopping_patience']:
                    recent_best = False
                    for i in range(config['early_stopping_patience']):
                        recent_epoch = epoch - i
                        recent_file = f"checkpoint_epoch_{recent_epoch:03d}.pth"
                        if recent_file + "_best.pth" in checkpoint_files:
                            recent_best = True
                            break

                    if not recent_best:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break

    print("\n" + "=" * 50)
    print(f"Training completed! Best Dice score: {best_dice:.4f}")
    print("=" * 50)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vesuvius Faz 1 Training')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/faz1_baseline.yaml',
        help='Path to config file'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        exit(1)

    print(">>> Vesuvius Faz 1 training script started", flush=True)


    # Start training
    main(args.config)