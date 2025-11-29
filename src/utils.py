"""
Vesuvius Faz 1 - Utility Functions
Metric hesaplama, logging, seed sabitleme ve diğer yardımcı fonksiyonlar
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import csv


def set_seed(seed: int = 42):
    """
    Reproducibility için random seed ayarla

    Args:
        seed: Random seed değeri
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")


def dice_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Binary Dice coefficient hesapla (metric olarak, loss değil)

    Args:
        pred: Predicted probabilities [B, 1, D, H, W] veya logits
        target: Ground truth [B, 1, D, H, W]
        threshold: Binary threshold
        eps: Numerical stability

    Returns:
        Dice score (0-1, 1 = perfect match)
    """
    # Eğer logits ise sigmoid uygula
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    # Binary prediction
    pred_binary = (pred > threshold).float()

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # Dice calculation
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return dice.item()


def iou_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Intersection over Union (IoU) metric

    Args:
        pred: Predicted probabilities or logits [B, 1, D, H, W]
        target: Ground truth [B, 1, D, H, W]
        threshold: Binary threshold
        eps: Numerical stability

    Returns:
        IoU score (0-1)
    """
    # Sigmoid if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    # Binary prediction
    pred_binary = (pred > threshold).float()

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # IoU calculation
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection

    iou = (intersection + eps) / (union + eps)

    return iou.item()


def f1_metric(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> dict:
    """
    F1, Precision, Recall hesapla

    Args:
        pred: Predicted probabilities or logits
        target: Ground truth
        threshold: Binary threshold
        eps: Numerical stability

    Returns:
        dict: {'f1': f1_score, 'precision': precision, 'recall': recall}
    """
    # Sigmoid if needed
    if pred.min() < 0 or pred.max() > 1:
        pred = torch.sigmoid(pred)

    # Binary prediction
    pred_binary = (pred > threshold).float()

    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)

    # TP, FP, FN calculation
    TP = (pred_flat * target_flat).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()

    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'f1': f1.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def get_device(device_str: Optional[str] = None) -> torch.device:
    """
    Uygun device seç

    Args:
        device_str: 'cuda', 'cpu', 'mps', veya None (otomatik)

    Returns:
        torch.device
    """
    if device_str:
        return torch.device(device_str)

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: nn.Module) -> int:
    """Model parametre sayısını hesapla"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MetricTracker:
    """Metrikleri takip etmek için yardımcı sınıf"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Tüm metrikleri sıfırla"""
        self.metrics = {}
        self.counts = {}

    def update(self, metric_name: str, value: float, count: int = 1):
        """
        Metrik güncelle

        Args:
            metric_name: Metrik adı
            value: Metrik değeri
            count: Batch size veya örnek sayısı
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
            self.counts[metric_name] = 0

        self.metrics[metric_name] += value * count
        self.counts[metric_name] += count

    def get_average(self, metric_name: str) -> float:
        """Ortalama metrik değerini al"""
        if metric_name not in self.metrics:
            return 0.0

        if self.counts[metric_name] == 0:
            return 0.0

        return self.metrics[metric_name] / self.counts[metric_name]

    def get_all_averages(self) -> dict:
        """Tüm metriklerin ortalamalarını al"""
        return {name: self.get_average(name) for name in self.metrics.keys()}


class Logger:
    """Basit bir logger sınıfı"""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: Log dizini
            experiment_name: Deneme adı
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name

        # Log dizinini oluştur
        os.makedirs(log_dir, exist_ok=True)

        # Timestamp ekle
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_prefix = f"{experiment_name}_{timestamp}"

        # CSV logger için dosya yolu
        self.csv_path = os.path.join(log_dir, f"{self.log_prefix}_metrics.csv")
        self.csv_initialized = False

    def log_metrics(self, epoch: int, metrics: dict, phase: str = 'train'):
        """
        Metrikleri CSV'ye kaydet

        Args:
            epoch: Epoch numarası
            metrics: Metrik dictionary
            phase: 'train' veya 'val'
        """
        # CSV header'ı oluştur
        if not self.csv_initialized:
            headers = ['epoch', 'phase'] + list(metrics.keys())
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
            self.csv_initialized = True

        # Metrikleri yaz
        row = {'epoch': epoch, 'phase': phase}
        row.update(metrics)

        with open(self.csv_path, 'a', newline='') as f:
            headers = ['epoch', 'phase'] + list(metrics.keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writerow(row)

    def log_config(self, config: dict):
        """Config'i JSON olarak kaydet"""
        config_path = os.path.join(self.log_dir, f"{self.log_prefix}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def log_text(self, text: str, filename: Optional[str] = None):
        """Text log kaydet"""
        if filename is None:
            filename = f"{self.log_prefix}_log.txt"

        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")


class AverageMeter:
    """Ortalama ve güncel değerleri takip eden basit meter"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss: float,
        metrics: dict,
        filepath: str,
        is_best: bool = False
):
    """
    Model checkpoint kaydet

    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Epoch numarası
        loss: Loss değeri
        metrics: Metrik dictionary
        filepath: Kayıt yolu
        is_best: En iyi model mi
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }

    torch.save(checkpoint, filepath)

    if is_best:
        # Best model'i ayrıca kaydet
        best_path = filepath.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        filepath: str,
        device: torch.device
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], int, dict]:
    """
    Checkpoint yükle

    Args:
        model: PyTorch model
        optimizer: Optimizer (None olabilir)
        filepath: Checkpoint dosya yolu
        device: Device

    Returns:
        model, optimizer, epoch, metrics
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})

    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")

    return model, optimizer, epoch, metrics


def print_metrics(epoch: int, phase: str, metrics: dict, time_elapsed: Optional[float] = None):
    """
    Metrikleri güzel formatta print et

    Args:
        epoch: Epoch numarası
        phase: 'train' veya 'val'
        metrics: Metrik dictionary
        time_elapsed: Geçen süre (saniye)
    """
    output = f"Epoch {epoch} [{phase.upper()}]:"

    for name, value in metrics.items():
        if isinstance(value, float):
            output += f" {name}: {value:.4f} |"
        else:
            output += f" {name}: {value} |"

    if time_elapsed is not None:
        output += f" Time: {time_elapsed:.2f}s"

    print(output)


def format_time(seconds: float) -> str:
    """Saniyeyi okunabilir formata çevir"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def test_utils():
    """Utility fonksiyonlarını test et"""
    print("Testing utility functions...")

    # Test seed
    set_seed(42)

    # Test device selection
    device = get_device()
    print(f"Selected device: {device}")

    # Test metrics
    pred = torch.rand(2, 1, 32, 32, 32)
    target = torch.randint(0, 2, (2, 1, 32, 32, 32)).float()

    dice = dice_metric(pred, target)
    print(f"Dice score: {dice:.4f}")

    iou = iou_metric(pred, target)
    print(f"IoU score: {iou:.4f}")

    f1_scores = f1_metric(pred, target)
    print(f"F1 metrics: {f1_scores}")

    # Test metric tracker
    tracker = MetricTracker()
    tracker.update('loss', 0.5, 32)
    tracker.update('loss', 0.3, 32)
    print(f"Average loss: {tracker.get_average('loss'):.4f}")

    # Test average meter
    meter = AverageMeter()
    meter.update(0.5)
    meter.update(0.3)
    print(f"Average meter: {meter.avg:.4f}")

    # Test time formatting
    print(f"Time format test: {format_time(3665)}")


if __name__ == "__main__":
    test_utils()