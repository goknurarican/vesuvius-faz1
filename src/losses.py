"""
Vesuvius Faz 1 - Loss Functions
BCE ve Dice loss implementasyonları
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice Loss hesaplama

    Args:
        pred: Predicted probabilities [B, C, D, H, W] (0-1 arası, sigmoid uygulanmış)
        target: Ground truth [B, C, D, H, W] (0 veya 1)
        smooth: Numerical stability için küçük değer

    Returns:
        Dice loss (0-1 arası, 0 = perfect match)
    """
    # Flatten spatial dimensions
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Dice coefficient hesapla
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Loss = 1 - Dice
    return 1.0 - dice.mean()


def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss - Class imbalance için kullanışlı (opsiyonel)

    Args:
        pred: Predicted probabilities [B, C, D, H, W] (sigmoid uygulanmış)
        target: Ground truth [B, C, D, H, W]
        alpha: Positive class weight
        gamma: Focusing parameter

    Returns:
        Focal loss
    """
    # Binary cross entropy
    bce = F.binary_cross_entropy(pred, target, reduction='none')

    # Focal weight
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (1 - pt) ** gamma

    # Alpha weight
    alpha_weight = torch.where(target == 1, alpha, 1 - alpha)

    # Final loss
    focal = alpha_weight * focal_weight * bce

    return focal.mean()


def tversky_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.5, beta: float = 0.5,
                 smooth: float = 1e-6) -> torch.Tensor:
    """
    Tversky Loss - Dice loss'un genelleştirilmiş hali
    alpha ve beta ile FP ve FN'lere farklı ağırlıklar verilebilir

    Args:
        pred: Predicted probabilities [B, C, D, H, W]
        target: Ground truth [B, C, D, H, W]
        alpha: False positive weight
        beta: False negative weight
        smooth: Numerical stability

    Returns:
        Tversky loss
    """
    # Flatten
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # True Positives, False Positives, False Negatives
    TP = (pred_flat * target_flat).sum(dim=1)
    FP = ((1 - target_flat) * pred_flat).sum(dim=1)
    FN = (target_flat * (1 - pred_flat)).sum(dim=1)

    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1.0 - tversky.mean()


class BCEDiceLoss(nn.Module):
    """
    BCE + Dice Loss kombinasyonu - Ana loss fonksiyonumuz
    """

    def __init__(
            self,
            bce_weight: float = 0.5,
            dice_weight: float = 0.5,
            smooth: float = 1e-6,
            pos_weight: Optional[torch.Tensor] = None
    ):
        """
        Args:
            bce_weight: BCE loss ağırlığı
            dice_weight: Dice loss ağırlığı
            smooth: Dice loss için smoothing
            pos_weight: BCE için positive class weight (class imbalance durumunda)
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Loss hesaplama

        Args:
            logits: Model çıkışı (raw logits, sigmoid uygulanmamış) [B, 1, D, H, W]
            target: Ground truth [B, 1, D, H, W]

        Returns:
            dict: {'total': total_loss, 'bce': bce_loss, 'dice': dice_loss}
        """
        # Sigmoid uygula
        pred_probs = torch.sigmoid(logits)

        # BCE Loss (logits üzerinden daha stabil)
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(logits, target)

        # Dice Loss
        dice_loss_val = dice_loss(pred_probs, target, smooth=self.smooth)

        # Kombine loss
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss_val

        return {
            'total': total_loss,
            'bce': bce_loss,
            'dice': dice_loss_val
        }


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 0.5,
                  dice_weight: float = 0.5) -> torch.Tensor:
    """
    Fonksiyonel BCE + Dice loss (class yerine fonksiyon)

    Args:
        logits: Model çıkışı (raw logits) [B, 1, D, H, W]
        target: Ground truth [B, 1, D, H, W]
        bce_weight: BCE loss ağırlığı
        dice_weight: Dice loss ağırlığı

    Returns:
        Combined loss
    """
    # Sigmoid
    pred_probs = torch.sigmoid(logits)

    # BCE Loss
    bce = F.binary_cross_entropy_with_logits(logits, target)

    # Dice Loss
    dice = dice_loss(pred_probs, target)

    # Weighted combination
    return bce_weight * bce + dice_weight * dice


class CombinedLoss(nn.Module):
    """
    Daha genel bir combined loss class - istediğin loss'ları ekleyebilirsin
    """

    def __init__(self, loss_config: dict):
        """
        Args:
            loss_config: Loss configuration
                Example: {
                    'bce': {'weight': 0.3, 'pos_weight': 2.0},
                    'dice': {'weight': 0.5, 'smooth': 1e-6},
                    'focal': {'weight': 0.2, 'alpha': 0.25, 'gamma': 2.0}
                }
        """
        super().__init__()
        self.loss_config = loss_config

        # Validate config
        total_weight = sum(cfg.get('weight', 0) for cfg in loss_config.values())
        assert abs(total_weight - 1.0) < 1e-6, f"Loss weights should sum to 1.0, got {total_weight}"

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Calculate combined loss

        Returns:
            dict with individual losses and total
        """
        pred_probs = torch.sigmoid(logits)
        losses = {}
        total = 0

        # BCE Loss
        if 'bce' in self.loss_config:
            cfg = self.loss_config['bce']
            pos_weight = cfg.get('pos_weight', None)
            if pos_weight:
                pos_weight = torch.tensor([pos_weight]).to(logits.device)
                bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            else:
                bce = F.binary_cross_entropy_with_logits(logits, target)

            losses['bce'] = bce
            total += cfg['weight'] * bce

        # Dice Loss
        if 'dice' in self.loss_config:
            cfg = self.loss_config['dice']
            dice = dice_loss(pred_probs, target, smooth=cfg.get('smooth', 1e-6))
            losses['dice'] = dice
            total += cfg['weight'] * dice

        # Focal Loss
        if 'focal' in self.loss_config:
            cfg = self.loss_config['focal']
            focal = focal_loss(
                pred_probs, target,
                alpha=cfg.get('alpha', 0.25),
                gamma=cfg.get('gamma', 2.0)
            )
            losses['focal'] = focal
            total += cfg['weight'] * focal

        # Tversky Loss
        if 'tversky' in self.loss_config:
            cfg = self.loss_config['tversky']
            tversky = tversky_loss(
                pred_probs, target,
                alpha=cfg.get('alpha', 0.5),
                beta=cfg.get('beta', 0.5),
                smooth=cfg.get('smooth', 1e-6)
            )
            losses['tversky'] = tversky
            total += cfg['weight'] * tversky

        losses['total'] = total
        return losses


def test_losses():
    """Loss fonksiyonlarını test et"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test tensors
    batch_size = 2
    logits = torch.randn(batch_size, 1, 32, 32, 32).to(device)
    target = torch.randint(0, 2, (batch_size, 1, 32, 32, 32)).float().to(device)

    # Test individual losses
    print("Testing individual losses:")

    pred_probs = torch.sigmoid(logits)
    dice_val = dice_loss(pred_probs, target)
    print(f"Dice loss: {dice_val.item():.4f}")

    focal_val = focal_loss(pred_probs, target)
    print(f"Focal loss: {focal_val.item():.4f}")

    tversky_val = tversky_loss(pred_probs, target)
    print(f"Tversky loss: {tversky_val.item():.4f}")

    # Test combined loss function
    print("\nTesting bce_dice_loss function:")
    combined_val = bce_dice_loss(logits, target)
    print(f"Combined loss: {combined_val.item():.4f}")

    # Test BCEDiceLoss class
    print("\nTesting BCEDiceLoss class:")
    criterion = BCEDiceLoss()
    loss_dict = criterion(logits, target)
    print(f"Total: {loss_dict['total'].item():.4f}")
    print(f"BCE: {loss_dict['bce'].item():.4f}")
    print(f"Dice: {loss_dict['dice'].item():.4f}")

    # Test CombinedLoss class
    print("\nTesting CombinedLoss class:")
    loss_config = {
        'bce': {'weight': 0.3},
        'dice': {'weight': 0.5, 'smooth': 1e-6},
        'focal': {'weight': 0.2, 'alpha': 0.25, 'gamma': 2.0}
    }

    combined_criterion = CombinedLoss(loss_config)
    combined_dict = combined_criterion(logits, target)

    for name, value in combined_dict.items():
        print(f"{name}: {value.item():.4f}")


if __name__ == "__main__":
    test_losses()