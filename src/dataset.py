"""
Vesuvius Faz 1 - 3D Patch Dataset
CT volume ve binary maske için patch-based dataset implementasyonu
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile
from typing import Dict, List, Tuple, Optional
import random


class VesuviusPatchDataset(Dataset):
    """
    3D CT volume için patch-based dataset.
    Her sample bir CT volume ve karşılık gelen binary maske içerir.
    """

    def __init__(
            self,
            data_root: str,
            sample_ids: List[str],
            patch_size: Tuple[int, int, int],
            patch_stride: Tuple[int, int, int],
            is_training: bool = True,
            augment: bool = True,
            normalize_method: str = 'minmax',  # 'minmax' veya 'zscore'
            cache_volumes: bool = True
    ):
        """
        Args:
            data_root: Veri kök dizini
            sample_ids: Sample ID listesi
            patch_size: (D, H, W) patch boyutları
            patch_stride: (D, H, W) patch stride değerleri
            is_training: Training modunda mı
            augment: Data augmentation uygulansın mı
            normalize_method: Normalizasyon metodu
            cache_volumes: Volumleri bellekte tut
        """
        self.data_root = data_root
        self.sample_ids = sample_ids
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.is_training = is_training
        self.augment = augment and is_training
        self.normalize_method = normalize_method
        self.cache_volumes = cache_volumes

        # Volume cache
        self.volume_cache = {}
        self.mask_cache = {}

        # Tüm patch pozisyonlarını önceden hesapla
        self.patch_positions = []
        self._prepare_patches()

    def _load_volume(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """CT volume ve maske yükle"""
        if self.cache_volumes and sample_id in self.volume_cache:
            return self.volume_cache[sample_id], self.mask_cache[sample_id]

        # CT volume yükle
        ct_path = os.path.join(self.data_root, sample_id, 'ct.tif')
        volume = tifffile.imread(ct_path).astype(np.float32)

        # Binary maske yükle
        mask_path = os.path.join(self.data_root, sample_id, 'mask.tif')
        mask = tifffile.imread(mask_path).astype(np.float32)

        # Cache'e ekle
        if self.cache_volumes:
            self.volume_cache[sample_id] = volume
            self.mask_cache[sample_id] = mask

        return volume, mask

    def _prepare_patches(self):
        """Tüm olası patch pozisyonlarını hesapla"""
        for sample_id in self.sample_ids:
            volume, mask = self._load_volume(sample_id)
            D, H, W = volume.shape
            pd, ph, pw = self.patch_size
            sd, sh, sw = self.patch_stride

            # Valid patch başlangıç pozisyonları
            d_positions = list(range(0, max(1, D - pd + 1), sd))
            h_positions = list(range(0, max(1, H - ph + 1), sh))
            w_positions = list(range(0, max(1, W - pw + 1), sw))

            # Son patch'leri de dahil et
            if D > pd and d_positions[-1] + pd < D:
                d_positions.append(D - pd)
            if H > ph and h_positions[-1] + ph < H:
                h_positions.append(H - ph)
            if W > pw and w_positions[-1] + pw < W:
                w_positions.append(W - pw)

            # Tüm kombinasyonları ekle
            for d in d_positions:
                for h in h_positions:
                    for w in w_positions:
                        self.patch_positions.append({
                            'sample_id': sample_id,
                            'position': (d, h, w)
                        })

        print(f"Toplam {len(self.patch_positions)} patch hazırlandı")

    def _normalize_ct(self, ct_patch: np.ndarray) -> np.ndarray:
        """CT patch normalizasyonu"""
        if self.normalize_method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = ct_patch.min()
            max_val = ct_patch.max()
            if max_val > min_val:
                ct_patch = (ct_patch - min_val) / (max_val - min_val)
            else:
                ct_patch = ct_patch * 0  # Sabit değerli patch
        elif self.normalize_method == 'zscore':
            # Z-score normalization
            mean_val = ct_patch.mean()
            std_val = ct_patch.std()
            if std_val > 0:
                ct_patch = (ct_patch - mean_val) / std_val
            else:
                ct_patch = ct_patch - mean_val

        return ct_patch

    def _augment_patch(self, ct_patch: np.ndarray, mask_patch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Basit data augmentation"""
        if not self.augment:
            return ct_patch, mask_patch

        # Random flip X
        if random.random() > 0.5:
            ct_patch = np.flip(ct_patch, axis=2).copy()
            mask_patch = np.flip(mask_patch, axis=2).copy()

        # Random flip Y
        if random.random() > 0.5:
            ct_patch = np.flip(ct_patch, axis=1).copy()
            mask_patch = np.flip(mask_patch, axis=1).copy()

        # Random flip Z (dikkatli kullan)
        if random.random() > 0.5:
            ct_patch = np.flip(ct_patch, axis=0).copy()
            mask_patch = np.flip(mask_patch, axis=0).copy()

        # Hafif Gaussian noise (sadece CT'ye)
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.01, ct_patch.shape).astype(np.float32)
            ct_patch = ct_patch + noise

        return ct_patch, mask_patch

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Bir patch döndür"""
        patch_info = self.patch_positions[idx]
        sample_id = patch_info['sample_id']
        d, h, w = patch_info['position']
        pd, ph, pw = self.patch_size

        # Volume ve maske yükle
        volume, mask = self._load_volume(sample_id)

        # Patch'leri çıkar
        ct_patch = volume[d:d + pd, h:h + ph, w:w + pw].copy()
        mask_patch = mask[d:d + pd, h:h + ph, w:w + pw].copy()

        # Normalizasyon
        ct_patch = self._normalize_ct(ct_patch)

        # Binary maske olduğundan emin ol
        mask_patch = (mask_patch > 0.5).astype(np.float32)

        # Augmentation
        ct_patch, mask_patch = self._augment_patch(ct_patch, mask_patch)

        # Torch tensor'a çevir ve channel dimension ekle [1, D, H, W]
        ct_tensor = torch.from_numpy(ct_patch).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)

        return {
            'ct': ct_tensor,
            'mask': mask_tensor,
            'sample_id': sample_id,
            'position': torch.tensor([d, h, w])
        }


def create_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Config'ten train ve validation DataLoader'ları oluştur

    Args:
        config: Configuration dictionary

    Returns:
        train_loader, val_loader
    """
    # Train dataset
    train_dataset = VesuviusPatchDataset(
        data_root=config['data_root'],
        sample_ids=config['train_samples'],
        patch_size=tuple(config['patch_size']),
        patch_stride=tuple(config['patch_stride']),
        is_training=True,
        augment=config.get('augment', True),
        normalize_method=config.get('normalize_method', 'minmax'),
        cache_volumes=config.get('cache_volumes', True)
    )

    # Validation dataset
    val_dataset = VesuviusPatchDataset(
        data_root=config['data_root'],
        sample_ids=config['val_samples'],
        patch_size=tuple(config['patch_size']),
        patch_stride=tuple(config.get('val_patch_stride', config['patch_stride'])),
        is_training=False,
        augment=False,
        normalize_method=config.get('normalize_method', 'minmax'),
        cache_volumes=config.get('cache_volumes', True)
    )

    # DataLoader'lar
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if config.get('device', 'cpu') != 'cpu' else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('val_batch_size', config['batch_size']),
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if config.get('device', 'cpu') != 'cpu' else False,
        drop_last=False
    )

    print(f"Train dataset: {len(train_dataset)} patches")
    print(f"Val dataset: {len(val_dataset)} patches")

    return train_loader, val_loader