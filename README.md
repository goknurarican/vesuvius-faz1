# Vesuvius Faz 1: 3D Surface Detection

## 📋 Proje Açıklaması

**Faz 1: Vesuvius 3D CT volume için temel 3D U-Net ile binary surface segmentation.**

Bu proje, Vesuvius Challenge benzeri 3D CT verilerinde papirüs/kağıt yüzeyini tespit etmek için geliştirilmiş bir deep learning pipeline'ıdır. Faz 1'de sadece binary segmentation yapıyoruz - CT volumünde yüzey olan voxelleri (1) ve arka planı (0) ayırıyoruz.

### 🎯 Hedefler
- 3D CT volumlerinden yüzey segmentasyonu
- Modüler ve genişletilebilir kod yapısı
- Hem lokal hem de Kaggle ortamında çalışabilme
- İleride teacher-student, affinity, graph network gibi gelişmiş tekniklere hazır altyapı

## 🏗️ Proje Yapısı

```
vesuvius_faz1/
├── src/
│   ├── dataset.py          # 3D patch-based dataset loader
│   ├── model_unet3d.py     # 3D U-Net model implementasyonu
│   ├── losses.py           # BCE + Dice loss fonksiyonları
│   ├── utils.py            # Metrik, logging ve yardımcı fonksiyonlar
│   └── train_faz1.py       # Ana training scripti
├── configs/
│   └── faz1_baseline.yaml  # Training konfigürasyonu
├── notebooks/              # Debug ve analiz için (opsiyonel)
├── requirements.txt        # Python bağımlılıkları
└── README.md              # Bu dosya
```

## 🚀 Hızlı Başlangıç

### Lokal Çalışma

1. **Repository'yi klonla:**
```bash
git clone https://github.com/yourusername/vesuvius_faz1.git
cd vesuvius_faz1
```

2. **Sanal ortam oluştur (önerilen):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

4. **Mini test verisi hazırla:**
```bash
# Örnek veri yapısı oluştur
mkdir -p data/vesuvius_mini/train/sample_1
mkdir -p data/vesuvius_mini/train/sample_2
mkdir -p data/vesuvius_mini/train/sample_3

# Test için sahte veri oluşturabilirsin (Python script ile)
python -c "
import numpy as np
import tifffile

# Sahte 3D volume oluştur
for i in range(1, 4):
    volume = np.random.randn(128, 256, 256).astype(np.float32)
    mask = (np.random.randn(128, 256, 256) > 0.5).astype(np.float32)
    
    tifffile.imwrite(f'data/vesuvius_mini/train/sample_{i}/ct.tif', volume)
    tifffile.imwrite(f'data/vesuvius_mini/train/sample_{i}/mask.tif', mask)
    
print('Test verisi oluşturuldu!')
"
```

5. **Config dosyasını düzenle:**
```bash
# configs/faz1_baseline.yaml dosyasında:
# data_root: "./data/vesuvius_mini/train"  # Lokal path
```

6. **Training başlat:**
```bash
python src/train_faz1.py --config configs/faz1_baseline.yaml
```

### Kaggle Üzerinde Çalışma

1. **GitHub'a yükle:**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Kaggle Notebook'ta:**

İlk hücre - Repository'yi klonla ve setup yap:
```python
# Repository'yi klonla
!git clone https://github.com/yourusername/vesuvius_faz1.git
%cd vesuvius_faz1

# Bağımlılıkları yükle
!pip install -q -r requirements.txt
```

İkinci hücre - Config'i Kaggle için güncelle:
```python
import yaml

# Config'i yükle
with open('configs/faz1_baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Kaggle path'lerini ayarla
config['data_root'] = '/kaggle/input/vesuvius-dataset/train'
config['output_dir'] = '/kaggle/working/outputs'
config['device'] = 'cuda'  # Kaggle GPU

# Güncellenmiş config'i kaydet
with open('configs/faz1_kaggle.yaml', 'w') as f:
    yaml.dump(config, f)

print("Config updated for Kaggle!")
```

Üçüncü hücre - Training başlat:
```python
# Training'i başlat
!python src/train_faz1.py --config configs/faz1_kaggle.yaml
```

## 📊 Veri Formatı

Beklenen veri yapısı:
```
data_root/
├── sample_1/
│   ├── ct.tif       # 3D CT volume [D, H, W]
│   └── mask.tif     # 3D binary mask [D, H, W]
├── sample_2/
│   ├── ct.tif
│   └── mask.tif
└── ...
```

- **ct.tif**: 3D CT volume, float32 format
- **mask.tif**: Binary segmentation mask (0=background, 1=surface)

## ⚙️ Konfigürasyon

Ana parametreler (`configs/faz1_baseline.yaml`):

### Data Ayarları
- `data_root`: Veri dizini
- `train_samples`: Training sample ID'leri
- `val_samples`: Validation sample ID'leri
- `patch_size`: 3D patch boyutu [D, H, W]
- `patch_stride`: Patch stride değerleri

### Model Ayarları
- `base_channels`: İlk katman kanal sayısı (16, 32, 64...)
- `num_levels`: U-Net derinliği (3, 4, 5...)
- `bilinear`: Upsampling metodu

### Training Ayarları
- `batch_size`: Batch boyutu (GPU belleğine göre)
- `epochs`: Epoch sayısı
- `learning_rate`: Öğrenme hızı
- `scheduler`: LR scheduler tipi ("reduce", "cosine", null)

### Loss Ayarları
- `bce_weight`: Binary Cross Entropy ağırlığı
- `dice_weight`: Dice loss ağırlığı

## 📈 Metrikler

Training sırasında takip edilen metrikler:
- **Loss**: BCE + Dice combined loss
- **Dice Score**: Overlap metriği (0-1, 1=perfect)
- **IoU**: Intersection over Union
- **F1 Score**: Precision ve Recall dengesi

Metrikler `outputs/` klasöründe CSV formatında kaydedilir.

## 💾 Checkpoint Sistemi

Model checkpoint'leri şu şekilde kaydedilir:
- `checkpoint_epoch_XXX.pth`: Her epoch checkpoint'i
- `checkpoint_epoch_XXX_best.pth`: En iyi model
- `last_checkpoint.pth`: Son epoch

Checkpoint yükleme:
```yaml
resume_checkpoint: "./outputs/checkpoints/last_checkpoint.pth"
```

## 🔧 Gelişmiş Özellikler

### Custom Loss Kombinasyonları
```yaml
loss_config:
  bce:
    weight: 0.3
    pos_weight: 2.0  # Class imbalance için
  dice:
    weight: 0.5
  focal:
    weight: 0.2
    alpha: 0.25
    gamma: 2.0
```

### Data Augmentation
- Random flips (X, Y, Z axes)
- Gaussian noise
- Config'ten kontrol edilebilir

### Memory Optimization
- `cache_volumes`: False yaparak RAM kullanımını azalt
- `batch_size`: GPU belleğine göre ayarla
- Gradient accumulation (gelecek sürüm)

## 🐛 Debug ve Test

Model test:
```python
python src/model_unet3d.py  # Model yapısını test et
```

Dataset test:
```python
python -c "
from src.dataset import VesuviusPatchDataset
import yaml

with open('configs/faz1_baseline.yaml') as f:
    config = yaml.safe_load(f)

dataset = VesuviusPatchDataset(
    config['data_root'],
    config['train_samples'],
    tuple(config['patch_size']),
    tuple(config['patch_stride'])
)

print(f'Dataset size: {len(dataset)}')
sample = dataset[0]
print(f'CT shape: {sample[\"ct\"].shape}')
print(f'Mask shape: {sample[\"mask\"].shape}')
"
```

## 📝 Notlar

### GPU Bellek Optimizasyonu
- Batch size = 2 için ~8GB GPU belleği gerekir
- Patch size küçültülerek bellek kullanımı azaltılabilir
- Mixed precision training eklenebilir (gelecek sürüm)

### Performans İpuçları
- `cache_volumes=True`: Hızlı ama RAM kullanır
- `num_workers`: CPU sayısına göre ayarla
- SSD disk kullanımı önerilir

### Gelecek Geliştirmeler (Faz 2+)
- [ ] Teacher-student learning
- [ ] Affinity field prediction
- [ ] Graph neural network integration
- [ ] Multi-scale training
- [ ] Advanced augmentations
- [ ] TensorBoard integration
- [ ] Mixed precision training
- [ ] Distributed training

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altındadır. Detaylar için LICENSE dosyasına bakın.

## 🙏 Teşekkürler

- Vesuvius Challenge organizatörleri
- PyTorch ekibi
- Kaggle community

## 📧 İletişim

Sorular için issue açabilir veya [email@example.com] adresinden iletişime geçebilirsiniz.

---

**Not**: Bu Faz 1 implementasyonu temel bir baseline sağlar. Gerçek Vesuvius verisi üzerinde fine-tuning gerekebilir.