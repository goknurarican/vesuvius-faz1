"""
Kaggle Notebook Setup Script
Bu scripti Kaggle notebook'ta çalıştır
"""

import os
import sys
import subprocess
import shutil


def setup_kaggle_environment():
    """Kaggle ortamını hazırla"""

    print("=" * 60)
    print("Vesuvius Faz 1 - Kaggle Setup")
    print("=" * 60)

    # 1. Çalışma dizinine geç
    working_dir = "/kaggle/working"
    os.chdir(working_dir)
    print(f"✓ Working directory: {os.getcwd()}")

    # 2. Eğer vesuvius-faz1 klasörü varsa sil (temiz başlangıç)
    if os.path.exists("vesuvius-faz1"):
        shutil.rmtree("vesuvius-faz1")
        print("✓ Cleaned existing directory")

    # 3. GitHub'dan clone et (senin repo URL'ni kullan)
    print("\n📥 Cloning repository...")
    repo_url = "https://github.com/goknurarican/vesuvius-faz1.git"  # URL'yi değiştir!
    result = subprocess.run(
        ["git", "clone", repo_url],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Git clone failed: {result.stderr}")
        return False

    print("✓ Repository cloned successfully")

    # 4. Proje dizinine geç
    os.chdir("vesuvius-faz1")
    print(f"✓ Changed to project directory: {os.getcwd()}")

    # 5. Dosyaları kontrol et
    print("\n📁 Project files:")
    for root, dirs, files in os.walk(".", topdown=True):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        level = root.replace(".", "", 1).count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            if not file.startswith('.'):
                print(f"{subindent}{file}")

    # 6. Requirements yükle
    print("\n📦 Installing requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"⚠️ Some packages failed to install: {result.stderr}")
    else:
        print("✓ Requirements installed")

    # 7. Config'i Kaggle için güncelle
    print("\n⚙️ Updating config for Kaggle...")

    import yaml

    config_path = "configs/faz1_baseline.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Kaggle path'lerini güncelle
        config['data_root'] = '/kaggle/input/vesuvius-challenge-surface-detection'
        config['output_dir'] = '/kaggle/working/outputs'
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['num_workers'] = 2  # Kaggle için optimize
        config['batch_size'] = 2  # GPU belleğine göre ayarla

        # Güncellenmiş config'i kaydet
        kaggle_config_path = "configs/faz1_kaggle.yaml"
        with open(kaggle_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✓ Config updated and saved to {kaggle_config_path}")
    else:
        print("❌ Config file not found!")
        return False

    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)

    return True


def test_imports():
    """Import'ları test et"""
    print("\n🔍 Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        import tifffile
        print("✓ tifffile imported")
    except ImportError as e:
        print(f"❌ tifffile import failed: {e}")
        return False

    try:
        import yaml
        print("✓ yaml imported")
    except ImportError as e:
        print(f"❌ yaml import failed: {e}")
        return False

    try:
        # Test local modules
        sys.path.insert(0, 'src')
        from dataset import VesuviusPatchDataset
        from model_unet3d import UNet3D
        from losses import BCEDiceLoss
        from utils import set_seed
        print("✓ All local modules imported successfully")
    except ImportError as e:
        print(f"❌ Local module import failed: {e}")
        return False

    return True


if __name__ == "__main__":
    import torch  # Test için

    # Setup çalıştır
    if setup_kaggle_environment():
        # Import'ları test et
        if test_imports():
            print("\n🚀 Ready to start training!")
            print("Run: python src/train_faz1.py --config configs/faz1_kaggle.yaml")
        else:
            print("\n⚠️ Some imports failed. Check the errors above.")
    else:
        print("\n❌ Setup failed. Check the errors above.")