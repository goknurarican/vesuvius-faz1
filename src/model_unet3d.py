"""
Vesuvius Faz 1 - 3D U-Net Model
Binary segmentation için 3D U-Net implementasyonu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DoubleConv3D(nn.Module):
    """Temel yapı taşı: Conv3D + BatchNorm + ReLU x2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling bloğu: MaxPool + DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, use_maxpool: bool = True):
        super().__init__()
        if use_maxpool:
            self.down = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv3D(in_channels, out_channels)
            )
        else:
            # Stride=2 convolution ile downsampling
            self.down = nn.Sequential(
                nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm3d(in_channels),
                nn.ReLU(inplace=True),
                DoubleConv3D(in_channels, out_channels)
            )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """Upsampling bloğu: ConvTranspose veya Upsample + DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            # Bilinear upsampling kullan
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            # Transposed convolution kullan
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Skip connection için boyut uyumsuzluklarını gider
        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])

        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net model implementasyonu

    Extensible yapı: İleride yeni head'ler eklenebilir
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            base_channels: int = 16,
            num_levels: int = 4,
            bilinear: bool = False,
            use_maxpool: bool = True
    ):
        """
        Args:
            in_channels: Giriş kanal sayısı (CT için genelde 1)
            out_channels: Çıkış kanal sayısı (binary mask için 1)
            base_channels: İlk katmandaki kanal sayısı
            num_levels: U-Net derinliği (downsampling sayısı)
            bilinear: Upsampling için bilinear interpolation kullan
            use_maxpool: Downsampling için MaxPool kullan (False ise strided conv)
        """
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = out_channels
        self.bilinear = bilinear

        # Kanal sayıları hesapla
        channels = [base_channels * (2 ** i) for i in range(num_levels + 1)]
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = DoubleConv3D(in_channels, channels[0])
        self.encoder = nn.ModuleList()

        for i in range(num_levels):
            self.encoder.append(
                Down(channels[i], channels[i + 1], use_maxpool=use_maxpool)
            )

        # Decoder path
        self.decoder = nn.ModuleList()

        for i in range(num_levels, 0, -1):
            in_ch = channels[i]
            out_ch = channels[i - 1] // factor if bilinear and i == 1 else channels[i - 1]
            self.decoder.append(
                Up(in_ch, out_ch, bilinear=bilinear)
            )

        # Output head (raw logits, sigmoid loss'ta uygulanacak)
        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)

        # İleride eklenebilecek ek head'ler için dict
        self.additional_heads = nn.ModuleDict()

    def add_head(self, name: str, head_module: nn.Module):
        """İleride yeni output head eklemek için"""
        self.additional_heads[name] = head_module

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor [B, 1, D, H, W]

        Returns:
            dict: {'mask': logits tensor [B, 1, D, H, W]}
                  İleride ek çıkışlar eklenebilir
        """
        # Encoder
        x_enc = [self.inc(x)]

        for encoder_block in self.encoder:
            x_enc.append(encoder_block(x_enc[-1]))

        # Decoder with skip connections
        x_dec = x_enc[-1]

        for i, decoder_block in enumerate(self.decoder):
            skip_idx = -(i + 2)  # İlgili encoder çıkışının indeksi
            x_dec = decoder_block(x_dec, x_enc[skip_idx])

        # Output
        logits = self.outc(x_dec)

        outputs = {'mask': logits}

        # Ek head'ler varsa onları da çalıştır
        for head_name, head_module in self.additional_heads.items():
            outputs[head_name] = head_module(x_dec)

        return outputs


class LightweightUNet3D(nn.Module):
    """
    Daha hafif bir 3D U-Net varyantı (opsiyonel)
    Daha az parametre, daha hızlı training
    """

    def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            features: List[int] = [16, 32, 64, 128]
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConv3D(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv3D(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose3d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(DoubleConv3D(feature * 2, feature))

        # Final conv
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            # Boyut uyumunu sağla
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        logits = self.final_conv(x)

        return {'mask': logits}


def test_model():
    """Model test fonksiyonu"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 64).to(device)

    # Model oluştur
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        num_levels=3
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output['mask'].shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Lightweight model test
    light_model = LightweightUNet3D(features=[8, 16, 32]).to(device)
    with torch.no_grad():
        light_output = light_model(input_tensor)

    print(f"\nLightweight model parameters: {sum(p.numel() for p in light_model.parameters()):,}")


if __name__ == "__main__":
    test_model()