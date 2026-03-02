"""
ESRGAN - Enhanced Super-Resolution Generative Adversarial Network
High quality super resolution with perceptual/adversarial training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for RRDB
    Uses dense connections within the block
    """
    
    def __init__(self, num_features: int = 64, growth_channels: int = 32):
        super(ResidualDenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_channels, num_features, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.beta = 0.2  # Residual scaling
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in [self.conv1, self.conv2, self.conv3, self.conv4]:
            nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Last conv initialized to zero for residual
        nn.init.zeros_(self.conv5.weight)
        if self.conv5.bias is not None:
            nn.init.zeros_(self.conv5.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * self.beta + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    The core building block of ESRGAN
    """
    
    def __init__(self, num_features: int = 64, growth_channels: int = 32):
        super(RRDB, self).__init__()
        
        self.rdb1 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb2 = ResidualDenseBlock(num_features, growth_channels)
        self.rdb3 = ResidualDenseBlock(num_features, growth_channels)
        self.beta = 0.2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.beta + x


class RRDBNet(nn.Module):
    """
    RRDB Network - Generator for ESRGAN
    
    Paper: "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
    https://arxiv.org/abs/1809.00219
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        growth_channels: int = 32,
        scale_factor: int = 4
    ):
        super(RRDBNet, self).__init__()
        self.scale_factor = scale_factor
        
        # First convolution
        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.Sequential(*[
            RRDB(num_features, growth_channels) for _ in range(num_blocks)
        ])
        
        # Conv after RRDB blocks
        self.conv_body = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Upsampling layers
        self.upsampler = self._make_upsampler(num_features, scale_factor)
        
        # Final reconstruction
        self.conv_hr = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self._initialize_weights()
    
    def _make_upsampler(self, num_features: int, scale_factor: int) -> nn.Sequential:
        """Create upsampling layers based on scale factor"""
        layers = []
        if scale_factor in [2, 4, 8]:
            for _ in range(int(torch.log2(torch.tensor(scale_factor)).item())):
                layers.append(nn.Conv2d(num_features, num_features * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif scale_factor == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsampling
        feat = self.upsampler(feat)
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class ESRGAN(nn.Module):
    """
    Complete ESRGAN with Generator and optional Discriminator
    """
    
    def __init__(
        self,
        scale_factor: int = 4,
        num_features: int = 64,
        num_blocks: int = 23,
        use_discriminator: bool = False
    ):
        super(ESRGAN, self).__init__()
        
        self.generator = RRDBNet(
            in_channels=3,
            out_channels=3,
            num_features=num_features,
            num_blocks=num_blocks,
            scale_factor=scale_factor
        )
        
        self.discriminator = None
        if use_discriminator:
            self.discriminator = Discriminator()
        
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)
    
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate super-resolved image"""
        return self.generator(x)


class Discriminator(nn.Module):
    """
    VGG-style discriminator for ESRGAN
    """
    
    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super(Discriminator, self).__init__()
        
        def conv_block(in_ch, out_ch, stride=1, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, stride, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        
        self.features = nn.Sequential(
            conv_block(in_channels, num_features, bn=False),
            conv_block(num_features, num_features, stride=2),
            conv_block(num_features, num_features * 2),
            conv_block(num_features * 2, num_features * 2, stride=2),
            conv_block(num_features * 2, num_features * 4),
            conv_block(num_features * 4, num_features * 4, stride=2),
            conv_block(num_features * 4, num_features * 8),
            conv_block(num_features * 8, num_features * 8, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


if __name__ == "__main__":
    # Test the models
    generator = RRDBNet(scale_factor=4, num_blocks=6)
    print(f"RRDBNet Parameters: {sum(p.numel() for p in generator.parameters()):,}")
    
    x = torch.randn(1, 3, 64, 64)
    y = generator(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
    
    # Test full ESRGAN
    esrgan = ESRGAN(scale_factor=4, num_blocks=6, use_discriminator=True)
    print(f"ESRGAN Generator Parameters: {sum(p.numel() for p in esrgan.generator.parameters()):,}")
    print(f"ESRGAN Discriminator Parameters: {sum(p.numel() for p in esrgan.discriminator.parameters()):,}")
