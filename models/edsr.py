"""
EDSR - Enhanced Deep Super-Resolution Network
A residual network optimized for super resolution.
"""
import torch
import torch.nn as nn
import math


class ResBlock(nn.Module):
    """
    Residual block without batch normalization
    EDSR removes BN to achieve better performance
    """
    
    def __init__(self, num_features: int, res_scale: float = 0.1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.res_scale = res_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.res_scale


class Upsampler(nn.Sequential):
    """
    Upsampler using pixel shuffle (sub-pixel convolution)
    """
    
    def __init__(self, scale_factor: int, num_features: int):
        layers = []
        
        if scale_factor in [2, 4, 8]:
            for _ in range(int(math.log2(scale_factor))):
                layers.append(nn.Conv2d(num_features, num_features * 4, 3, 1, 1))
                layers.append(nn.PixelShuffle(2))
        elif scale_factor == 3:
            layers.append(nn.Conv2d(num_features, num_features * 9, 3, 1, 1))
            layers.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale factor: {scale_factor}")
        
        super(Upsampler, self).__init__(*layers)


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution Network
    
    Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution"
    https://arxiv.org/abs/1707.02921
    
    Key features:
    - No batch normalization (for better SR performance)
    - Residual scaling (0.1) for training stability
    - Pixel shuffle upsampling
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 16,
        scale_factor: int = 4,
        res_scale: float = 0.1
    ):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        
        # Head
        self.head = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        # Body - residual blocks
        self.body = nn.Sequential(*[
            ResBlock(num_features, res_scale) for _ in range(num_blocks)
        ])
        self.body_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        # Tail - upsampler
        self.upsampler = Upsampler(scale_factor, num_features)
        self.tail = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Head
        head = self.head(x)
        
        # Body with global residual
        body = self.body_conv(self.body(head))
        res = head + body
        
        # Upsampling and reconstruction
        out = self.tail(self.upsampler(res))
        return out


class EDSRLite(nn.Module):
    """
    Lightweight EDSR variant for faster inference
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 32,
        num_blocks: int = 8,
        scale_factor: int = 4
    ):
        super(EDSRLite, self).__init__()
        
        self.head = nn.Conv2d(in_channels, num_features, 3, 1, 1)
        
        self.body = nn.Sequential(*[
            ResBlock(num_features, res_scale=1.0) for _ in range(num_blocks)
        ])
        self.body_conv = nn.Conv2d(num_features, num_features, 3, 1, 1)
        
        self.upsampler = Upsampler(scale_factor, num_features)
        self.tail = nn.Conv2d(num_features, out_channels, 3, 1, 1)
        self.scale_factor = scale_factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head = self.head(x)
        body = self.body_conv(self.body(head))
        res = head + body
        out = self.tail(self.upsampler(res))
        return out


if __name__ == "__main__":
    # Test the models
    model = EDSR(scale_factor=4, num_blocks=16)
    print(f"EDSR Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
    
    # Test lite version
    model_lite = EDSRLite(scale_factor=4, num_blocks=8)
    print(f"EDSR Lite Parameters: {sum(p.numel() for p in model_lite.parameters()):,}")
