"""
SRCNN - Super-Resolution Convolutional Neural Network
A lightweight baseline model for super resolution.
"""
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """
    Super-Resolution Convolutional Neural Network
    
    Paper: "Image Super-Resolution Using Deep Convolutional Networks"
    https://arxiv.org/abs/1501.00092
    
    This is a lightweight 3-layer CNN that performs:
    1. Patch extraction and representation
    2. Non-linear mapping
    3. Reconstruction
    """
    
    def __init__(self, num_channels: int = 3, scale_factor: int = 4):
        super(SRCNN, self).__init__()
        self.scale_factor = scale_factor
        
        # Upscale input first using bicubic interpolation
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False
        )
        
        # Feature extraction layer
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Non-linear mapping
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Reconstruction layer
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Low resolution input tensor [B, C, H, W]
            
        Returns:
            Super resolved output tensor [B, C, H*scale, W*scale]
        """
        x = self.upsample(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


class SRCNNLarge(nn.Module):
    """
    Larger version of SRCNN with more layers for better quality
    """
    
    def __init__(self, num_channels: int = 3, scale_factor: int = 4, num_features: int = 64):
        super(SRCNNLarge, self).__init__()
        self.scale_factor = scale_factor
        
        self.upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bicubic',
            align_corners=False
        )
        
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, num_features, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 2, num_channels, kernel_size=5, padding=2),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.features(x)


if __name__ == "__main__":
    # Test the model
    model = SRCNN(scale_factor=4)
    print(f"SRCNN Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")
