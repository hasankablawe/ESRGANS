"""
Real-ESRGAN compatible architecture
Matches the official Real-ESRGAN weight structure exactly
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block - matches Real-ESRGAN structure"""
    
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block - matches Real-ESRGAN"""
    
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RealESRGANer(nn.Module):
    """
    Real-ESRGAN Generator - EXACTLY matches official weight structure
    
    This is compatible with weights from:
    - RealESRGAN_x4plus.pth
    - RealESRGAN_x4plus_anime_6B.pth (with num_block=6)
    """
    
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3, 
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32
    ):
        super(RealESRGANer, self).__init__()
        self.scale = scale
        
        # First conv
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Body - RRDB blocks
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch))
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling - uses NAMED layers (conv_up1, conv_up2) not Sequential!
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # For 8x scale
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # Upsample 2x
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # Upsample 2x again (total 4x)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        # For 8x
        if self.scale == 8:
            feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_realesrgan(model_path: str, scale: int = 4, device='cuda'):
    """
    Load Real-ESRGAN model with proper architecture
    
    Args:
        model_path: Path to .pth weights
        scale: 4 or 8
        device: torch device
        
    Returns:
        Loaded model ready for inference
    """
    # Check if it's the anime model (6 blocks) or regular (23 blocks)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle different checkpoint formats
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    
    # Detect number of blocks from weights
    num_block = 0
    for key in state_dict.keys():
        if key.startswith('body.') and '.rdb' in key:
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                idx = int(parts[1])
                num_block = max(num_block, idx + 1)
    
    # Default to 23 if detection failed
    if num_block == 0:
        num_block = 23
    
    print(f"Detected {num_block} RRDB blocks")
    
    # Create model
    model = RealESRGANer(
        num_in_ch=3,
        num_out_ch=3,
        scale=scale,
        num_feat=64,
        num_block=num_block,
        num_grow_ch=32
    )
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        model = load_realesrgan(model_path)
        print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test
        x = torch.randn(1, 3, 64, 64).cuda()
        with torch.no_grad():
            y = model(x)
        print(f"Test: {x.shape} -> {y.shape}")
    else:
        # Quick test
        model = RealESRGANer(num_block=23)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        x = torch.randn(1, 3, 64, 64)
        y = model(x)
        print(f"Input: {x.shape} -> Output: {y.shape}")
