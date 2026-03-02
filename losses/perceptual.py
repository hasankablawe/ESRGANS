"""
Perceptual Loss using VGG features
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights
from typing import List, Optional


class VGGFeatureExtractor(nn.Module):
    """
    Extract features from VGG19 for perceptual loss
    
    Uses features before activation (as in ESRGAN paper)
    """
    
    def __init__(
        self,
        feature_layers: List[int] = [2, 7, 16, 25, 34],
        use_input_norm: bool = True,
        requires_grad: bool = False
    ):
        """
        Args:
            feature_layers: Layer indices to extract features from
            use_input_norm: Normalize input to ImageNet stats
            requires_grad: Whether to compute gradients for VGG
        """
        super(VGGFeatureExtractor, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        
        self.feature_layers = feature_layers
        self.use_input_norm = use_input_norm
        
        # Create sequential modules up to each feature layer
        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer_idx in sorted(feature_layers):
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:layer_idx + 1]))
            prev_idx = layer_idx + 1
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract VGG features
        
        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            List of feature tensors from each layer
        """
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        
        features = []
        for slice_module in self.slices:
            x = slice_module(x)
            features.append(x)
        
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    
    Combines:
    - L1/L2 pixel loss
    - VGG feature matching loss
    """
    
    def __init__(
        self,
        feature_layers: List[int] = [34],  # conv5_4 before activation
        feature_weights: Optional[List[float]] = None,
        pixel_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        criterion: str = "l1"
    ):
        """
        Args:
            feature_layers: VGG layers to extract features from
            feature_weights: Weights for each feature layer
            pixel_weight: Weight for pixel loss
            perceptual_weight: Weight for perceptual loss
            criterion: 'l1' or 'l2'
        """
        super(PerceptualLoss, self).__init__()
        
        self.feature_extractor = VGGFeatureExtractor(feature_layers)
        self.feature_weights = feature_weights or [1.0] * len(feature_layers)
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        
        if criterion == "l1":
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.MSELoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> tuple:
        """
        Calculate perceptual loss
        
        Args:
            pred: Predicted SR image [B, 3, H, W]
            target: Ground truth HR image [B, 3, H, W]
            
        Returns:
            Tuple of (total_loss, pixel_loss, perceptual_loss)
        """
        # Pixel loss
        pixel_loss = self.criterion(pred, target)
        
        # Perceptual loss
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        
        perceptual_loss = 0
        for i, (pred_feat, target_feat) in enumerate(zip(pred_features, target_features)):
            perceptual_loss += self.feature_weights[i] * self.criterion(pred_feat, target_feat)
        
        # Total loss
        total_loss = self.pixel_weight * pixel_loss + self.perceptual_weight * perceptual_loss
        
        return total_loss, pixel_loss, perceptual_loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1)
    Better for training stability than L1
    """
    
    def __init__(self, eps: float = 1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


if __name__ == "__main__":
    # Test perceptual loss
    loss_fn = PerceptualLoss()
    
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    
    total_loss, pixel_loss, perceptual_loss = loss_fn(pred, target)
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Pixel Loss: {pixel_loss.item():.4f}")
    print(f"Perceptual Loss: {perceptual_loss.item():.4f}")
