"""
Adversarial (GAN) losses for super resolution
"""
import torch
import torch.nn as nn
from torch import autograd
from typing import Optional


class GANLoss(nn.Module):
    """
    GAN loss supporting various GAN types
    
    Supports:
    - vanilla: Binary cross-entropy
    - lsgan: Least squares GAN
    - wgan: Wasserstein GAN
    - wgan-gp: Wasserstein GAN with gradient penalty
    - hinge: Hinge loss
    """
    
    def __init__(
        self,
        gan_type: str = "vanilla",
        real_label: float = 1.0,
        fake_label: float = 0.0,
        loss_weight: float = 1.0
    ):
        super(GANLoss, self).__init__()
        
        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label
        self.loss_weight = loss_weight
        
        if gan_type == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_type == "lsgan":
            self.criterion = nn.MSELoss()
        elif gan_type in ["wgan", "wgan-gp"]:
            self.criterion = None  # Computed differently
        elif gan_type == "hinge":
            self.criterion = None
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
    
    def _get_target_tensor(
        self,
        pred: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """Create target tensor with same shape as prediction"""
        if target_is_real:
            return torch.full_like(pred, self.real_label)
        return torch.full_like(pred, self.fake_label)
    
    def forward(
        self,
        pred: torch.Tensor,
        target_is_real: bool,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Calculate GAN loss
        
        Args:
            pred: Discriminator output
            target_is_real: Whether the target is real
            for_discriminator: Whether this is for discriminator training
            
        Returns:
            GAN loss value
        """
        if self.gan_type in ["vanilla", "lsgan"]:
            target = self._get_target_tensor(pred, target_is_real)
            loss = self.criterion(pred, target)
            
        elif self.gan_type in ["wgan", "wgan-gp"]:
            if target_is_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()
                
        elif self.gan_type == "hinge":
            if for_discriminator:
                if target_is_real:
                    loss = nn.functional.relu(1 - pred).mean()
                else:
                    loss = nn.functional.relu(1 + pred).mean()
            else:
                loss = -pred.mean()
        else:
            raise NotImplementedError(f"GAN type {self.gan_type} not implemented")
        
        return loss * self.loss_weight


def gradient_penalty(
    discriminator: nn.Module,
    real_data: torch.Tensor,
    fake_data: torch.Tensor,
    penalty_weight: float = 10.0
) -> torch.Tensor:
    """
    Compute gradient penalty for WGAN-GP
    
    Args:
        discriminator: Discriminator network
        real_data: Real images
        fake_data: Generated images
        penalty_weight: Weight for gradient penalty
        
    Returns:
        Gradient penalty loss
    """
    batch_size = real_data.size(0)
    device = real_data.device
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Discriminator output for interpolated data
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = penalty_weight * ((gradient_norm - 1) ** 2).mean()
    
    return penalty


class RelativisticGANLoss(nn.Module):
    """
    Relativistic GAN loss as used in ESRGAN
    
    The discriminator estimates probability that real data
    is more realistic than fake data
    """
    
    def __init__(self, loss_weight: float = 1.0):
        super(RelativisticGANLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.loss_weight = loss_weight
    
    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Calculate relativistic GAN loss
        
        Args:
            real_pred: Discriminator output for real images
            fake_pred: Discriminator output for fake images
            for_discriminator: Whether this is for discriminator
            
        Returns:
            Loss value
        """
        real_ones = torch.ones_like(real_pred)
        fake_zeros = torch.zeros_like(fake_pred)
        
        if for_discriminator:
            # D should recognize real > fake
            loss_real = self.criterion(real_pred - fake_pred.mean(), real_ones)
            loss_fake = self.criterion(fake_pred - real_pred.mean(), fake_zeros)
            loss = (loss_real + loss_fake) / 2
        else:
            # G should make fake > real
            loss_real = self.criterion(real_pred - fake_pred.mean(), fake_zeros)
            loss_fake = self.criterion(fake_pred - real_pred.mean(), real_ones)
            loss = (loss_real + loss_fake) / 2
        
        return loss * self.loss_weight


if __name__ == "__main__":
    # Test GAN losses
    gan_loss = GANLoss(gan_type="vanilla")
    
    pred = torch.randn(4, 1)
    loss_real = gan_loss(pred, target_is_real=True)
    loss_fake = gan_loss(pred, target_is_real=False)
    
    print(f"Vanilla GAN - Real loss: {loss_real.item():.4f}, Fake loss: {loss_fake.item():.4f}")
    
    # Test relativistic loss
    rel_loss = RelativisticGANLoss()
    real_pred = torch.randn(4, 1)
    fake_pred = torch.randn(4, 1)
    
    d_loss = rel_loss(real_pred, fake_pred, for_discriminator=True)
    g_loss = rel_loss(real_pred, fake_pred, for_discriminator=False)
    
    print(f"Relativistic GAN - D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}")
