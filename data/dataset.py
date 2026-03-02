"""
Dataset classes for Super Resolution training
"""
import os
import random
from pathlib import Path
from typing import Tuple, Optional, Callable, List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


class SuperResolutionDataset(Dataset):
    """
    Dataset for training super resolution models
    
    Supports two modes:
    1. Paired mode: Separate LR and HR directories
    2. Single mode: HR images only (LR generated on-the-fly)
    """
    
    def __init__(
        self,
        hr_dir: str,
        lr_dir: Optional[str] = None,
        scale_factor: int = 4,
        patch_size: int = 96,
        transform: Optional[Callable] = None,
        mode: str = "train"
    ):
        """
        Args:
            hr_dir: Directory containing high-resolution images
            lr_dir: Directory containing low-resolution images (optional)
            scale_factor: Downscaling factor for generating LR images
            patch_size: Size of HR patches to extract during training
            transform: Optional transforms to apply
            mode: 'train' or 'val'
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.transform = transform
        self.mode = mode
        
        # Supported image formats
        self.extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        # Get image list
        self.hr_images = self._get_image_list(self.hr_dir)
        
        if len(self.hr_images) == 0:
            raise ValueError(f"No images found in {hr_dir}")
        
        print(f"Found {len(self.hr_images)} images in {hr_dir}")
    
    def _get_image_list(self, directory: Path) -> List[Path]:
        """Get list of image files in directory"""
        images = []
        for ext in self.extensions:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))
        return sorted(images)
    
    def _load_image(self, path: Path) -> Image.Image:
        """Load image as RGB"""
        return Image.open(path).convert('RGB')
    
    def _generate_lr(self, hr_img: Image.Image) -> Image.Image:
        """Generate LR image by downscaling"""
        w, h = hr_img.size
        lr_w, lr_h = w // self.scale_factor, h // self.scale_factor
        return hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    
    def _get_patch(
        self, 
        hr_img: Image.Image, 
        lr_img: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Extract random patch pair for training"""
        hr_w, hr_h = hr_img.size
        lr_patch_size = self.patch_size // self.scale_factor
        
        # Random position
        lr_x = random.randint(0, max(0, hr_w // self.scale_factor - lr_patch_size))
        lr_y = random.randint(0, max(0, hr_h // self.scale_factor - lr_patch_size))
        
        hr_x, hr_y = lr_x * self.scale_factor, lr_y * self.scale_factor
        
        lr_patch = lr_img.crop((lr_x, lr_y, lr_x + lr_patch_size, lr_y + lr_patch_size))
        hr_patch = hr_img.crop((hr_x, hr_y, hr_x + self.patch_size, hr_y + self.patch_size))
        
        return hr_patch, lr_patch
    
    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to tensor [0, 1]"""
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        return tensor
    
    def __len__(self) -> int:
        return len(self.hr_images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load HR image
        hr_path = self.hr_images[idx]
        hr_img = self._load_image(hr_path)
        
        # Load or generate LR image
        if self.lr_dir:
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                lr_img = self._load_image(lr_path)
            else:
                lr_img = self._generate_lr(hr_img)
        else:
            lr_img = self._generate_lr(hr_img)
        
        # Extract patches during training
        if self.mode == "train":
            hr_img, lr_img = self._get_patch(hr_img, lr_img)
        
        # Apply transforms if provided
        if self.transform:
            hr_img, lr_img = self.transform(hr_img, lr_img)
        
        # Convert to tensors
        hr_tensor = self._to_tensor(hr_img)
        lr_tensor = self._to_tensor(lr_img)
        
        return lr_tensor, hr_tensor


class ImageFolderDataset(Dataset):
    """
    Simple dataset for inference - loads all images from a folder
    """
    
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        self.extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        self.images = []
        for ext in self.extensions:
            self.images.extend(self.image_dir.glob(f"*{ext}"))
            self.images.extend(self.image_dir.glob(f"*{ext.upper()}"))
        self.images = sorted(self.images)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
        return tensor, img_path.name


def create_dataloader(
    hr_dir: str,
    lr_dir: Optional[str] = None,
    batch_size: int = 16,
    scale_factor: int = 4,
    patch_size: int = 96,
    num_workers: int = 4,
    mode: str = "train"
) -> DataLoader:
    """
    Create a dataloader for super resolution training
    
    Args:
        hr_dir: Path to high-resolution images
        lr_dir: Path to low-resolution images (optional)
        batch_size: Batch size
        scale_factor: Upscaling factor
        patch_size: Size of patches to extract
        num_workers: Number of data loading workers
        mode: 'train' or 'val'
    
    Returns:
        DataLoader instance
    """
    from .transforms import get_train_transforms, get_val_transforms
    
    transform = get_train_transforms() if mode == "train" else get_val_transforms()
    
    dataset = SuperResolutionDataset(
        hr_dir=hr_dir,
        lr_dir=lr_dir,
        scale_factor=scale_factor,
        patch_size=patch_size,
        transform=transform,
        mode=mode
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == "train")
    )
