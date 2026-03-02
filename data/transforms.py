"""
Data augmentation transforms for super resolution
"""
import random
from typing import Tuple
from PIL import Image


class PairedTransform:
    """Base class for transforms that operate on paired LR/HR images"""
    
    def __call__(self, hr_img: Image.Image, lr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        raise NotImplementedError


class RandomHorizontalFlip(PairedTransform):
    """Randomly flip images horizontally"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, hr_img: Image.Image, lr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return hr_img.transpose(Image.FLIP_LEFT_RIGHT), lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        return hr_img, lr_img


class RandomVerticalFlip(PairedTransform):
    """Randomly flip images vertically"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, hr_img: Image.Image, lr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            return hr_img.transpose(Image.FLIP_TOP_BOTTOM), lr_img.transpose(Image.FLIP_TOP_BOTTOM)
        return hr_img, lr_img


class RandomRotation90(PairedTransform):
    """Randomly rotate images by 90, 180, or 270 degrees"""
    
    def __call__(self, hr_img: Image.Image, lr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return hr_img, lr_img
        elif angle == 90:
            return hr_img.transpose(Image.ROTATE_90), lr_img.transpose(Image.ROTATE_90)
        elif angle == 180:
            return hr_img.transpose(Image.ROTATE_180), lr_img.transpose(Image.ROTATE_180)
        else:
            return hr_img.transpose(Image.ROTATE_270), lr_img.transpose(Image.ROTATE_270)


class Compose(PairedTransform):
    """Compose multiple transforms"""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, hr_img: Image.Image, lr_img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            hr_img, lr_img = t(hr_img, lr_img)
        return hr_img, lr_img


def get_train_transforms() -> Compose:
    """Get transforms for training"""
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
    ])


def get_val_transforms() -> Compose:
    """Get transforms for validation (no augmentation)"""
    return Compose([])
