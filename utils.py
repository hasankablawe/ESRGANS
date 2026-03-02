"""
Utility functions for super resolution
"""
import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union
from pathlib import Path


def load_image(path: str) -> Image.Image:
    """Load image as RGB PIL Image"""
    return Image.open(path).convert('RGB')


def save_image(image: Union[torch.Tensor, np.ndarray, Image.Image], path: str):
    """Save image to file"""
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Create directory if needed
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    image.save(path)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to tensor [0, 1]"""
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    return tensor.unsqueeze(0)  # Add batch dimension


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp to valid range
    tensor = tensor.clamp(0, 1)
    
    # CHW -> HWC
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255).astype(np.uint8)
    
    return Image.fromarray(arr)


def get_device() -> torch.device:
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pad_to_multiple(
    image: torch.Tensor,
    multiple: int = 8
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Pad image to be a multiple of given size
    Required for some architectures
    
    Returns:
        Padded image and original size
    """
    _, _, h, w = image.shape
    
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    
    return image, (h, w)


def unpad(image: torch.Tensor, original_size: Tuple[int, int], scale: int = 1) -> torch.Tensor:
    """Remove padding from image"""
    h, w = original_size
    return image[:, :, :h * scale, :w * scale]


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def download_pretrained(model_name: str, save_dir: str = "./pretrained") -> str:
    """
    Download pretrained model weights
    
    Available models:
    - esrgan_x4: ESRGAN 4x upscaling
    - realesrgan_x4: Real-ESRGAN 4x upscaling
    - edsr_x4: EDSR 4x upscaling
    """
    import urllib.request
    
    urls = {
        "esrgan_x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        "realesrgan_x4": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        # We can add more specific variants here if needed
    }
    
    if model_name not in urls:
        # Check if it might be a close match or just pass if we want strict
        available = list(urls.keys())
        # Try to fuzzy match or just fail
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    
    if not os.path.exists(save_path):
        print(f"Downloading {model_name}...")
        try:
            urllib.request.urlretrieve(urls[model_name], save_path)
            print(f"Saved to {save_path}")
        except Exception as e:
            # Clean up partial file
            if os.path.exists(save_path):
                os.remove(save_path)
            raise e
    
    return save_path


def create_comparison_image(
    lr: Image.Image,
    sr: Image.Image,
    hr: Optional[Image.Image] = None
) -> Image.Image:
    """Create side-by-side comparison image"""
    # Resize LR to match SR size
    lr_upscaled = lr.resize(sr.size, Image.BICUBIC)
    
    images = [lr_upscaled, sr]
    labels = ["LR (Bicubic)", "Super Resolution"]
    
    if hr is not None:
        images.append(hr)
        labels.append("Ground Truth")
    
    # Create composite
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    
    total_width = sum(widths) + 10 * (len(images) - 1)  # 10px gap
    max_height = max(heights) + 30  # Space for labels
    
    composite = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    
    x_offset = 0
    for img, label in zip(images, labels):
        composite.paste(img, (x_offset, 30))
        x_offset += img.width + 10
    
    return composite
