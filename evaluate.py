"""
Evaluation metrics for Super Resolution
PSNR and SSIM calculation
"""
import torch
import numpy as np
from typing import Union, Optional
from PIL import Image
import math


def calculate_psnr(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    max_val: float = 1.0,
    crop_border: int = 0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        img1: First image (prediction)
        img2: Second image (ground truth)
        max_val: Maximum pixel value (1.0 for normalized, 255 for uint8)
        crop_border: Pixels to crop from border before calculation
        
    Returns:
        PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:
        psnr_values = []
        for i in range(img1.shape[0]):
            psnr_values.append(calculate_psnr(img1[i], img2[i], max_val, crop_border))
        return np.mean(psnr_values)
    
    # CHW -> HWC
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Crop border
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * math.log10(max_val / math.sqrt(mse))
    return psnr


def calculate_ssim(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    crop_border: int = 0,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03
) -> float:
    """
    Calculate Structural Similarity Index (SSIM)
    
    Args:
        img1: First image (prediction)
        img2: Second image (ground truth)
        crop_border: Pixels to crop from border
        window_size: Size of sliding window
        K1, K2: SSIM constants
        
    Returns:
        SSIM value (0 to 1)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if img1.ndim == 4:
        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_values.append(calculate_ssim(img1[i], img2[i], crop_border, window_size))
        return np.mean(ssim_values)
    
    # CHW -> HWC
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Crop border
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
    
    # Calculate SSIM for each channel and average
    ssim_values = []
    for i in range(img1.shape[2]):
        ssim_values.append(_ssim_single_channel(
            img1[:, :, i], img2[:, :, i], window_size, K1, K2
        ))
    
    return np.mean(ssim_values)


def _ssim_single_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    K1: float = 0.01,
    K2: float = 0.03
) -> float:
    """Calculate SSIM for a single channel"""
    from scipy import ndimage
    
    # Constants
    C1 = (K1 * 1.0) ** 2  # Assuming max value is 1.0
    C2 = (K2 * 1.0) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    x = np.arange(window_size) - (window_size - 1) / 2
    gauss = np.exp(-x ** 2 / (2 * sigma ** 2))
    gauss /= gauss.sum()
    window = np.outer(gauss, gauss)
    
    # Compute means
    mu1 = ndimage.convolve(img1, window, mode='reflect')
    mu2 = ndimage.convolve(img2, window, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = ndimage.convolve(img1 ** 2, window, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(img2 ** 2, window, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(img1 * img2, window, mode='reflect') - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def evaluate_images(
    sr_path: str,
    hr_path: str,
    crop_border: int = 4
) -> dict:
    """
    Evaluate a super-resolved image against ground truth
    
    Args:
        sr_path: Path to super-resolved image
        hr_path: Path to ground truth HR image
        crop_border: Border pixels to crop
        
    Returns:
        Dictionary with PSNR and SSIM values
    """
    sr_img = Image.open(sr_path).convert('RGB')
    hr_img = Image.open(hr_path).convert('RGB')
    
    # Convert to numpy arrays (normalized to 0-1)
    sr_arr = np.array(sr_img).astype(np.float32) / 255.0
    hr_arr = np.array(hr_img).astype(np.float32) / 255.0
    
    # Ensure same size
    if sr_arr.shape != hr_arr.shape:
        print(f"Warning: Size mismatch. SR: {sr_arr.shape}, HR: {hr_arr.shape}")
        # Resize SR to match HR
        sr_img = sr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)
        sr_arr = np.array(sr_img).astype(np.float32) / 255.0
    
    psnr = calculate_psnr(sr_arr, hr_arr, max_val=1.0, crop_border=crop_border)
    ssim = calculate_ssim(sr_arr, hr_arr, crop_border=crop_border)
    
    return {
        'psnr': psnr,
        'ssim': ssim
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Super Resolution Results")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Super-resolved image path")
    parser.add_argument("--reference", "-r", type=str, required=True,
                        help="Ground truth HR image path")
    parser.add_argument("--crop-border", type=int, default=4,
                        help="Border pixels to crop before evaluation")
    
    args = parser.parse_args()
    
    results = evaluate_images(args.input, args.reference, args.crop_border)
    
    print(f"PSNR: {results['psnr']:.2f} dB")
    print(f"SSIM: {results['ssim']:.4f}")


if __name__ == "__main__":
    main()
