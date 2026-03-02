"""
Download and use pre-trained super resolution models
"""
import os
import torch
from pathlib import Path


PRETRAINED_URLS = {
    # Real-ESRGAN models
    "realesrgan_x4": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "model_type": "esrgan"
    },
    "realesrgan_x4_anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "model_type": "esrgan"
    },
    # ESRGAN original
    "esrgan_x4": {
        "url": "https://github.com/xinntao/ESRGAN/releases/download/v0.1.0/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
        "scale": 4,
        "model_type": "esrgan"
    },
}


def download_pretrained(model_name: str, save_dir: str = "./pretrained") -> str:
    """
    Download pre-trained model weights
    
    Available models:
    - realesrgan_x4: Real-ESRGAN 4x (general images) 
    - realesrgan_x4_anime: Real-ESRGAN 4x (anime/illustrations)
    - esrgan_x4: Original ESRGAN 4x
    
    Args:
        model_name: Name of the pretrained model
        save_dir: Directory to save weights
        
    Returns:
        Path to downloaded weights
    """
    import urllib.request
    
    if model_name not in PRETRAINED_URLS:
        available = list(PRETRAINED_URLS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")
    
    info = PRETRAINED_URLS[model_name]
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    
    if not os.path.exists(save_path):
        print(f"Downloading {model_name}...")
        print(f"URL: {info['url']}")
        
        try:
            urllib.request.urlretrieve(info['url'], save_path)
            print(f"✓ Saved to {save_path}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("You can manually download from the URL above.")
            return None
    else:
        print(f"Model already exists at {save_path}")
    
    return save_path


def list_pretrained():
    """List available pre-trained models"""
    print("Available pre-trained models:")
    print("-" * 50)
    for name, info in PRETRAINED_URLS.items():
        print(f"  {name}")
        print(f"    Scale: {info['scale']}x")
        print(f"    Type: {info['model_type']}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pre-trained SR models")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name to download")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available models")
    parser.add_argument("--output", "-o", type=str, default="./pretrained",
                        help="Output directory")
    
    args = parser.parse_args()
    
    if args.list or args.model is None:
        list_pretrained()
    else:
        download_pretrained(args.model, args.output)
