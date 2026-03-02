"""
Inference script for Super Resolution
Supports image and video upscaling
"""
import os
import argparse
from pathlib import Path
from typing import Optional, Union
import time

import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

from models import SRCNN, ESRGAN, EDSR, RRDBNet
from models.realesrgan import RealESRGANer, load_realesrgan
from utils import (
    get_device, pil_to_tensor, tensor_to_pil, 
    pad_to_multiple, unpad, load_image, save_image
)


class SuperResolutionInference:
    """
    Inference class for super resolution models
    
    Supports:
    - Single image upscaling
    - Batch image processing
    - Video upscaling
    - Tiled processing for large images
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "esrgan",
        scale_factor: int = 4,
        device: Optional[torch.device] = None,
        tile_size: int = 512,  # Larger tiles for better quality
        tile_overlap: int = 32,  # More overlap for smoother blending
        auto_tile: bool = True,  # Auto-enable tiling to prevent OOM
        fp16: bool = True,
        use_compile: bool = False
    ):
        """
        Args:
            model_path: Path to model weights (None for random init)
            model_type: Type of model ('srcnn', 'esrgan', 'edsr')
            scale_factor: Upscaling factor
            device: Torch device
            tile_size: Size of tiles for large images
            tile_overlap: Overlap between tiles
            auto_tile: Automatically use tiling based on GPU memory
            fp16: Use half precision for inference (faster, less VRAM)
            use_compile: Use torch.compile (requires Linux/Triton)
        """
        self.device = device or get_device()
        self.scale_factor = scale_factor
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.auto_tile = auto_tile
        self.fp16 = fp16 and self.device.type == 'cuda'
        
        # Optimize
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Adjust tile size for low VRAM GPUs
        if self.device.type == 'cuda':
            # Use the correct device index
            device_idx = self.device.index if self.device.index is not None else 0
            gpu_mem_gb = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
            if gpu_mem_gb < 6:
                self.tile_size = min(self.tile_size, 256)  # Smaller tiles for low VRAM
                self.auto_tile = True
        
        # Load model
        self.model = self._load_model(model_type, model_path, scale_factor)
        
        if self.fp16:
            self.model = self.model.half()
            
        if hasattr(self, 'use_compile') and self.use_compile:
            print("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"Warning: Compilation failed: {e}")
                
        self.model.eval()
    
    def _load_model(
        self,
        model_type: str,
        model_path: Optional[str],
        scale_factor: int
    ) -> torch.nn.Module:
        """Load model architecture and weights"""
        
        # Auto-detect Real-ESRGAN weights
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Check for Real-ESRGAN format (has params_ema or params keys)
            if 'params_ema' in state_dict or 'params' in state_dict:
                model = load_realesrgan(model_path, scale=scale_factor, device=self.device)
                return model

        # Check for auto-download if path not provided but type is known
        if model_path is None:
            from utils import download_pretrained
            try:
                if model_type == "realesrgan":
                    model_name = f"realesrgan_x{scale_factor}"
                    print(f"No model path provided. downloading {model_name}...")
                    model_path = download_pretrained(model_name)
                elif model_type == "esrgan":
                    model_name = f"esrgan_x{scale_factor}"
                    print(f"No model path provided. downloading {model_name}...")
                    model_path = download_pretrained(model_name)
            except Exception as e:
                print(f"Could not download pretrained model: {e}")
                print("Falling back to random initialization.")

        # If we downloaded a Real-ESRGAN model, load it now
        if model_path and os.path.exists(model_path) and model_type == "realesrgan":
             model = load_realesrgan(model_path, scale=scale_factor, device=self.device)
             return model
        
        # Otherwise use our models
        if model_type == "srcnn":
            model = SRCNN(scale_factor=scale_factor)
        elif model_type == "esrgan":
            model = RRDBNet(scale_factor=scale_factor, num_blocks=23)
        elif model_type == "edsr":
            model = EDSR(scale_factor=scale_factor, num_blocks=16)
        elif model_type == "realesrgan":
             # Fallback if download failed or random init desired
             # RealESRGAN uses the same architecture as ESRGAN (RRDBNet) essentially, 
             # but we use the specific loader for weights. For random init:
             from models.realesrgan import RealESRGANer
             model = RealESRGANer(scale=scale_factor, num_block=23)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'generator' in state_dict:
                state_dict = state_dict['generator']
            
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {model_path}")
                
        return model.to(self.device)
    
    @torch.no_grad()
    def upscale(
        self,
        image: Union[str, Image.Image, np.ndarray],
        use_tiling: Optional[bool] = None
    ) -> Image.Image:
        """
        Upscale a single image
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            use_tiling: Use tiled processing for large images (None=auto)
            
        Returns:
            Upscaled PIL Image
        """
        # Load image
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get the device the model is actually on
        model_device = next(self.model.parameters()).device
        
        # Convert to tensor and move to model's device
        lr_tensor = pil_to_tensor(image).to(model_device)
        if self.fp16:
            lr_tensor = lr_tensor.half()
        
        # Auto-detect if tiling is needed
        _, _, h, w = lr_tensor.shape
        if use_tiling is None:
            # More aggressive tiling threshold if not FP16
            use_tiling = self.auto_tile or (h > self.tile_size or w > self.tile_size)
        
        # Upscale
        if use_tiling:
            sr_tensor = self._tile_process(lr_tensor, model_device)
        else:
            try:
                lr_padded, original_size = pad_to_multiple(lr_tensor, multiple=8)
                sr_tensor = self.model(lr_padded)
                sr_tensor = unpad(sr_tensor, original_size, self.scale_factor)
            except torch.cuda.OutOfMemoryError:
                print("GPU OOM - falling back to tiled processing...")
                torch.cuda.empty_cache()
                sr_tensor = self._tile_process(lr_tensor, model_device)
        
        # Float for saving
        return tensor_to_pil(sr_tensor.float())

    def upscale_with_tta(self, image: Union[str, Image.Image, np.ndarray], use_tiling: Optional[bool] = None) -> Image.Image:
        """
        Upscale with Test-Time Augmentation (8x slower, better quality)
        Averages 8 rotations/flips of the input.
        """
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # 8 augmentations: index 0-7
        # 0: original
        # 1: rot90
        # 2: rot180
        # 3: rot270
        # 4: flip_lr
        # 5: flip_lr + rot90
        # 6: flip_lr + rot180
        # 7: flip_lr + rot270
        
        results = []
        
        print("Running TTA (8 passes)...")
        for i in range(8):
            # Augment
            img_aug = image
            if i >= 4:
                img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)
            
            rot = i % 4
            if rot == 1:
                img_aug = img_aug.transpose(Image.ROTATE_90)
            elif rot == 2:
                img_aug = img_aug.transpose(Image.ROTATE_180)
            elif rot == 3:
                img_aug = img_aug.transpose(Image.ROTATE_270)
            
            # Upscale
            sr_aug = self.upscale(img_aug, use_tiling)
            
            # Inverse Augment (to bring back to original orientation)
            if rot == 1:
                sr_aug = sr_aug.transpose(Image.ROTATE_270) # -90
            elif rot == 2:
                sr_aug = sr_aug.transpose(Image.ROTATE_180) # -180
            elif rot == 3:
                sr_aug = sr_aug.transpose(Image.ROTATE_90)  # -270
                
            if i >= 4:
                sr_aug = sr_aug.transpose(Image.FLIP_LEFT_RIGHT)
            
            results.append(np.array(sr_aug).astype(np.float32))
            
        # Average
        avg_img = np.mean(results, axis=0)
        return Image.fromarray(avg_img.astype(np.uint8))
    
    def _tile_process(self, lr_tensor: torch.Tensor, device: torch.device = None, tile_size: int = None) -> torch.Tensor:
        """
        Process large images using overlapping tiles
        Prevents memory issues with high-resolution images
        Recursive OOM handling.
        """
        if device is None:
            device = lr_tensor.device
            
        _, _, h, w = lr_tensor.shape
        
        # Default tile size from self if not provided (recursive calls provide tile_size)
        if tile_size is None:
            tile_size = self.tile_size
            
        # Recursive base case: checks reasonable minimum
        if tile_size < 64:
             raise RuntimeError("Tile size too small, cannot process image even with tiling.")

        tile = tile_size
        overlap = self.tile_overlap
        stride = tile - overlap
        
        # Calculate output size
        out_h = h * self.scale_factor
        out_w = w * self.scale_factor
        
        try:
             # Initialize output tensors on same device as input
            out_tensor = torch.zeros(1, 3, out_h, out_w, device=device, dtype=lr_tensor.dtype)
            weight_tensor = torch.zeros(1, 1, out_h, out_w, device=device, dtype=lr_tensor.dtype)
            
            # Create weight mask for blending (feathered edges)
            weight = self._create_weight_mask(tile * self.scale_factor, device, dtype=lr_tensor.dtype)
            
            # Process tiles
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    # Extract tile
                    x_end = min(x + tile, w)
                    y_end = min(y + tile, h)
                    x_start = max(0, x_end - tile)
                    y_start = max(0, y_end - tile)
                    
                    tile_tensor = lr_tensor[:, :, y_start:y_end, x_start:x_end]
                    
                    # Pad if necessary
                    tile_padded, orig_size = pad_to_multiple(tile_tensor, 8)
                    
                    # Process tile
                    sr_tile = self.model(tile_padded)
                    sr_tile = unpad(sr_tile, orig_size, self.scale_factor)
                    
                    # Output coordinates
                    out_y_start = y_start * self.scale_factor
                    out_y_end = y_end * self.scale_factor
                    out_x_start = x_start * self.scale_factor
                    out_x_end = x_end * self.scale_factor
                    
                    # Get weight for this tile size
                    tile_weight = weight[:, :, :out_y_end - out_y_start, :out_x_end - out_x_start]
                    
                    # Accumulate
                    out_tensor[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += sr_tile * tile_weight
                    weight_tensor[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += tile_weight
            
            # Normalize by weights
            out_tensor = out_tensor / weight_tensor.clamp(min=1e-8)
            return out_tensor

        except torch.cuda.OutOfMemoryError:
            # Recursive fallback
            new_tile_size = tile_size // 2
            print(f"OOM with tile_size={tile_size}. Retrying with tile_size={new_tile_size}...")
            torch.cuda.empty_cache()
            return self._tile_process(lr_tensor, device, tile_size=new_tile_size)
    
    def _create_weight_mask(self, size: int, device: torch.device = None, dtype=None) -> torch.Tensor:
        """Create a weight mask with feathered edges for blending"""
        if device is None:
            device = self.device
        weight = torch.ones(1, 1, size, size, device=device, dtype=dtype)
        
        # Create linear ramp for edges
        ramp_size = min(size // 4, 64)
        ramp = torch.linspace(0, 1, ramp_size, device=device)
        
        # Apply to edges
        weight[:, :, :ramp_size, :] *= ramp.view(1, 1, -1, 1)
        weight[:, :, -ramp_size:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        weight[:, :, :, :ramp_size] *= ramp.view(1, 1, 1, -1)
        weight[:, :, :, -ramp_size:] *= ramp.flip(0).view(1, 1, 1, -1)
        
        return weight
    
    def upscale_batch(
        self,
        input_dir: str,
        output_dir: str,
        use_tiling: bool = False
    ) -> None:
        """
        Upscale all images in a directory
        
        Args:
            input_dir: Input directory with LR images
            output_dir: Output directory for SR images
            use_tiling: Use tiled processing
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        
        print(f"Processing {len(images)} images...")
        
        for img_path in tqdm(images):
            try:
                if hasattr(self, 'use_tta') and self.use_tta:
                    sr_image = self.upscale_with_tta(str(img_path), use_tiling=use_tiling)
                else:
                    sr_image = self.upscale(str(img_path), use_tiling=use_tiling)
                output_file = output_path / f"{img_path.stem}_sr{img_path.suffix}"
                sr_image.save(output_file)
            except Exception as e:
                print(f"Failed to process {img_path}: {e}")
        
        print(f"Saved results to {output_dir}")
    

    def upscale_video(
        self,
        input_path: str,
        output_path: str,
        use_tiling: bool = False,
        max_frames: Optional[int] = None
    ) -> None:
        """
        Upscale a video file
        
        Args:
            input_path: Path to input video
            output_path: Path for output video
            use_tiling: Use tiled processing
            max_frames: Maximum frames to process (None for all)
        """
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Input: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"Output: {width * self.scale_factor}x{height * self.scale_factor}")
        
        # Setup output video
        out_width = width * self.scale_factor
        out_height = height * self.scale_factor
        
        # Ensure directory exists
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Process frames
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while cap.isOpened() and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            
            # Upscale
            if hasattr(self, 'use_tta') and self.use_tta:
                sr_frame = self.upscale_with_tta(pil_frame, use_tiling=use_tiling)
            else:
                sr_frame = self.upscale(pil_frame, use_tiling=use_tiling)
            
            # Convert back to BGR for OpenCV
            sr_array = np.array(sr_frame)
            sr_bgr = cv2.cvtColor(sr_array, cv2.COLOR_RGB2BGR)
            
            out.write(sr_bgr)
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"Saved video to {output_path}")

def depixelate(
    image_path: str,
    sr_model: SuperResolutionInference,
    target_width: int = 48
) -> Image.Image:
    """
    Aggressive de-pixelization strategy:
    1. Downscale to very low resolution (merging pixel blocks)
    2. Super-resolve multiple times to rebuild lost detail
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # 1. Downscale to merge blocks
    aspect = h / w
    target_h = int(target_width * aspect)
    
    # Use AREA interpolation for best downscaling (pixel merging)
    img_np = np.array(img)
    small_np = cv2.resize(img_np, (target_width, target_h), interpolation=cv2.INTER_AREA)
    small_pil = Image.fromarray(small_np)
    
    print(f"De-pixelization: Downscaled to {target_width}x{target_h} to remove grid")
    
    # 2. First Pass SR (Restores shapes)
    print("Pass 1: Rebuilding structure...")
    if hasattr(sr_model, 'use_tta') and sr_model.use_tta:
        pass1 = sr_model.upscale_with_tta(small_pil, use_tiling=True)
    else:
        pass1 = sr_model.upscale(small_pil, use_tiling=True)
    
    # 3. Second Pass SR (Adds details)
    # Only do second pass if we need more resolution
    if pass1.width < w * 2: # Heuristic: if still smaller than 2x original
        print("Pass 2: Adding texture details...")
        if hasattr(sr_model, 'use_tta') and sr_model.use_tta:
            final = sr_model.upscale_with_tta(pass1, use_tiling=True)
        else:
            final = sr_model.upscale(pass1, use_tiling=True)
    else:
        final = pass1
        
    return final


def main():
    parser = argparse.ArgumentParser(description="Super Resolution Inference")
    
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input image/directory/video")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output image/directory/video")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model weights")
    parser.add_argument("--model-type", type=str, default="esrgan",
                        choices=["srcnn", "esrgan", "edsr", "realesrgan"],
                        help="Model architecture")
    parser.add_argument("--scale", type=int, default=4,
                        choices=[2, 4, 8],
                        help="Upscaling factor")
    parser.add_argument("--tile", action="store_true",
                        help="Use tiled processing for large images")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size for tiled processing")
    parser.add_argument("--smooth", action="store_true",
                        help="Apply aggressive de-pixelization (downscale-upscale)")
    
    parser.add_argument("--fp32", action="store_true",
                        help="Force FP32 (Full Precision) inference")
    parser.add_argument("--tta", action="store_true",
                        help="Enable Test-Time Augmentation (Best Quality, 8x slower)")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile optimization (Linux only)")
    
    args = parser.parse_args()
    
    # Initialize inference
    sr = SuperResolutionInference(
        model_path=args.model,
        model_type=args.model_type,
        scale_factor=args.scale,
        tile_size=args.tile_size,
        fp16=not args.fp32,
        use_compile=args.compile
    )
    sr.use_tta = args.tta
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Check if video
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        if input_path.suffix.lower() in video_extensions:
            sr.upscale_video(str(input_path), args.output, use_tiling=args.tile)
        else:
            # Single image
            start_time = time.time()
            
            if args.smooth:
                print("Mode: Aggressive De-pixelization")
                result = depixelate(str(input_path), sr)
            else:
                if args.tta:
                    result = sr.upscale_with_tta(str(input_path), use_tiling=args.tile)
                else:
                    result = sr.upscale(str(input_path), use_tiling=args.tile)
                
            elapsed = time.time() - start_time
            
            save_image(result, args.output)
            print(f"Saved to {args.output} ({elapsed:.2f}s)")
    
    elif input_path.is_dir():
        # Batch processing
        if args.smooth:
            print("Warning: Smoothing not yet implemented for batch processing")
        sr.upscale_batch(str(input_path), args.output, use_tiling=args.tile)
    
    else:
        print(f"Error: {args.input} not found")
        return

if __name__ == "__main__":
    main()
