# Super Resolution System

A complete deep learning-based super resolution system for enhancing image and video resolution.

## Features

- **Multiple Models**: SRCNN (fast), ESRGAN (high quality), EDSR (balanced)
- **Training Pipeline**: Perceptual + adversarial losses, data augmentation
- **Flexible Inference**: Single images, batch processing, video upscaling
- **Tiled Processing**: Handle large images without memory issues
- **Quality Metrics**: PSNR and SSIM evaluation

## Installation

```bash
cd super_resolution
pip install -r requirements.txt
```

## Quick Start

### Inference (Upscale an image)

```bash
# Basic usage (4x upscaling with ESRGAN)
python inference.py --input low_res.jpg --output high_res.jpg

# Use specific model and scale
python inference.py --input low_res.jpg --output high_res.jpg --model-type edsr --scale 2

# Upscale a video
python inference.py --input video.mp4 --output video_sr.mp4

# Batch process a folder
python inference.py --input ./input_folder --output ./output_folder

# Use tiled processing for large images (reduces memory)
python inference.py --input large_image.jpg --output sr_image.jpg --tile
```

### Training (Custom dataset)

```bash
# Train ESRGAN on your dataset
python train.py \
    --train-hr /path/to/hr_images \
    --val-hr /path/to/val_hr_images \
    --model esrgan \
    --scale 4 \
    --epochs 100 \
    --batch-size 16 \
    --output-dir ./output

# Train with adversarial loss (GAN)
python train.py \
    --train-hr /path/to/hr_images \
    --model esrgan \
    --use-gan \
    --gan-weight 0.005 \
    --epochs 200

# Train lightweight SRCNN
python train.py \
    --train-hr /path/to/hr_images \
    --model srcnn \
    --scale 4 \
    --epochs 50
```

### Evaluation

```bash
# Evaluate a super-resolved image
python evaluate.py --input sr_image.jpg --reference hr_ground_truth.jpg
```

## Models

| Model | Parameters | Quality | Speed | Best For |
|-------|------------|---------|-------|----------|
| SRCNN | ~57K | ★★☆ | ★★★ | Real-time, mobile |
| EDSR | ~1.5M | ★★☆ | ★★☆ | Balanced quality/speed |
| ESRGAN | ~16M | ★★★ | ★☆☆ | Maximum quality |

## Python API

```python
from super_resolution import SuperResolutionInference

# Initialize
sr = SuperResolutionInference(
    model_path="path/to/weights.pth",  # Optional
    model_type="esrgan",
    scale_factor=4
)

# Upscale single image
result = sr.upscale("input.jpg")
result.save("output.jpg")

# Batch process
sr.upscale_batch("./input_dir", "./output_dir")

# Video upscaling
sr.upscale_video("input.mp4", "output.mp4")
```

## Project Structure

```
super_resolution/
├── models/
│   ├── srcnn.py      # SRCNN model
│   ├── esrgan.py     # ESRGAN/RRDB model
│   └── edsr.py       # EDSR model
├── data/
│   ├── dataset.py    # Dataset classes
│   └── transforms.py # Data augmentation
├── losses/
│   ├── perceptual.py # VGG perceptual loss
│   └── adversarial.py# GAN losses
├── train.py          # Training script
├── inference.py      # Inference script
├── evaluate.py       # PSNR/SSIM metrics
└── utils.py          # Utilities
```

## License

MIT License
