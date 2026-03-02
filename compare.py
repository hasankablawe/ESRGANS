import os
from PIL import Image, ImageDraw, ImageFont, ImageOps

def create_comparison(output_path="super_resolution/comparison_grid.png"):
    # Define pairs (Before, After)
    pairs = [
        ("super_resolution/test2.png", "super_resolution/output2.png")
    ]
    
    # Check if files exist
    valid_pairs = []
    for before_path, after_path in pairs:
        if os.path.exists(before_path) and os.path.exists(after_path):
            valid_pairs.append((before_path, after_path))
        else:
            print(f"Warning: Could not find pair {before_path} -> {after_path}")
            
    if not valid_pairs:
        print("No valid image pairs found.")
        return

    # Load images and determine target dimensions based on "After" images
    loaded_pairs = []
    max_w = 0
    max_h = 0
    
    for before_path, after_path in valid_pairs:
        before_img = Image.open(before_path).convert("RGB")
        after_img = Image.open(after_path).convert("RGB")
        
        # Resize before image to match after image dimensions (Nearest Neighbor to show pixelation)
        # Or Bicubic to show "standard upscale". 
        # Let's use Bicubic for a fairer baseline comparison, 
        # or actually let's use NEAREST if we want to emphasize the super resolution effect against "raw pixels"
        # But usually users compare against standard upsampling.
        # Let's resize it with BICUBIC which is a standard simple upscaler.
        before_upscaled = before_img.resize(after_img.size, Image.BICUBIC)
        
        loaded_pairs.append((before_upscaled, after_img))
        
        max_w = max(max_w, after_img.width)
        max_h = max(max_h, after_img.height)

    # Grid settings
    margin = 20
    label_height = 40
    
    # Total canvas size
    # 2 columns (Before, After), N rows
    # Width = max_w * 2 + margin * 3
    # Height = (max_h + label_height + margin) * len(valid_pairs) + margin
    
    # Wait, images might have different sizes. 
    # To make a nice grid, we should probably resize everything to the max width/height or similar.
    # For now, let's assume valid_pairs[0] and valid_pairs[1] might differ sizes.
    # We will build a canvas big enough for the widest logic.
    
    # Actually, simpler approach: Create a row for each pair.
    # Width of canvas = max( (pair_w1 + pair_w2) ) + margins
    
    canvas_w = 0
    canvas_h = margin
    
    row_images = []
    
    for before, after in loaded_pairs:
        # If one is taller than the other? They are same size now.
        w = before.width * 2 + margin
        h = before.height + label_height
        
        canvas_w = max(canvas_w, w)
        canvas_h += h + margin
        
        row_images.append( (before, after) )
        
    canvas_w += margin * 2 # Side margins
    
    # Create canvas
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        # Try to load a font, otherwise default
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()

    y_offset = margin
    
    for i, (before, after) in enumerate(row_images):
        # Draw labels
        draw.text((margin, y_offset), "Original (Scaled)", font=font, fill=(0, 0, 0))
        draw.text((margin + before.width + margin, y_offset), "Super Resolution", font=font, fill=(0, 0, 0))
        
        y_offset += label_height
        
        # Paste images
        canvas.paste(before, (margin, y_offset))
        canvas.paste(after, (margin + before.width + margin, y_offset))
        
        y_offset += before.height + margin
        
    canvas.save(output_path)
    print(f"Comparison saved to {output_path}")

if __name__ == "__main__":
    create_comparison()
