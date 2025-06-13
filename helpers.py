"""
helpers.py - Common helper functions used across different modules.
"""
import os
import numpy as np
from typing import Dict
from PIL import Image, ImageDraw, ImageFont

def build_image_path_map(data_dir: str, dataset_type: str) -> Dict[str, str]:
    """Scans the data directory to build a map from item_id to its image path."""
    print(f"[INFO] Building image path map from {os.path.join(data_dir, dataset_type)}...")
    path_map = {}
    target_dir = os.path.join(data_dir, dataset_type)
    if not os.path.exists(target_dir):
        print(f"[WARN] Image directory not found: {target_dir}")
        return {}
        
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                item_id = os.path.splitext(file)[0].replace('_m', '').replace('_s', '').replace('_l', '')
                path_map[item_id] = os.path.join(root, file)
    print(f"[INFO] Image path map built with {len(path_map)} items.")
    return path_map

def safe_load_image(item_id: str, image_path_map: Dict[str, str], thumb_size=(150, 150)):
    """Loads an image from the pre-built path map, with a placeholder fallback."""
    path = image_path_map.get(str(item_id))
    if path and os.path.exists(path):
        try:
            with Image.open(path).convert("RGB") as img:
                img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
                return img
        except Exception:
            pass
    
    # Fallback placeholder
    img = Image.new("RGB", thumb_size, "#EAEAEA")
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 10)
    except: font = ImageFont.load_default()
    draw.text((10, 10), f"ID: {item_id}\n(No Image)", font=font, fill="#666666")
    return img