"""
Utility Helper Functions
Common utilities for the project
"""

import os
import yaml
import json
import torch
import cv2
import numpy as np
from pathlib import Path


def load_yaml(file_path):
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    """Save data to YAML file"""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)


def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data, file_path, indent=2):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  No GPU available, using CPU")
        return False


def create_directory(path):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def count_files(directory, extensions=['.jpg', '.png']):
    """Count files with specific extensions in directory"""
    count = 0
    for ext in extensions:
        count += len(list(Path(directory).rglob(f'*{ext}')))
    return count


def resize_image(image, target_size=640):
    """Resize image maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    return resized


def draw_boxes(image, boxes, labels, colors):
    """Draw bounding boxes on image"""
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, str(label), (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


if __name__ == "__main__":
    print("Utility Helper Functions")
    check_gpu()
