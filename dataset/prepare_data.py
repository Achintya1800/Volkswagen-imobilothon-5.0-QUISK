

import os
import shutil
import yaml
from pathlib import Path
import cv2
from tqdm import tqdm
import json


def remap_label_classes(label_path, class_remapping):
    """Remap class IDs"""
    remapped_annotations = []
    
    if not os.path.exists(label_path):
        return remapped_annotations
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            old_class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            
            # Remap class ID
            if old_class_id in class_remapping:
                new_class_id = class_remapping[old_class_id]
                remapped_annotations.append([new_class_id, x_center, y_center, width, height])
    except:
        pass
    
    return remapped_annotations


def process_dataset_fixed(dataset_name, dataset_path, output_path, class_remapping):
    """Process dataset"""
    
    print(f"\n📂 Processing: {dataset_name}")
    print(f"   Path: {dataset_path}")
    print(f"   Class Remapping: {class_remapping}")
    
    if not os.path.exists(dataset_path):
        print(f"   ⚠️  Not found, skipping")
        return 0, 0
    
    train_img_dir = None
    val_img_dir = None
    train_label_dir = None
    val_label_dir = None
    
    # Find directories
    if os.path.exists(f"{dataset_path}/train/images"):
        train_img_dir = f"{dataset_path}/train/images"
        train_label_dir = f"{dataset_path}/train/labels"
    elif os.path.exists(f"{dataset_path}/images/train"):
        train_img_dir = f"{dataset_path}/images/train"
        train_label_dir = f"{dataset_path}/labels/train"
    
    if os.path.exists(f"{dataset_path}/val/images"):
        val_img_dir = f"{dataset_path}/val/images"
        val_label_dir = f"{dataset_path}/val/labels"
    elif os.path.exists(f"{dataset_path}/valid/images"):
        val_img_dir = f"{dataset_path}/valid/images"
        val_label_dir = f"{dataset_path}/valid/labels"
    elif os.path.exists(f"{dataset_path}/images/val"):
        val_img_dir = f"{dataset_path}/images/val"
        val_label_dir = f"{dataset_path}/labels/val"
    
    if not train_img_dir:
        print(f"   ⚠️  No train images found")
        return 0, 0
    
    # Process train
    train_count = process_split_fixed(
        train_img_dir, train_label_dir, output_path, "train", class_remapping, dataset_name
    )
    
    # Process val
    val_count = 0
    if val_img_dir:
        val_count = process_split_fixed(
            val_img_dir, val_label_dir, output_path, "val", class_remapping, dataset_name
        )
    
    print(f"   ✅ {train_count} train, {val_count} val")
    return train_count, val_count


def process_split_fixed(img_dir, label_dir, output_path, split, class_remapping, dataset_name):
    """Process split"""
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(img_dir).glob(f"*{ext}"))
    
    if not image_files:
        return 0
    
    count = 0
    
    for img_path in tqdm(image_files, desc=f"    {split}", leave=False):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
        except:
            continue
        
        unique_name = f"{dataset_name}_{count}"
        
        # Copy image
        img_dst = f"{output_path}/images/{split}/{unique_name}.jpg"
        shutil.copy(str(img_path), img_dst)
        
        # Process label
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        label_dst = f"{output_path}/labels/{split}/{unique_name}.txt"
        
        # Remap and save
        remapped = remap_label_classes(label_path, class_remapping)
        
        with open(label_dst, 'w') as f:
            for ann in remapped:
                f.write(' '.join(map(str, ann)) + '\n')
        
        count += 1
    
    return count


def create_unified_dataset():
    """Create unified dataset with CORRECTED remapping"""
    
    print("="*80)
    print("🔧 YOLOv10 Dataset Preparation (CORRECTED)")
    print("="*80)
    
    unified_classes = ['pothole', 'speed_bump', 'debris']
    output_path = "datasets/unified_road_hazards_v10"
    
    # Create output directories
    for split in ['train', 'val']:
        os.makedirs(f"{output_path}/images/{split}", exist_ok=True)
        os.makedirs(f"{output_path}/labels/{split}", exist_ok=True)
    
    print(f"\n📊 Classes: {unified_classes}")
    print("   0: pothole")
    print("   1: speed_bump")
    print("   2: debris")
    
    # CORRECTED class remapping based on actual Roboflow structure
    datasets = [
        {
            'name': 'pothole_raw',
            'path': 'datasets/pothole_raw',
            'class_remapping': {0: 0},  # pothole class 0 → 0
        },
        {
            'name': 'debris',
            'path': 'datasets/debris',
            'class_remapping': {2: 2},  # ✅ FIXED: debris already class 2 → 2
        },
        {
            'name': 'speed_bump',
            'path': 'datasets/speed_bump',
            'class_remapping': {1: 1},  # speed_bump class 0 → 1
        },
        {
            'name': 'combined_hazards',
            'path': 'datasets/combined_hazards',
            'class_remapping': {0: 0, 1: 1},  # combined: 0→0, 1→1
        }
    ]
    
    print("\n📦 Processing Datasets:")
    total_train = 0
    total_val = 0
    
    for dataset_config in datasets:
        train_count, val_count = process_dataset_fixed(
            dataset_name=dataset_config['name'],
            dataset_path=dataset_config['path'],
            output_path=output_path,
            class_remapping=dataset_config['class_remapping']
        )
        total_train += train_count
        total_val += val_count
    
    # Create data.yaml
    data_yaml = {
        'path': os.path.abspath(output_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': unified_classes
    }
    
    yaml_path = f"{output_path}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print("\n" + "="*80)
    print("✅ Dataset Preparation Complete!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"   Training images: {total_train}")
    print(f"   Validation images: {total_val}")
    print(f"   Total: {total_train + total_val}")
    
    # Analyze class distribution
    print("\n📊 Class Distribution:")
    analyze_class_distribution(output_path)
    
    print(f"\n🎯 Next: python training/train_distillation_v10.py")


def analyze_class_distribution(output_path):
    """Analyze class distribution"""
    class_counts = {0: 0, 1: 0, 2: 0}
    
    for split in ['train', 'val']:
        label_dir = f"{output_path}/labels/{split}"
        if not os.path.exists(label_dir):
            continue
        
        for label_file in Path(label_dir).glob("*.txt"):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
    
    print(f"   Pothole (0): {class_counts[0]:,}")
    print(f"   Speed Bump (1): {class_counts[1]:,}")
    print(f"   Debris (2): {class_counts[2]:,}")
    print(f"   Total: {sum(class_counts.values()):,}")


if __name__ == "__main__":
    create_unified_dataset()
