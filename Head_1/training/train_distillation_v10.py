"""
YOLOv10 + DINOv3 Distillation Training Script
Complete training pipeline with knowledge distillation
"""

import torch
import os
import sys
import shutil
import yaml
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def load_config(config_path="models/config.yaml"):
    """Load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_yolov10_with_dinov3(config_path="models/config.yaml"):
    """
    Main training function for YOLOv10 + DINOv3 distillation
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    print("="*80)
    print("🚀 YOLOv10 + DINOv3 Knowledge Distillation Training")
    print("="*80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Model: YOLOv10-{config['model']['variant'].upper()}")
    print(f"🎓 Teacher: {config['distillation']['teacher']}")
    print(f"📦 Dataset: {config['data']['path']}")
    print("="*80)
    
    # Setup paths
    model_variant = config['model']['variant']
    yolo_model_path = f"yolov10{model_variant}.pt"
    output_dir = config['paths']['project']
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset path
    data_yaml = os.path.join(config['data']['path'], 'data.yaml')
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        print("Run: python datasets/prepare_data.py")
        return None
    
    # =============================================================================
    # PHASE 1: DINOv3 Feature Distillation (Pre-training)
    # =============================================================================
    if config['distillation']['enabled']:
        print("\n" + "="*80)
        print("📚 PHASE 1: DINOv3 Knowledge Distillation (Backbone Pre-training)")
        print("="*80)
        
        try:
            import lightly_train
            
            print(f"🎓 Distilling {config['distillation']['teacher']} into YOLOv10...")
            print(f"   Method: {config['distillation']['method']}")
            print(f"   Weight: {config['distillation']['weight']}")
            
            # Run distillation
            lightly_train.train(
                out=f"{output_dir}/distillation",
                data=str(Path(data_yaml).parent / config['data']['train']),
                model=yolo_model_path,
                method=config['distillation']['method'],
                method_args={
                    "teacher": config['distillation']['teacher'],
                },
                trainer_args={
                    "max_epochs": 100,
                    "batch_size": config['train']['batch_size'],
                    "devices": [config['train']['device']] if config['train']['device'] != 'cpu' else 'cpu',
                }
            )
            
            distilled_checkpoint = f"{output_dir}/distillation/last.ckpt"
            print(f"✅ Distillation complete: {distilled_checkpoint}")
            
        except Exception as e:
            print(f"⚠️  Distillation phase encountered issue: {e}")
            print("Continuing with standard YOLOv10 training...")
    
    # =============================================================================
    # PHASE 2: YOLOv10 Object Detection Fine-tuning
    # =============================================================================
    print("\n" + "="*80)
    print("🎯 PHASE 2: YOLOv10 Fine-tuning for Road Hazard Detection")
    print("="*80)
    
    # Load YOLOv10 model
    print(f"📦 Loading YOLOv10-{model_variant.upper()}...")
    model = YOLO(yolo_model_path)
    
    # Training parameters
    train_args = {
        'data': data_yaml,
        'epochs': config['train']['epochs'],
        'imgsz': config['train']['imgsz'],
        'batch': config['train']['batch_size'],
        'device': config['train']['device'],
        'workers': config['train']['workers'],
        'project': output_dir,
        'name': 'finetune',
        'exist_ok': config['paths']['exist_ok'],
        
        # Optimizer
        'optimizer': config['train']['optimizer'],
        'lr0': config['train']['lr0'],
        'lrf': config['train']['lrf'],
        'momentum': config['train']['momentum'],
        'weight_decay': config['train']['weight_decay'],
        
        # Scheduler
        'warmup_epochs': config['train']['warmup_epochs'],
        'warmup_momentum': config['train']['warmup_momentum'],
        'warmup_bias_lr': config['train']['warmup_bias_lr'],
        
        # Augmentation
        'hsv_h': config['train']['hsv_h'],
        'hsv_s': config['train']['hsv_s'],
        'hsv_v': config['train']['hsv_v'],
        'degrees': config['train']['degrees'],
        'translate': config['train']['translate'],
        'scale': config['train']['scale'],
        'shear': config['train']['shear'],
        'perspective': config['train']['perspective'],
        'flipud': config['train']['flipud'],
        'fliplr': config['train']['fliplr'],
        'mosaic': config['train']['mosaic'],
        'mixup': config['train']['mixup'],
        
        # Advanced
        'close_mosaic': config['train']['close_mosaic'],
        'amp': config['train']['amp'],
        
        # Validation & Checkpointing
        'val': True,
        'save': True,
        'save_period': config['paths']['save_period'],
        'plots': config['val']['plots'],
        'verbose': True,
    }
    
    print("\n🏋️  Starting training...")
    print(f"   Epochs: {train_args['epochs']}")
    print(f"   Batch size: {train_args['batch']}")
    print(f"   Image size: {train_args['imgsz']}")
    print(f"   Device: {train_args['device']}")
    
    # Train model
    results = model.train(**train_args)
    
    # =============================================================================
    # Save Final Model
    # =============================================================================
    final_model_path = f"{output_dir}/yolov10{model_variant}_best.pt"
    best_weights = f"{output_dir}/finetune/weights/best.pt"
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_model_path)
        print(f"\n✅ Best model saved: {final_model_path}")
    
    # =============================================================================
    # Training Summary
    # =============================================================================
    print("\n" + "="*80)
    print("📊 YOLOv10 Training Results")
    print("="*80)
    
    try:
        metrics = results.results_dict
        print(f"   • mAP@0.5:     {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"   • mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"   • Precision:    {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"   • Recall:       {metrics.get('metrics/recall(B)', 0):.4f}")
    except:
        print("   • Check training logs for detailed metrics")
    
    print("\n🎉 YOLOv10 Advantages:")
    print("   ✓ No NMS overhead (dual assignments)")
    print("   ✓ 46% lower latency vs YOLOv8")
    print("   ✓ 25% fewer parameters")
    print("   ✓ Higher accuracy at same size")
    
    print("\n📁 Output Locations:")
    print(f"   • Best model: {final_model_path}")
    print(f"   • Training logs: {output_dir}/finetune")
    print(f"   • Plots & metrics: {output_dir}/finetune/plots")
    
    print("\n🔍 Next Steps:")
    print("   1. Evaluate: python evaluation/evaluate_v10.py")
    print("   2. Inference: python inference/batch_inference_v10.py")
    print("="*80)
    
    return final_model_path


def quick_train(model_size='s', epochs=150, batch_size=16):
    """
    Quick training function with minimal configuration
    
    Args:
        model_size: Model variant (n, s, m, b, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
    """
    print(f"🚀 Quick Train: YOLOv10-{model_size.upper()}")
    
    data_yaml = "datasets/unified_road_hazards_v10/data.yaml"
    
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found. Run: python datasets/prepare_data.py")
        return None
    
    model = YOLO(f"yolov10{model_size}.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=0,
        project=f"experiments/yolov10{model_size}_quick",
        name="train",
        plots=True,
        save=True,
        val=True
    )
    
    print("✅ Training complete!")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv10 with DINOv3 Distillation')
    parser.add_argument('--config', type=str, default='models/config.yaml', 
                       help='Path to config file')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick training mode (skip distillation)')
    parser.add_argument('--model', type=str, default='s', choices=['n', 's', 'm', 'b', 'l', 'x'],
                       help='Model size for quick mode')
    
    args = parser.parse_args()
    
    if args.quick:
        print("🏃 Quick Training Mode")
        quick_train(model_size=args.model)
    else:
        print("🎓 Full Training with Distillation")
        train_yolov10_with_dinov3(config_path=args.config)
