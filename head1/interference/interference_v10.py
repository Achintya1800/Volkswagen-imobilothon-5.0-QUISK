"""
Single Image Inference with YOLOv10
Quick inference script for testing
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class YOLOv10Inference:
    """YOLOv10 inference wrapper"""
    
    def __init__(self, model_path, conf_threshold=0.25):
        """
        Initialize inference
        
        Args:
            model_path: Path to YOLOv10 model
            conf_threshold: Confidence threshold
        """
        print(f"🚀 Loading YOLOv10 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        self.classes = {
            0: 'pothole',
            1: 'speed_bump',
            2: 'debris',
            3: 'crack'
        }
        
        self.colors = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (255, 0, 0),
            3: (0, 165, 255)
        }
        
        print("✅ Model loaded successfully!")
    
    def predict(self, image_path, save_path=None):
        """
        Run inference on single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save output (optional)
        
        Returns:
            detections: List of detections
        """
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': self.classes[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': {
                    'x1': float(box.xyxy[0][0]),
                    'y1': float(box.xyxy[0][1]),
                    'x2': float(box.xyxy[0][2]),
                    'y2': float(box.xyxy[0][3])
                }
            }
            detections.append(detection)
        
        # Visualize and save
        if save_path:
            self.visualize(image_path, detections, save_path)
        
        return detections
    
    def visualize(self, image_path, detections, output_path):
        """Draw bounding boxes on image"""
        image = cv2.imread(str(image_path))
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
            
            color = self.colors[det['class_id']]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(str(output_path), image)
        print(f"💾 Saved to: {output_path}")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv10 Single Image Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--output', type=str, default='output.jpg', help='Output path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Run inference
    inference = YOLOv10Inference(args.model, args.conf)
    detections = inference.predict(args.image, args.output)
    
    # Print results
    print(f"\n📊 Detected {len(detections)} objects:")
    for i, det in enumerate(detections, 1):
        print(f"   {i}. {det['class_name']}: {det['confidence']:.3f}")


if __name__ == "__main__":
    main()