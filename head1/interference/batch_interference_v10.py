# inference/batch_inference_v10.py
import os
import json
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time

class RoadHazardDetectorV10:
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize YOLOv10 road hazard detector
        
        YOLOv10 advantages:
        - No NMS required (built-in dual assignments)
        - 46% faster inference than YOLOv8
        - Better accuracy with fewer parameters
        """
        print(f"🚀 Loading YOLOv10 model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Class names
        self.classes = {
            0: 'pothole',
            1: 'speed_bump',
            2: 'debris',
            3: 'crack'
        }
        
        # Colors for visualization (BGR)
        self.colors = {
            0: (0, 0, 255),      # Red for potholes
            1: (0, 255, 255),    # Yellow for speed bumps
            2: (255, 0, 0),      # Blue for debris
            3: (0, 165, 255)     # Orange for cracks
        }
        
        print("✅ YOLOv10 model loaded successfully!")
        print(f"   • Confidence threshold: {conf_threshold}")
        print(f"   • IoU threshold: {iou_threshold}")
    
    def detect_image(self, image_path, return_timing=False):
        """Run YOLOv10 detection on single image"""
        start_time = time.time()
        
        # YOLOv10 inference (no NMS overhead!)
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=0 if torch.cuda.is_available() else 'cpu'
        )[0]
        
        inference_time = time.time() - start_time
        
        detections = self.parse_results(results)
        
        if return_timing:
            return detections, inference_time
        return detections
    
    def parse_results(self, results):
        """Parse YOLOv10 results into structured format"""
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
        
        return detections
    
    def visualize_detections(self, image_path, detections, output_path):
        """Draw bounding boxes with YOLOv10 styling"""
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Draw box with YOLOv10 style
            color = self.colors[det['class_id']]
            thickness = max(2, int(min(w, h) / 300))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label with background
            label = f"{det['class_name']} {det['confidence']:.2f}"
            font_scale = min(w, h) / 1000
            font_thickness = max(1, int(font_scale * 2))
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            label_y = max(y1 - 10, label_size[1] + 10)
            
            # Background rectangle
            cv2.rectangle(image, 
                         (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0] + 5, label_y + 5), 
                         color, -1)
            
            # White text
            cv2.putText(image, label, 
                       (x1 + 2, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       (255, 255, 255), 
                       font_thickness)
        
        cv2.imwrite(str(output_path), image)
    
    def batch_inference(self, input_folder, output_folder, save_json=True, save_images=True):
        """
        Run YOLOv10 batch inference on folder of images
        
        Returns:
            dict: All detection results with timing statistics
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"\n🔍 YOLOv10 Processing {len(image_files)} images...")
        
        all_results = {}
        total_time = 0
        
        for img_path in tqdm(image_files, desc="Detecting hazards"):
            # Run detection with timing
            detections, inference_time = self.detect_image(str(img_path), return_timing=True)
            total_time += inference_time
            
            # Save to results
            all_results[img_path.name] = {
                'file': img_path.name,
                'detections': detections,
                'num_detections': len(detections),
                'inference_time_ms': round(inference_time * 1000, 2)
            }
            
            # Save visualized image
            if save_images:
                vis_path = output_path / f"detected_{img_path.name}"
                self.visualize_detections(img_path, detections, vis_path)
        
        # Calculate performance metrics
        avg_inference_time = (total_time / len(image_files)) * 1000  # ms
        fps = len(image_files) / total_time if total_time > 0 else 0
        
        # Save JSON results
        if save_json:
            summary = {
                'model': 'YOLOv10',
                'total_images': len(image_files),
                'total_detections': sum(r['num_detections'] for r in all_results.values()),
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'fps': round(fps, 2),
                'results': all_results
            }
            
            json_path = output_path / "detections_v10.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"💾 Results saved to: {json_path}")
        
        # Print detailed summary
        print(f"\n📊 YOLOv10 Detection Summary:")
        print(f"   • Total images: {len(image_files)}")
        print(f"   • Total detections: {sum(r['num_detections'] for r in all_results.values())}")
        print(f"   • Avg inference time: {avg_inference_time:.2f}ms")
        print(f"   • FPS: {fps:.2f}")
        print(f"   • Output folder: {output_path}")
        
        # Per-class statistics
        class_counts = {name: 0 for name in self.classes.values()}
        for result in all_results.values():
            for det in result['detections']:
                class_counts[det['class_name']] += 1
        
        print(f"\n📈 Per-Class Detections:")
        for class_name, count in class_counts.items():
            print(f"   • {class_name}: {count}")
        
        return all_results


def main():
    """Main YOLOv10 inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv10 Road Hazard Detection Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained YOLOv10 model')
    parser.add_argument('--input', type=str, required=True, help='Input folder with images')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    
    args = parser.parse_args()
    
    # Initialize YOLOv10 detector
    detector = RoadHazardDetectorV10(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Run batch inference
    detector.batch_inference(
        input_folder=args.input,
        output_folder=args.output
    )


if __name__ == "__main__":
    main()