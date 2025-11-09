import os
import cv2
from ultralytics import YOLO

# ========= CONFIG =========
MODEL_PATH = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H4RoadMask\model\yolo11n-seg.pt"
INPUT_FOLDER = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H4RoadMask\input"
OUTPUT_FOLDER = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H4RoadMask\output_RoadSegment"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= LOAD MODEL =========
print("🚀 Loading YOLOv11 segmentation model...")
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")

# Vehicle classes in COCO dataset
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck

# ========= PROCESS EACH IMAGE =========
for file in os.listdir(INPUT_FOLDER):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_FOLDER, file)
    print(f"\n🧠 Processing {file}...")

    # Run inference
    results = model(img_path, conf=0.3)

    annotated = results[0].plot()  # get annotated image
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, file.replace(".jpg", "_annotated.jpg")), annotated)

    # Filter for vehicles only
    boxes = results[0].boxes
    masks = results[0].masks

    if boxes is not None and len(boxes) > 0:
        for i, cls in enumerate(boxes.cls):
            if int(cls) in VEHICLE_CLASSES:
                label = results[0].names[int(cls)]
                print(f"   🚗 Detected: {label} (confidence={boxes.conf[i]:.2f})")
    else:
        print("   ⚠️ No objects detected.")

print("\n🎯 Vehicle segmentation complete!")
print(f"📁 Results saved in: {OUTPUT_FOLDER}")
