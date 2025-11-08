import os
import cv2
import torch
import json
import numpy as np
from ultralytics import YOLO
import sys

# ========= CONFIG =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAST_SCNN_PATH = os.path.join(BASE_DIR, "model", "fast_scnn_citys.pth")
POTHOLE_MODEL_PATH = os.path.join(BASE_DIR, "model", "YoloV8Segmented.pt")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "model", "yolo11n-seg.pt")
INPUT_FOLDER = os.path.join(BASE_DIR, "input")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

device = torch.device("cpu")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========= IMPORT FAST-SCNN =========
fast_scnn_path = os.path.join(BASE_DIR, "Fast_SCNN")
sys.path.append(fast_scnn_path)
from fast_scnn import FastSCNN

# ========= LOAD MODELS =========
print("🚀 Loading models (CPU mode)...")
road_model = FastSCNN(num_classes=19)
road_model.load_state_dict(torch.load(FAST_SCNN_PATH, map_location=device))
road_model.eval()

pothole_model = YOLO(POTHOLE_MODEL_PATH)
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
print("✅ Models loaded successfully!\n")

# ========= UTILS =========
def transparent_overlay(base, mask, color, alpha=0.5):
    overlay = base.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(base, 1 - alpha, overlay, alpha, 0)

VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
OBJECT_DATA = {}

# ========= PROCESS EACH IMAGE =========
for file in os.listdir(INPUT_FOLDER):
    if not file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(INPUT_FOLDER, file)
    print(f"\n🧠 Processing: {file}")
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    # ===== 1️⃣ ROAD MASK =====
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img_rgb, (1024, 512))
    inp_t = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        output = road_model(inp_t)[0]
        road_mask = torch.argmax(output, dim=1).squeeze().numpy()
    road_binary = (road_mask == 0).astype(np.uint8)
    road_binary = cv2.resize(road_binary, (w, h), interpolation=cv2.INTER_NEAREST)
    road_dilated = cv2.dilate(road_binary, np.ones((15, 15), np.uint8), iterations=1)

    # ===== 2️⃣ POTHOLE DETECTION =====
    results_pothole = pothole_model(img)
    vis = img.copy()
    objects = []

    if results_pothole[0].masks is not None:
        for i, mask in enumerate(results_pothole[0].masks.data):
            conf = float(results_pothole[0].boxes.conf[i])
            cls = results_pothole[0].names[int(results_pothole[0].boxes.cls[i])]
            mask_np = cv2.resize(mask.cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            intersection = np.logical_and(mask_np > 0, road_dilated > 0)
            ratio = intersection.sum() / (mask_np.sum() + 1e-5)
            status = "INSIDE ROAD" if ratio > 0.3 else "OUTSIDE ROAD"

            objects.append({
                "type": cls,
                "confidence": round(conf, 3),
                "on_road_ratio": round(ratio, 3),
                "status": status
            })

            vis = transparent_overlay(vis, mask_np, (0, 0, 255), 0.5)
            color = (0, 255, 255) if "INSIDE" in status else (0, 0, 255)
            x1, y1, x2, y2 = map(int, results_pothole[0].boxes.xyxy[i])
            cv2.putText(vis, f"{cls} ({status})", (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ===== 3️⃣ VEHICLE DETECTION =====
    results_vehicle = vehicle_model(img)
    if results_vehicle[0].masks is not None:
        for i, cls in enumerate(results_vehicle[0].boxes.cls):
            if int(cls) not in VEHICLE_CLASSES:
                continue
            conf = float(results_vehicle[0].boxes.conf[i])
            name = results_vehicle[0].names[int(cls)]
            mask_np = cv2.resize(results_vehicle[0].masks.data[i].cpu().numpy(), (w, h), interpolation=cv2.INTER_NEAREST)
            intersection = np.logical_and(mask_np > 0, road_dilated > 0)
            ratio = intersection.sum() / (mask_np.sum() + 1e-5)
            status = "ON ROAD" if ratio > 0.15 else "OFF ROAD"

            objects.append({
                "type": name,
                "confidence": round(conf, 3),
                "on_road_ratio": round(ratio, 3),
                "status": status
            })

            vis = transparent_overlay(vis, mask_np, (255, 165, 0), 0.45)
            color = (0, 255, 255) if "ON" in status else (0, 0, 255)
            x1, y1, x2, y2 = map(int, results_vehicle[0].boxes.xyxy[i])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{name} ({status})", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save info
    OBJECT_DATA[file] = objects
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, file.replace(".jpg", "_annotated.png")), vis)
    print(f"💾 Saved → {file.replace('.jpg', '_annotated.png')}")

# ===== SAVE JSON =====
json_path = os.path.join(OUTPUT_FOLDER, "status_data.json")
with open(json_path, "w") as f:
    json.dump(OBJECT_DATA, f, indent=4)

print("\n📊 Detailed detection data saved in:", json_path)
print("🎯 Per-object status computed successfully!")
