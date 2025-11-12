"""
Privacy Blur System (Hybrid: ONNX + YOLOv8)
-------------------------------------------
Detects and blurs faces and license plates in images/videos.
Author: Nidhi Patel + ChatGPT | 2025
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

# ==========================================
# 1️⃣ Model Paths
# ==========================================
# ==========================================
# 1️⃣ Model Paths (repo-relative, robust)
# ==========================================
from pathlib import Path
HERE = Path(__file__).resolve().parent

# model files expected inside privacy/models/
FACE_MODEL_PATH = str(HERE / "models" / "face_detection_model.onnx")
PLATE_MODEL_PATH = str(HERE / "models" / "license_plate_detector.pt")

INPUT_DIR = str(HERE / "input")
OUTPUT_DIR = str(HERE / "output_blur")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2️⃣ Load Models
# ==========================================

# --- FACE: ONNX ---
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
face_session = ort.InferenceSession(FACE_MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"])
face_input = face_session.get_inputs()[0].name
face_output = face_session.get_outputs()[0].name

# --- LICENSE PLATE: YOLOv8 (.pt) ---
plate_model = YOLO(PLATE_MODEL_PATH)

print("✅ Models loaded successfully!")
print(f"   Face model : {FACE_MODEL_PATH}")
print(f"   Plate model: {PLATE_MODEL_PATH}")

# ==========================================
# 3️⃣ Preprocessing (for ONNX Face Model)
# ==========================================
def preprocess(frame, size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    return img

# ==========================================
# 4️⃣ Postprocess (for ONNX)
# ==========================================
def postprocess(outputs, orig_w, orig_h, conf_thres=0.35):
    preds = np.squeeze(outputs)
    preds = np.transpose(preds)
    boxes = []
    for p in preds:
        if len(p) < 5:
            continue
        x_center, y_center, w, h, conf = p[:5]
        if conf > conf_thres:
            x1 = int((x_center - w / 2) * orig_w / 640)
            y1 = int((y_center - h / 2) * orig_h / 640)
            x2 = int((x_center + w / 2) * orig_w / 640)
            y2 = int((y_center + h / 2) * orig_h / 640)
            boxes.append([x1, y1, x2, y2])
    return boxes

# ==========================================
# 5️⃣ Blur Helper
# ==========================================
def blur_region(frame, x1, y1, x2, y2, ksize=(55, 55)):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    roi = frame[y1:y2, x1:x2]
    blur = cv2.GaussianBlur(roi, ksize, 30)
    frame[y1:y2, x1:x2] = blur
    return frame

# ==========================================
# 6️⃣ Core Logic – Run Both Models & Blur
# ==========================================
def blur_privacy_objects(frame):
    orig_h, orig_w = frame.shape[:2]

    # --- FACE DETECTION (ONNX) ---
    face_inp = preprocess(frame)
    face_out = face_session.run([face_output], {face_input: face_inp})
    face_boxes = postprocess(face_out[0], orig_w, orig_h, conf_thres=0.3)

    # --- LICENSE PLATE DETECTION (YOLOv8) ---
    results = plate_model(frame, verbose=False)
    plate_boxes = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        plate_boxes.append([x1, y1, x2, y2])

    # --- Combine Detections ---
    all_boxes = [*face_boxes, *plate_boxes]

    # --- Apply Blur ---
    for (x1, y1, x2, y2) in all_boxes:
        frame = blur_region(frame, x1, y1, x2, y2)

    return frame

# ==========================================
# 7️⃣ Process Image Files
# ==========================================
def process_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return
    blurred = blur_privacy_objects(img)
    cv2.imwrite(save_path, blurred)
    print(f"🖼️ Saved blurred image → {save_path}")

# ==========================================
# 8️⃣ Process Video Files
# ==========================================
def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = blur_privacy_objects(frame)
        out.write(frame)

        if frame_count % 10 == 0:
            print(f"[VID] Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"🎥 Saved blurred video → {save_path}")

# ==========================================
# 9️⃣ Main Runner
# ==========================================
def main():
    for file_name in os.listdir(INPUT_DIR):
        in_path = os.path.join(INPUT_DIR, file_name)
        out_path = os.path.join(OUTPUT_DIR, f"blurred_{file_name}")

        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(in_path, out_path)
        elif file_name.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
            process_video(in_path, out_path)
        else:
            print(f"[SKIP] Unsupported file: {file_name}")

if __name__ == "__main__":
    main()
