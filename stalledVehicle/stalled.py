"""
Stalled Vehicle Detection with Ego-Motion Compensation (Affine RANSAC)
Uses YOLO11 + SORT + Optical Flow
Author: Nino + ChatGPT | 2025
"""

import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
from sort import Sort  # local file

# ---------------- CONFIG ----------------
MODEL_PATH = r"stalledVehicle/models/yolo11n.pt"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Stalled detection thresholds
STALLED_FRAMES = 20
RESIDUAL_THRESHOLD_PX = 1.5
GOOD_FEATURE_PARAMS = dict(maxCorners=1500, qualityLevel=0.01, minDistance=7, blockSize=7)
LK_PARAMS = dict(winSize=(21,21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
FARB_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15,
                   iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# ----------------------------------------

model = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)
previous_positions = {}
stalled_counters = {}

# ---------------- Helper functions ----------------
def build_detection_mask(shape, detections):
    """Build background mask (1=background, 0=detected vehicles)."""
    mask = np.ones(shape, dtype=np.uint8)
    for box in detections:
        x1, y1, x2, y2 = map(int, box[:4])
        mask[y1:y2+1, x1:x2+1] = 0
    return mask

def estimate_ego_motion_affine(prev_gray, curr_gray, detections):
    """
    Estimate ego-motion using affine transform (RANSAC).
    Returns (M, success_flag)
    """
    mask = build_detection_mask(prev_gray.shape, detections)
    pts_prev = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **GOOD_FEATURE_PARAMS)
    if pts_prev is None or len(pts_prev) < 10:
        return None, False

    pts_curr, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None, **LK_PARAMS)
    if pts_curr is None:
        return None, False

    st = st.reshape(-1)
    good_prev = pts_prev[st == 1]
    good_curr = pts_curr[st == 1]
    if len(good_prev) < 10:
        return None, False

    # RANSAC affine estimation
    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3)
    return M, M is not None

def avg_flow_in_bbox(flow, bbox):
    """Compute average (median) flow inside bounding box."""
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = flow.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w-1, x2); y2 = min(h-1, y2)
    roi = flow[y1:y2+1, x1:x2+1]
    if roi.size == 0:
        return 0.0, 0.0, 0.0
    fx = roi[...,0].flatten()
    fy = roi[...,1].flatten()
    dx = np.median(fx)
    dy = np.median(fy)
    mag = np.median(np.sqrt(fx**2 + fy**2))
    return float(dx), float(dy), float(mag)
# --------------------------------------------------

def process_video(video_path):
    print(f"[INFO] Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(OUTPUT_DIR, f"stalled_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    prev_gray = None
    frame_idx = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run YOLO detection
        results = model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
            label = model.names[cls]
            conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
            if label in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf])
        dets = np.array(detections) if len(detections) else np.empty((0,5))

        # Run tracker
        tracks = tracker.update(dets, frame_shape=(height, width))

        # Ego motion compensation
        if prev_gray is not None:
            M, ok = estimate_ego_motion_affine(prev_gray, curr_gray, [d[:4] for d in detections])
        else:
            M, ok = None, False

        # Warp prev frame to compensate ego motion
        if ok and M is not None:
            warped_prev = cv2.warpAffine(prev_gray, M, (width, height))
            flow = cv2.calcOpticalFlowFarneback(warped_prev, curr_gray, None, **FARB_PARAMS)
        elif prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **FARB_PARAMS)
        else:
            flow = np.zeros((height, width, 2), dtype=np.float32)

        # Process each track
        for trk in tracks:
            x1, y1, x2, y2, tid = map(int, trk)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            dx, dy, mag = avg_flow_in_bbox(flow, (x1,y1,x2,y2))

            # Residual flow magnitude after ego compensation
            residual_mag = np.sqrt(dx**2 + dy**2)

            # Track persistence
            if residual_mag < RESIDUAL_THRESHOLD_PX:
                stalled_counters[tid] = stalled_counters.get(tid, 0) + 1
            else:
                stalled_counters[tid] = 0

            is_stalled = stalled_counters[tid] >= STALLED_FRAMES
            color = (0,0,255) if is_stalled else (0,255,0)
            label_text = f"ID {tid} {'Stalled' if is_stalled else 'Moving'} {residual_mag:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label_text, (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            previous_positions[tid] = (cx, cy)

        # Overlay ego motion info
        if ok and M is not None:
            dx, dy = M[0,2], M[1,2]
            cv2.putText(frame, f"EgoMotion dx={dx:.2f} dy={dy:.2f}", (20,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        out.write(frame)
        prev_gray = curr_gray.copy()

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"[DONE] Saved: {out_path} | Frames: {frame_idx} | FPS: {frame_idx/elapsed:.1f}")
# --------------------------------------------------

if __name__ == "__main__":
    videos = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not videos:
        print(f"[INFO] No videos found in {INPUT_DIR}")
    for v in videos:
        process_video(os.path.join(INPUT_DIR, v))
    print("[✅] All videos processed.")
