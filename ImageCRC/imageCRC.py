#!/usr/bin/env python3
"""
imageCRC.py — AKS keyframe selector (headless)

Usage example:
python ImageCRC/imageCRC.py --video "path/to/video.mp4" --output "storage/demo1/aks" --motion-threshold 12 --fast-skip 1 --slow-skip 5 --max-frames 500 --save-visuals
"""

import argparse
from pathlib import Path
import os
import cv2
import numpy as np
import json
import time
import matplotlib
matplotlib.use("Agg")   # headless backend
import matplotlib.pyplot as plt
from collections import deque

def parse_args():
    p = argparse.ArgumentParser(description="AKS keyframe selector (ImageCRC) — headless")
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument("--output", required=False, default=None, help="Output folder for keyframes and visuals")
    p.add_argument("--motion-threshold", type=float, default=12.0, help="Motion threshold (mean abs diff)")
    p.add_argument("--fast-skip", type=int, default=1, help="Skip rate when motion > threshold (smaller => more keyframes)")
    p.add_argument("--slow-skip", type=int, default=5, help="Skip rate when motion <= threshold")
    p.add_argument("--max-frames", type=int, default=500, help="Max input frames to process (use small number for quick tests)")
    p.add_argument("--save-visuals", action="store_true", help="Save overlay visualization frames")
    p.add_argument("--save-plot", action="store_true", help="Save motion & fps plot as PNG")
    return p.parse_args()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def draw_overlay(frame, text_lines):
    img = frame.copy()
    y0, dy = 30, 28
    for i, line in enumerate(text_lines):
        y = y0 + i*dy
        cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    return img

def save_plot(motions, fps_vals, out_path):
    plt.figure(figsize=(10,4))
    frames = np.arange(1, len(motions)+1)
    plt.plot(frames, motions, label="Motion (mean abs diff)")
    if fps_vals:
        plt.plot(frames, fps_vals, label="Processing FPS")
    plt.xlabel("Frame")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()

def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        return

    # Setup output folder structure
    if args.output:
        out_root = Path(args.output)
    else:
        out_root = Path(__file__).resolve().parent / "output_realtime"
    frames_out = out_root / "frames"
    visuals_out = out_root / "visuals"
    ensure_dir(out_root)
    ensure_dir(frames_out)
    ensure_dir(visuals_out)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: Cannot open video file.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"Loaded video: {video_path} — total_frames={total_frames}, fps={input_fps:.2f}")

    # Buffers & params
    motion_threshold = args.motion_threshold
    fast_skip = max(1, args.fast_skip)
    slow_skip = max(1, args.slow_skip)
    max_frames = args.max_frames

    prev_gray = None
    frame_idx = 0
    saved_idx = 0
    motions = []
    speeds = []
    fps_vals = []
    timestamps = []

    prev_time = time.time()
    start_time = prev_time

    # Process loop
    while True:
        if frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        ts = time.time()
        timestamps.append(ts)

        # convert to gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            # treat first frame as non-key by default (or you may choose to save it)
            motions.append(0.0)
            speeds.append(0.0)
            fps_vals.append(0.0)
            # prepare next iteration
            prev_time = ts
            continue

        diff = cv2.absdiff(gray, prev_gray)
        motion = float(np.mean(diff))
        motions.append(motion)

        # speed estimation (relative / heuristic)
        speed = motion * 1.2
        speeds.append(speed)

        # processing FPS estimate
        curr_time = ts
        fps_live = 1.0 / (curr_time - prev_time) if (curr_time > prev_time) else 0.0
        fps_vals.append(fps_live)
        prev_time = curr_time

        # decide skip rate
        skip_rate = fast_skip if motion > motion_threshold else slow_skip
        is_key = (frame_idx % skip_rate) == 0

        # Save keyframe
        if is_key:
            key_path = frames_out / f"frame_{saved_idx:04d}.jpg"
            cv2.imwrite(str(key_path), frame)
            saved_idx += 1

        # Save overlay visualization if requested
        if args.save_visuals:
            text_lines = [
                f"Frame: {frame_idx}/{total_frames or '??'}",
                f"Decision: {'KEYFRAME' if is_key else 'SKIP'} (skip_rate={skip_rate})",
                f"Motion: {motion:.2f}",
                f"Speed (est): {speed:.2f}",
                f"FPS: {fps_live:.2f}"
            ]
            vis = draw_overlay(frame, text_lines)
            vis_path = visuals_out / f"aks_vis_{frame_idx:04d}.jpg"
            cv2.imwrite(str(vis_path), vis)

        # progress printing every 50 frames
        if frame_idx % 50 == 0:
            print(f"[{frame_idx}] motion={motion:.2f} fps={fps_live:.2f} saved_keyframes={saved_idx}")

        prev_gray = gray

    cap.release()
    end_time = time.time()

    # Summary
    elapsed = end_time - start_time
    avg_fps = float(np.mean(fps_vals)) if fps_vals else 0.0
    avg_motion = float(np.mean(motions)) if motions else 0.0
    saved_ratio = (saved_idx / (frame_idx or 1)) * 100.0

    status = {
        "video": str(video_path),
        "total_input_frames": frame_idx,
        "input_fps": input_fps,
        "keyframes_saved": saved_idx,
        "saved_ratio_percent": saved_ratio,
        "avg_motion": avg_motion,
        "avg_processing_fps": avg_fps,
        "motion_threshold": motion_threshold,
        "fast_skip": fast_skip,
        "slow_skip": slow_skip,
        "output_folder": str(out_root),
        "frames_folder": str(frames_out),
        "visuals_folder": str(visuals_out) if args.save_visuals else None,
        "elapsed_seconds": elapsed
    }

    status_path = out_root / "status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)

    print("\n=== AKS SUMMARY ===")
    print(f"Processed frames: {frame_idx}")
    print(f"Keyframes saved: {saved_idx} ({saved_ratio:.2f}%)")
    print(f"Avg processing FPS: {avg_fps:.2f}")
    print(f"Avg motion: {avg_motion:.2f}")
    print(f"Output folder: {out_root}")
    print(f"Status file: {status_path}")
    print("===================")

    # optional: save plot
    if args.save_plot:
        plot_path = out_root / "motion_fps_plot.png"
        try:
            save_plot(motions, fps_vals, plot_path)
            print(f"Saved plot: {plot_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

if __name__ == "__main__":
    main()
