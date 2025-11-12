# orchestrator.py
import argparse
from pathlib import Path
import cv2
import json
import sys

def extract_frames(video_path: str, out_dir: Path, max_frames: int = 5):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = []
    frame_id = 0
    saved_count = 0

    while saved_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        # Save the frame as jpg
        out_path = out_dir / f"frame_{frame_id:04d}.jpg"
        # cv2.imwrite expects BGR array; frame is as read
        cv2.imwrite(str(out_path), frame)
        saved.append(str(out_path))
        saved_count += 1

    cap.release()
    return saved

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video (mp4)")
    parser.add_argument("--job", default="test1", help="Job id / folder name")
    parser.add_argument("--max-frames", type=int, default=5, help="Number of frames to extract")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video path does not exist: {video_path}", file=sys.stderr)
        sys.exit(1)

    storage_root = Path("storage")
    job_dir = storage_root / args.job
    frames_dir = job_dir / "frames"
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting up to {args.max_frames} frames from {video_path} into {frames_dir} ...")
    saved_files = extract_frames(str(video_path), frames_dir, max_frames=args.max_frames)
    print(f"Saved {len(saved_files)} frames.")

    status = {
        "job": args.job,
        "video": str(video_path),
        "frames_saved": len(saved_files),
        "frames": saved_files
    }
    status_path = job_dir / "status.json"
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote status file: {status_path}")
    print("Done.")

if __name__ == "__main__":
    main()
    