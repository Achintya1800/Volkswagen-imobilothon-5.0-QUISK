# pipeline_integration/adapters/stalled_adapter.py
"""
Adapter for stalledVehicle/stalled.py

Creates a temporary video from frames, runs stalled.py, collects outputs.

Usage:
python pipeline_integration/adapters/stalled_adapter.py --job demo1 --root storage --fps 10
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import json
import time
import cv2
import os
import glob

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def frames_to_video(frames_dir: Path, out_video: Path, fps=10):
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return False, "no_frames"
    # read first frame to get size
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False, "cannot_read_first_frame"
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_video), fourcc, float(fps), (w, h))
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if (img.shape[1], img.shape[0]) != (w, h):
            img = cv2.resize(img, (w, h))
        vw.write(img)
    vw.release()
    return True, None

def run_script(cmd, cwd=None, timeout=1200):
    print("Running:", " ".join(cmd), "cwd=", cwd)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd, timeout=timeout)
    except Exception as e:
        print("Subprocess exception:", e)
        return 1, "", str(e)
    out = res.stdout or ""
    err = res.stderr or ""
    print("stdout (tail):", out[-800:])
    print("stderr (tail):", err[-800:])
    return res.returncode, out, err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="demo1")
    parser.add_argument("--root", default="storage")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max", type=int, default=1000)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    job_dir = Path(args.root) / args.job
    frames_dir = job_dir / "frames"
    stalled_out_dir = job_dir / "stalled"
    ensure_dir(stalled_out_dir)

    stalled_repo = repo_root / "stalledVehicle"
    stalled_input = stalled_repo / "input"
    stalled_output = stalled_repo / "output"

    ensure_dir(stalled_repo)
    ensure_dir(stalled_input)
    ensure_dir(stalled_output)

    if not frames_dir.exists():
        print("ERROR: frames folder not found:", frames_dir)
        return

    # clear stalled input and output folders
    for f in stalled_input.glob("*"):
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
        except Exception:
            pass
    for f in stalled_output.glob("*"):
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
        except Exception:
            pass

    # Copy up to max frames to local tmp folder OR build video
    tmp_video = stalled_input / f"input_for_{args.job}.mp4"
    ok, err = frames_to_video(frames_dir, tmp_video, fps=args.fps)
    if not ok:
        print("Failed creating video from frames:", err)
        return
    print("Wrote temporary video:", tmp_video)

    # Ensure stalled.py will see file: stalled.py looks into INPUT_DIR for .mp4 files
    # Leave the tmp_video in stalled_input and call stalled.py with cwd=stalled_repo
    # Run stalled.py
    stalled_script = stalled_repo / "stalled.py"
    if not stalled_script.exists():
        print("ERROR: stalled.py not found at", stalled_script)
        return

    rc, out, err = run_script([sys.executable, str(stalled_script)], cwd=str(stalled_repo), timeout=1800)
    if rc != 0:
        print("stalled.py returned non-zero code:", rc)
        # continue to collect whatever outputs exist

    # Collect output video(s) from stalledVehicle/output -> storage/<job>/stalled
    collected = 0
    if stalled_output.exists():
        for f in sorted(stalled_output.glob("*")):
            if f.is_file():
                try:
                    dst = stalled_out_dir / f.name
                    shutil.copyfile(f, dst)
                    collected += 1
                except Exception as e:
                    print("Warning copying output:", e)
    else:
        print("Warning: stalled output folder not found:", stalled_output)

    # Optionally, you can try to parse logs or produce per-track JSON, but for now we copy outputs
    status = {
        "job": args.job,
        "video_created": str(tmp_video),
        "collected_outputs": collected,
        "stalled_output_folder": str(stalled_output),
        "collected_to": str(stalled_out_dir),
        "timestamp": time.time()
    }
    with open(stalled_out_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)

    print("Stalled adapter finished. Collected", collected, "files into", stalled_out_dir)

if __name__ == "__main__":
    main()
