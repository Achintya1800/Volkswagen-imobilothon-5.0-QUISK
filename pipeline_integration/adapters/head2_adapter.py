# pipeline_integration/adapters/head2_adapter.py
"""
Head-2 adapter: runs segmentation then depth on the AKS frames.

Workflow:
- Copies frames from storage/<job>/frames -> H2Segmentation/input
- Runs H2Segmentation/Segment.py (YOLOv8Seg) to produce segmented images in segmented_output/segmented
- Copies segmented images back to storage/<job>/head2/segmented
- Copies segmented images into H2Segmentation/input (overwrite) and runs Run_midas.py to compute depth/area outputs
- Collects depth outputs from H2Segmentation/depth_area_output -> storage/<job>/head2/depth
- Writes status JSON
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import json
import os
from pathlib import Path
import cv2
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Head2 Adapter — Segmentation + Depth")
parser.add_argument("--job", required=True, help="Job ID folder name under storage/")
parser.add_argument("--root", required=False, default="storage", help="Root folder for storage")
parser.add_argument("--max", type=int, default=1000, help="Max frames to process")
args = parser.parse_args()
# --- repo-relative model path + UTF-8 stdout fix ---


# Allow safe UTF-8 console printing on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Compute correct path to H2Segmentation/model/YoloV8Segmented.pt
HERE = Path(__file__).resolve().parent
# two levels up from adapters → pipeline_integration → project root
MODEL_DIR = HERE.parent.parent / "H2Segmentation" / "model"
MODEL_FILE = MODEL_DIR / "YoloV8Segmented.pt"  # exact filename from your dir output

print("Resolved model path:", MODEL_FILE)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Segmentation model not found at {MODEL_FILE!s}. "
        f"Please ensure YoloV8Segmented.pt exists in {MODEL_DIR!s}"
    )
# make sure folders exist
root_dir = Path(args.root) if hasattr(args, "root") else Path("storage")
job_id = args.job if hasattr(args, "job") else "default_job"

# Final output folder for Head2 results
output_folder = root_dir / job_id / "head2"
os.makedirs(output_folder, exist_ok=True)
print(f"Head2 output folder set to: {output_folder}")
os.makedirs(output_folder, exist_ok=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd, timeout=600):
    """
    Run a subprocess and return (returncode, stdout, stderr).
    Uses utf-8 decoding with errors='replace' to avoid UnicodeDecodeError on Windows.
    """
    print("Running:", " ".join(cmd))
    try:
        # capture_output + text + encoding/errors makes decoding robust
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout
        )
    except Exception as e:
        # subprocess failed to start or timed out
        print("Subprocess execution exception:", e)
        return 1, "", str(e)

    out = res.stdout or ""
    err = res.stderr or ""
    # safe tail-print
    print("stdout (tail):", out[-400:])
    print("stderr (tail):", err[-400:])
    return res.returncode, out, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="test1")
    parser.add_argument("--root", default="storage")
    parser.add_argument("--max", type=int, default=500)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    job_dir = Path(args.root) / args.job
    frames_dir = job_dir / "frames"
    head2_dir = job_dir / "head2"
    seg_out_storage = head2_dir / "segmented"
    depth_out_storage = head2_dir / "depth"

    # H2Segmentation paths (in repo)
    h2_dir = repo_root / "H2Segmentation"
    h2_input = h2_dir / "input"
    h2_segmented_out = h2_dir / "segmented_output" / "segmented"
    h2_depth_out = h2_dir / "depth_area_output"

    ensure_dir(head2_dir)
    ensure_dir(seg_out_storage)
    ensure_dir(depth_out_storage)
    ensure_dir(h2_input)

    if not frames_dir.exists():
        print("ERROR: frames folder not found:", frames_dir)
        return

    # 1) copy up to max frames into H2Segmentation/input
    print("Copying frames to H2Segmentation/input ...")
    # clear input folder first
    for f in h2_input.glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception as e:
            print("Warning clearing input:", e)
    copied = 0
    for p in sorted(frames_dir.glob("frame_*.jpg"))[: args.max]:
        shutil.copyfile(p, h2_input / p.name)
        copied += 1
    print(f"Copied {copied} frames into {h2_input}")

    # 2) Run segmentation script
    seg_script = repo_root / "H2Segmentation" / "Segment.py"
    if not seg_script.exists():
        print("ERROR: Segment.py not found at", seg_script)
    else:
        cmd = [sys.executable, str(seg_script)]
        # run; script uses internal hardcoded paths relative to H2Segmentation
        rc, out, err = run_cmd(cmd, timeout=600)
        if rc != 0:
            print("Segmentation script returned non-zero:", rc)

    # 3) Collect segmented output
    if h2_segmented_out.exists():
        print("Collecting segmented images to storage ...")
        for f in sorted(h2_segmented_out.glob("*")):
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                dst = seg_out_storage / f.name
                shutil.copyfile(f, dst)
        seg_count = len(list(seg_out_storage.glob("*.*")))
        print("Collected segmented images:", seg_count)
    else:
        print("Warning: segmentation output folder not found:", h2_segmented_out)

    # 4) Prepare Run_midas input: copy segmented images to H2Segmentation/input (overwrite)
    print("Preparing input for Run_midas (copy segmented images into H2Segmentation/input)...")
    # clear input folder then copy segmented images if any; else reuse original frames
    for f in h2_input.glob("*"):
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception:
            pass
    src_for_midas = seg_out_storage if seg_out_storage.exists() and any(seg_out_storage.iterdir()) else frames_dir
    for f in sorted(src_for_midas.glob("*"))[: args.max]:
        shutil.copyfile(f, h2_input / f.name)
    print("Copied images for midas:", len(list(h2_input.glob("*"))))

    # 5) Run Run_midas.py
    midas_script = repo_root / "H2Segmentation" / "Run_midas.py"
    if not midas_script.exists():
        print("ERROR: Run_midas.py not found at", midas_script)
    else:
        cmd2 = [sys.executable, str(midas_script)]
        rc2, out2, err2 = run_cmd(cmd2, timeout=600)
        if rc2 != 0:
            print("Run_midas script returned non-zero:", rc2)

    # 6) Collect midas outputs
    if h2_depth_out.exists():
        for f in sorted(h2_depth_out.glob("*")):
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                dst = depth_out_storage / f.name
                shutil.copyfile(f, dst)
        depth_count = len(list(depth_out_storage.glob("*.*")))
        print("Collected depth outputs:", depth_count)
    else:
        print("Warning: midas output folder not found:", h2_depth_out)

    # 7) Write status
    status = {
        "job": args.job,
        "segmented_count": len(list(seg_out_storage.glob("*.*"))),
        "depth_outputs": len(list(depth_out_storage.glob("*.*"))),
        "seg_storage": str(seg_out_storage),
        "depth_storage": str(depth_out_storage),
        "h2_input": str(h2_input),
        "timestamp": time.time()
    }
    with open(head2_dir / "status.json", "w") as fh:
        json.dump(status, fh, indent=2)

    print("Head-2 adapter finished. Status:", status)

if __name__ == "__main__":
    main()
