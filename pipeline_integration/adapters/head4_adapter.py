# pipeline_integration/adapters/head4_adapter.py
"""
Head-4 adapter — Road mask & segmentation integration.

What it does:
- Copies frames from storage/<job>/frames -> H4RoadMask/input
- Runs H4RoadMask/mask.py (the script you provided)
- Collects annotated images and status JSON from H4RoadMask/output -> storage/<job>/head4/
- Writes a simple aggregated status.json in storage/<job>/head4/

Usage:
python pipeline_integration/adapters/head4_adapter.py --job demo1 --root storage --max 200
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import json
import time
import os

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_script(cmd, timeout=600):
    print("Running:", " ".join(cmd))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
    parser.add_argument("--job", default="test1")
    parser.add_argument("--root", default="storage")
    parser.add_argument("--max", type=int, default=500)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    job_dir = Path(args.root) / args.job
    frames_dir = job_dir / "frames"
    head4_dir = job_dir / "head4"
    overlay_dir = head4_dir / "overlay"
    status_out_path = head4_dir / "status.json"

    ensure_dir(job_dir)
    ensure_dir(head4_dir)
    ensure_dir(overlay_dir)

    # H4 repo paths
    h4_dir = repo_root / "H4RoadMask"
    h4_input = h4_dir / "input"
    h4_output = h4_dir / "output"
    # mask.py writes annotated images into OUTPUT_FOLDER defined inside its file (H4RoadMask/output)
    # and produces status_data.json inside that folder (as seen in your mask.py)

    if not frames_dir.exists():
        print("ERROR: frames folder not found:", frames_dir)
        return

    # 1) clear H4RoadMask/input and copy frames
    print("Copying frames to H4RoadMask/input ...")
    ensure_dir(h4_input)
    for f in sorted(h4_input.glob("*")):
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
        except Exception:
            pass

    copied = 0
    for p in sorted(frames_dir.glob("*.jpg"))[: args.max]:
        shutil.copyfile(p, h4_input / p.name)
        copied += 1
    print(f"Copied {copied} frames into {h4_input}")

    # 2) Run the mask script
    mask_script = h4_dir / "mask.py"
    if not mask_script.exists():
        print("ERROR: mask.py not found at", mask_script)
        return

    # run mask.py using repo_root as cwd so relative BASE_DIR works nicely
    cmd = [sys.executable, str(mask_script)]
    rc, out, err = run_script(cmd, timeout=900)
    if rc != 0:
        print("mask.py returned non-zero; check stderr above. Continuing to collect any outputs.")

    # 3) collect outputs
    # mask.py writes output images to H4RoadMask/output and a JSON at output/status_data.json
    expected_output_folder = h4_dir / "output"
    json_path = expected_output_folder / "status_data.json"

    # copy annotated images
    if expected_output_folder.exists():
        images = list(expected_output_folder.glob("*_annotated.*"))
        for img in images:
            dst = overlay_dir / img.name
            try:
                shutil.copyfile(img, dst)
            except Exception as e:
                print("Warning copying annotated image:", e)
    else:
        print("Warning: expected output folder not found:", expected_output_folder)

    # read status JSON and write into job folder
    aggregated = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                aggregated = json.load(fh)
        except Exception as e:
            print("Warning reading status_data.json:", e)
    else:
        print("Warning: status_data.json not found in H4RoadMask/output. Creating empty status.")

    # save aggregated status to storage
    out_status = {
        "job": args.job,
        "copied_frames": copied,
        "annotated_count": len(list(overlay_dir.glob("*_annotated.*"))),
        "raw_status": aggregated,
        "output_folder": str(expected_output_folder),
        "timestamp": time.time()
    }
    with open(status_out_path, "w", encoding="utf-8") as fh:
        json.dump(out_status, fh, indent=2)

    print("Head-4 adapter finished. Outputs written to:", overlay_dir)
    print("status:", out_status)

if __name__ == "__main__":
    main()
