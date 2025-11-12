# pipeline_integration/adapters/privacy_adapter.py
"""
Privacy adapter — runs privacyonnx to blur faces/license-plates.
Robust to Unicode in subprocess output (Windows).
Usage:
    python pipeline_integration/adapters/privacy_adapter.py --job demo1 --root storage --max 500
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import json
import os
import sys
# ensure stdout/stderr use utf-8 on Windows so printing special chars (emojis) won't crash

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_script(cmd, cwd=None, timeout=1200):
    """
    Run subprocess and return (returncode, stdout_text, stderr_text).
    Use utf-8 decoding + replace so Windows cp1252 won't raise UnicodeDecodeError.
    """
    print("Running:", " ".join(cmd), "cwd=", cwd)
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd,
            timeout=timeout
        )
    except Exception as e:
        print("Subprocess exception:", e)
        return 1, "", str(e)

    out = res.stdout or ""
    err = res.stderr or ""
    tail_out = out[-2000:]
    tail_err = err[-2000:]
    # safe print of tails
    print("stdout (tail):", tail_out)
    print("stderr (tail):", tail_err)
    return res.returncode, out, err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="demo1", help="job id under storage/")
    parser.add_argument("--root", default="storage", help="root storage folder")
    parser.add_argument("--max", type=int, default=500, help="max frames to process")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    job_dir = Path(args.root) / args.job
    frames_dir = job_dir / "frames"
    out_job_dir = job_dir / "privacy"
    blurred_out = out_job_dir / "blurred"
    ensure_dir(job_dir)
    ensure_dir(out_job_dir)
    ensure_dir(blurred_out)

    # privacy repo locations (assume folder name 'privacy' at repo root)
    privacy_dir = repo_root / "privacy"
    privacy_input = privacy_dir / "input"
    privacy_output = privacy_dir / "output_blur"
    privacy_script = privacy_dir / "privacyonnx.py"

    # safety checks
    if not frames_dir.exists():
        print("ERROR: frames folder not found:", frames_dir)
        # write an empty status and return nonzero so pipeline can decide
        status = {"job": args.job, "error": "frames_not_found", "frames_dir": str(frames_dir), "timestamp": time.time()}
        with open(out_job_dir / "status.json", "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        return 1

    # clear privacy/input and copy frames (up to max)
    ensure_dir(privacy_input)
    for f in sorted(privacy_input.glob("*")):
        try:
            if f.is_file():
                f.unlink()
            else:
                shutil.rmtree(f)
        except Exception:
            pass

    copied = 0
    for i, p in enumerate(sorted(frames_dir.glob("*"))):
        if i >= args.max:
            break
        # only copy common image/video types
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi"]:
            dst = privacy_input / p.name
            try:
                shutil.copyfile(p, dst)
                copied += 1
            except Exception as e:
                print("Warning copying frame", p, e)
    print(f"Copied {copied} frames into {privacy_input}")

    # run privacy script (in privacy folder so relative paths resolve)
    if not privacy_script.exists():
        print("ERROR: privacyonnx.py not found at", privacy_script)
        status = {"job": args.job, "error": "privacy_script_missing", "timestamp": time.time()}
        with open(out_job_dir / "status.json", "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        return 1

    # run with cwd=privacy_dir so script can use relative model paths
    cmd = [sys.executable, str(privacy_script)]
    rc, out, err = run_script(cmd, timeout=1800, cwd=str(privacy_dir))
    if rc != 0:
        print("privacyonnx.py returned non-zero code:", rc)
        # continue to collect outputs (some partial outputs may exist)

    # collect blurred outputs (if any)
    collected = 0
    if privacy_output.exists():
        for f in sorted(privacy_output.glob("*")):
            if f.is_file():
                try:
                    dest = blurred_out / f.name
                    shutil.copyfile(f, dest)
                    collected += 1
                except Exception as e:
                    print("Warning copying", f, e)
    else:
        print("Warning: privacy output folder not found:", privacy_output)

    # write status.json
    status = {
        "job": args.job,
        "frames_copied": copied,
        "blurred_collected": collected,
        "privacy_input": str(privacy_input),
        "privacy_output": str(privacy_output),
        "collected_to": str(blurred_out),
        "rc": rc,
        "timestamp": time.time()
    }
    with open(out_job_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)

    if rc != 0:
        print("Privacy adapter finished with errors. See status.json for details.")
        return rc

    print("Privacy adapter finished successfully. Collected", collected, "blurred images into", blurred_out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
