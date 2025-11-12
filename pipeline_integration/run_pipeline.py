# pipeline_integration/run_pipeline.py
"""
Run the demo pipeline in order:
  1) AKS/keyframe selector (ImageCRC -> writes frames to storage/<job>/frames)
  2) Dehaze (pipeline_integration/adapters/dehaze_adapter.py)
  3) Head1 detection (pipeline_integration/adapters/head1_adapter.py)
  4) Head2 segmentation+depth (pipeline_integration/adapters/head2_adapter.py)
  5) Head4 road mask, Head5 stalled, Privacy etc (adapters already made)
Usage:
python pipeline_integration/run_pipeline.py --video "path/to/video.mp4" --job demo1
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time

def run_cmd(cmd, timeout=1200):
    print("\n>>> RUN:", " ".join(cmd))
    try:
        # force utf-8 decoding of stdout/stderr and replace invalid bytes to avoid
        # UnicodeDecodeError on Windows consoles using cp1252
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except Exception as e:
        print("Command exception:", e)
        return 1, "", str(e)
    out = res.stdout or ""
    err = res.stderr or ""
    print("---- stdout (tail) ----")
    try:
        # write bytes directly to avoid UnicodeEncodeError when console encoding
        # cannot represent certain characters (Windows cp1252)
        sys.stdout.buffer.write(out[-800:].encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")
    except Exception:
        # fallback to safe print
        print(out[-800:].encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    print("---- stderr (tail) ----")
    try:
        sys.stderr.buffer.write(err[-800:].encode("utf-8", errors="replace"))
        sys.stderr.buffer.write(b"\n")
    except Exception:
        print(err[-800:].encode("utf-8", errors="replace").decode("utf-8", errors="replace"))
    return res.returncode, out, err

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--job", default="demo1", help="Job id (storage/<job>)")
    parser.add_argument("--max-frames", type=int, default=500, help="max frames for AKS/imageCRC")
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    storage_job = repo_root / "storage" / args.job
    storage_job.mkdir(parents=True, exist_ok=True)

    # 1) AKS / ImageCRC: write keyframes directly into storage/<job>/frames
    frames_out = storage_job / "frames"
    frames_out.mkdir(parents=True, exist_ok=True)
    imagecrc_script = repo_root / "ImageCRC" / "imageCRC.py"
    if not imagecrc_script.exists():
        print("ERROR: ImageCRC script not found:", imagecrc_script)
        return 1

    cmd1 = [sys.executable, str(imagecrc_script),
            "--video", str(args.video),
            "--output", str(frames_out)]
    rc, _, _ = run_cmd(cmd1, timeout=600)
    if rc != 0:
        print("ImageCRC failed (rc=%d). Aborting pipeline." % rc)
        return rc
    print("AKS done: frames in", frames_out)

    # 2) Dehaze adapter
    dehaze_adapter = repo_root / "pipeline_integration" / "adapters" / "dehaze_adapter.py"
    if dehaze_adapter.exists():
        cmd2 = [sys.executable, str(dehaze_adapter), "--job", args.job, "--root", "storage", "--max", "1000"]
        rc, _, _ = run_cmd(cmd2, timeout=900)
        if rc != 0:
            print("Dehaze adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Dehaze done: outputs in", storage_job / "dehaze")
    else:
        print("Dehaze adapter not found, skipping.")

    # 3) Head1 adapter (detection)
    head1_adapter = repo_root / "pipeline_integration" / "adapters" / "head1_adapter.py"
    if head1_adapter.exists():
        cmd3 = [sys.executable, str(head1_adapter), "--job", args.job, "--root", "storage", "--max", "1000"]
        rc, _, _ = run_cmd(cmd3, timeout=900)
        if rc != 0:
            print("Head1 adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Head1 done: overlays+detections in", storage_job / "head1")
    else:
        print("Head1 adapter not found, skipping.")

    # 4) Head2 adapter (segmentation + depth)
    head2_adapter = repo_root / "pipeline_integration" / "adapters" / "head2_adapter.py"
    if head2_adapter.exists():
        cmd4 = [sys.executable, str(head2_adapter), "--job", args.job, "--root", "storage", "--max", "1000"]
        rc, _, _ = run_cmd(cmd4, timeout=1200)
        if rc != 0:
            print("Head2 adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Head2 done: segmented+depth in", storage_job / "head2")
    else:
        print("Head2 adapter not found, skipping.")

    # 5) Head4 adapter
    head4_adapter = repo_root / "pipeline_integration" / "adapters" / "head4_adapter.py"
    if head4_adapter.exists():
        cmd5 = [sys.executable, str(head4_adapter), "--job", args.job, "--root", "storage", "--max", "1000"]
        rc, _, _ = run_cmd(cmd5, timeout=900)
        if rc != 0:
            print("Head4 adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Head4 done: outputs in", storage_job / "head4")
    else:
        print("Head4 adapter not found, skipping.")

    # 6) Stalled adapter
    stalled_adapter = repo_root / "pipeline_integration" / "adapters" / "stalled_adapter.py"
    if stalled_adapter.exists():
        cmd6 = [sys.executable, str(stalled_adapter), "--job", args.job, "--root", "storage", "--fps", "10"]
        rc, _, _ = run_cmd(cmd6, timeout=1200)
        if rc != 0:
            print("Stalled adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Stalled adapter done: outputs in", storage_job / "stalled")
    else:
        print("Stalled adapter not found, skipping.")

    # 7) Privacy adapter
    privacy_adapter = repo_root / "pipeline_integration" / "adapters" / "privacy_adapter.py"
    if privacy_adapter.exists():
        cmd7 = [sys.executable, str(privacy_adapter), "--job", args.job, "--root", "storage", "--max", "1000"]
        rc, _, _ = run_cmd(cmd7, timeout=900)
        if rc != 0:
            print("Privacy adapter failed (rc=%d). Aborting pipeline." % rc)
            return rc
        print("Privacy adapter done: outputs in", storage_job / "privacy")
    else:
        print("Privacy adapter not found, skipping.")

    # 8) Fusion engine
    fusion_script = repo_root / "pipeline_integration" / "fusion_engine.py"
    if fusion_script.exists():
        cmdf = [sys.executable, str(fusion_script), "--job", args.job, "--root", "storage"]
        rc, _, _ = run_cmd(cmdf, timeout=300)
        if rc != 0:
            print("Fusion engine failed (rc=%d)." % rc)
            return rc
        print("Fusion done: outputs in", storage_job / "fusion")
    else:
        print("Fusion script not found, skipping fusion.")

    print("PIPELINE FINISHED SUCCESSFULLY")
    return 0

if __name__ == "__main__":
    main()
