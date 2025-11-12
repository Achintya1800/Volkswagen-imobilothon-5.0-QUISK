# pipeline_integration/adapters/head1_adapter.py
"""
Head-1 adapter
- Runs repository detection scripts (batch_interference_v10.py or interference_v10.py)
- Collects produced overlays/detections and copies them into storage/<job>/head1 and head1/overlay
Usage:
    python pipeline_integration/adapters/head1_adapter.py --job <job> --root storage --max 1000
"""
import argparse
from pathlib import Path
import subprocess
import sys
import shutil
import json
import time
import glob
import sys
import subprocess
from pathlib import Path

# ensure utf-8 prints on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def run_script(cmd, cwd=None, timeout=900):
    print("Running:", " ".join(cmd), "cwd=", cwd)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd, timeout=timeout)
    except Exception as e:
        print("Subprocess exception:", e)
        return 1, "", str(e)
    out = res.stdout or ""
    err = res.stderr or ""
    print("Subprocess stdout (tail):", out[-800:])
    print("Subprocess stderr (tail):", err[-800:])
    return res.returncode, out, err

def collect_outputs(repo_root: Path, job_dir: Path, candidates):
    """
    Search repository and candidate directories for probable outputs and copy into job_dir/head1
    """
    head1_dir = job_dir / "head1"
    overlay_dir = head1_dir / "overlay"
    head1_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # patterns to search for (relative to repo root and candidate script locations)
    patterns = [
        "**/detections.json",
        "**/detections_v10.json",
        "**/detections_*.json",
        "**/overlay_frame_*.jpg",
        "**/detected_frame_*.jpg",
        "**/overlay/*.jpg",
        "**/overlay_frame_*.png",
        "**/detected_frame_*.png",
    ]

    found_json = None
    copied_images = 0
    # search repo-wide first
    for pat in patterns:
        for p in repo_root.glob(pat):
            p = p.resolve()
            # skip if file is already inside storage/<job>/head1 (avoid copying twice)
            if str(p).startswith(str(head1_dir.resolve())):
                continue
            if p.suffix.lower() in [".json"]:
                if found_json is None:
                    # copy detections file
                    dest = head1_dir / "detections.json"
                    try:
                        shutil.copyfile(p, dest)
                        found_json = dest
                        print("Copied detections json from:", p, "->", dest)
                    except Exception as e:
                        print("Failed copy json", p, e)
            else:
                # images: copy to overlay dir
                try:
                    dest = overlay_dir / p.name
                    shutil.copyfile(p, dest)
                    copied_images += 1
                except Exception as e:
                    print("Failed copy image", p, e)

    # Also inspect candidate script folders (where dynamic import/subprocess may produce outputs)
    for script in candidates:
        script_dir = script.resolve().parent
        for pat in patterns:
            for p in script_dir.glob(pat):
                if str(p).startswith(str(head1_dir.resolve())):
                    continue
                if p.suffix.lower() == ".json" and found_json is None:
                    try:
                        dest = head1_dir / "detections.json"
                        shutil.copyfile(p, dest)
                        found_json = dest
                        print("Copied detections json from script folder:", p, "->", dest)
                    except Exception as e:
                        print("Failed copy json from script folder", p, e)
                else:
                    try:
                        dest = overlay_dir / p.name
                        shutil.copyfile(p, dest)
                        copied_images += 1
                    except Exception as e:
                        print("Failed copy image from script folder", p, e)

    # If nothing found, we will create a minimal detections.json (empty)
    if found_json is None:
        print("No detections.json found; writing an empty detections.json to allow fusion to run.")
        minimal = {}
        with open(head1_dir / "detections.json", "w", encoding="utf-8") as fh:
            json.dump(minimal, fh, indent=2)
        found_json = head1_dir / "detections.json"

    print(f"Collected images: {copied_images}, detections.json at {found_json}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", required=True, help="job id under storage/")
    parser.add_argument("--root", default="storage", help="root storage folder")
    parser.add_argument("--max", type=int, default=500, help="max frames to process")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]  # repo / pipeline_integration
    # climb to repo root
    repo_root = repo_root.parent
    job_dir = Path(args.root) / args.job
    print("Repository root:", repo_root)
    print("Job dir:", job_dir)

    # find candidate detector scripts
    candidates = list(repo_root.glob("**/batch_interference_v10.py")) + list(repo_root.glob("**/interference_v10.py"))
    print("Found candidate detector scripts:", candidates)

    any_success = False
    # try to run subprocess if script exists (prefer batch_interference)
    for script in candidates:
        script_path = script.resolve()
        # prepare frames/input and outputs for the detector script
        # use job/root to compute paths
        frames_dir = (Path(args.root) / args.job / "frames").resolve()
        detector_output = (job_dir / "head1" / "detector_out").resolve()
        detector_output.mkdir(parents=True, exist_ok=True)

        # quick sanity check: ensure frames exist before calling detector scripts
        image_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
        if not frames_dir.exists() or len(image_files) == 0:
            print("No frames found in", frames_dir, "— skipping head1 detector and writing empty detections.json")
            # ensure head1 output dirs exist and write empty detections.json so downstream stages can continue
            head1_dir = job_dir / "head1"
            overlay_dir = head1_dir / "overlay"
            head1_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)
            det_file = head1_dir / "detections.json"
            with open(det_file, "w", encoding="utf-8") as fh:
                json.dump({}, fh, indent=2)
            print("Wrote empty detections.json to", det_file)
            rc = 0
            any_success = True
            # no candidate run — collect_outputs will find nothing but keep structure
            collect_outputs(repo_root, job_dir, candidates)
            break

        # try to auto-discover a model in the script folder (common places)
        model_candidates = []
        cand_dirs = [
            script_path.parent,
            script_path.parent / "model",
            repo_root / "head1" / "interference",
            repo_root / "models",
            repo_root / "pipeline_integration" / "adapters" / "model",
        ]
        for d in cand_dirs:
            d = Path(d)
            if d.exists():
                model_candidates += list(d.glob("*.pt")) + list(d.glob("*.onnx")) + list(d.glob("*.pth"))

        # Auto-discover model (always use discovery, no explicit --model flag)
        model_file = model_candidates[0] if model_candidates else None
        if model_file:
            print("Auto-discovered model for detector:", model_file)
        else:
            print("No model file found near detector. Will attempt to run but script may require --model flag.")

        # Build command according to script name and discovered model
        if "batch_interference" in script_path.name.lower():
            # this script expects --model --input --output (batch mode)
            # build command and forward user-supplied model/input/output if present
            cmd = [sys.executable, str(script_path)]
            if model_file is not None:
                cmd += ["--model", str(model_file)]
            if frames_dir is not None:
                cmd += ["--input", str(frames_dir)]
            if detector_output is not None:
                cmd += ["--output", str(detector_output)]
            # If no model was found or supplied, warn but still try
            if model_file is None:
                print("WARNING: No model provided to batch_interference; subprocess may fail with usage error.")
            print("Attempting subprocess run (batch mode):", " ".join(cmd))
            rc, out, err = run_script(cmd, cwd=str(script_path.parent), timeout=900)
        elif "interference_v10" in script_path.name.lower() or "interference" in script_path.name.lower():
            # older script expects single image mode (--image). We'll try to call batch if available OR
            # if model exists, call the script for each frame (slow), else skip.
            if (script_path.parent / "batch_interference_v10.py").exists():
                # prefer batch script in same folder
                batch_script = script_path.parent / "batch_interference_v10.py"
                # build batch command and forward model/input/output if present
                cmd = [sys.executable, str(batch_script)]
                if model_file is not None:
                    cmd += ["--model", str(model_file)]
                if frames_dir is not None:
                    cmd += ["--input", str(frames_dir)]
                if detector_output is not None:
                    cmd += ["--output", str(detector_output)]
                if model_file is None:
                    print("WARNING: batch_interference found but no model detected. Will attempt anyway (may fail).")
                print("Attempting subprocess run (batch script found):", " ".join(cmd))
                rc, out, err = run_script(cmd, cwd=str(batch_script.parent), timeout=1200)
            else:
                # fallback: call interference_v10 per-image (only if model present)
                if model_file is None:
                    print("No model found and no batch script — skipping this detector.")
                    rc = 2
                else:
                    # build a command that processes all images by invoking script once per image
                    image_files = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
                    rc = 0
                    for img in image_files:
                        cmd = [sys.executable, str(script_path),
                               "--model", str(model_file),
                               "--image", str(img),
                               "--output", str(detector_output)]
                        print("Running per-image detector:", " ".join(cmd))
                        _rc, out, err = run_script(cmd, cwd=str(script_path.parent), timeout=600)
                        if _rc != 0:
                            print("Per-image detector failed for", img, "rc=", _rc)
                            rc = _rc
                            # continue to try other images but record failure
        else:
            # unknown script type — try a generic invocation passing job/root
            cmd = [sys.executable, str(script_path), "--job", args.job, "--root", args.root]
            print("Attempting generic subprocess run:", " ".join(cmd))
            rc, out, err = run_script(cmd, cwd=str(script_path.parent), timeout=900)

        if rc == 0:
            print("Subprocess succeeded for:", script_path)
            any_success = True
            # after success, collect outputs
            collect_outputs(repo_root, job_dir, candidates)
            break
        else:
            print("Subprocess returned non-zero rc:", rc, "for", script_path)
            # try next candidate


    # If none succeeded, attempt dynamic import + call (best-effort)
    if not any_success and candidates:
        for script in candidates:
            try:
                # dynamic import
                import importlib.util
                spec = importlib.util.spec_from_file_location(script.stem, str(script))
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                # try some common entrypoints
                if hasattr(mod, "main"):
                    print("Calling dynamic main() for", script)
                    try:
                        mod.main(job=args.job, root=args.root)
                    except TypeError:
                        # fallback: call with no args
                        mod.main()
                    any_success = True
                    collect_outputs(repo_root, job_dir, candidates)
                    break
                elif hasattr(mod, "RoadHazardDetectorV10"):
                    print("Instantiating RoadHazardDetectorV10")
                    try:
                        detector = getattr(mod, "RoadHazardDetectorV10")()
                        if hasattr(detector, "run_batch"):
                            detector.run_batch(str(job_dir))
                        any_success = True
                        collect_outputs(repo_root, job_dir, candidates)
                        break
                    except Exception as e:
                        print("Dynamic call failed:", e)
                else:
                    print("No known entrypoint in", script)
            except Exception as e:
                print("Dynamic import failed for", script, e)

    # final fallback: if nothing ran, still ensure head1 dir exists and write empty detections.json
    head1_dir = job_dir / "head1"
    overlay_dir = head1_dir / "overlay"
    head1_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    det_file = head1_dir / "detections.json"
    if not det_file.exists():
        with open(det_file, "w", encoding="utf-8") as fh:
            json.dump({}, fh, indent=2)

    print("Head1 adapter finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
