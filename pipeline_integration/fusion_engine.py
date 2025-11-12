# pipeline_integration/fusion_engine.py
"""
Fusion engine: combine head1/head2/head4/head5/privacy outputs into final_report.json
Usage:
    python pipeline_integration/fusion_engine.py --job demo1 --root storage
"""
import argparse
from pathlib import Path
import json
import math
import statistics

def load_json_safe(p: Path):
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        print("Failed to read json", p, e)
        return None

def find_depth_jsons(head2_dir: Path):
    out = []
    ddir = head2_dir / "depth"
    if not ddir.exists():
        return out
    for p in sorted(ddir.glob("*.json")):
        out.append((p.name, load_json_safe(p)))
    return out

def safe_get(dct, key, default=None):
    if not dct:
        return default
    return dct.get(key, default)

def bbox_area(bbox):
    # bbox expected [x1,y1,x2,y2]
    try:
        x1,y1,x2,y2 = bbox
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return w*h
    except Exception:
        return 0

def score_to_label(s):
    if s >= 0.75:
        return "High"
    if s >= 0.4:
        return "Medium"
    return "Low"

def normalize_depth_value(v, minv=0.0, maxv=3.0):
    # depth normalization assumption: depth values in meters; clip
    try:
        x = float(v)
    except Exception:
        return 0.5
    x = max(minv, min(maxv, x))
    return (x - minv) / (maxv - minv) if maxv>minv else 0.5

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="demo1")
    parser.add_argument("--root", default="storage")
    args = parser.parse_args()

    root = Path(args.root)
    job = args.job
    job_dir = root / job

    # load head1 detections (required)
    head1_json = job_dir / "head1" / "detections.json"
    head1 = load_json_safe(head1_json)
    if head1 is None:
        print("ERROR: head1 detections not found at", head1_json)
        return

    # optional: head4 road mask status
    head4_status = load_json_safe(job_dir / "head4" / "status.json") or {}
    head4_raw = head4_status.get("raw_status", {})

    # optional: stalled
    stalled_status = load_json_safe(job_dir / "stalled" / "status.json") or {}

    # optional: privacy
    privacy_status = load_json_safe(job_dir / "privacy" / "status.json") or {}

    # optional: head2 depth JSONs
    head2_dir = job_dir / "head2"
    depth_jsons = find_depth_jsons(head2_dir)
    depth_map_by_frame = {}
    # if depth jsons exist, try to populate a simple mapping frame->avg_depth_in_bbox per file structure
    for fname, dj in depth_jsons:
        if not dj:
            continue
        # assumed structure: top-level mapping frame-> { maybe detections or depth array }
        # We'll look for keys like frame_0001.jpg -> {"avg_depth": ...} or similar.
        if isinstance(dj, dict):
            for fkey, val in dj.items():
                # make a best-effort: if val has 'avg_depth' or 'depth' or number
                if isinstance(val, dict):
                    if "avg_depth" in val:
                        depth_map_by_frame.setdefault(fkey, []).append(val["avg_depth"])
                    elif "depth" in val and isinstance(val["depth"], (int, float)):
                        depth_map_by_frame.setdefault(fkey, []).append(val["depth"])
                    else:
                        # try to find numeric leaf
                        for kk, vv in val.items():
                            if isinstance(vv, (int, float)):
                                depth_map_by_frame.setdefault(fkey, []).append(vv)
                elif isinstance(val, (int, float)):
                    depth_map_by_frame.setdefault(fkey, []).append(val)
    # collapse to mean
    depth_mean = {}
    for k,v in depth_map_by_frame.items():
        try:
            depth_mean[k] = float(statistics.mean(v))
        except Exception:
            pass

    # Build final report
    fusion_dir = job_dir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)

    final = {}
    per_frame_summary = {}

    for frame_name, dets in head1.items():
        frame_entry = []
        # head4 info per frame (raw_status seems like a mapping frame->list)
        head4_frame_info = head4_raw.get(frame_name, [])
        # head2 depth mean (if present)
        depth_val = depth_mean.get(frame_name, None)

        # stalled check: basic lookup in stalled_status raw JSON if present
        stalled_flag = False
        stalled_raw = stalled_status.get("raw", {}) if isinstance(stalled_status.get("raw", {}), dict) else stalled_status
        # best-effort: check if stalled_status has 'stalled_frames' or similar
        if isinstance(stalled_status, dict):
            # check keys
            if "stalled_frames" in stalled_status:
                if frame_name in stalled_status["stalled_frames"]:
                    stalled_flag = True
        # fallback: check if any filename in stalled output has the frame name
        # (we won't parse videos here)

        # for each detection, compute combined score
        for d in dets:
            bbox = d.get("bbox", d.get("box", d.get("bbox_xyxy", None)))
            base_score = float(d.get("score", d.get("confidence", 0.5)))
            cls = d.get("class_name", d.get("type", "obj"))

            # on-road multiplier: try to find a matching item in head4_frame_info
            on_road_ratio = 1.0
            if isinstance(head4_frame_info, list) and head4_frame_info:
                # head4 raw structure in your screenshots: head4/raw_status -> frame: list of items with on_road_ratio
                # we'll take average of on_road_ratio if multiple entries
                try:
                    vals = [float(x.get("on_road_ratio", 1.0)) for x in head4_frame_info if isinstance(x, dict)]
                    if vals:
                        on_road_ratio = float(sum(vals) / len(vals))
                except Exception:
                    on_road_ratio = 1.0

            # depth multiplier
            depth_mul = 1.0
            if depth_val is not None:
                # normalize depth (assume lower depth => deeper pothole? if your depth is inverse, adapt)
                nd = normalize_depth_value(depth_val, minv=0.0, maxv=3.0)
                # heuristic: deeper (higher normalized) -> higher multiplier
                depth_mul = 0.8 + 0.9 * nd  # in [0.8,1.7]

            # stalled multiplier
            stalled_mul = 1.0
            if stalled_flag:
                stalled_mul += 0.25

            # compute final numeric score
            raw_score = base_score * on_road_ratio * depth_mul * stalled_mul
            score = max(0.0, min(1.0, raw_score))
            label = score_to_label(score)

            frame_entry.append({
                "bbox": bbox,
                "class": cls,
                "base_score": base_score,
                "on_road_ratio": on_road_ratio,
                "depth_mean": depth_val,
                "stalled_flag": stalled_flag,
                "final_score": round(score, 3),
                "severity": label
            })

        # store
        final[frame_name] = frame_entry

        # produce a short human-readable summary
        if frame_entry:
            top = sorted(frame_entry, key=lambda x: x["final_score"], reverse=True)[0]
            per_frame_summary[frame_name] = {
                "top_class": top["class"],
                "top_score": top["final_score"],
                "severity": top["severity"],
                "on_road_ratio": top["on_road_ratio"],
                "depth_mean": top["depth_mean"]
            }
        else:
            per_frame_summary[frame_name] = {"message": "no detections"}

    # write outputs
    final_path = fusion_dir / "final_report.json"
    with open(final_path, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2)

    pf_path = fusion_dir / "per_frame_summary.json"
    with open(pf_path, "w", encoding="utf-8") as fh:
        json.dump(per_frame_summary, fh, indent=2)

    print("Fusion complete. final_report:", final_path)
    print("Per-frame summary:", pf_path)

if __name__ == "__main__":
    main()
