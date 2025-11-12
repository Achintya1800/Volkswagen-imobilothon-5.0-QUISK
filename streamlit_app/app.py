"""
QUISK Streamlit Demo App
Zero-inference demo showcasing all 5 heads and 3 utilities using precomputed assets.
"""

import os
import json
import time
import io
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# OpenCV is optional - only used for placeholder video generation
# Make it optional to avoid libGL.so.1 errors on headless servers
try:
    import cv2
    _HAS_CV2 = True
except (ImportError, OSError, AttributeError):
    # OSError catches missing system libraries like libGL.so.1
    # AttributeError catches partial imports that fail
    _HAS_CV2 = False
    cv2 = None

# ------- Demo defaults (no sidebar slider anymore) -------
DEMO_DELAY_SECONDS = 0.8  # previously controlled by the sidebar slider
# keep the variable name 'delay' so existing calls keep working
delay = DEMO_DELAY_SECONDS
# also maintain 'loading_delay' for compatibility with existing code
loading_delay = DEMO_DELAY_SECONDS
# ---------------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="QUISK — Feature Extractor (Demo)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for rounded images
st.markdown("""
<style>
    img {
        border-radius: 10px;
        max-width: 100%;
    }
    .status-chip {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 500;
        margin: 10px 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Asset paths mapping - use script directory as base
ASSETS_BASE = Path(__file__).parent / "assets"
ASSETS = {
    "h1": {
        "pothole": {
            "input": ASSETS_BASE / "h1" / "pothole" / "inputt.jpg",
            "detection": ASSETS_BASE / "h1" / "pothole" / "detetctionn.jpg",
            "segmentation": ASSETS_BASE / "h1" / "pothole" / "segmentation.png",
            "depth": ASSETS_BASE / "h1" / "pothole" / "depth.png",
            "metrics": ASSETS_BASE / "h1" / "pothole" / "metrics.json"
        },
        "debris": {
            "input": ASSETS_BASE / "h1" / "debris" / "input.jpg",
            "detection": ASSETS_BASE / "h1" / "debris" / "detection.jpg",
            "segmentation": ASSETS_BASE / "h1" / "debris" / "segmentation.png",
            "depth": ASSETS_BASE / "h1" / "debris" / "depth.png",
            "metrics": ASSETS_BASE / "h1" / "debris" / "metrics.json"
        },
        "speed_bump": {
            "input": ASSETS_BASE / "h1" / "speed_bump" / "input.jpg",
            "detection": ASSETS_BASE / "h1" / "speed_bump" / "detection.jpg",
            "segmentation": ASSETS_BASE / "h1" / "speed_bump" / "segmentation.png",
            "depth": ASSETS_BASE / "h1" / "speed_bump" / "depth.png",
            "metrics": ASSETS_BASE / "h1" / "speed_bump" / "metrics.json"
        }
    },
    "h4": {
        "pothole": {
            "input": ASSETS_BASE / "h4" / "pothole" / "input.png",
            "road_mask": ASSETS_BASE / "h4" / "pothole" / "road_mask.png"
        },
        "debris": {
            "input": ASSETS_BASE / "h4" / "debris" / "input.jpg",
            "road_mask": ASSETS_BASE / "h4" / "debris" / "road_mask.png",
            "filtered": ASSETS_BASE / "h4" / "debris" / "filtered.jpg"
        },
        "speed_bump": {
            "input": ASSETS_BASE / "h4" / "speed_bump" / "input.jpg",
            "road_mask": ASSETS_BASE / "h4" / "speed_bump" / "road_mask.png",
            "filtered": ASSETS_BASE / "h4" / "speed_bump" / "filtered.jpg"
        }
    },
    "h5": {
        "input": ASSETS_BASE / "h5" / "input.mp4",
        "output": ASSETS_BASE / "h5" / "output_annotated.mp4",
        "tracks": ASSETS_BASE / "h5" / "tracks.csv"
    },
    "util": {
        "crc": {
            "input": ASSETS_BASE / "util" / "crc" / "input.mp4",
            "output": ASSETS_BASE / "util" / "crc" / "output_sampled.mp4",
            "stats": ASSETS_BASE / "util" / "crc" / "stats.csv"
        },
        "dehaze": {
            "before": ASSETS_BASE / "util" / "dehaze" / "before.png",
            "after": ASSETS_BASE / "util" / "dehaze" / "after.png"
        },
        "privacy": {
            "before": ASSETS_BASE / "util" / "privacy" / "before.png",
            "after": ASSETS_BASE / "util" / "privacy" / "after.png"
        }
    }
}

# --- Utilities: CRC Scenarios helpers & safe imports ---
# matplotlib is required for the reference look; fall back to st.line_chart if missing
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# OpenCV availability already checked at top-level import
# _HAS_CV2 is set above

CRC_BASE = ASSETS_BASE / "util" / "crc"
SCENES = {
    "Scene-1: Traffic": CRC_BASE / "traffic",
    "Scene-2: Highway": CRC_BASE / "highway",
}

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def synth_stats(scene_name: str, n_frames: int = 360, seed: int = 42) -> pd.DataFrame:
    """Create synthetic stats that look like the reference chart."""
    rng = np.random.default_rng(seed if "Traffic" in scene_name else seed + 1)
    frames = np.arange(n_frames)

    if "Traffic" in scene_name:
        speed = 12 + 4*np.sin(0.35*frames) + rng.normal(0, 1.2, size=n_frames)
        fps   = 10 + 2*np.sin(0.50*frames + 1.2) + rng.normal(0, 0.8, size=n_frames)
    else:  # Highway
        speed = 16 + 2.5*np.sin(0.25*frames) + rng.normal(0, 0.6, size=n_frames)
        fps   = 12 + 1.5*np.sin(0.45*frames + 0.8) + rng.normal(0, 0.5, size=n_frames)

    speed = np.clip(speed, 0.0, 19.5).round(2)
    fps   = np.clip(fps,   1.0, 19.5).round(2)
    kept  = ((frames % 5) == 0).astype(int)  # keep ~20% frames

    return pd.DataFrame({"frame": frames, "speed_kmh": speed, "fps": fps, "kept": kept})

def gen_placeholder_video(path: Path, scene_name: str, width=1280, height=720, secs=8, fps=30):
    """Create a short placeholder MP4 with moving shapes and title text."""
    if not _HAS_CV2:
        return
    _ensure_dir(path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(secs * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # road-like gray strip
        cv2.rectangle(frame, (0, int(height*0.65)), (width, height), (40, 40, 40), -1)
        # moving car-like rectangle
        x = int((i * 8) % (width + 200)) - 100
        cv2.rectangle(frame, (x, int(height*0.62)), (x+120, int(height*0.68)), (255, 255, 255), -1)
        # scene label
        cv2.putText(frame, f"{scene_name} (PLACEHOLDER)", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(frame, "Replace with real video", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()

def gen_placeholder_graph_video(path: Path, scene_name: str, df: pd.DataFrame = None,
                                width=960, height=540, fps=30, secs=8):
    """Create a short placeholder MP4 showing an animated chart-like visualization."""
    if not _HAS_CV2:
        return
    _ensure_dir(path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    # If stats not supplied, synthesize a small set to animate
    if df is None or df.empty:
        df = synth_stats(scene_name, n_frames=secs*fps)

    # Normalize data to fit chart box
    frames = min(len(df), secs*fps)
    margin = 60
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin
    W = x1 - x0
    H = y0 - y1

    # Scale helpers
    def xmap(i): 
        return int(x0 + (i / max(1, frames - 1)) * W)
    
    def ymap(val, vmin=0, vmax=20): 
        v = max(vmin, min(vmax, float(val)))
        return int(y0 - ((v - vmin) / (vmax - vmin)) * H)

    for i in range(frames):
        frame = np.full((height, width, 3), 20, dtype=np.uint8)

        # Chart area
        cv2.rectangle(frame, (x0, y1), (x1, y0), (60, 60, 60), 2)
        # Gridlines
        for gy in range(0, 21, 4):
            yy = ymap(gy)
            cv2.line(frame, (x0, yy), (x1, yy), (50, 50, 50), 1)

        # Title + labels
        title = "⚡ AKS Real-Time Speed & FPS Monitor"
        cv2.putText(frame, title, (margin, margin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)
        cv2.putText(frame, "Speed (km/h)", (x1-220, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 36, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "FPS", (x1-220, y1+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 144, 0), 2, cv2.LINE_AA)

        # Draw progressive lines up to frame i
        for j in range(1, min(i+1, len(df))):
            if j >= len(df):
                break
            x_prev, x_cur = xmap(j-1), xmap(j)
            # speed in red
            try:
                cv2.line(
                    frame,
                    (x_prev, ymap(df.iloc[j-1]["speed_kmh"])),
                    (x_cur,  ymap(df.iloc[j]["speed_kmh"])),
                    (36, 36, 255), 2
                )
                # fps in blue-ish (distinct from red)
                cv2.line(
                    frame,
                    (x_prev, ymap(df.iloc[j-1]["fps"])),
                    (x_cur,  ymap(df.iloc[j]["fps"])),
                    (255, 144, 0), 2
                )
            except (IndexError, KeyError):
                break

        # Footer watermark
        cv2.putText(frame, f"{scene_name} — Sampling Stats (PLACEHOLDER)", 
                    (margin, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
        out.write(frame)

    out.release()

def ensure_crc_placeholders():
    """Ensure traffic/highway folders, CSVs, and videos exist; never overwrite real files."""
    for scene_name, folder in SCENES.items():
        _ensure_dir(folder)
        csv_p = folder / "stats.csv"
        if not csv_p.exists():
            df = synth_stats(scene_name)
            df.to_csv(csv_p, index=False)
        else:
            try:
                df = pd.read_csv(csv_p)
            except Exception:
                df = synth_stats(scene_name)

        vid_p = folder / "input.mp4"
        if not vid_p.exists():
            gen_placeholder_video(vid_p, scene_name)

        graph_p = folder / "graph.mp4"
        if not graph_p.exists():
            gen_placeholder_graph_video(graph_p, scene_name, df=df)

def compute_report(df: pd.DataFrame):
    """Compute performance report metrics from DataFrame. Returns safe defaults if df is None or empty."""
    if df is None or df.empty:
        return 0, 0.0, 0, 0.0, 0.0, 0.0
    
    try:
        frames = int(df["frame"].max() + 1) if "frame" in df.columns else len(df)
        avg_speed = float(df["speed_kmh"].mean()) if "speed_kmh" in df.columns else 0.0
        avg_fps = float(df["fps"].mean()) if "fps" in df.columns else 0.0
        total_time = (frames / avg_fps) if avg_fps > 0 else 0.0
        saved_count = int(df["kept"].sum()) if "kept" in df.columns else 0
        saved_pct = 100.0 * saved_count / max(frames, 1)
        return frames, total_time, saved_count, saved_pct, avg_speed, avg_fps
    except Exception as e:
        # Return safe defaults if any column is missing
        return 0, 0.0, 0, 0.0, 0.0, 0.0

def build_report_text(frames, total_time, saved_count, saved_pct, avg_speed, avg_fps):
    return (
        f"📼 Loaded video with {frames} frames @ {avg_fps:.2f} FPS\n\n"
        "================ PERFORMANCE REPORT ================\n\n"
        f"🕒 Total Time: {total_time:.2f}s\n"
        f"🎞️ Frames: {frames}\n"
        f"🖼️ Saved Keyframes: {saved_count} ({saved_pct:.2f}%)\n"
        f"🚗 Avg Speed: {avg_speed:.2f} km/h\n"
        f"⚡ Avg Processing FPS: {avg_fps:.2f}\n"
    )

def render_crc_scene(title: str, folder: Path, delay_seconds: float):
    st.markdown(f"### {title}")
    simulate_processing(f"Loading {title}", delay_seconds)

    vid_p = folder / "input.mp4"
    csv_p = folder / "stats.csv"

    col_video, col_plot = st.columns([2, 1], vertical_alignment="top")
    with col_video:
        st.write("**Input Video**")
        if vid_p.exists():
            with open(vid_p, 'rb') as f:
                data = f.read()
            st.video(data)
            st.caption("Demo input – replace with real footage.")
            st.download_button("Download input video", data, 
                             file_name=f"{title.lower().replace(' ', '_').replace(':', '')}_input.mp4",
                             mime="video/mp4")
        else:
            st.warning(f"Missing video at: {vid_p}. Drop your MP4 here to replace the placeholder.")

    with col_plot:
        # NEW: show demo video for Sampling Statistics
        graph_video_p = folder / "graph.mp4"
        if graph_video_p.exists():
            st.write("**Sampling Statistics — Demo Video**")
            try:
                with open(graph_video_p, 'rb') as f:
                    gv_bytes = f.read()
                st.video(gv_bytes)
                st.download_button("Download stats demo video", gv_bytes,
                                 file_name=f"{title.lower().replace(' ', '_').replace(':', '')}_stats_demo.mp4",
                                 mime="video/mp4")
            except Exception as e:
                st.warning(f"Could not load demo video: {str(e)}")
        else:
            st.info("Sampling stats demo video not found. A placeholder will be generated on next run.")
        
        st.write("**Sampling Statistics**")
        # Load CSV with cache if available
        df = pd.DataFrame()
        try:
            if csv_p.exists():
                df = load_csv(str(csv_p))
                # Ensure df is not None
                if df is None:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
        except Exception as e:
            st.warning(f"Error loading CSV: {str(e)}")
            df = pd.DataFrame()

        if df is not None and not df.empty:
            # Check if required columns exist
            required_cols = ["frame", "speed_kmh", "fps"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"CSV is missing required columns: {', '.join(missing_cols)}")
                st.info(f"Expected columns: frame, speed_kmh, fps, kept")
                st.dataframe(df, use_container_width=True, height=300)
            else:
                # Render chart
                try:
                    if _HAS_MPL:
                        fig, ax = plt.subplots(figsize=(6.5, 4.0))
                        ax.plot(df["frame"], df["speed_kmh"], label="Speed (km/h)", color="red", linewidth=2.0)
                        ax.plot(df["frame"], df["fps"], label="FPS", color="blue", linewidth=2.0)
                        ax.set_title("⚡ AKS Real-Time Speed & FPS Monitor")
                        ax.set_xlabel("Frame")
                        ax.set_ylabel("Value")
                        ax.set_ylim(0, 20)
                        ax.grid(axis="y", linestyle=":", alpha=0.5)
                        ax.legend(loc="lower right")
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("matplotlib not installed; showing fallback chart. (We added matplotlib to requirements to fix this permanently.)")
                        st.line_chart(df.set_index("frame")[["speed_kmh", "fps"]])
                except Exception as e:
                    st.error(f"Error rendering chart: {str(e)}")
                    st.info("Showing data table instead.")
                
                # Show dataframe
                st.dataframe(df, use_container_width=True, height=300)
                
                # Download button
                try:
                    csv_data = df.to_csv(index=False)
                    st.download_button("Download stats CSV", csv_data.encode("utf-8"),
                                     file_name=f"{title.lower().replace(' ', '_').replace(':', '')}_stats.csv",
                                     mime="text/csv")
                except Exception as e:
                    st.warning(f"Could not generate CSV download: {str(e)}")
        else:
            st.warning(f"Missing or empty stats CSV at: {csv_p}. A placeholder will be generated on next run.")
            st.info("Click 'Always rerun' in the Streamlit menu to trigger placeholder generation.")

    # Performance report
    if df is None or df.empty:
        df = pd.DataFrame({"frame": [0], "speed_kmh": [0], "fps": [1], "kept": [0]})
    
    try:
        frames, total_time, saved_count, saved_pct, avg_speed, avg_fps = compute_report(df)
        report = build_report_text(frames, total_time, saved_count, saved_pct, avg_speed, avg_fps)
        st.markdown("### Performance Report")
        st.code(report, language=None)
        st.download_button("Download Performance Report", report.encode("utf-8"),
                           file_name=f"{title.lower().replace(' ', '_').replace(':', '')}_report.txt",
                           mime="text/plain")
    except Exception as e:
        st.error(f"Error generating performance report: {str(e)}")
        st.info("Please ensure the stats CSV has the required columns: frame, speed_kmh, fps, kept")

    st.divider()
# --- end CRC Scenarios helpers ---

def create_placeholder_image(width, height, role_text, poi_text, is_depth=False):
    """Create a placeholder image with centered text."""
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font_large = ImageFont.truetype("arial.ttf", 48)
        font_medium = ImageFont.truetype("arial.ttf", 36)
        font_small = ImageFont.truetype("arial.ttf", 28)
    except:
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Draw background rectangle
    draw.rectangle([50, 50, width-50, height-50], fill=(255, 255, 255), outline=(200, 200, 200), width=3)
    
    # Draw text
    text_y = height // 2 - 100
    draw.text((width//2, text_y), role_text, fill=(50, 50, 50), font=font_large, anchor="mm")
    text_y += 80
    draw.text((width//2, text_y), poi_text.upper(), fill=(100, 100, 200), font=font_medium, anchor="mm")
    text_y += 60
    draw.text((width//2, text_y), "PLACEHOLDER – Replace with model output", fill=(150, 150, 150), font=font_small, anchor="mm")
    
    # Add visual elements
    if "Detection" in role_text:
        # Draw bounding box
        draw.rectangle([width//4, height//3, 3*width//4, 2*height//3], outline=(255, 0, 0), width=5)
    elif "Segmentation" in role_text:
        # Draw ellipse mask
        draw.ellipse([width//4, height//3, 3*width//4, 2*height//3], fill=(0, 255, 0, 100), outline=(0, 200, 0), width=3)
    elif is_depth:
        # Add gradient heatmap stripe
        for x in range(width//4, 3*width//4):
            intensity = int(255 * (x - width//4) / (width//2))
            draw.rectangle([x, height//2-50, x+1, height//2+50], fill=(intensity, 255-intensity, 0))
    
    return img


def create_placeholder_video(output_path, width, height, fps, duration, is_annotated=False):
    """Create a placeholder video with moving rectangles."""
    if not _HAS_CV2:
        # Skip video generation if OpenCV is not available (e.g., on headless servers)
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = int(fps * duration)
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame.fill(240)  # Light gray background
        
        # Moving rectangle
        x = int((i / total_frames) * (width - 200)) + 100
        y = height // 2 - 50
        
        color = (0, 255, 0) if not is_annotated or i < total_frames // 2 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + 200, y + 100), color, 3)
        
        # Text overlay
        text = "Annotated Video" if is_annotated else "Sampled Video"
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (50, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        out.write(frame)
    
    out.release()


def bootstrap_assets():
    """Auto-generate placeholder assets if they don't exist."""
    # H1 assets for each POI type
    for poi_type in ["pothole", "debris", "speed_bump"]:
        base_path = ASSETS_BASE / "h1" / poi_type
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Input image - use path from ASSETS dictionary
        input_path = ASSETS["h1"][poi_type]["input"]
        if not input_path.exists():
            img = create_placeholder_image(1280, 720, "H1 – Input Image", poi_type)
            img.save(input_path)
        
        # Detection output - use path from ASSETS dictionary
        detection_path = ASSETS["h1"][poi_type]["detection"]
        if not detection_path.exists():
            img = create_placeholder_image(1280, 720, "H1 – Detection Output", poi_type)
            img.save(detection_path)
        
        # Segmentation
        seg_path = base_path / "segmentation.png"
        if not seg_path.exists():
            img = create_placeholder_image(1280, 720, "H2 – Segmentation Mask", poi_type)
            img.save(seg_path)
        
        # Depth image
        depth_path = base_path / "depth.png"
        if not depth_path.exists():
            img = create_placeholder_image(1280, 720, "H3 – Depth & Heatmap", poi_type, is_depth=True)
            img.save(depth_path)
        
        # Metrics JSON
        metrics_path = base_path / "metrics.json"
        if not metrics_path.exists():
            metrics = {
                "area": 15234,
                "mean_depth": 0.42,
                "depth_range": 0.31,
                "height": 0.12,
                "severity": "Moderate"
            }
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
    
    # H4 assets
    for poi_type in ["pothole", "debris", "speed_bump"]:
        base_path = ASSETS_BASE / "h4" / poi_type
        base_path.mkdir(parents=True, exist_ok=True)
        
        # Use input.png for pothole, input.jpg for others
        input_ext = "png" if poi_type == "pothole" else "jpg"
        input_path = base_path / f"input.{input_ext}"
        if not input_path.exists():
            img = create_placeholder_image(1280, 720, "H4 – Input Image", poi_type)
            img.save(input_path)
        
        mask_path = base_path / "road_mask.png"
        if not mask_path.exists():
            img = create_placeholder_image(1280, 720, "H4 – Road Mask", poi_type)
            img.save(mask_path)
        
        # Only create filtered image for debris and speed_bump (not pothole)
        if poi_type != "pothole":
            filtered_path = base_path / "filtered.jpg"
            if not filtered_path.exists():
                img = create_placeholder_image(1280, 720, "H4 – Filtered Detections", poi_type)
                img.save(filtered_path)
    
    # H5 assets
    h5_path = ASSETS_BASE / "h5"
    h5_path.mkdir(parents=True, exist_ok=True)
    
    input_video = h5_path / "input.mp4"
    if not input_video.exists():
        create_placeholder_video(input_video, 1280, 720, 24, 3, is_annotated=False)
    
    output_video = h5_path / "output_annotated.mp4"
    if not output_video.exists():
        create_placeholder_video(output_video, 1280, 720, 24, 3, is_annotated=True)
    
    tracks_csv = h5_path / "tracks.csv"
    if not tracks_csv.exists():
        tracks_data = {
            "track_id": [12, 21, 34, 47, 53],
            "frames_stalled": [48, 15, 60, 8, 30],
            "avg_motion": [0.8, 3.1, 0.5, 5.2, 1.9],
            "status": ["STALLED", "MOVING", "STALLED", "MOVING", "STALLED"]
        }
        df = pd.DataFrame(tracks_data)
        df.to_csv(tracks_csv, index=False)
    
    # Utility assets - CRC
    crc_path = ASSETS_BASE / "util" / "crc"
    crc_path.mkdir(parents=True, exist_ok=True)
    
    crc_input = crc_path / "input.mp4"
    if not crc_input.exists():
        create_placeholder_video(crc_input, 1280, 720, 24, 3, is_annotated=False)
    
    crc_output = crc_path / "output_sampled.mp4"
    if not crc_output.exists():
        create_placeholder_video(crc_output, 1280, 720, 12, 3, is_annotated=False)  # Lower fps for sampled
    
    stats_csv = crc_path / "stats.csv"
    if not stats_csv.exists():
        stats_data = {
            "time_s": [round(i * 0.1, 1) for i in range(30)],
            "frame_idx": list(range(0, 30, 1)),
            "kept": [1 if i % 3 != 0 else 0 for i in range(30)]
        }
        df = pd.DataFrame(stats_data)
        df.to_csv(stats_csv, index=False)
    
    # Utility assets - Dehaze
    dehaze_path = ASSETS_BASE / "util" / "dehaze"
    dehaze_path.mkdir(parents=True, exist_ok=True)
    
    dehaze_before = dehaze_path / "before.png"
    if not dehaze_before.exists():
        img = create_placeholder_image(1280, 720, "Dehaze – Before", "HAZY IMAGE")
        img.save(dehaze_before)
    
    dehaze_after = dehaze_path / "after.png"
    if not dehaze_after.exists():
        img = create_placeholder_image(1280, 720, "Dehaze – After", "DEHAZED IMAGE")
        img.save(dehaze_after)
    
    # Utility assets - Privacy
    privacy_path = ASSETS_BASE / "util" / "privacy"
    privacy_path.mkdir(parents=True, exist_ok=True)
    
    privacy_before = privacy_path / "before.png"
    if not privacy_before.exists():
        img = create_placeholder_image(1280, 720, "Privacy Blur – Before", "ORIGINAL")
        img.save(privacy_before)
    
    privacy_after = privacy_path / "after.png"
    if not privacy_after.exists():
        img = create_placeholder_image(1280, 720, "Privacy Blur – After", "BLURRED")
        img.save(privacy_after)


# Cached loaders
@st.cache_data
def load_image(path_str, mtime=0):
    """Load an image with error handling. Cache key includes file modification time."""
    path = Path(path_str)
    try:
        if path.exists():
            img = Image.open(path)
            return img
        else:
            st.warning(f"Image file not found: {path}")
            return None
    except Exception as e:
        st.warning(f"Could not load image: {path} - {str(e)}")
        return None

def load_image_with_mtime(path):
    """Helper to load image with automatic mtime detection for cache invalidation."""
    path_obj = Path(path) if isinstance(path, str) else path
    mtime = path_obj.stat().st_mtime if path_obj.exists() else 0
    return load_image(str(path_obj), mtime)


@st.cache_data
def load_json(path_str, mtime=0):
    """Load JSON file with error handling. Cache key includes file modification time."""
    path = Path(path_str)
    try:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        st.warning(f"Could not load JSON: {path} - {str(e)}")
        return None

def load_json_with_mtime(path):
    """Helper to load JSON with automatic mtime detection for cache invalidation."""
    path_obj = Path(path) if isinstance(path, str) else path
    mtime = path_obj.stat().st_mtime if path_obj.exists() else 0
    return load_json(str(path_obj), mtime)


@st.cache_data
def load_csv(path):
    """Load CSV file with error handling. Returns empty DataFrame if file doesn't exist or fails to load."""
    try:
        path_obj = Path(path) if isinstance(path, str) else path
        if path_obj.exists():
            return pd.read_csv(path_obj)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load CSV: {path} - {str(e)}")
        return pd.DataFrame()


def simulate_processing(label, seconds):
    """Show a spinner with configurable delay."""
    with st.spinner(label):
        time.sleep(seconds)


def status_chip(text, success=True):
    """Render a status chip with HTML."""
    color_class = "status-success" if success else "status-error"
    st.markdown(f'<div class="status-chip {color_class}">{text}</div>', unsafe_allow_html=True)


# ----------------- H1→H3 multi-variant helpers -----------------
H1_BASE = ASSETS_BASE / "h1"
H1_CLASSES = ["pothole", "debris", "speed_bump"]

# ---------- H1 Diagnostics (graphs) ----------
GRAPH_FILES = {
    "recall_conf": "graph_recall_conf.png",
    "pr":          "graph_precision_recall.png",
    "prec_conf":   "graph_precision_conf.png",
    "f1_conf":     "graph_f1_conf.png",
    "cm":          "graph_confusion_matrix.png",
}

def _gdir(cls: str) -> Path:
    """Get graph directory for a class, using absolute path."""
    base = H1_BASE if isinstance(H1_BASE, Path) and H1_BASE.is_absolute() else Path(__file__).parent / "assets" / "h1"
    return base / cls / "graphs"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _placeholder_jpeg(path: Path, title: str, size=(1200, 800)):
    img = Image.new("RGB", size, (34,36,40))
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("arial.ttf", 44)
    except Exception:
        f1 = None
    d.text((40, 40), f"{title}\n(PLACEHOLDER — replace with real plot)", (240,240,240), font=f1)
    _ensure_dir(path.parent)
    if not path.exists():
        img.save(path)

def _make_curves(seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, 200)
    # pretend performance per class (fixed shapes)
    r_pothole     = 0.65 - 0.62*x**1.6
    r_speed_bump  = 0.72 - 0.70*x**1.8
    r_debris      = 0.40 - 0.39*x**1.3
    p_pothole     = 0.78 + 0.20*x**2.0
    p_speed_bump  = 0.76 + 0.24*x**2.2
    p_debris      = 0.64 + 0.15*x**1.7
    f1_pothole    = 2*(p_pothole*r_pothole)/(p_pothole+r_pothole+1e-9)
    f1_speed_bump = 2*(p_speed_bump*r_speed_bump)/(p_speed_bump+r_speed_bump+1e-9)
    f1_debris     = 2*(p_debris*r_debris)/(p_debris+r_debris+1e-9)
    return x, (r_pothole, r_speed_bump, r_debris), (p_pothole, p_speed_bump, p_debris), (f1_pothole, f1_speed_bump, f1_debris)

def _plot_or_placeholder(path: Path, make_fig_fn):
    _ensure_dir(path.parent)
    if _HAS_MPL:
        if not path.exists():
            fig = make_fig_fn()
            fig.savefig(path, bbox_inches="tight", dpi=140)
            plt.close(fig)
    else:
        _placeholder_jpeg(path, path.stem)

def _make_fig_recall_conf(highlight: str):
    x, R, P, F1 = _make_curves()
    cls = ["pothole", "speed_bump", "debris"]
    colors = {"pothole":"#5B9BD5","speed_bump":"#ED7D31","debris":"#70AD47"}
    fig, ax = plt.subplots(figsize=(8,5.2))
    for curve,c in zip(R, cls):
        lw = 4 if c == highlight else 2
        ax.plot(x, curve, label=c, linewidth=lw, color=colors[c], alpha=0.9)
    ax.plot(x, np.mean(np.vstack(R), axis=0), label="all classes", color="#0041ff", linewidth=5)
    ax.set_title("Recall-Confidence Curve"); ax.set_xlabel("Confidence"); ax.set_ylabel("Recall"); ax.set_ylim(0,1)
    ax.legend(loc="lower right")
    return fig

def _make_fig_pr(highlight: str):
    x, R, P, F1 = _make_curves()
    cls = ["pothole", "speed_bump", "debris"]
    colors = {"pothole":"#5B9BD5","speed_bump":"#ED7D31","debris":"#70AD47"}
    fig, ax = plt.subplots(figsize=(8,5.2))
    for pr, rc, c in zip(P, R, cls):
        lw = 4 if c == highlight else 2
        ax.plot(rc, pr, label=c, linewidth=lw, color=colors[c], alpha=0.9)
    ax.plot(np.mean(np.vstack(R), axis=0), np.mean(np.vstack(P), axis=0),
            label="all classes", color="#0041ff", linewidth=5)
    ax.set_title("Precision-Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.legend(loc="lower left")
    return fig

def _make_fig_prec_conf(highlight: str):
    x, R, P, F1 = _make_curves()
    cls = ["pothole", "speed_bump", "debris"]
    colors = {"pothole":"#5B9BD5","speed_bump":"#ED7D31","debris":"#70AD47"}
    fig, ax = plt.subplots(figsize=(8,5.2))
    for curve,c in zip(P, cls):
        lw = 4 if c == highlight else 2
        ax.plot(x, curve, label=c, linewidth=lw, color=colors[c], alpha=0.9)
    ax.plot(x, np.mean(np.vstack(P), axis=0), label="all classes", color="#0041ff", linewidth=5)
    ax.set_title("Precision-Confidence Curve"); ax.set_xlabel("Confidence"); ax.set_ylabel("Precision"); ax.set_ylim(0,1)
    ax.legend(loc="lower right")
    return fig

def _make_fig_f1_conf(highlight: str):
    x, R, P, F1 = _make_curves()
    cls = ["pothole", "speed_bump", "debris"]
    colors = {"pothole":"#5B9BD5","speed_bump":"#ED7D31","debris":"#70AD47"}
    fig, ax = plt.subplots(figsize=(8,5.2))
    for curve,c in zip(F1, cls):
        lw = 4 if c == highlight else 2
        ax.plot(x, curve, label=c, linewidth=lw, color=colors[c], alpha=0.9)
    ax.plot(x, np.mean(np.vstack(F1), axis=0), label="all classes", color="#0041ff", linewidth=5)
    ax.set_title("F1-Confidence Curve"); ax.set_xlabel("Confidence"); ax.set_ylabel("F1"); ax.set_ylim(0,1)
    ax.legend(loc="lower right")
    return fig

def _make_fig_cm(highlight: str):
    # 4x4 normalized matrix (pothole, speed_bump, debris, background)
    labels = ["pothole","speed_bump","debris","background"]
    mat = np.array([[0.65,0.05,0.10,0.20],
                    [0.05,0.75,0.08,0.12],
                    [0.07,0.10,0.40,0.43],
                    [0.12,0.10,0.15,0.63]])
    fig, ax = plt.subplots(figsize=(7.2,5.6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion Matrix Normalized")
    ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="#0b2545" if mat[i,j]>0.5 else "#0b2545")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig

def ensure_h1_graph_placeholders(current_class: str):
    """Create the 5 graph PNGs for the current class if missing. Never overwrite."""
    gdir = _gdir(current_class)
    _ensure_dir(gdir)
    # choose maker by key
    makers = {
        "recall_conf": lambda: _make_fig_recall_conf(current_class),
        "pr":          lambda: _make_fig_pr(current_class),
        "prec_conf":   lambda: _make_fig_prec_conf(current_class),
        "f1_conf":     lambda: _make_fig_f1_conf(current_class),
        "cm":          lambda: _make_fig_cm(current_class),
    }
    for key, fname in GRAPH_FILES.items():
        out = gdir / fname
        _plot_or_placeholder(out, makers[key])

def get_graph_paths(current_class: str) -> dict:
    """Get graph paths for current class, falling back to pothole graphs if not available."""
    # Use absolute paths to ensure correct resolution
    script_dir = Path(__file__).parent
    assets_base = script_dir / "assets" / "h1"
    
    gdir = assets_base / current_class / "graphs"
    pothole_gdir = assets_base / "pothole" / "graphs"  # Use pothole graphs as fallback
    
    paths = {}
    for k, fname in GRAPH_FILES.items():
        class_path = gdir / fname
        # Use class-specific graph if exists, otherwise fall back to pothole graphs
        if class_path.exists():
            paths[k] = str(class_path.resolve())  # Use absolute path
        else:
            fallback_path = pothole_gdir / fname
            if fallback_path.exists():
                paths[k] = str(fallback_path.resolve())  # Use absolute path
            else:
                # If neither exists, use class path (will generate placeholder)
                paths[k] = str(class_path.resolve())
    return paths

def zip_graphs(graph_paths: dict, zip_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, p in graph_paths.items():
            if Path(p).exists():
                zf.write(p, arcname=f"{zip_name}_{Path(p).name}")
    return buf.getvalue()
# ---------- end H1 Diagnostics ----------

def get_variants_for_class(cls: str):
    """Return variants available for a given class. Pothole has 2 cases, others have 1."""
    if cls == "pothole":
        return ["case1", "case2"]
    else:
        return ["case1"]

def _ensure(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _placeholder_img(text_top: str, text_mid: str, size=(1280, 720)):
    img = Image.new("RGB", size, (34, 36, 40))
    d = ImageDraw.Draw(img)
    try:
        font_big = ImageFont.truetype("arial.ttf", 48)
        font_sm = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font_big = None
        font_sm = None
    d.text((40, 40), text_top, fill=(240,240,240), font=font_big)
    d.text((40, 110), text_mid, fill=(200,200,200), font=font_sm)
    return img

def _save_if_missing(p: Path, img: Image.Image):
    if not p.exists():
        _ensure(p.parent)
        img.save(p)

def ensure_h1_placeholders():
    """Create demo images for each class/variant if missing; also a simple metrics.json."""
    for cls in H1_CLASSES:
        variants = get_variants_for_class(cls)
        for var in variants:
            base = H1_BASE / cls / var
            _ensure(base)
            inp = base / "input.jpg"
            det = base / "detection.jpg"
            seg = base / "segmentation.png"
            dep = base / "depth.png"
            met = base / "metrics.json"
            
            # base background
            img_in = _placeholder_img(f"{cls.upper()} – {var}", "Input (PLACEHOLDER)")
            _save_if_missing(inp, img_in)
            
            # detection: add red boxes
            img_det = img_in.copy()
            d = ImageDraw.Draw(img_det)
            for k in range(3):
                x0 = 100 + k*220
                y0 = 200 + (k%2)*60
                x1 = x0 + 260
                y1 = y0 + 140
                d.rectangle([x0,y0,x1,y1], outline=(220,60,60), width=6)
            _save_if_missing(det, img_det)
            
            # segmentation and depth only for pothole
            if cls == "pothole":
                # segmentation: blue filled polygons
                img_seg = img_in.copy()
                d = ImageDraw.Draw(img_seg, "RGBA")
                d.polygon([(160,260),(460,260),(430,360),(180,360)], fill=(60,90,255,120), outline=(60,90,255,220))
                d.polygon([(520,300),(820,300),(780,380),(560,380)], fill=(60,90,255,120), outline=(60,90,255,220))
                _save_if_missing(seg, img_seg)
                
                # depth: red overlays
                img_dep = img_in.copy()
                d = ImageDraw.Draw(img_dep, "RGBA")
                d.polygon([(160,260),(460,260),(430,360),(180,360)], fill=(220,60,60,120))
                d.polygon([(520,300),(820,300),(780,380),(560,380)], fill=(220,60,60,120))
                _save_if_missing(dep, img_dep)
                
                # metrics.json only for pothole
                if not met.exists():
                    met.write_text(json.dumps({
                        "area": int(15000 + 1000*np.random.rand()),
                        "mean_depth": round(0.4 + 0.1*np.random.rand(), 2),
                        "depth_range": round(0.3 + 0.1*np.random.rand(), 2),
                        "height": round(0.1 + 0.05*np.random.rand(), 2),
                        "severity": np.random.choice(["Shallow","Moderate","Deep"], p=[0.25,0.5,0.25])
                    }, indent=2))

def get_h1_paths(cls: str, var: str):
    base = H1_BASE / cls / var
    # Check for depth image in both formats (png or jpg)
    depth_path = None
    if (base / "depth.png").exists():
        depth_path = str(base / "depth.png")
    elif (base / "depth.jpg").exists():
        depth_path = str(base / "depth.jpg")
    else:
        depth_path = str(base / "depth.png")  # Default fallback
    
    return {
        "input": str(base / "input.jpg"),
        "detection": str(base / "detection.jpg"),
        "segmentation": str(base / "segmentation.png"),
        "depth": depth_path,
        "metrics": str(base / "metrics.json"),
    }

# session state & slower progress
def init_h13_state():
    ss = st.session_state
    ss.setdefault("current_class", "pothole")   # radio
    ss.setdefault("current_variant", "case1")   # thumbnail choice inside class
    ss.setdefault("h1_status", "idle")          # idle|running|done
    ss.setdefault("h2_status", "hidden")        # hidden|running|done
    ss.setdefault("h3_status", "hidden")        # hidden|running|done
    
    # Ensure variant is valid for current class
    variants = get_variants_for_class(ss.get("current_class", "pothole"))
    if ss.get("current_variant") not in variants:
        ss.current_variant = variants[0]

def reset_h13_state(keep_class=True, keep_variant=False):
    if not keep_class:
        st.session_state.current_class = "pothole"
    if not keep_variant:
        st.session_state.current_variant = "case1"
    st.session_state.h1_status = "idle"
    st.session_state.h2_status = "hidden"
    st.session_state.h3_status = "hidden"

def step_progress(label: str, slider_delay: float):
    """Slower, more 'real' animation: ~4–6s total regardless of slider (but obeys it if larger)."""
    base = 4.0                      # target total seconds for realism
    total = max(slider_delay*1.75, base)
    steps = ["Loading model", "Preprocessing image", "Running inference", "Post-processing results"]
    per = total / len(steps)
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))

@st.cache_data(show_spinner=False)
def load_image_safe(path: str, mtime=0):
    """Load image with caching. Cache key includes file modification time for auto-reload."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as e:
        return None

def load_image_safe_with_mtime(path: str):
    """Helper to load image with automatic mtime detection for cache invalidation."""
    p = Path(path) if isinstance(path, str) else Path(path)
    if not p.exists():
        return None
    try:
        # Use mtime as integer milliseconds for better cache granularity
        mtime = int(p.stat().st_mtime * 1000)
    except Exception:
        mtime = 0
    return load_image_safe(str(p), mtime)

@st.cache_data(show_spinner=False)
def load_json_safe(path: str, mtime=0):
    """Load JSON with caching. Cache key includes file modification time for auto-reload."""
    p = Path(path) if isinstance(path, str) else Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {}

def load_json_safe_with_mtime(path: str):
    """Helper to load JSON with automatic mtime detection for cache invalidation."""
    p = Path(path) if isinstance(path, str) else Path(path)
    if not p.exists():
        return {}
    try:
        # Use mtime as integer milliseconds for better cache granularity
        mtime = int(p.stat().st_mtime * 1000)
    except Exception:
        mtime = 0
    return load_json_safe(str(p), mtime)
# -------------------------------------------------------------


# ===== H4 helpers: variants + placeholders + progress =====
H4_BASE = ASSETS_BASE / "h4"
H4_CLASSES = ["pothole", "debris", "speed_bump"]
H4_VARIANTS = ["case1", "case2"]

def _h4_ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _h4_placeholder(title_top: str, subtitle: str, size=(1280, 720)):
    img = Image.new("RGB", size, (34, 36, 40))
    d = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.truetype("arial.ttf", 44)
        f_sm  = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        f_big = None
        f_sm = None
    d.text((40, 40), title_top, fill=(240,240,240), font=f_big)
    d.text((40, 100), subtitle, fill=(200,200,200), font=f_sm)
    return img

def _h4_save_if_missing(path: Path, img: Image.Image):
    if not path.exists():
        _h4_ensure_dir(path.parent)
        img.save(path)

def ensure_h4_placeholders_variants():
    """Create demo images for each class & variant if missing. Non-variant fallback kept."""
    for cls in H4_CLASSES:
        for var in H4_VARIANTS:
            base = H4_BASE / cls / var
            _h4_ensure_dir(base)
            inp = base / "input.jpg"
            msk = base / "road_mask.png"

            # Input
            _h4_save_if_missing(inp, _h4_placeholder(f"{cls.upper()} – {var.title()}", "Input (PLACEHOLDER)"))

            # Road mask overlay
            img = _h4_placeholder(f"{cls.upper()} – Road Mask", "Fast-SCNN mask + detections (PLACEHOLDER)")
            d = ImageDraw.Draw(img, "RGBA")
            d.rectangle([(0, int(img.size[1]*0.62)), (img.size[0], img.size[1])], fill=(60, 120, 255, 80))
            for x0 in [220, 640, 980]:
                d.rectangle([x0, int(img.size[1]*0.55), x0+220, int(img.size[1]*0.65)],
                            outline=(240, 60, 60, 255), width=6)
            _h4_save_if_missing(msk, img)

@st.cache_data(show_spinner=False)
def h4_load_image(path: str, mtime=0):
    """Load image with caching. Cache key includes file modification time for auto-reload."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None

def h4_load_image_with_mtime(path: str):
    """Helper to load image with automatic mtime detection for cache invalidation."""
    p = Path(path) if isinstance(path, str) else Path(path)
    if not p.exists():
        return None
    try:
        # Use mtime as integer milliseconds for better cache granularity
        mtime = int(p.stat().st_mtime * 1000)
    except Exception:
        mtime = 0
    return h4_load_image(str(p), mtime)

def h4_paths(cls: str, var: str) -> dict:
    """
    Prefer variant folder (assets/h4/<cls>/<var>/...), fall back to non-variant.
    Be tolerant to common filename variants: input.jpg/jpeg/png and inputt.jpg.
    Likewise for mask: road_mask.png/jpg and mask.png/output.png.
    """
    def first_existing(base: Path, names: list[str]) -> Path | None:
        for n in names:
            p = base / n
            if p.exists():
                return p
        return None

    # try variant first
    base = H4_BASE / cls / var
    # Prefer .png over .jpg since users often update .png files
    p_in = first_existing(base, ["input.png", "input.jpg", "input.jpeg", "inputt.jpg"])
    p_mk = first_existing(base, ["road_mask.png", "road_mask.jpg", "mask.png", "output.png"])

    # if variant missing, try non-variant fallback
    if p_in is None or p_mk is None:
        base = H4_BASE / cls
        if p_in is None:
            # Prefer .png over .jpg since users often update .png files
            p_in = first_existing(base, ["input.png", "input.jpg", "input.jpeg", "inputt.jpg"])
        if p_mk is None:
            p_mk = first_existing(base, ["road_mask.png", "road_mask.jpg", "mask.png", "output.png"])

    # last resort to avoid None
    p_in = p_in or (H4_BASE / cls / var / "input.jpg")
    p_mk = p_mk or (H4_BASE / cls / var / "road_mask.png")

    return {"input": str(p_in), "mask": str(p_mk)}

def h4_step_progress(label: str, slider_delay: float):
    """Slower progress to feel real (~4–6s min)."""
    total = max(slider_delay * 1.75, 4.5)
    steps = [
        "Loading Fast-SCNN (road segmentation)",
        "Running YOLOv10 (hazard detection)",
        "Filtering detections to road region",
        "Rendering overlays"
    ]
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        per = total / len(steps)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))

def h4_init_state():
    st.session_state.setdefault("h4_class", "pothole")
    st.session_state.setdefault("h4_variant", "case1")
    st.session_state.setdefault("h4_status", "idle")  # idle | running | done
    # Guard: if a previous session had case3, fall back to case1
    if st.session_state.h4_variant not in ["case1", "case2"]:
        st.session_state.h4_variant = "case1"

def h4_reset(keep_class: bool = True, keep_variant: bool = True):
    """Reset run status; optionally reset class/variant."""
    if not keep_class:
        st.session_state.h4_class = "pothole"
    if not keep_variant:
        st.session_state.h4_variant = "case1"
    st.session_state.h4_status = "idle"
# ===== end H4 helpers =====

# ===== H5 helpers (state + progress) =====
def h5_init_state():
    st.session_state.setdefault("h5_status", "idle")  # idle | running | done

def h5_reset():
    st.session_state.h5_status = "idle"

def h5_step_progress(label: str, slider_delay: float):
    """Believable loader (~3–5s) that respects your sidebar 'delay' slider if longer."""
    total = max(slider_delay * 1.5, 3.6)
    steps = [
        "Loading YOLO11 vehicle detector",
        "Starting SORT multi-object tracker",
        "Analyzing optical flow & ego-motion",
        "Flagging stalled tracks"
    ]
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        per = total / len(steps)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))
# ===== end H5 helpers =====

# ===== ICRC (Image Capture Rate Controller) helpers =====
def icrc_init_state():
    # statuses: idle | running | done
    st.session_state.setdefault("icrc_s1", "idle")
    st.session_state.setdefault("icrc_s2", "idle")

def icrc_reset(scene_key: str):
    st.session_state[scene_key] = "idle"

def icrc_progress(label: str, slider_delay: float):
    """Believable loader that honors the sidebar delay slider if longer (~3–5s)."""
    total = max(slider_delay * 1.25, 3.2)
    steps = [
        "Reading frames & timestamps",
        "Estimating speed & capturing FPS",
        "Sampling frames by speed policy",
        "Writing stats & building report"
    ]
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        per = total / len(steps)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))
# ===== end ICRC helpers =====

# ===== Dehaze helpers (state + progress) =====
def dehaze_init_state():
    st.session_state.setdefault("dehaze_status", "idle")  # idle | running | done

def dehaze_reset():
    st.session_state.dehaze_status = "idle"

def dehaze_progress(label: str, slider_delay: float):
    """Believable loader (~2.5–4s) honoring your sidebar 'delay' if longer."""
    total = max(slider_delay * 1.25, 2.8)
    steps = [
        "Estimating atmospheric light",
        "Computing transmission map",
        "Recovering scene radiance"
    ]
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        per = total / len(steps)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))
# ===== end Dehaze helpers =====

# ===== Privacy Blur helpers (state + progress) =====
def privacy_init_state():
    st.session_state.setdefault("privacy_status", "idle")  # idle | running | done

def privacy_reset():
    st.session_state.privacy_status = "idle"

def privacy_progress(label: str, slider_delay: float):
    """Believable loader (~2.5–4s) honoring your sidebar 'delay' if longer."""
    total = max(slider_delay * 1.25, 2.8)
    steps = [
        "Detecting faces (YuNet, ONNX)",
        "Detecting license plates (YOLOv8)",
        "Applying Gaussian blur to sensitive regions"
    ]
    with st.spinner(f"{label}…"):
        prog = st.progress(0)
        per = total / len(steps)
        for i, s in enumerate(steps, start=1):
            st.write(f"• {s}…")
            time.sleep(per)
            prog.progress(int(i/len(steps)*100))
# ===== end Privacy Blur helpers =====

# Bootstrap assets on startup
bootstrap_assets()

# ===================== SIDEBAR (Navigation only) =====================
with st.sidebar:
    # Minimal CSS polish (no external deps)
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background: #101214;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        .nav-header {
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 0.02em;
            margin: .25rem 0 1rem 0;
        }
        .nav-help {
            color: rgba(255,255,255,0.6);
            font-size: 0.9rem;
            margin: -0.5rem 0 1rem 0;
        }
        .nav-box {
            padding: .6rem .75rem;
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            margin-bottom: 1rem;
        }
        .nav-note {
            color: rgba(255,255,255,0.55);
            font-size: .85rem;
            margin-top: .5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-header">Navigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-help">Pick a section to preview the demo outputs.</div>', unsafe_allow_html=True)

    # group radio inside a subtle card
    with st.container():
        st.markdown('<div class="nav-box">', unsafe_allow_html=True)

        # Keep your exact page labels/keys; just add emojis for readability
        page_choice = st.radio(
            "Select Page",
            options=[
                "H1→H3: Detection → Segmentation → Depth & Severity",
                "H4: Road Mask",
                "H5: Stalled Vehicles",
                "Utilities: Image Capture / Dehaze / Privacy",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Store the selected page exactly as before (so routing below remains unchanged)
    page = page_choice

    # Small hint; no slider / cache controls anymore
    st.markdown('<div class="nav-note">Tip: sections now use a fixed demo loading effect for realism.</div>', unsafe_allow_html=True)
# =====================================================================

# Main content based on selected page
if page == "H1→H3: Detection → Segmentation → Depth & Severity":
    # prepare placeholders on first run
    ensure_h1_placeholders()
    init_h13_state()

    st.title("QUISK — Feature Extractor (Demo)")
    st.caption("H1→H3: Detection → Segmentation → Depth & Severity")

    # ===== Top controls: keep existing Select Sample (class) =====
    st.markdown("**Select Sample**")
    cls = st.radio(
        "",
        options=H1_CLASSES,
        index=H1_CLASSES.index(st.session_state.current_class) if st.session_state.current_class in H1_CLASSES else 0,
        horizontal=True,
        key="class_selector"
    )
    if cls != st.session_state.current_class:
        st.session_state.current_class = cls
        reset_h13_state(keep_class=True)
        st.rerun()

    # ===== Show thumbnails for the selected class (variant chooser) =====
    variants = get_variants_for_class(st.session_state.current_class)
    
    # Ensure current variant is valid for this class
    if st.session_state.current_variant not in variants:
        st.session_state.current_variant = variants[0]
    
    if len(variants) > 1:
        st.markdown("**Choose an example image**")
        vcols = st.columns(len(variants))
    else:
        # Single variant - no need to show selection UI
        st.markdown("**Example image**")
        vcols = [st.container()]  # Single container
    
    for i, var in enumerate(variants):
        with vcols[i]:
            p = get_h1_paths(st.session_state.current_class, var)["input"]
            img = load_image_safe_with_mtime(p)
            if img:
                # Highlight selected variant (only if multiple variants)
                is_selected = (st.session_state.current_variant == var)
                if len(variants) > 1:
                    border_color = "#FF4B4B" if is_selected else "transparent"
                    border_width = "3px" if is_selected else "1px"
                    st.markdown(f'<div style="border: {border_width} solid {border_color}; border-radius: 8px; padding: 4px;">', unsafe_allow_html=True)
                    st.image(img, caption=f"{st.session_state.current_class.replace('_',' ').title()} – {var.title()}" + (" ✓" if is_selected else ""),
                             use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # Single variant - just show image
                    st.image(img, caption=f"{st.session_state.current_class.replace('_',' ').title()} – {var.title()}",
                             use_container_width=True)
            else:
                st.warning(f"Image not found: {p}")
            
            # Only show "Use" button if multiple variants
            if len(variants) > 1:
                button_type = "primary" if is_selected else "secondary"
                if st.button(f"Use {var.title()}", key=f"use_{var}", type=button_type):
                    st.session_state.current_variant = var
                    reset_h13_state(keep_class=True, keep_variant=True)
                    st.rerun()

    # convenience vars - ensure variant is set
    variants = get_variants_for_class(st.session_state.current_class)
    if "current_variant" not in st.session_state or st.session_state.current_variant not in variants:
        st.session_state.current_variant = variants[0]
    
    paths = get_h1_paths(st.session_state.current_class, st.session_state.current_variant)
    delay_effect = delay  # use the global default delay
    
    # Check if this class supports H2/H3 (only pothole does)
    supports_h2_h3 = (st.session_state.current_class == "pothole")

    # ==================== H1 ====================
    st.header("H1: Detection")
    if st.session_state.h1_status in ("idle","running"):
        if st.button("Run H1 – Detection", type="primary", disabled=st.session_state.h1_status=="running"):
            st.session_state.h1_status = "running"
            step_progress("Model is analyzing (H1)", delay_effect)
            st.session_state.h1_status = "done"
            st.rerun()

        if st.session_state.h1_status == "idle":
            st.info("Pick an example above, then click **Run H1 – Detection**.")

    if st.session_state.h1_status == "done":
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Input Image")
            input_path = paths["input"]
            im = load_image_safe_with_mtime(input_path)
            if im:
                st.image(im, use_container_width=True)
                # Download button for input image
                try:
                    with open(input_path, 'rb') as f:
                        img_data = f.read()
                    st.download_button(
                        "Download input image",
                        img_data,
                        file_name=f"{st.session_state.current_class}_{st.session_state.current_variant}_input.jpg",
                        mime="image/jpeg"
                    )
                except Exception:
                    pass
            else:
                st.warning(f"Input image not found at: {input_path}")
        with c2:
            st.subheader("Detection Output")
            detection_path = paths["detection"]
            im = load_image_safe_with_mtime(detection_path)
            if im:
                st.image(im, use_container_width=True)
                # Download button for detection image
                try:
                    with open(detection_path, 'rb') as f:
                        img_data = f.read()
                    st.download_button(
                        "Download detection image",
                        img_data,
                        file_name=f"{st.session_state.current_class}_{st.session_state.current_variant}_detection.jpg",
                        mime="image/jpeg"
                    )
                except Exception:
                    pass
            else:
                st.warning(f"Detection image not found at: {detection_path}")
            msg = {
                "pothole": "A pothole has been detected.",
                "debris": "Debris has been detected.",
                "speed_bump": "A speed bump has been detected."
            }[st.session_state.current_class]
            status_chip(msg, success=True)

        # === NEW: Diagnostics (after H1) ===
        # Get graph paths first (will use existing images if available)
        graph_paths = get_graph_paths(st.session_state.current_class)
        
        # Only generate placeholders if graphs don't exist (never overwrite real images)
        # Check if pothole graphs exist (used as fallback)
        script_dir = Path(__file__).parent
        pothole_gdir = script_dir / "assets" / "h1" / "pothole" / "graphs"
        if not all((pothole_gdir / fname).exists() for fname in GRAPH_FILES.values()):
            ensure_h1_graph_placeholders("pothole")
        
        # Check if current class graphs exist
        current_gdir = script_dir / "assets" / "h1" / st.session_state.current_class / "graphs"
        if not all((current_gdir / fname).exists() for fname in GRAPH_FILES.values()):
            ensure_h1_graph_placeholders(st.session_state.current_class)
        
        # Refresh paths after potential placeholder generation
        graph_paths = get_graph_paths(st.session_state.current_class)

        with st.expander("Model Diagnostics (H1) — Quality Curves & Confusion", expanded=True):
            # small spinner to look 'live'
            with st.spinner("Fetching diagnostics…"):
                time.sleep(max(1.2, delay_effect * 1.25))

            tabs = st.tabs([
                "Recall–Confidence",
                "Precision–Recall",
                "Precision–Confidence",
                "F1–Confidence",
                "Confusion Matrix",
            ])

            with tabs[0]:
                recall_path = graph_paths["recall_conf"]
                recall_path_obj = Path(recall_path)
                if recall_path_obj.exists():
                    # Use absolute path and clear any caching issues
                    st.image(str(recall_path_obj.resolve()), use_container_width=True)
                    try:
                        st.download_button("Download image", Path(recall_path).read_bytes(),
                                         file_name=f"{st.session_state.current_class}_recall_conf.png",
                                         mime="image/png")
                    except Exception:
                        pass
                else:
                    st.warning(f"Graph not found: {recall_path}")

            with tabs[1]:
                pr_path = graph_paths["pr"]
                pr_path_obj = Path(pr_path)
                if pr_path_obj.exists():
                    st.image(str(pr_path_obj.resolve()), use_container_width=True)
                    try:
                        st.download_button("Download image", Path(pr_path).read_bytes(),
                                         file_name=f"{st.session_state.current_class}_precision_recall.png",
                                         mime="image/png")
                    except Exception:
                        pass
                else:
                    st.warning(f"Graph not found: {pr_path}")

            with tabs[2]:
                prec_conf_path = graph_paths["prec_conf"]
                prec_conf_path_obj = Path(prec_conf_path)
                if prec_conf_path_obj.exists():
                    st.image(str(prec_conf_path_obj.resolve()), use_container_width=True)
                    try:
                        st.download_button("Download image", Path(prec_conf_path).read_bytes(),
                                         file_name=f"{st.session_state.current_class}_precision_conf.png",
                                         mime="image/png")
                    except Exception:
                        pass
                else:
                    st.warning(f"Graph not found: {prec_conf_path}")

            with tabs[3]:
                f1_conf_path = graph_paths["f1_conf"]
                f1_conf_path_obj = Path(f1_conf_path)
                if f1_conf_path_obj.exists():
                    st.image(str(f1_conf_path_obj.resolve()), use_container_width=True)
                    try:
                        st.download_button("Download image", Path(f1_conf_path).read_bytes(),
                                         file_name=f"{st.session_state.current_class}_f1_conf.png",
                                         mime="image/png")
                    except Exception:
                        pass
                else:
                    st.warning(f"Graph not found: {f1_conf_path}")

            with tabs[4]:
                cm_path = graph_paths["cm"]
                cm_path_obj = Path(cm_path)
                if cm_path_obj.exists():
                    st.image(str(cm_path_obj.resolve()), use_container_width=True)
                    try:
                        st.download_button("Download image", Path(cm_path).read_bytes(),
                                         file_name=f"{st.session_state.current_class}_confusion_matrix.png",
                                         mime="image/png")
                    except Exception:
                        pass
                else:
                    st.warning(f"Graph not found: {cm_path}")

            # optional: one-click ZIP
            try:
                all_zip = zip_graphs(graph_paths, f"{st.session_state.current_class}_h1_diagnostics")
                st.download_button("Download all graphs (.zip)", all_zip,
                                 file_name=f"{st.session_state.current_class}_h1_diagnostics.zip",
                                 mime="application/zip")
            except Exception:
                pass

        if supports_h2_h3:
            st.success("H1 complete. You can proceed to H2.")
        else:
            st.success("H1 complete. Detection finished.")
        st.divider()

    # ==================== H2 ====================
    # Only show H2/H3 for pothole
    if supports_h2_h3 and st.session_state.h1_status == "done":
        st.header("H2: Segmentation")

        if st.session_state.h2_status in ("hidden","running"):
            if st.button("Run H2 – Segmentation", type="primary", disabled=st.session_state.h2_status=="running"):
                st.session_state.h2_status = "running"
                step_progress("Generating segmentation mask (H2)", delay_effect)
                st.session_state.h2_status = "done"
                st.rerun()

        if st.session_state.h2_status == "done":
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Original Image")
                im = load_image_safe_with_mtime(paths["input"])
                if im:
                    st.image(im, use_container_width=True)
                else:
                    st.warning("Input image not found")
            with c4:
                st.subheader("Segmentation Mask")
                im = load_image_safe_with_mtime(paths["segmentation"])
                if im:
                    st.image(im, use_container_width=True)
                else:
                    st.warning("Segmentation image not found")

            st.success("H2 complete. Proceed to H3.")
            st.divider()

    # ==================== H3 ====================
    # Only show H3 for pothole
    if supports_h2_h3 and st.session_state.h2_status == "done":
        st.header("H3: Depth & Severity")

        if st.session_state.h3_status in ("hidden","running"):
            if st.button("Run H3 – Depth & Severity", type="primary", disabled=st.session_state.h3_status=="running"):
                st.session_state.h3_status = "running"
                step_progress("Estimating depth & severity (H3)", delay_effect)
                st.session_state.h3_status = "done"
                st.rerun()

        if st.session_state.h3_status == "done":
            left, right = st.columns([2,1])
            with left:
                st.subheader("Depth & Heatmap")
                depth_path = paths["depth"]
                im = load_image_safe_with_mtime(depth_path)
                if im:
                    st.image(im, use_container_width=True)
                    # Download button for depth image
                    try:
                        if Path(depth_path).exists():
                            with open(depth_path, 'rb') as f:
                                img_data = f.read()
                            file_ext = Path(depth_path).suffix.lower()
                            mime_type = "image/jpeg" if file_ext == ".jpg" else "image/png"
                            st.download_button(
                                "Download depth image",
                                img_data,
                                file_name=f"{st.session_state.current_class}_{st.session_state.current_variant}_depth{file_ext}",
                                mime=mime_type
                            )
                    except Exception:
                        pass
                else:
                    st.warning(f"Depth image not found at: {depth_path}")
            with right:
                st.subheader("Metrics")
                m = load_json_safe_with_mtime(paths["metrics"])
                if m:
                    area_val = m.get('area', '—')
                    if isinstance(area_val, (int, float)):
                        st.metric("Area", f"{area_val:,} px²")
                    else:
                        st.metric("Area", str(area_val))
                    if 'mean_depth' in m:  st.metric("Mean Depth",  f"{m['mean_depth']:.2f}")
                    if 'depth_range' in m: st.metric("Depth Range", f"{m['depth_range']:.2f}")
                    if 'height' in m:      st.metric("Height",      f"{m['height']:.2f}")
                    if 'width' in m:      st.metric("Width",       f"{m['width']:,} px")
                    if 'depth' in m:      st.metric("Depth",       f"{m['depth']:.2f}")
                    if 'severity' in m:    st.metric("Severity",    m['severity'])
                    st.download_button("Download metrics (JSON)",
                        data=json.dumps(m, indent=2).encode("utf-8"),
                        file_name=f"{st.session_state.current_class}_{st.session_state.current_variant}_metrics.json",
                        mime="application/json")
                else:
                    st.warning("Metrics not found for this example.")

    st.caption("Use **Use Case buttons** above to switch examples; click **Run** on each step to continue.")

elif page == "H4: Road Mask":
    ensure_h4_placeholders_variants()
    h4_init_state()

    st.title("QUISK — Feature Extractor (Demo)")
    st.caption("H4: Road Mask")

    # --- Class selector (kept) ---
    st.markdown("**Select Sample**")
    new_cls = st.radio(
        "",
        options=H4_CLASSES,
        index=H4_CLASSES.index(st.session_state.h4_class) if st.session_state.h4_class in H4_CLASSES else 0,
        horizontal=True
    )
    if new_cls != st.session_state.h4_class:
        st.session_state.h4_class = new_cls
        h4_reset(keep_class=True, keep_variant=False)  # reset variant to case1 on class change
        st.rerun()

    # --- Minimal variant UI: dropdown + big preview (like screenshot layout) ---
    left, right = st.columns([2, 1], vertical_alignment="center")
    
    with left:
        st.markdown("**Selected Input**")
        preview = h4_load_image_with_mtime(h4_paths(st.session_state.h4_class, st.session_state.h4_variant)["input"])
        if preview:
            st.image(preview, use_container_width=True)
        else:
            st.warning("Preview image not found")
    
    with right:
        st.markdown("**Variant**")
        sel_var = st.selectbox("Choose example", ["case1", "case2"],
                               index=(0 if st.session_state.h4_variant == "case1" else 1),
                               label_visibility="collapsed")
        if sel_var != st.session_state.h4_variant:
            # keep the chosen variant; just reset the run status so the user can re-run
            st.session_state.h4_variant = sel_var
            st.session_state.h4_status = "idle"   # don't call h4_reset() here
            st.rerun()
        
        st.write("")  # spacer
        
        run_clicked = st.button("Run H4 – Road Mask", type="primary", use_container_width=True,
                        disabled=st.session_state.h4_status=="running")
        st.write("")
        
        if st.button("Reset H4", use_container_width=True):
            h4_reset(keep_class=True, keep_variant=True)  # reset only run status, keep variant
            st.rerun()

    # --- Gate: run → progress → reveal outputs ---
    if run_clicked and st.session_state.h4_status in ("idle", "running"):
        st.session_state.h4_status = "running"
        h4_step_progress("Generating road mask & filtering detections", loading_delay)
        st.session_state.h4_status = "done"
        st.rerun()

    st.subheader("Road Mask Analysis")

    if st.session_state.h4_status == "idle":
        st.info("Pick a class and example, then click **Run H4 – Road Mask** to see the results.")

    if st.session_state.h4_status == "done":
        paths = h4_paths(st.session_state.h4_class, st.session_state.h4_variant)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Original**")
            input_path = paths["input"]
            im = h4_load_image_with_mtime(input_path)
            if im:
                st.image(im, use_container_width=True)
                try:
                    p = Path(input_path)
                    file_ext = p.suffix.lower()
                    mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
                    st.download_button("Download original", p.read_bytes(),
                                     file_name=f"{st.session_state.h4_class}_{st.session_state.h4_variant}_original{file_ext}",
                                     mime=mime_type,
                                     use_container_width=True)
                except Exception:
                    pass
            else:
                st.warning(f"Input image not found: {input_path}")

        with c2:
            st.markdown("**Road Mask**")
            mask_path = paths["mask"]
            im = h4_load_image_with_mtime(mask_path)
            if im:
                st.image(im, use_container_width=True)
                try:
                    p = Path(mask_path)
                    file_ext = p.suffix.lower()
                    mime_type = "image/png" if file_ext == ".png" else "image/jpeg"
                    st.download_button("Download road mask", p.read_bytes(),
                                     file_name=f"{st.session_state.h4_class}_{st.session_state.h4_variant}_road_mask{file_ext}",
                                     mime=mime_type,
                                     use_container_width=True)
                except Exception:
                    pass
            else:
                st.warning(f"Road mask not found: {mask_path}")

        st.success("H4 complete. Road-only hazards are shown above.")

elif page == "H5: Stalled Vehicles":
    h5_init_state()
    
    st.title("QUISK — Feature Extractor (Demo)")
    st.caption("H5: Stalled Vehicles")
    
    # IMPORTANT: reuse your exact variables/assignments for these paths.
    input_path = ASSETS["h5"]["input"]
    annot_path = ASSETS["h5"]["output"]
    
    # --- IDLE/RUNNING: show ONLY the input video full-width; put buttons directly below it ---
    if st.session_state.h5_status in ("idle", "running"):
        st.subheader("Input Video")
        if input_path.exists():
            with open(input_path, "rb") as f:
                st.video(f.read())
            
            st.download_button("Download input video", open(input_path, "rb").read(),
                             file_name="input_video.mp4",
                             mime="video/mp4")
        else:
            st.warning("Input video not found")
        
        # 👉 Run button directly under the input video (as requested)
        run_clicked = st.button("Run H5 – Detect Stalled Vehicles",
                                type="primary",
                                key="h5_run_below_input",
                                disabled=(st.session_state.h5_status == "running"))
        
        # Optional reset under the run button (kept local to H5)
        reset_clicked = st.button("Reset H5", key="h5_reset_below_input")
        if reset_clicked:
            h5_reset()
            st.rerun()
        
        if run_clicked:
            st.session_state.h5_status = "running"
            h5_step_progress("Processing video", loading_delay)  # uses global default delay
            st.session_state.h5_status = "done"
            st.rerun()
        
        if st.session_state.h5_status == "idle":
            st.info("Click **Run H5 – Detect Stalled Vehicles** to generate the annotated output.")
    
    # --- DONE: show input (left) and annotated (right) videos side-by-side ---
    else:
        left, right = st.columns(2)
        
        with left:
            st.subheader("Input Video")
            if input_path.exists():
                with open(input_path, "rb") as f:
                    st.video(f.read())
                st.download_button("Download input video", open(input_path, "rb").read(),
                                 file_name="input_video.mp4",
                                 mime="video/mp4")
            else:
                st.warning("Input video not found")
            
            # keep reset here too for convenience
            if st.button("Reset H5", key="h5_reset_done_left"):
                h5_reset()
                st.rerun()
        
        with right:
            st.subheader("Annotated Video")
            if annot_path.exists():
                with open(annot_path, "rb") as f:
                    st.video(f.read())
                st.download_button("Download annotated video", open(annot_path, "rb").read(),
                                 file_name="annotated_video.mp4",
                                 mime="video/mp4")
            else:
                st.warning("Annotated video not found")
        
        st.divider()
        
        # Track Data section below videos (only shown after run)
        st.header("Track Data")
        tracks_df = load_csv(ASSETS["h5"]["tracks"])
        if tracks_df is not None and not tracks_df.empty:
            st.dataframe(tracks_df, use_container_width=True)
            # Download CSV button
            csv = tracks_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="tracks.csv",
                mime="text/csv"
            )
        else:
            st.info("No track data available.")

elif page == "Utilities: Image Capture / Dehaze / Privacy":
    st.title("QUISK — Feature Extractor (Demo)")
    st.caption("Utilities: Image Capture / Dehaze / Privacy")
    
    tab1, tab2, tab3 = st.tabs([
        "Image Capture Rate Controller",
        "Dehaze Image",
        "Privacy Blur"
    ])
    
    with tab1:
        st.write("**Demo of speed-adaptive keyframe sampling — two scenarios**")
        # Make sure placeholder assets exist (only created if missing)
        ensure_crc_placeholders()
        
        icrc_init_state()  # safe no-op if already called above
        
        # ---------------- Scene-1: Traffic ----------------
        st.subheader("Scene-1: Traffic")
        
        # Reuse existing assignments for these three items (do not rename/replace):
        scene1_folder = SCENES["Scene-1: Traffic"]
        scene1_input_path = scene1_folder / "input.mp4"
        scene1_stats_video_path = scene1_folder / "graph.mp4"
        scene1_csv_path = scene1_folder / "stats.csv"
        
        # Performance report with exact values
        scene1_performance_markdown = """\
================ PERFORMANCE REPORT =================

🕒 Total Time: 40.87s
🎞️ Frames: 473
💾 Saved Keyframes: 95 (20.08%)
🚗 Avg Speed: 0.18 km/h
⚡ Avg Processing FPS: 11.28
"""
        
        st.markdown("**Input Video**")
        if scene1_input_path.exists():
            with open(scene1_input_path, "rb") as f:
                st.video(f.read())
            st.download_button("Download input video", open(scene1_input_path, "rb").read(),
                             file_name="scene1_input.mp4",
                             mime="video/mp4")
        else:
            st.warning("Input video not found")
        
        # Run/Reset directly under the input
        c1, c2 = st.columns([1,1])
        with c1:
            run1 = st.button("Run Scene-1 – Image Capture", type="primary", key="icrc_run_s1",
                            disabled=(st.session_state.icrc_s1 == "running"))
        with c2:
            if st.button("Reset Scene-1", key="icrc_reset_s1"):
                icrc_reset("icrc_s1")
                st.rerun()
        
        if run1:
            st.session_state.icrc_s1 = "running"
            icrc_progress("Running speed-adaptive sampling", loading_delay)  # uses global default delay
            st.session_state.icrc_s1 = "done"
            st.rerun()
        
        # IDLE hint
        if st.session_state.icrc_s1 == "idle":
            st.info("Click **Run Scene-1 – Image Capture** to generate sampling stats and the performance report.")
        
        # After run: show ONLY (a) the stats video and (b) the performance report
        if st.session_state.icrc_s1 == "done":
            st.markdown("**Sampling Statistics — Demo Video**")
            if scene1_stats_video_path.exists():
                with open(scene1_stats_video_path, "rb") as f:
                    st.video(f.read())
                st.download_button("Download stats demo video", open(scene1_stats_video_path, "rb").read(),
                                 file_name="scene1_stats_demo.mp4",
                                 mime="video/mp4")
            else:
                st.warning("Stats demo video not found")
            
            st.subheader("Performance Report")
            st.code(scene1_performance_markdown, language="text")
        
        st.divider()
        
        # ---------------- Scene-2: Highway ----------------
        st.subheader("Scene-2: Highway")
        
        # Reuse existing variables (do not rename):
        scene2_folder = SCENES["Scene-2: Highway"]
        scene2_input_path = scene2_folder / "input.mp4"
        scene2_stats_video_path = scene2_folder / "graph.mp4"
        scene2_csv_path = scene2_folder / "stats.csv"
        
        # Performance report with exact values
        scene2_performance_markdown = """\
================ PERFORMANCE REPORT =================

🕒 Total Time: 44.60s
🎞️ Frames: 466
💾 Saved Keyframes: 111 (23.82%)
🚗 Avg Speed: 13.07 km/h
⚡ Avg Processing FPS: 10.96
"""
        
        st.markdown("**Input Video**")
        if scene2_input_path.exists():
            with open(scene2_input_path, "rb") as f:
                st.video(f.read())
            st.download_button("Download input video", open(scene2_input_path, "rb").read(),
                             file_name="scene2_input.mp4",
                             mime="video/mp4")
        else:
            st.warning("Input video not found")
        
        c1, c2 = st.columns([1,1])
        with c1:
            run2 = st.button("Run Scene-2 – Image Capture", type="primary", key="icrc_run_s2",
                            disabled=(st.session_state.icrc_s2 == "running"))
        with c2:
            if st.button("Reset Scene-2", key="icrc_reset_s2"):
                icrc_reset("icrc_s2")
                st.rerun()
        
        if run2:
            st.session_state.icrc_s2 = "running"
            icrc_progress("Running speed-adaptive sampling", loading_delay)
            st.session_state.icrc_s2 = "done"
            st.rerun()
        
        if st.session_state.icrc_s2 == "idle":
            st.info("Click **Run Scene-2 – Image Capture** to generate sampling stats and the performance report.")
        
        if st.session_state.icrc_s2 == "done":
            st.markdown("**Sampling Statistics — Demo Video**")
            if scene2_stats_video_path.exists():
                with open(scene2_stats_video_path, "rb") as f:
                    st.video(f.read())
                st.download_button("Download stats demo video", open(scene2_stats_video_path, "rb").read(),
                                 file_name="scene2_stats_demo.mp4",
                                 mime="video/mp4")
            else:
                st.warning("Stats demo video not found")
            
            st.subheader("Performance Report")
            st.code(scene2_performance_markdown, language="text")
    
    with tab2:
        # ---------------- Utilities → Dehaze Image (gated) ----------------
        dehaze_init_state()
        
        st.subheader("Dehaze Image")
        
        # Reuse your existing variables/assignments (do NOT change filenames/locations).
        dehaze_input_path = ASSETS["util"]["dehaze"]["before"]
        dehaze_output_path = ASSETS["util"]["dehaze"]["after"]
        
        # Always show the input image first
        left, right = st.columns([1, 1])
        
        with left:
            st.markdown("**Before**")
            before_img = load_image_with_mtime(dehaze_input_path)
            if before_img:
                st.image(before_img, use_container_width=True)
            else:
                st.warning("Before image not found")
        
        with right:
            st.markdown("**After**")
            if st.session_state.dehaze_status == "done":
                after_img = load_image_with_mtime(dehaze_output_path)
                if after_img:
                    st.image(after_img, use_container_width=True)
                    # Keep your existing download button style
                    from io import BytesIO
                    buf = BytesIO()
                    after_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button("Download dehazed image",
                                     data=buf,
                                     file_name="dehazed.png",
                                     mime="image/png")
                else:
                    st.warning("After image not found")
            else:
                st.info("Press **Run Dehaze** to generate and reveal the output image.")
        
        # Buttons directly under the images
        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            run_clicked = st.button("Run Dehaze", type="primary", key="btn_dehaze_run",
                                  disabled=(st.session_state.dehaze_status == "running"))
        with bcol2:
            if st.button("Reset Dehaze", key="btn_dehaze_reset"):
                dehaze_reset()
                st.rerun()
        
        # Run → progress → reveal
        if run_clicked:
            st.session_state.dehaze_status = "running"
            # uses global default delay
            dehaze_progress("Applying dehaze pipeline", loading_delay)
            st.session_state.dehaze_status = "done"
            st.rerun()
        # ------------------------------------------------------------------
    
    with tab3:
        # ---------------- Utilities → Privacy Blur (gated) ----------------
        privacy_init_state()
        
        st.subheader("Privacy Blur")
        
        # Reuse your existing variables/assignments (do NOT change filenames/locations).
        privacy_input_path = ASSETS["util"]["privacy"]["before"]
        privacy_output_path = ASSETS["util"]["privacy"]["after"]
        
        # Always show the input image first
        left, right = st.columns([1, 1])
        
        with left:
            st.markdown("**Before**")
            before_img = load_image_with_mtime(privacy_input_path)
            if before_img:
                st.image(before_img, use_container_width=True)
            else:
                st.warning("Before image not found")
        
        with right:
            st.markdown("**After**")
            if st.session_state.privacy_status == "done":
                after_img = load_image_with_mtime(privacy_output_path)
                if after_img:
                    st.image(after_img, use_container_width=True)
                    # Keep your existing download button style
                    from io import BytesIO
                    buf = BytesIO()
                    after_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button("Download blurred image",
                                     data=buf,
                                     file_name="blurred.png",
                                     mime="image/png")
                else:
                    st.warning("After image not found")
            else:
                st.info("Press **Run Privacy Blur** to reveal the blurred output image.")
        
        # Buttons directly under the images
        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            run_clicked = st.button("Run Privacy Blur", type="primary", key="btn_privacy_run",
                                  disabled=(st.session_state.privacy_status == "running"))
        with bcol2:
            if st.button("Reset Privacy Blur", key="btn_privacy_reset"):
                privacy_reset()
                st.rerun()
        
        # Run → progress → reveal
        if run_clicked:
            st.session_state.privacy_status = "running"
            # uses global default delay
            privacy_progress("Running face & plate blurring", loading_delay)
            st.session_state.privacy_status = "done"
            st.rerun()
        # ------------------------------------------------------------------

# Footer
st.divider()
st.caption("This is a zero-inference demo. All results are produced offline by our QUISK models and displayed here for a smooth review experience.")

