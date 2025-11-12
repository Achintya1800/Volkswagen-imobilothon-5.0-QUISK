# pipeline_integration/streamlit_app.py
# Modern, clean Streamlit UI with beautiful visuals

import streamlit as st
from pathlib import Path
import subprocess, sys, time, json
from typing import Optional

# Page config with custom theme
st.set_page_config(
    layout="wide", 
    page_title="Road Analysis Pipeline",
    page_icon="🛣️",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Headers */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #1e40af;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #2563eb;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    /* Cards */
    .status-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metrics */
    .metric-container {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 3px solid #3b82f6;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #1e293b;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    /* JSON viewer */
    .json-container {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Image containers */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        background: white;
        padding: 1rem;
    }
    
    .image-label {
        color: #475569;
        font-weight: 600;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-pending {
        background: #fef3c7;
        color: #92400e;
    }
    
    /* Log viewer */
    .log-viewer {
        background: #0f172a;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        color: #e2e8f0;
        max-height: 500px;
        overflow-y: auto;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

ROOT = Path(".").resolve()
STORAGE = ROOT / "storage"
RUNNER = ROOT / "pipeline_integration" / "run_pipeline.py"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ---------- Utility Functions ----------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def contains_images(p: Path) -> bool:
    if not p or not p.exists() or not p.is_dir():
        return False
    return any(e.is_file() and e.suffix.lower() in IMAGE_EXTS for e in p.iterdir())

def find_frames_folder(job_dir: Path) -> Optional[Path]:
    if not job_dir.exists():
        return None
    
    candidates = [
        job_dir / "frames",
        job_dir / "frames" / "frames",
        job_dir / "frames" / "visuals",
        job_dir / "visuals",
        job_dir / "images",
    ]
    
    for c in candidates:
        if contains_images(c):
            return c
    
    for d in job_dir.iterdir():
        if d.is_dir() and contains_images(d):
            return d
        if d.is_dir():
            for d2 in d.iterdir():
                if d2.is_dir() and contains_images(d2):
                    return d2
    return None

def list_images_in(folder: Optional[Path], extensions=IMAGE_EXTS):
    if not folder or not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])

def load_json_safe(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        st.error(f"❌ Failed to load {p.name}: {e}")
        return None

def render_json_pretty(data, title="Data"):
    """Render JSON data in a nice, collapsible format"""
    if data:
        with st.expander(f"📋 {title}", expanded=False):
            st.json(data)

def render_status_badge(exists: bool, label: str):
    """Render a status badge"""
    if exists:
        return f'<span class="status-badge status-success">✓ {label}</span>'
    else:
        return f'<span class="status-badge status-pending">○ {label}</span>'

# ---------- MAIN UI ----------
st.markdown("# 🛣️ Road Analysis Pipeline")
st.markdown("**Upload, Process, and Analyze Road Video Data**")

# Sidebar controls
with st.sidebar:
    st.markdown("## 🎛️ Pipeline Controls")
    st.markdown("---")
    
    uploaded = st.file_uploader(
        "📹 Upload Video File",
        type=["mp4", "avi", "mov", "mkv"],
        help="Select a road video to analyze"
    )
    
    max_frames = st.number_input(
        "🎞️ Max Frames",
        min_value=10,
        max_value=2000,
        value=500,
        help="Maximum number of frames to process"
    )
    
    run_btn = st.button("▶️ Run Pipeline", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 📁 Recent Jobs")
    
    ensure_dir(STORAGE)
    jobs = sorted([p.name for p in STORAGE.iterdir() if p.is_dir()], reverse=True)
    
    if jobs:
        sel_job = st.selectbox(
            "Select Job",
            options=[""] + jobs,
            format_func=lambda x: "Choose a job..." if x == "" else x
        )
    else:
        sel_job = ""
        st.info("No jobs yet. Upload a video to start!")
    
    st.markdown("---")
    st.markdown("### 💡 About")
    st.markdown("""
    This pipeline processes road videos through multiple analysis stages:
    - Frame extraction
    - Dehazing
    - Object detection
    - Road segmentation
    - Privacy protection
    - Data fusion
    """)

# Log area setup
log_session_key = "pipeline_log_area"
if log_session_key not in st.session_state:
    st.session_state[log_session_key] = ""

# Run pipeline
if run_btn:
    if uploaded is None:
        st.warning("⚠️ Please upload a video first.")
    else:
        ts = int(time.time())
        job = f"webjob_{ts}"
        job_dir = STORAGE / job
        ensure_dir(job_dir)
        frames_dir = job_dir / "frames"
        ensure_dir(frames_dir)

        # Save uploaded video
        ext = Path(uploaded.name).suffix or ".mp4"
        upload_path = job_dir / f"input_video{ext}"
        with open(upload_path, "wb") as fh:
            fh.write(uploaded.getbuffer())
        
        st.success(f"✅ Video saved: `{uploaded.name}`")

        # Build command
        cmd = [
            sys.executable,
            str(RUNNER),
            "--video", str(upload_path),
            "--job", job,
            "--max-frames", str(int(max_frames))
        ]
        
        st.info("🚀 Starting pipeline execution...")
        
        with st.expander("🔍 Command Details", expanded=False):
            st.code(" ".join(cmd), language="bash")

        # Start subprocess
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as e:
            st.error(f"❌ Failed to start pipeline: {e}")
            proc = None

        if proc:
            st.markdown("### 📊 Pipeline Logs")
            log_area = st.empty()
            out_lines = []
            
            try:
                with st.spinner("⏳ Pipeline running..."):
                    while True:
                        line = proc.stdout.readline()
                        if line:
                            out_lines.append(line.rstrip())
                            recent = "\n".join(out_lines[-800:])
                            st.session_state[log_session_key] = recent
                            log_area.code(recent, language="log", line_numbers=False)
                        elif proc.poll() is not None:
                            remainder = proc.stdout.read()
                            if remainder:
                                out_lines.append(remainder.rstrip())
                            final = "\n".join(out_lines[-800:])
                            st.session_state[log_session_key] = final
                            log_area.code(final, language="log", line_numbers=False)
                            break
                
                rc = proc.returncode
                if rc == 0:
                    st.success("✅ Pipeline completed successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Pipeline failed with return code {rc}")
            except Exception as e:
                st.error(f"❌ Error during execution: {e}")

        st.session_state["last_job"] = job
        st.info(f"💾 Job ID: `{job}` — Select it from the sidebar to view results")

# Viewer area
viewer_job = sel_job if sel_job else st.session_state.get("last_job", None)

if viewer_job:
    st.markdown("---")
    st.markdown(f"## 📊 Results Viewer")
    
    job_dir = STORAGE / viewer_job
    
    if not job_dir.exists():
        st.error(f"❌ Job folder not found: `{viewer_job}`")
    else:
        # Job header with metadata
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 🔖 Job: `{viewer_job}`")
        with col2:
            if (job_dir / "input_video.mp4").exists():
                st.metric("📹 Input", "Video Available")
        with col3:
            frame_count = len(list_images_in(find_frames_folder(job_dir)))
            st.metric("🎞️ Frames", frame_count)
        
        # Pipeline stages status
        st.markdown("#### 🔄 Pipeline Stages")
        stages = {
            "frames": "Frame Extraction",
            "dehaze": "Dehazing",
            "head1": "Object Detection",
            "head2": "Segmentation/Depth",
            "head4": "Road Segmentation",
            "stalled": "Stalled Tracking",
            "privacy": "Privacy Protection",
            "fusion": "Data Fusion"
        }
        
        status_html = "<div style='margin: 1rem 0;'>"
        for key, label in stages.items():
            exists = (job_dir / key).exists()
            status_html += render_status_badge(exists, label) + " "
        status_html += "</div>"
        st.markdown(status_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Frame viewer
        frames_folder = find_frames_folder(job_dir)
        
        if not frames_folder:
            st.warning("⚠️ No frames found for this job yet")
        else:
            frame_paths = list_images_in(frames_folder)
            
            if not frame_paths:
                st.warning("⚠️ Frames folder exists but contains no images yet")
            else:
                st.markdown("### 🖼️ Frame Analysis")
                
                # Frame selector
                frame_names = [p.name for p in frame_paths]
                col1, col2 = st.columns([3, 1])
                with col1:
                    sel_frame = st.select_slider(
                        "Select Frame",
                        options=frame_names,
                        value=frame_names[0]
                    )
                with col2:
                    st.metric("Frame Index", frame_names.index(sel_frame) + 1)
                
                # Main content area
                tab1, tab2, tab3, tab4 = st.tabs([
                    "🎯 Detection & Analysis",
                    "🌫️ Preprocessing",
                    "🛣️ Segmentation",
                    "📈 Metrics & Data"
                ])
                
                # Tab 1: Detection & Analysis
                with tab1:
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        st.markdown('<div class="image-label">📸 Original Frame</div>', unsafe_allow_html=True)
                        aksp = frames_folder / sel_frame
                        if aksp.exists():
                            st.image(str(aksp), use_container_width=True)
                        else:
                            st.info("Frame not available")
                    
                    with col_right:
                        st.markdown('<div class="image-label">🎯 Object Detection (Head-1)</div>', unsafe_allow_html=True)
                        head1_overlay = job_dir / "head1" / "overlay"
                        fallback_head1 = job_dir / "head1" / "detector_out"
                        
                        if not head1_overlay.exists() and fallback_head1.exists():
                            head1_overlay = fallback_head1
                        
                        if head1_overlay.exists():
                            found = list(head1_overlay.glob(f"*{Path(sel_frame).stem}*.jpg"))
                            if found:
                                st.image(str(found[0]), use_container_width=True)
                            else:
                                dets = load_json_safe(job_dir / "head1" / "detections.json") or {}
                                if dets:
                                    render_json_pretty(dets.get(sel_frame, dets), "Detection Data")
                                else:
                                    st.info("No detections for this frame")
                        else:
                            st.info("Detection results not available yet")
                
                # Tab 2: Preprocessing
                with tab2:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown('<div class="image-label">🌫️ Dehazed Image</div>', unsafe_allow_html=True)
                        dehaze_folder = job_dir / "dehaze"
                        if dehaze_folder.exists():
                            cand = list(dehaze_folder.glob(f"*{Path(sel_frame).stem}*"))
                            if cand:
                                st.image(str(cand[0]), use_container_width=True)
                            else:
                                st.info("Dehazed version not available")
                        else:
                            st.info("Dehazing not completed yet")
                    
                    with col_b:
                        st.markdown('<div class="image-label">🔒 Privacy Protected</div>', unsafe_allow_html=True)
                        privacy_folder = job_dir / "privacy" / "blurred"
                        if privacy_folder.exists():
                            cand = list(privacy_folder.glob(f"*{Path(sel_frame).stem}*"))
                            if cand:
                                st.image(str(cand[0]), use_container_width=True)
                            else:
                                st.info("Privacy protection not applied to this frame")
                        else:
                            st.info("Privacy protection not completed yet")
                
                # Tab 3: Segmentation
                with tab3:
                    col_x, col_y = st.columns(2)
                    
                    with col_x:
                        st.markdown('<div class="image-label">🛣️ Road Mask (Head-4)</div>', unsafe_allow_html=True)
                        head4_overlay = job_dir / "head4" / "overlay"
                        if head4_overlay.exists():
                            cand = list(head4_overlay.glob(f"*{Path(sel_frame).stem}*_annotated.*"))
                            if cand:
                                st.image(str(cand[0]), use_container_width=True)
                                
                                head4_status = load_json_safe(job_dir / "head4" / "status.json")
                                if head4_status:
                                    render_json_pretty(
                                        head4_status.get("raw_status", {}).get(sel_frame),
                                        "Road Mask Data"
                                    )
                            else:
                                st.info("Road mask not available for this frame")
                        else:
                            st.info("Road segmentation not completed yet")
                    
                    with col_y:
                        st.markdown('<div class="image-label">📏 Depth Estimation (Head-2)</div>', unsafe_allow_html=True)
                        head2_depth = job_dir / "head2" / "depth"
                        if head2_depth.exists():
                            shown = False
                            for p in sorted(head2_depth.glob("*.json")):
                                dd = load_json_safe(p)
                                if dd and sel_frame in dd:
                                    render_json_pretty({sel_frame: dd[sel_frame]}, f"Depth Data ({p.name})")
                                    shown = True
                                    break
                            if not shown:
                                st.info("No depth data for this frame")
                        else:
                            st.info("Depth estimation not completed yet")
                
                # Tab 4: Metrics & Data
                with tab4:
                    st.markdown("#### 📊 Frame Metrics")
                    
                    # Fusion summary
                    fusion_summary = load_json_safe(job_dir / "fusion" / "per_frame_summary.json")
                    if fusion_summary and sel_frame in fusion_summary:
                        render_json_pretty(fusion_summary[sel_frame], "Fusion Summary")
                    elif fusion_summary:
                        st.info("No fusion data for this specific frame")
                        render_json_pretty(fusion_summary, "Overall Fusion Data")
                    else:
                        st.info("Fusion analysis not completed yet")
                    
                    # Stalled tracking
                    st.markdown("#### 🚗 Stalled Vehicle Tracking")
                    stalled_folder = job_dir / "stalled"
                    if stalled_folder.exists():
                        stalled_status = load_json_safe(stalled_folder / "status.json")
                        if stalled_status:
                            render_json_pretty(stalled_status, "Stalled Vehicle Data")
                        
                        vids = list(stalled_folder.glob("*.mp4")) + list(stalled_folder.glob("*.avi"))
                        if vids:
                            st.markdown("**Tracking Videos:**")
                            for v in vids:
                                st.video(str(v))
                    else:
                        st.info("Stalled vehicle tracking not completed yet")
        
        # Final report download
        st.markdown("---")
        st.markdown("### 📥 Export Results")
        
        final_report = job_dir / "fusion" / "final_report.json"
        if final_report.exists():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.success("✅ Final report available")
            with col2:
                report_size = final_report.stat().st_size / 1024
                st.metric("Size", f"{report_size:.1f} KB")
            with col3:
                with open(final_report, "rb") as fh:
                    st.download_button(
                        "📄 Download Report",
                        data=fh.read(),
                        file_name=f"{viewer_job}_final_report.json",
                        mime="application/json",
                        use_container_width=True
                    )
        else:
            st.info("💭 Final report will be available when pipeline completes")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0;'>
    <p>🛣️ Road Analysis Pipeline | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)