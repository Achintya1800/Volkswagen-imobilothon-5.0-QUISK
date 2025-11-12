# app.py
import streamlit as st
import tempfile
import cv2
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Imobilothon Demo", layout="centered")
st.title("Imobilothon Pipeline — Demo (Step 1)")

st.markdown(
    """
    **What this demo does (first step)**  
    - Upload a short video (mp4).  
    - Shows a preview (first extracted frame).  
    - Displays basic file info and a button to "Run pipeline" (placeholder for next steps).
    """
)

uploaded = st.file_uploader("Upload a short MP4 video (1-10s)", type=["mp4", "mov", "avi"])

def extract_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame from video.")
    # Convert BGR -> RGB for display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

if uploaded is not None:
    # Save to a temp file
    tmpdir = tempfile.mkdtemp(prefix="imobilothon_")
    local_video_path = Path(tmpdir) / uploaded.name
    with open(local_video_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved upload to: {local_video_path}")
    try:
        frame = extract_first_frame(str(local_video_path))
        st.subheader("First frame preview")
        st.image(frame, use_column_width=True)
        st.write("Video info:")
        st.write(f"- Filename: `{uploaded.name}`")
        cap = cv2.VideoCapture(str(local_video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else "unknown"
        cap.release()
        st.write(f"- FPS: `{fps}`")
        st.write(f"- Frames: `{frame_count}`")
        st.write(f"- Duration (s): `{duration}`")
    except Exception as e:
        st.error(f"Failed to process video: {e}")

    st.write("---")
    if st.button("Run pipeline (demo placeholder)"):
        st.info("Pipeline run placeholder — in the next steps we'll hook each module and show per-layer outputs.")
        # Save metadata for orchestrator to pick up (simple marker file)
        marker = Path(tmpdir) / "job_ready.txt"
        marker.write_text("ready")
        st.success(f"Created job marker: {marker}")
else:
    st.info("Upload a short video to see a quick preview and enable the pipeline button.")
