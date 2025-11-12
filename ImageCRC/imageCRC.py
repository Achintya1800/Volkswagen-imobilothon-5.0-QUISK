import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import io
from PIL import Image

# =========================================================
# 1️⃣ Input / Output Paths
# =========================================================
input_video = r"input_frames\frame2.mp4"
output_folder = r"Output\output_realtime_frame2"
os.makedirs(output_folder, exist_ok=True)

# =========================================================
# 2️⃣ Initialize Video
# =========================================================
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎬 Loaded video with {total_frames} frames @ {fps:.2f} FPS")

# =========================================================
# 3️⃣ Parameters
# =========================================================
motion_threshold = 12
fast_skip = 1
slow_skip = 5
window_length = 50    # points shown in graph window

# =========================================================
# 4️⃣ Setup Live Plot
# =========================================================
plt.ion()
fig, ax = plt.subplots(figsize=(12,6))
ax.set_title("⚡ AKS Real-Time Speed & FPS Monitor", fontsize=16, fontweight='bold')
ax.set_xlabel("Frame", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
line1, = ax.plot([], [], label='Speed (km/h)', color='red', linewidth=2)
line2, = ax.plot([], [], label='FPS', color='blue', linewidth=2)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.show(block=False)

# =========================================================
# 4.5️⃣ Setup Video Writer for Graph Recording
# =========================================================
graph_video_path = os.path.join(output_folder, "AKS_graph_visualization.mp4")
# Get figure size in pixels
fig.canvas.draw()
graph_width = fig.canvas.get_width_height()[0]
graph_height = fig.canvas.get_width_height()[1]
graph_writer = cv2.VideoWriter(
    graph_video_path,
    cv2.VideoWriter_fourcc(*'mp4v'),
    10.0,  # 10 FPS for graph video
    (graph_width, graph_height)
)
print(f"📹 Recording graph video: {graph_video_path}")

# =========================================================
# 5️⃣ Buffers
# =========================================================
frame_idx = 0
saved_idx = 0
prev_gray = None
motion_values = []
speeds = deque(maxlen=window_length)
fps_vals = deque(maxlen=window_length)
frames = deque(maxlen=window_length)

prev_time = time.time()
start_time = prev_time

# =========================================================
# 6️⃣ Loop
# =========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue

    diff = cv2.absdiff(gray, prev_gray)
    motion = np.mean(diff)
    motion_values.append(motion)

    # Estimate speed (scaled motion)
    speed = motion * 1.2
    speeds.append(speed)

    # FPS tracking
    curr_time = time.time()
    fps_live = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
    prev_time = curr_time
    fps_vals.append(fps_live)
    frames.append(frame_idx)

    # Adaptive keyframe logic
    skip_rate = fast_skip if motion > motion_threshold else slow_skip
    decision = "KEYFRAME" if frame_idx % skip_rate == 0 else "SKIPPED"

    # Overlay info on frame
    overlay = frame.copy()
    color = (0, 255, 0) if decision == "KEYFRAME" else (0, 0, 255)
    cv2.putText(overlay, decision, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(overlay, f"Speed: {speed:.1f} km/h", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Save only keyframes
    if decision == "KEYFRAME":
        cv2.imwrite(os.path.join(output_folder, f"frame_{saved_idx:04d}.jpg"), frame)
        saved_idx += 1

    # Show frame live
    cv2.imshow("AKS Live Feed", overlay)

    # Update live plot
    line1.set_data(frames, speeds)
    line2.set_data(frames, fps_vals)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # =========================================================
    # 📹 Capture Graph Frame and Write to Video
    # =========================================================
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    # Convert to image array
    graph_pil = Image.open(buf)
    graph_img = np.array(graph_pil)
    # Convert RGB to BGR for OpenCV
    graph_img_bgr = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
    # Resize to match video writer dimensions if needed
    if graph_img_bgr.shape[:2] != (graph_height, graph_width):
        graph_img_bgr = cv2.resize(graph_img_bgr, (graph_width, graph_height))
    graph_writer.write(graph_img_bgr)
    buf.close()

    prev_gray = gray
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
graph_writer.release()  # Release graph video writer
plt.ioff()
plt.close()
end_time = time.time()

# =========================================================
# 7️⃣ Performance Summary
# =========================================================
elapsed = end_time - start_time
saved_ratio = (saved_idx / total_frames) * 100 if total_frames > 0 else 0
avg_fps = np.mean(list(fps_vals)) if fps_vals else 0
avg_speed = np.mean(list(speeds)) if speeds else 0

print("\n================ PERFORMANCE REPORT ================\n")
print(f"🕒 Total Time: {elapsed:.2f}s")
print(f"🎞️ Frames: {total_frames}")
print(f"💾 Saved Keyframes: {saved_idx} ({saved_ratio:.2f}%)")
print(f"🚗 Avg Speed: {avg_speed:.2f} km/h")
print(f"⚡ Avg Processing FPS: {avg_fps:.2f}")
print(f"✅ Output Folder: {output_folder}")
print(f"📹 Graph Video: {graph_video_path}")
print("=====================================================")
