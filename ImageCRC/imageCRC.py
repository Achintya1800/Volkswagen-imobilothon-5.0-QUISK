import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

# =========================================================
# 1️⃣ Input / Output Paths
# =========================================================
input_video = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\ImageCRC\input_frames\frame1.mp4"
output_folder = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\ImageCRC\output_realtime"
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
fig, ax = plt.subplots(figsize=(8,4))
ax.set_title("⚡ AKS Real-Time Speed & FPS Monitor")
ax.set_xlabel("Frame")
ax.set_ylabel("Value")
line1, = ax.plot([], [], label='Speed (km/h)', color='red')
line2, = ax.plot([], [], label='FPS', color='blue')
ax.legend()
plt.show(block=False)

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

    prev_gray = gray
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
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
print("✅ Output Folder:", output_folder)
print("=====================================================")
