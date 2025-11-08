import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Base directory (folder where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative paths
model_path_yolo  = os.path.join(BASE_DIR, "model", "YoloV8Segmented.pt")
model_path_midas = os.path.join(BASE_DIR, "model", "midas_v21_small_256.pt")
input_folder     = os.path.join(BASE_DIR, "input")
output_folder    = os.path.join(BASE_DIR, "depth_area_output")

# Create output folder if missing
os.makedirs(output_folder, exist_ok=True)
# --- Load Models ---
yolo_model = YOLO(model_path_yolo)

# Load MiDaS (Depth Model)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.load_state_dict(torch.load(model_path_midas, map_location="cpu"))
midas.eval()

# Preprocessing
transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# --- Process Each Image ---
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(input_folder, file_name)
    print(f"\n🖼️ Processing: {file_name}")

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run YOLO segmentation
    results = yolo_model.predict(source=image_path, conf=0.5, save=False, verbose=False)
    if len(results) == 0 or results[0].masks is None:
        print("No potholes detected.")
        continue

    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Run MiDaS depth estimation
    input_batch = transform(img_rgb)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bilinear",
            align_corners=False
        ).squeeze()
    depth_map = prediction.cpu().numpy()

    # Normalize for visualization
    depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)

    # Process each pothole
    for i, mask in enumerate(masks):
        # Resize mask
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Calculate geometric parameters
        area_px = np.sum(mask_resized > 0.5)
        y, x = np.where(mask_resized > 0.5)
        if len(x) == 0 or len(y) == 0:
            continue

        cx, cy = int(np.mean(x)), int(np.mean(y))
        xmin, xmax = np.min(x), np.max(x)
        ymin, ymax = np.min(y), np.max(y)
        width_px, height_px = xmax - xmin, ymax - ymin

        # Calculate depth inside mask
        masked_depth = depth_map * mask_resized
        mean_depth = np.mean(masked_depth[mask_resized > 0.5])

        # Convert to cm (optional, assuming 1px ≈ 0.5cm)
        px_to_cm = 0.5
        width_cm = width_px * px_to_cm
        height_cm = height_px * px_to_cm
        area_cm2 = area_px * (px_to_cm ** 2)

        # Overlay mask
        mask_vis = np.stack((mask_resized,)*3, axis=-1)
        color = (0, 0, 255)
        img = np.where(mask_vis, img * 0.5 + np.array(color) * 0.5, img).astype(np.uint8)

        # Text Display
        cv2.putText(img, f"Area: {area_px}px ({area_cm2:.1f}cm²)", (cx-120, cy-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Width: {width_px}px ({width_cm:.1f}cm)", (cx-120, cy-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Depth: {mean_depth:.3f}", (cx-120, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        print(f"  ➤ Pothole {i+1}: Area={area_px}px ({area_cm2:.1f}cm²), Width={width_px}px, Depth={mean_depth:.3f}")

    # Save result
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, img)
    print(f"✅ Saved result with depth & area at: {output_path}")

print("\n🎉 All images analyzed and saved with width, depth & area annotations!")
