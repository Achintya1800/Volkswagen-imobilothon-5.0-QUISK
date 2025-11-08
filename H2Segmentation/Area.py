from ultralytics import YOLO
import cv2
import numpy as np
import os

# Load trained model
model_path = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\model\YoloV8Segmented.pt"
model = YOLO(model_path)

# Input and output folders
input_folder = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\input"
output_folder = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\segmented_area_output"

os.makedirs(output_folder, exist_ok=True)

# Loop through all test images
for file_name in os.listdir(input_folder):
    if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(input_folder, file_name)
    print(f"\n🖼️ Processing: {file_name}")

    # Run inference
    results = model.predict(source=image_path, conf=0.5, save=False, verbose=False)
    img = cv2.imread(image_path)

    if len(results) == 0 or results[0].masks is None:
        print("No potholes detected.")
        continue

    masks = results[0].masks.data.cpu().numpy()

    for i, mask in enumerate(masks):
        # Compute area (number of pixels inside mask)
        area_pixels = np.sum(mask > 0.5)

        # Resize mask to match original image size
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Compute centroid for labeling
        y, x = np.where(mask_resized > 0.5)
        if len(x) == 0 or len(y) == 0:
            continue
        cx, cy = int(np.mean(x)), int(np.mean(y))

        # Create colored overlay for mask
        color = (0, 0, 255)  # red mask
        mask_vis = np.stack((mask_resized,)*3, axis=-1)
        img = np.where(mask_vis, img * 0.5 + np.array(color) * 0.5, img).astype(np.uint8)

        # Display area on the image
        cv2.putText(img, f"Area: {area_pixels}px", (cx-60, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        print(f"  ➤ Pothole {i+1}: Area = {area_pixels} pixels")

    # Save the annotated image
    output_path = os.path.join(output_folder, file_name)
    cv2.imwrite(output_path, img)
    print(f"✅ Saved annotated image with area to: {output_path}")

print("\n🎉 All images processed! Check your 'segmented_area_output' folder.")
