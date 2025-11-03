# import cv2
# import os
# from ultralytics import YOLO
# import numpy as np

# # ---------------------------------------------------
# # 1. Setup model (YOLOv8 number plate detector)
# # ---------------------------------------------------
# plate_model = YOLO(r"C:\Users\Nidhi Patel\OneDrive\Desktop\Mobile\backend\privacy\models\license_plate_detector.pt")

# # ---------------------------------------------------
# # 2. Folder paths
# # ---------------------------------------------------
# INPUT_DIR = "input"
# OUTPUT_DIR = "output"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------------------------------------------------
# # 3. Helper: Apply Gaussian blur
# # ---------------------------------------------------
# def blur_region(frame, x1, y1, x2, y2, ksize=(35, 35)):
#     region = frame[y1:y2, x1:x2]
#     if region.size == 0:
#         return frame
#     blurred = cv2.GaussianBlur(region, ksize, 30)
#     frame[y1:y2, x1:x2] = blurred
#     return frame

# # ---------------------------------------------------
# # 4. Process single image
# # ---------------------------------------------------
# def process_image(img_path, save_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"[WARN] Cannot read {img_path}")
#         return
#     results = plate_model(img)
#     detections = results[0].boxes.xyxy.cpu().numpy().astype(int)
#     for (x1, y1, x2, y2) in detections:
#         img = blur_region(img, x1, y1, x2, y2)
#     cv2.imwrite(save_path, img)
#     print(f"[IMG] Saved blurred image → {save_path}")

# # ---------------------------------------------------
# # 5. Process video
# # ---------------------------------------------------
# def process_video(video_path, save_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[WARN] Cannot open {video_path}")
#         return

#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

#     frame_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_count += 1

#         # Run detection every frame
#         results = plate_model(frame)
#         detections = results[0].boxes.xyxy.cpu().numpy().astype(int)

#         for (x1, y1, x2, y2) in detections:
#             frame = blur_region(frame, x1, y1, x2, y2)

#         out.write(frame)

#         if frame_count % 10 == 0:
#             print(f"[VID] Processed {frame_count} frames...")

#     cap.release()
#     out.release()
#     print(f"[VID] Saved blurred video → {save_path}")

# # ---------------------------------------------------
# # 6. Run over all files in input/
# # ---------------------------------------------------
# def main():
#     for file_name in os.listdir(INPUT_DIR):
#         in_path = os.path.join(INPUT_DIR, file_name)
#         out_path = os.path.join(OUTPUT_DIR, file_name)

#         # Image formats
#         if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
#             process_image(in_path, out_path)
#         # Video formats
#         elif file_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
#             process_video(in_path, out_path)
#         else:
#             print(f"[SKIP] Unsupported file type: {file_name}")

# if __name__ == "__main__":
#     main()
import cv2
import os
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------
# 1. Setup ONNX model
# ---------------------------------------------------
MODEL_PATH = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Mobile\backend\privacy\models\license_plate_detector.onnx"
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize ONNX session
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------------------------------------------------
# 2. Preprocess for YOLOv8
# ---------------------------------------------------
def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ---------------------------------------------------
# 3. Postprocess (decode YOLOv8 ONNX output)
# ---------------------------------------------------
def postprocess(preds, orig_w, orig_h, conf_thres=0.35):
    preds = np.squeeze(preds)
    preds = np.transpose(preds)  # (8400, 5)
    boxes = []
    for p in preds:
        # Your model gives 5 values: x_center, y_center, w, h, conf
        x_center, y_center, w, h, conf = p
        if conf > conf_thres:
            x1 = int((x_center - w / 2) * orig_w / 640)
            y1 = int((y_center - h / 2) * orig_h / 640)
            x2 = int((x_center + w / 2) * orig_w / 640)
            y2 = int((y_center + h / 2) * orig_h / 640)
            boxes.append([x1, y1, x2, y2])
    return boxes


# ---------------------------------------------------
# 4. Blur helper
# ---------------------------------------------------
def blur_region(frame, x1, y1, x2, y2, ksize=(35, 35)):
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return frame
    region = frame[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(region, ksize, 30)
    frame[y1:y2, x1:x2] = blurred
    return frame

# ---------------------------------------------------
# 5. Image processing
# ---------------------------------------------------
def process_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return

    orig_h, orig_w = img.shape[:2]
    inp = preprocess(img)
    outputs = session.run([output_name], {input_name: inp})
    boxes = postprocess(outputs[0], orig_w, orig_h)

    if not boxes:
        print(f"[NO DETECTION] {img_path}")
    else:
        for (x1, y1, x2, y2) in boxes:
            img = blur_region(img, x1, y1, x2, y2)

    cv2.imwrite(save_path, img)
    print(f"[IMG] Saved blurred image → {save_path}")

# ---------------------------------------------------
# 6. Video processing
# ---------------------------------------------------
def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        inp = preprocess(frame)
        outputs = session.run([output_name], {input_name: inp})
        boxes = postprocess(outputs[0], w, h)

        for (x1, y1, x2, y2) in boxes:
            frame = blur_region(frame, x1, y1, x2, y2)

        out.write(frame)

        if frame_count % 10 == 0:
            print(f"[VID] Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"[VID] Saved blurred video → {save_path}")

# ---------------------------------------------------
# 7. Run all files
# ---------------------------------------------------
def main():
    for file_name in os.listdir(INPUT_DIR):
        in_path = os.path.join(INPUT_DIR, file_name)
        out_path = os.path.join(OUTPUT_DIR, file_name)

        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(in_path, out_path)
        elif file_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            process_video(in_path, out_path)
        else:
            print(f"[SKIP] Unsupported file type: {file_name}")

if __name__ == "__main__":
    main()
