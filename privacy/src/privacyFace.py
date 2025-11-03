# import os
# import cv2
# from ultralytics import YOLO

# # ---------------------------------------------------
# # 1. Load your YOLO face detection model
# # ---------------------------------------------------
# model_path = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Mobile\backend\privacy\models\face_detection_model.pt"
# face_model = YOLO(model_path)

# # ---------------------------------------------------
# # 2. Function to blur detected faces in one frame
# # ---------------------------------------------------
# def blur_faces(frame):
#     results = face_model(frame, verbose=False)
#     for result in results:
#         boxes = result.boxes.xyxy  # (x1, y1, x2, y2)
#         for box in boxes:
#             x1, y1, x2, y2 = map(int, box[:4])
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
#             face_roi = frame[y1:y2, x1:x2]
#             if face_roi.size > 0:
#                 blur = cv2.GaussianBlur(face_roi, (99, 99), 30)
#                 frame[y1:y2, x1:x2] = blur
#     return frame

# # ---------------------------------------------------
# # 3. Process all images/videos
# # ---------------------------------------------------
# def process_all(input_dir="input", output_dir="output_faces"):
#     os.makedirs(output_dir, exist_ok=True)
#     files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]

#     for f in files:
#         in_path = os.path.join(input_dir, f)
#         out_path = os.path.join(output_dir, "face_blurred_" + f)

#         # ----- Image -----
#         if f.lower().endswith(('.jpg', '.jpeg', '.png')):
#             frame = cv2.imread(in_path)
#             if frame is None:
#                 print(f"⚠️ Skipped unreadable file: {f}")
#                 continue
#             blurred = blur_faces(frame)
#             cv2.imwrite(out_path, blurred)
#             print(f"✅ Image saved → {out_path}")

#         # ----- Video -----
#         elif f.lower().endswith('.mp4'):
#             cap = cv2.VideoCapture(in_path)
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             fps = int(cap.get(cv2.CAP_PROP_FPS))
#             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#                 frame = blur_faces(frame)
#                 out.write(frame)

#             cap.release()
#             out.release()
#             print(f"🎥 Video saved → {out_path}")

# # ---------------------------------------------------
# # 4. Run
# # ---------------------------------------------------
# if __name__ == "__main__":
#     process_all()


import os
import cv2
import numpy as np
import onnxruntime as ort

# ---------------------------------------------------
# 1. Load the ONNX model
# ---------------------------------------------------
model_path = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Mobile\backend\privacy\models\face_detection_model.onnx"
session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"📦 Model loaded successfully: {model_path}")
print(f"🔹 Input name: {input_name}")
print(f"🔹 Output name: {output_name}")

# ---------------------------------------------------
# 2. Preprocess
# ---------------------------------------------------
def preprocess(frame, size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.transpose(2, 0, 1) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

# ---------------------------------------------------
# 3. Postprocess (auto-detects YOLOv5/v8 output type)
# ---------------------------------------------------
def postprocess(outputs, frame, conf_thresh=0.3, img_size=640):
    preds = outputs[0]

    # Debug info
    print("🧾 Raw output shape:", preds.shape)

    # Handle (1, 84, N) -> YOLOv8 style
    if len(preds.shape) == 3 and preds.shape[1] < preds.shape[2]:
        preds = preds.transpose(0, 2, 1)
        print("🔄 Transposed predictions for YOLOv8 format.")

    h, w = frame.shape[:2]
    boxes = []

    for det in preds[0]:
        conf = det[4]
        if conf > conf_thresh:
            x, y, bw, bh = det[:4]
            # YOLO outputs are normalized center coordinates in [0, 1]
            x1 = int((x - bw / 2) * w / img_size)
            y1 = int((y - bh / 2) * h / img_size)
            x2 = int((x + bw / 2) * w / img_size)
            y2 = int((y + bh / 2) * h / img_size)
            boxes.append((x1, y1, x2, y2))

    print(f"📦 Found {len(boxes)} boxes with conf > {conf_thresh}")
    return boxes

# ---------------------------------------------------
# 4. Blur faces in a frame
# ---------------------------------------------------
def blur_faces(frame):
    img = preprocess(frame)
    outputs = session.run([output_name], {input_name: img})
    boxes = postprocess(outputs, frame)

    for (x1, y1, x2, y2) in boxes:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size > 0:
            blur = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y1:y2, x1:x2] = blur
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # optional for debugging
    return frame

# ---------------------------------------------------
# 5. Process all images/videos
# ---------------------------------------------------
def process_all(input_dir="input", output_dir="output_faces"):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4'))]

    for f in files:
        in_path = os.path.join(input_dir, f)
        out_path = os.path.join(output_dir, "face_blurred_" + f)

        # ---------- IMAGE ----------
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            frame = cv2.imread(in_path)
            if frame is None:
                print(f"⚠️ Skipped unreadable file: {f}")
                continue
            blurred = blur_faces(frame)
            cv2.imwrite(out_path, blurred)
            print(f"✅ Image saved → {out_path}")

        # ---------- VIDEO ----------
        elif f.lower().endswith('.mp4'):
            cap = cv2.VideoCapture(in_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame = blur_faces(frame)
                out.write(frame)
                if frame_count % 10 == 0:
                    print(f"🟢 Processed {frame_count} frames...")

            cap.release()
            out.release()
            print(f"🎥 Video saved → {out_path}")

# ---------------------------------------------------
# 6. Run on input folder
# ---------------------------------------------------
if __name__ == "__main__":
    process_all()
