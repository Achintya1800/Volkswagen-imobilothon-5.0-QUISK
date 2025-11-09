from ultralytics import YOLO
import os

# Load your trained YOLOv8 segmentation model
model_path = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\model\YoloV8Segmented.pt"
model = YOLO(model_path)

# Input and output paths
input_folder = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\input"
output_folder = r"C:\Users\Nidhi Patel\OneDrive\Desktop\Volkswagen-imobilothon-5.0-QUISK\H2Segmentation\segmented_output"

# Run inference
results = model.predict(
    source=input_folder,      # can be folder or single image
    conf=0.5,                 # confidence threshold
    save=True,                # saves annotated images
    save_txt=False,           # skip YOLO txt labels if you don’t need them
    project=output_folder,    # output folder for results
    name="segmented"          # subfolder name
)

print(f"\n✅ Segmented images saved at: {output_folder}\\segmented")
