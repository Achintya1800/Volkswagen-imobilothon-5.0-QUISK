# pipeline_integration/adapters/dehaze_adapter.py
"""
Dehaze adapter for the pipeline.

Provides:
- dehaze_frame(cv2_bgr_array) -> cv2_bgr_array  (callable from orchestrator)
- CLI mode: process all frames in storage/<job>/frames -> storage/<job>/dehaze

Usage (CLI):
python pipeline_integration/adapters/dehaze_adapter.py --job demo1 --root storage --max 200

"""
from pathlib import Path
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import json

# ------------------------
# Model (same as your AOD-Net)
# ------------------------
class DehazeNet(nn.Module):
    def __init__(self):
        super(DehazeNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(torch.cat((x1, x2), 1)))
        x4 = self.relu(self.e_conv4(torch.cat((x2, x3), 1)))
        x5 = self.relu(self.e_conv5(torch.cat((x1, x2, x3, x4), 1)))
        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image

# ------------------------
# Globals / transforms
# ------------------------
_transform_to_tensor = transforms.ToTensor()
_to_pil = transforms.ToPILImage()

# Lazy model holder
_MODEL = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_PATHS = [
    Path(__file__).resolve().parents[2] / "dehazing" / "models" / "dehazer.pth",
    Path(__file__).resolve().parents[2] / "dehazing" / "models" / "dehazer.pt",
    Path(__file__).resolve().parents[2] / "dehazing" / "model.pth",
]

def _find_model_path():
    for p in _MODEL_PATHS:
        if p.exists():
            return p
    # fallback to models folder next to this adapter
    alt = Path(__file__).resolve().parent / "models" / "dehazer.pth"
    if alt.exists():
        return alt
    return None

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    model_path = _find_model_path()
    if model_path is None:
        raise FileNotFoundError("Dehaze model not found. Expected in dehazing/models/dehazer.pth or pipeline_integration/adapters/models/")
    model = DehazeNet().to(_DEVICE)
    checkpoint = torch.load(str(model_path), map_location=_DEVICE)
    # If checkpoint is a state_dict inside dict, handle both cases
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()
    _MODEL = model
    return _MODEL

# ------------------------
# Utility: convert between cv2 (BGR numpy) and PIL
# ------------------------
def _cv2_to_pil(bgr: np.ndarray) -> Image.Image:
    # convert BGR->RGB then to PIL
    rgb = bgr[..., ::-1]
    return Image.fromarray(rgb)

def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img)  # RGB
    bgr = arr[..., ::-1]
    return bgr

# ------------------------
# Public: dehaze a single frame (numpy BGR)
# ------------------------
def dehaze_frame(bgr_frame: np.ndarray) -> np.ndarray:
    """
    Input: bgr_frame (H,W,3) uint8 (OpenCV)
    Output: bgr_frame_dehazed (H,W,3) uint8
    """
    model = _load_model()
    # PIL conversion -> tensor -> batch
    pil = _cv2_to_pil(bgr_frame).convert("RGB")
    tensor = _transform_to_tensor(pil).unsqueeze(0).to(_DEVICE)  # 1,C,H,W
    with torch.no_grad():
        out = model(tensor)
        out = out.squeeze(0).cpu().clamp(0.0, 1.0)
    pil_out = _to_pil(out)
    bgr_out = _pil_to_cv2(pil_out)
    # Convert float array [0..255] if needed and dtype
    if bgr_out.dtype != bgr_frame.dtype:
        bgr_out = bgr_out.astype(bgr_frame.dtype)
    return bgr_out

# ------------------------
# CLI: process a folder of frames for a job
# ------------------------
def process_job(job: str = "test1", root: str = "storage", max_frames: int = 1000):
    storage_root = Path(root)
    job_dir = storage_root / job
    frames_dir = job_dir / "frames"
    out_dir = job_dir / "dehaze"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = _find_model_path()
    print(f"Device: {_DEVICE}   Model: {model_path}")
    _load_model()

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    processed = []
    for i, p in enumerate(frame_paths):
        if i >= max_frames:
            break
        try:
            import cv2
            img = cv2.imread(str(p))
            if img is None:
                print(f"Warning: cannot read {p}, skipping")
                continue
            out_img = dehaze_frame(img)
            out_path = out_dir / p.name.replace("frame_", "dehaze_")
            cv2.imwrite(str(out_path), out_img)
            processed.append(str(out_path))
            if (i+1) % 20 == 0:
                print(f"Processed {i+1}/{len(frame_paths)}")
        except Exception as e:
            print(f"ERROR processing {p}: {e}")

    # write simple status
    status = {
        "job": job,
        "processed_count": len(processed),
        "processed": processed
    }
    status_path = job_dir / "dehaze_status.json"
    with open(status_path, "w") as fh:
        json.dump(status, fh, indent=2)
    print(f"Wrote status to {status_path}")
    return status

# ------------------------
# CLI entrypoint
# ------------------------
def _cli_main():
    p = argparse.ArgumentParser()
    p.add_argument("--job", default="test1")
    p.add_argument("--root", default="storage")
    p.add_argument("--max", type=int, default=1000)
    args = p.parse_args()
    process_job(args.job, args.root, args.max)

if __name__ == "__main__":
    _cli_main()
