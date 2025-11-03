import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# =========================================================
# ✅ MODEL DEFINITION (exact same as original AOD-Net impl)
# =========================================================
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


# =========================================================
# ✅ PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(BASE_DIR, "input")
output_folder = os.path.join(BASE_DIR, "output_GAN")
model_path = os.path.join(BASE_DIR, "models", "dehazer.pth")

os.makedirs(output_folder, exist_ok=True)

# =========================================================
# ✅ DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# =========================================================
# ✅ LOAD MODEL
# =========================================================
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found at {model_path}")

model = DehazeNet().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# =========================================================
# ✅ IMAGE TRANSFORMS
# =========================================================
transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# =========================================================
# ✅ RUN DEHAZING ON ALL IMAGES
# =========================================================
for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, file)
        img = Image.open(img_path).convert("RGB")

        inp = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(inp)

        out_img = to_pil(output.squeeze(0).cpu().clamp(0, 1))
        out_path = os.path.join(output_folder, file)
        out_img.save(out_path)
        print(f"✅ Saved dehazed image: {out_path}")

print("🎉 All images processed successfully!")
