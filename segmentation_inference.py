import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from model.segmentation_model import SimpleSegmentationModel

# === Paths ===
model_path = "segmentation_model.pth"
image_dir = "datasets/COCO2017/images/train2017"
output_dir = "segmentation_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Device ===
device = torch.device("cpu")

# === Load Model ===
print("[🧠] Loading model...")
model = SimpleSegmentationModel(num_classes=91)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Decode Segmentation Mask ===
def decode_segmap(mask):
    np.random.seed(42)
    label_colors = np.random.randint(0, 255, (91, 3))
    color_mask = label_colors[mask]
    return color_mask.astype(np.uint8)

# === Get first 50 images ===
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')][15:20]

# === Process Images ===
print(f"[🚀] Processing {len(image_files)} images...")

for img_file in tqdm(image_files):
    img_path = os.path.join(image_dir, img_file)
    image = Image.open(img_path).convert("RGB")
    original_size = image.size

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    decoded_mask = decode_segmap(predicted_mask)
    decoded_mask_resized = cv2.resize(decoded_mask, original_size)

    original_np = np.array(image)
    combined = np.concatenate((original_np, decoded_mask_resized), axis=1)

    out_path = os.path.join(output_dir, f"output_{img_file}")
    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

print(f"[✅] All images saved to: {output_dir}")
