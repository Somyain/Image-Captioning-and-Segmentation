import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from model.segmentation_model import SimpleSegmentationModel
from data_loader_segment import CocoSegmentationDataset

# === Paths ===
image_dir = r"datasets\COCO2017\images\train2017"
ann_path = r"datasets\COCO2017\annotations\instances_train2017.json"
model_save_path = "segmentation_model_epoch_final.pth"

# === Hyperparameters ===
batch_size = 4
num_epochs = 20
learning_rate = 0.001
image_size = (128, 128)
# subset_limit = 118000  
device = torch.device("cpu")

# === Transforms ===
transform = transforms.Compose([
    transforms.ToTensor(),
])

# === Dataset & DataLoader ===
print("[🧠] Preparing dataset...")
full_dataset = CocoSegmentationDataset(
    image_dir=image_dir,
    ann_path=ann_path,
    image_size=image_size,
    transform=transform
)

# Subset of 20,000 samples
# subset_indices = list(range(min(subset_limit, len(full_dataset))))
# subset_dataset = Subset(full_dataset, subset_indices)
# dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
total_steps = len(dataloader)

# === Model, Loss, Optimizer ===
num_classes = 91  # COCO has 91 categories in segmentation
model = SimpleSegmentationModel(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
print(f"[🚀] Starting training on {device} using 20,000 images...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs} started...")

    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        percent = (i + 1) / total_steps * 100
        print(f"  ⏳ Batch [{i+1}/{total_steps}] - Loss: {loss.item():.4f} ({percent:.1f}% done)", end="\r")

    avg_loss = total_loss / total_steps
    elapsed = time.time() - start_time
    print(f"\n✅ Epoch [{epoch+1}/{num_epochs}] completed - Avg Loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")

    # Save checkpoint
    epoch_path = f"segmentation_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), epoch_path)
    print(f"💾 Saved model checkpoint to {epoch_path}")

# === Final Save ===
torch.save(model.state_dict(), model_save_path)
print(f"\n🎯 Epoch {epoch+1} complete. Model saved to {model_save_path}")

