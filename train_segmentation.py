import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_coco_loader
from model import UNet
import os

# Paths
data_dir = r"E:\ImageProject\datasets\processed\COCO2017"
model_dir = r"E:\ImageProject\models"
os.makedirs(model_dir, exist_ok=True)

# Hyperparameters
batch_size = 4  # Small batch size for 8GB RAM
num_epochs = 5
learning_rate = 0.001
device = torch.device("cpu")  # Use CPU

# Initialize data loader
train_loader = get_coco_loader(
    os.path.join(data_dir, "images.npz"),
    os.path.join(data_dir, "masks.npz"),
    batch_size=batch_size
)

# Initialize model
model = UNet().to(device)
criterion = nn.BCELoss()  # Binary cross-entropy for segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(model_dir, f"segmentation_model_epoch_{epoch+1}.pth"))

print("Segmentation model training complete.")