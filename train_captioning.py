import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vocab import Vocabulary
# Make sure the class name matches the actual class in data_loader_caption.py
from data_loader_caption import Flickr8kDataset  # Change 'Flickr8kDataset' to the correct class name if needed
from model.captioning_model import CaptioningModel

# === Paths ===
image_dir = r"datasets\Flickr8k\Images"
caption_file = r"datasets\Flickr8k\captions.txt"
model_save_path = "captioning_model_final.pth"

# === Hyperparameters ===
batch_size = 64
num_epochs = 10
learning_rate = 0.001
embed_size = 256
hidden_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Prepare dataset and vocabulary ===
print("[🧠] Building vocabulary and loading dataset...")
vocab = Vocabulary(caption_file, freq_threshold=5)
dataset = Flickr8kDataset(image_dir, caption_file, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

# === Model, Loss, Optimizer ===
model = CaptioningModel(embed_size, hidden_size, len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
print(f"[🚀] Starting training on {device}...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    print(f"\n🔁 Epoch {epoch+1}/{num_epochs}")

    for i, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)

        outputs = model(images, captions[:, :-1])
        loss = criterion(outputs[:, 1:, :].reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"  ⏳ Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}", end="\r")

    avg_loss = total_loss / len(dataloader)
    elapsed = time.time() - start_time
    print(f"\n✅ Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f} - Time: {elapsed:.1f}s")

    # Save model checkpoint
    checkpoint_path = f"captioning_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Saved checkpoint: {checkpoint_path}")

# === Final Save ===
torch.save(model.state_dict(), model_save_path)
print(f"\n🎯 Final model saved to {model_save_path}")
