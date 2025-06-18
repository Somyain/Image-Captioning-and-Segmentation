import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import get_flickr8k_loader
from model import CaptioningModel
import pickle
import os

# Paths
data_dir = r"E:\ImageProject\datasets\processed\Flickr8k"
model_dir = r"E:\ImageProject\models"
os.makedirs(model_dir, exist_ok=True)

# Hyperparameters
batch_size = 8  # Small batch size for 8GB RAM
num_epochs = 5
learning_rate = 0.001
device = torch.device("cpu")  # Use CPU

# Load vocabulary
with open(os.path.join(data_dir, "vocab.pkl"), 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# Initialize data loader
train_loader = get_flickr8k_loader(
    os.path.join(data_dir, "images.npz"),
    os.path.join(data_dir, "captions.pkl"),
    os.path.join(data_dir, "vocab.pkl"),
    batch_size=batch_size
)

# Initialize model
model = CaptioningModel(vocab_size=vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])  # Ignore padding
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (images, captions) in enumerate(train_loader):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs = model(images, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), os.path.join(model_dir, f"caption_model_epoch_{epoch+1}.pth"))

print("Captioning model training complete.")