import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (B, 32, H, W)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 32, H/2, W/2)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B, 64, H/4, W/4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (B, 32, H/2, W/2)
            nn.ReLU(),

            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),  # (B, num_classes, H, W)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # shape: (B, num_classes, H, W)
