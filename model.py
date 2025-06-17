import torch
import torch.nn as nn
import torchvision.models as models

# Image Captioning Model (CNN-LSTM)
class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super(CaptioningModel, self).__init__()
        # CNN: ResNet50
        self.cnn = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        for param in self.cnn.parameters():
            param.requires_grad = False  # Freeze CNN
        # LSTM
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.cnn(images)
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings[:, :-1]), dim=1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs

# Segmentation Model (U-Net)
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(256, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.conv_final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        bottleneck = self.bottleneck(self.pool(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        return torch.sigmoid(self.conv_final(dec1))

if __name__ == "__main__":
    # Test models
    caption_model = CaptioningModel(vocab_size=10000)
    seg_model = UNet()
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_caption = torch.ones(1, 10, dtype=torch.long)
    print("Captioning model output shape:", caption_model(dummy_image, dummy_caption).shape)
    print("Segmentation model output shape:", seg_model(dummy_image).shape)