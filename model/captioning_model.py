import torch
import torch.nn as nn

class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CaptioningModel, self).__init__()

        # === CNN Encoder from scratch ===
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 224 → 112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 112 → 56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56 → 28
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # output: (B, 64, 1, 1)
        )
        self.enc_fc = nn.Linear(64, embed_size)  # project to embedding size

        # === Decoder ===
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images).squeeze()  # (B, 64)
        features = self.enc_fc(features)           # (B, embed_size)

        embeddings = self.embed(captions)          # (B, T, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        hiddens, _ = self.lstm(embeddings)         # (B, T+1, hidden_size)
        outputs = self.fc(hiddens)                 # (B, T+1, vocab_size)

        return outputs