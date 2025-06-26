import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, caption_file, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.image_captions = self._load_captions(caption_file)
        print(f"[📂] Found {len(self.image_captions)} valid image-caption pairs.")

    def _load_captions(self, file_path):
        data = []
        missing = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["image"].strip()
                caption = row["caption"].strip()
                img_path = os.path.join(self.image_dir, filename)
                if os.path.exists(img_path):
                    data.append((filename, caption))
                else:
                    missing += 1
        print(f"[🔍] Loaded {len(data)} captions. Missing images: {missing}")
        return data

    def __len__(self):
        return len(self.image_captions)

    def __getitem__(self, idx):
        filename, caption = self.image_captions[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Convert caption to numerical tokens
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)
        tokens += [self.vocab.stoi["<EOS>"]]

        return image, torch.tensor(tokens)

    def collate_fn(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images)

        lengths = [len(cap) for cap in captions]
        max_len = max(lengths)
        padded = torch.zeros(len(captions), max_len).long()

        for i, cap in enumerate(captions):
            end = lengths[i]
            padded[i, :end] = cap[:end]

        return images, padded
