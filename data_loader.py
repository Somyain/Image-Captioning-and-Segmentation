import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

# Flickr8k Dataset
class Flickr8kDataset(Dataset):
    def __init__(self, image_npz, captions_pkl, vocab_pkl, transform=None):
        self.images = np.load(image_npz, allow_pickle=True)
        self.captions = pd.read_pickle(captions_pkl)
        with open(vocab_pkl, 'rb') as f:
            self.vocab = pickle.load(f)
        self.transform = transform
        self.image_names = list(self.images.files)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions.iloc[idx]
        img_name = caption_data['image']
        caption = caption_data['indices']
        image = self.images[img_name]
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(caption, dtype=torch.long)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    # Pad captions to the maximum length in the batch
    max_len = max(len(c) for c in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions

def get_flickr8k_loader(image_npz, captions_pkl, vocab_pkl, batch_size=32):
    dataset = Flickr8kDataset(image_npz, captions_pkl, vocab_pkl)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# COCO Dataset
class COCODataset(Dataset):
    def __init__(self, image_npz, mask_npz, transform=None):
        self.images = np.load(image_npz, allow_pickle=True)
        self.masks = np.load(mask_npz, allow_pickle=True)
        self.transform = transform
        self.image_ids = list(self.images.files)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        image = self.images[img_id]
        mask = self.masks[img_id]
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def get_coco_loader(image_npz, mask_npz, batch_size=32):
    dataset = COCODataset(image_npz, mask_npz)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    flickr_loader = get_flickr8k_loader(
        r"E:\ImageProject\datasets\processed\Flickr8k\images.npz",
        r"E:\ImageProject\datasets\processed\Flickr8k\captions.pkl",
        r"E:\ImageProject\datasets\processed\Flickr8k\vocab.pkl"
    )
    coco_loader = get_coco_loader(
        r"E:\ImageProject\datasets\processed\COCO2017\images.npz",
        r"E:\ImageProject\datasets\processed\COCO2017\masks.npz"
    )
    for images, captions in flickr_loader:
        print("Flickr8k batch:", images.shape, captions.shape, images.dtype, captions.dtype)
        break
    for images, masks in coco_loader:
        print("COCO batch:", images.shape, masks.shape, images.dtype, masks.dtype)
        break