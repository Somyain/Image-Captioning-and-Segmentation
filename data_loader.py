import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import logging

# Set up logging
logging.basicConfig(filename=r"E:\ImageProject\data_loader.log", level=logging.WARNING, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Flickr8k Dataset
class Flickr8kDataset(Dataset):
    def __init__(self, image_npz, captions_pkl, vocab_pkl, transform=None):
        self.images = np.load(image_npz, allow_pickle=True)
        self.captions = pd.read_pickle(captions_pkl)
        with open(vocab_pkl, 'rb') as f:
            self.vocab = pickle.load(f)
        self.transform = transform
        self.image_names = [name for name in self.images.files if name != 'allow_pickle']
        if not self.image_names:
            raise ValueError("No valid images found in Flickr8k dataset")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption_data = self.captions.iloc[idx]
        img_name = caption_data['image']
        caption = caption_data['indices']
        image = self.images[img_name]
        if not isinstance(image, np.ndarray) or image.shape != (224, 224, 3):
            logging.error(f"Invalid Flickr8k image {img_name}: {image.shape if isinstance(image, np.ndarray) else type(image)}")
            raise ValueError(f"Invalid image {img_name}")
        if self.transform:
            image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(caption, dtype=torch.long)

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    max_len = max(len(c) for c in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    return images, padded_captions

def get_flickr8k_loader(image_npz, captions_pkl, vocab_pkl, batch_size=8):
    dataset = Flickr8kDataset(image_npz, captions_pkl, vocab_pkl)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# COCO Dataset
class COCODataset(Dataset):
    def __init__(self, image_npz, mask_npz, transform=None):
        self.images = np.load(image_npz, allow_pickle=True)
        self.masks = np.load(mask_npz, allow_pickle=True)
        self.transform = transform
        self.image_ids = [img_id for img_id in self.images.files if img_id != 'allow_pickle']
        # Validate images and masks
        self.valid_ids = []
        for img_id in self.image_ids:
            try:
                img = self.images[img_id]
                mask = self.masks[img_id]
                if (isinstance(img, np.ndarray) and img.shape == (224, 224, 3) and img.size > 0 and
                    isinstance(mask, np.ndarray) and mask.shape == (224, 224) and mask.size > 0):
                    self.valid_ids.append(img_id)
                else:
                    logging.warning(f"Invalid image/mask for ID {img_id}: img={img.shape if isinstance(img, np.ndarray) else type(img)}, mask={mask.shape if isinstance(mask, np.ndarray) else type(mask)}")
            except Exception as e:
                logging.error(f"Error processing ID {img_id}: {str(e)}")
        print(f"Valid COCO images: {len(self.valid_ids)}/{len(self.image_ids)}")
        if not self.valid_ids:
            raise ValueError("No valid images found in COCO dataset")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        image = self.images[img_id]
        mask = self.masks[img_id]
        if not (isinstance(image, np.ndarray) and image.shape == (224, 224, 3)):
            logging.error(f"Invalid COCO image {img_id}: {image.shape if isinstance(image, np.ndarray) else type(image)}")
            raise ValueError(f"Invalid image {img_id}")
        if not (isinstance(mask, np.ndarray) and mask.shape == (224, 224)):
            logging.error(f"Invalid COCO mask {img_id}: {mask.shape if isinstance(mask, np.ndarray) else type(mask)}")
            raise ValueError(f"Invalid mask {img_id}")
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def get_coco_loader(image_npz, mask_npz, batch_size=4):
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