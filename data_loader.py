import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import logging

logging.basicConfig(
    filename=r"E:\ImageProject\data_loader.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Flickr8kDataset(Dataset):
    def __init__(self, image_path, caption_path, vocab_path, split='train'):
        try:
            self.images = np.load(image_path, allow_pickle=True)
            with open(caption_path, 'rb') as f:
                self.captions = pickle.load(f)
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading Flickr8k data: {e}")
            raise
        self.keys = [k for k in self.images.keys() if isinstance(k, str)]
        total_len = len(self.keys)
        val_size = int(0.1 * total_len)
        if split == 'val':
            self.keys = self.keys[:val_size]
        else:
            self.keys = self.keys[val_size:]
        logging.info(f"Flickr8k {split} split: {len(self.keys)} images")
        if len(self.keys) == 0:
            logging.error(f"No valid keys found for Flickr8k {split} split")
            raise ValueError(f"No valid keys found for Flickr8k {split} split")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_key = self.keys[idx]
        try:
            image = self.images[img_key]
            captions = self.captions[img_key]
            caption = random.choice(captions)
            if len(caption) != 20:
                logging.warning(f"Caption length {len(caption)} for {img_key}, expected 20")
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            caption = torch.tensor(caption, dtype=torch.long)
            return image, caption
        except Exception as e:
            logging.error(f"Error accessing {img_key}: {e}")
            raise

class COCODataset(Dataset):
    def __init__(self, image_path, mask_path, split='train'):
        try:
            self.images = np.load(image_path, allow_pickle=True)
            self.masks = np.load(mask_path, allow_pickle=True)
        except Exception as e:
            logging.error(f"Error loading COCO data: {e}")
            raise
        self.keys = [k for k in self.images.keys() if isinstance(k, str)]
        total_len = len(self.keys)
        val_size = int(0.1 * total_len)
        if split == 'val':
            self.keys = self.keys[:val_size]
        else:
            self.keys = self.keys[val_size:]
        logging.info(f"COCO {split} split: {len(self.keys)} images")
        if len(self.keys) == 0:
            logging.error(f"No valid keys found for COCO {split} split")
            raise ValueError(f"No valid keys found for COCO {split} split")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_key = self.keys[idx]
        try:
            image = self.images[img_key]
            mask = self.masks[img_key]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
            return image, mask
        except Exception as e:
            logging.error(f"Error accessing {img_key}: {e}")
            raise

def get_flickr8k_loader(image_path, caption_path, vocab_path, batch_size=8, split='train'):
    dataset = Flickr8kDataset(image_path, caption_path, vocab_path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))

def get_coco_loader(image_path, mask_path, batch_size=4, split='train'):
    dataset = COCODataset(image_path, mask_path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))

if __name__ == "__main__":
    logging.info("Testing data loaders")
    try:
        flickr_loader = get_flickr8k_loader(
            r"E:\ImageProject\datasets\processed\Flickr8k\images.npz",
            r"E:\ImageProject\datasets\processed\Flickr8k\captions.pkl",
            r"E:\ImageProject\datasets\processed\Flickr8k\vocab.pkl",
            split='train'
        )
        coco_loader = get_coco_loader(
            r"E:\ImageProject\datasets\processed\COCO2017\images.npz",
            r"E:\ImageProject\datasets\processed\COCO2017\masks.npz",
            split='train'
        )
        for images, captions in flickr_loader:
            print(f"Flickr8k batch: {images.shape} {captions.shape} {images.dtype} {captions.dtype}")
            logging.info(f"Flickr8k batch: {images.shape} {captions.shape}")
            break
        for images, masks in coco_loader:
            print(f"COCO batch: {images.shape} {masks.shape} {images.dtype} {masks.dtype}")
            logging.info(f"COCO batch: {images.shape} {masks.shape}")
            break
        logging.info("Data loader test completed")
    except Exception as e:
        logging.error(f"Data loader test failed: {e}")
        raise