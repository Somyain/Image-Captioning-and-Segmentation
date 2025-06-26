import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from collections import Counter
from pycocotools.coco import COCO
import numpy as np
import pickle

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, captions, save_path):
        counter = Counter()
        for caption in captions:
            counter.update(caption.lower().split())
        for word, count in counter.items():
            if count >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def numericalize(self, caption):
        tokens = caption.lower().split()
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class Flicker8kDataset(Dataset):
    def __init__(self, img_dir, caption_file, transform=None, vocab=None):
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab
        self.captions = pd.read_csv(caption_file, sep=',', header=0, names=['image', 'caption'])
        self.captions['image'] = self.captions['image'].str.strip()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        caption = row['caption']
        if self.transform:
            image = self.transform(image)
        if self.vocab:
            caption = [1] + self.vocab.numericalize(caption) + [2]
            caption = torch.tensor(caption)
        return image, caption

class CocoSegmentation(Dataset):
    def __init__(self, root, ann_file, transform=None, max_classes=10):
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.max_classes = max_classes
        self.ids = list(self.coco.imgs.keys())[:100]  # Reduced for testing

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(path).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros((img_info['height'], img_info['width']))
        for ann in anns:
            cat_id = min(ann['category_id'], self.max_classes - 1)
            mask[self.coco.annToMask(ann) == 1] = cat_id
        mask = Image.fromarray(mask.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((224, 224))(mask)
            mask = torch.tensor(np.array(mask), dtype=torch.long)
        return image, mask

    def __len__(self):
        return len(self.ids)

def get_flicker_loader(img_dir, caption_file, vocab, batch_size=16, train=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = Flicker8kDataset(img_dir, caption_file, transform=transform, vocab=vocab if train else None)
    if train:
        def collate_fn(batch):
            images, captions = zip(*batch)
            images = torch.stack(images)
            max_len = 20
            padded_captions = []
            for cap in captions:
                if len(cap) < max_len:
                    pad = torch.zeros(max_len - len(cap), dtype=torch.long)
                    cap = torch.cat((cap[:max_len], pad))
                else:
                    cap = cap[:max_len]
                padded_captions.append(cap)
            captions = torch.stack(padded_captions)
            return images, captions
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    return dataset

def get_coco_loader(root, ann_file, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = CocoSegmentation(root, ann_file, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
