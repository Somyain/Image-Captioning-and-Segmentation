import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import cv2


class CocoSegmentationDataset(Dataset):
    def __init__(self, image_dir, ann_path, image_size=(256, 256), transform=None):
        self.image_dir = image_dir
        self.coco = COCO(ann_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        # Load and resize image
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.image_size)

        # Load and build the segmentation mask
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        mask = np.zeros(self.image_size, dtype=np.uint8)

        for ann in anns:
            if ann['iscrowd']:
                continue
            category_id = ann['category_id']
            rle = self.coco.annToMask(ann)
            rle = cv2.resize(rle, self.image_size, interpolation=cv2.INTER_NEAREST)
            mask[rle == 1] = category_id

        # Apply optional transform
        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask
