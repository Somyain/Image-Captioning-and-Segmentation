import numpy as np
import os

images_path = r"E:\ImageProject\datasets\processed\COCO2017\images.npz"
masks_path = r"E:\ImageProject\datasets\processed\COCO2017\masks.npz"

# Load data
images = np.load(images_path, allow_pickle=True)
masks = np.load(masks_path, allow_pickle=True)

# Check images
print(f"Total images: {len(images.files)}")
invalid_images = []
for img_id in images.files:
    img = images[img_id]
    if not (isinstance(img, np.ndarray) and img.shape == (224, 224, 3)):
        invalid_images.append((img_id, img.shape if isinstance(img, np.ndarray) else type(img)))
print(f"Invalid images: {len(invalid_images)}")
if invalid_images:
    print("Sample invalid images:", invalid_images[:5])

# Check masks
print(f"Total masks: {len(masks.files)}")
invalid_masks = []
for mask_id in masks.files:
    mask = masks[mask_id]
    if not (isinstance(mask, np.ndarray) and mask.shape == (224, 224)):
        invalid_masks.append((mask_id, mask.shape if isinstance(mask, np.ndarray) else type(mask)))
print(f"Invalid masks: {len(invalid_masks)}")
if invalid_masks:
    print("Sample invalid masks:", invalid_masks[:5])

# Check matching IDs
mismatched_ids = set(images.files) ^ set(masks.files)
print(f"Mismatched image/mask IDs: {len(mismatched_ids)}")
if mismatched_ids:
    print("Sample mismatched IDs:", list(mismatched_ids)[:5])