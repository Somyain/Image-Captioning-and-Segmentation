import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import pickle

# Paths
base_dir = r"E:\ImageProject\datasets\COCO2017"
output_dir = r"E:\ImageProject\datasets\processed\COCO2017"
os.makedirs(output_dir, exist_ok=True)

# Initialize COCO API
annot_file = os.path.join(base_dir, "annotations", "instances_train2017.json")
coco = COCO(annot_file)

# Get image IDs (limit to 10,000 for 8GB RAM)
img_ids = coco.getImgIds()[:10000]
print(f"Processing {len(img_ids)} images")

# Preprocess images and masks
def preprocess_image(img_path, size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    return np.array(img) / 255.0

def get_segmentation_mask(coco, img_id, size=(224, 224)):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    img_info = coco.loadImgs(img_id)[0]
    mask = np.zeros((img_info['height'], img_info['width']))  # Fix shape tuple
    for ann in anns:
        if 'segmentation' in ann:
            mask += coco.annToMask(ann)
    mask = (mask > 0).astype(np.uint8)  # Binary mask
    # Resize mask to match image size
    mask_img = Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)
    return np.array(mask_img)

# Process in batches
batch_size = 1000
image_arrays = {}
mask_arrays = {}
image_output_file = os.path.join(output_dir, "images.npz")
mask_output_file = os.path.join(output_dir, "masks.npz")

for i in range(0, len(img_ids), batch_size):
    batch_ids = img_ids[i:i + batch_size]
    batch_images = {}
    batch_masks = {}
    for img_id in batch_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(base_dir, "images", "train2017", img_info['file_name'])
        if os.path.exists(img_path):
            batch_images[str(img_id)] = preprocess_image(img_path)
            batch_masks[str(img_id)] = get_segmentation_mask(coco, img_id)
        else:
            print(f"Image not found: {img_path}")
    image_arrays.update(batch_images)
    mask_arrays.update(batch_masks)
    print(f"Processed batch {i // batch_size + 1}/{(len(img_ids) // batch_size) + 1}, Images in batch: {len(batch_images)}")

# Save all data
np.savez(image_output_file, **image_arrays, allow_pickle=True)
np.savez(mask_output_file, **mask_arrays, allow_pickle=True)
print(f"Saved {len(image_arrays)} images and {len(mask_arrays)} masks")

print("COCO 2017 preprocessing complete.")