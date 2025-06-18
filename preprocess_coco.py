import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import logging

logging.basicConfig(filename=r"E:\ImageProject\preprocess_coco.log", level=logging.WARNING,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def get_segmentation_mask(coco, img_id, target_size=(224, 224)):
    img_info = coco.loadImgs(img_id)[0]
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    mask = np.zeros(target_size, dtype=np.uint8)  # Use uint8 for binary mask
    for ann in anns:
        if 'segmentation' in ann:
            # Get mask and resize to 224x224
            ann_mask = coco.annToMask(ann)
            ann_mask = Image.fromarray(ann_mask).resize(target_size, Image.Resampling.NEAREST)
            ann_mask = np.array(ann_mask, dtype=np.uint8)
            mask = np.maximum(mask, ann_mask)  # Combine masks
    return mask

def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
        return img_array
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None

if __name__ == "__main__":
    data_dir = r"E:\ImageProject\datasets\COCO2017"
    annotation_file = os.path.join(data_dir, "annotations", "instances_train2017.json")
    image_dir = os.path.join(data_dir, "images", "train2017")
    output_dir = r"E:\ImageProject\datasets\processed\COCO2017"
    os.makedirs(output_dir, exist_ok=True)

    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()[:10000]  # Limit to 10,000 images

    batch_images = {}
    batch_masks = {}
    valid_count = 0

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        img_array = preprocess_image(img_path)
        if img_array is None:
            continue
        mask = get_segmentation_mask(coco, img_id)
        if mask is None or mask.shape != (224, 224):
            logging.warning(f"Invalid mask for {img_id}: {mask.shape if mask is not None else None}")
            continue
        batch_images[str(img_id)] = img_array
        batch_masks[str(img_id)] = mask
        valid_count += 1
        if valid_count % 100 == 0:
            logging.info(f"Processed {valid_count} valid images")

    np.savez(os.path.join(output_dir, "images.npz"), **batch_images)
    np.savez(os.path.join(output_dir, "masks.npz"), **batch_masks)
    print(f"Saved {valid_count} valid images and masks")
    logging.info(f"Saved {valid_count} valid images and masks")