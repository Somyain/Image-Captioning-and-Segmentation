import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import logging
import zipfile

logging.basicConfig(
    filename=r"E:\ImageProject\preprocess_coco.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_image(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        logging.error(f"Image file does not exist: {img_path}")
        return None
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        logging.debug(f"Processed image: {img_path}, shape: {img_array.shape}")
        return img_array
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None

def preprocess_mask(ann, img_shape, target_size=(224, 224)):
    try:
        from pycocotools import mask as mask_utils
        rle = ann['segmentation']
        if isinstance(rle, list):
            rle = mask_utils.frPyObjects(rle, img_shape[0], img_shape[1])
        decoded_mask = mask_utils.decode(rle)
        if decoded_mask.ndim == 2:
            decoded_mask = decoded_mask[:, :, np.newaxis]
        mask_img = Image.fromarray(decoded_mask[:, :, 0].astype(np.uint8) * 255)
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)
        mask_array = np.array(mask_img, dtype=np.float32) / 255.0
        logging.debug(f"Processed mask for ann {ann['id']}, shape: {mask_array.shape}")
        return mask_array
    except Exception as e:
        logging.error(f"Error processing mask for ann {ann['id']}: {e}")
        return None

def verify_npz(file_path):
    try:
        with zipfile.ZipFile(file_path, 'r') as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                logging.error(f"Corrupted file in {file_path}: {bad_file}")
                return False
        return True
    except Exception as e:
        logging.error(f"Error verifying {file_path}: {e}")
        return False

if __name__ == "__main__":
    data_dir = r"E:\ImageProject\datasets\COCO2017"
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")
    image_dir = os.path.join(data_dir, "images", "val2017")
    output_dir = r"E:\ImageProject\datasets\processed\COCO2017"
    os.makedirs(output_dir, exist_ok=True)

    # Verify paths
    if not os.path.exists(ann_file):
        logging.error(f"Annotation file not found: {ann_file}")
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not os.path.exists(image_dir):
        logging.error(f"Image directory not found: {image_dir}")
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    logging.info(f"Loading annotations from {ann_file}")
    try:
        coco = COCO(ann_file)
    except Exception as e:
        logging.error(f"Error loading COCO annotations: {e}")
        raise
    img_ids = coco.getImgIds()
    logging.info(f"Found {len(img_ids)} image IDs")

    batch_images = {}
    batch_masks = {}
    valid_count = 0
    for img_id in img_ids[:10001]:  # Limit for testing
        img_info = coco.loadImgs(img_id)[0]
        img_name = img_info['file_name']
        img_path = os.path.join(image_dir, img_name)
        img_array = preprocess_image(img_path)
        if img_array is None:
            logging.warning(f"Skipping image {img_name} due to processing error")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            logging.warning(f"No annotations for image {img_name}")
            continue

        mask_array = None
        for ann in anns:
            if 'segmentation' in ann:
                temp_mask = preprocess_mask(ann, (img_info['height'], img_info['width']))
                if temp_mask is not None:
                    mask_array = temp_mask if mask_array is None else np.maximum(mask_array, temp_mask)
        if mask_array is None:
            logging.warning(f"No valid masks for image {img_name}")
            continue

        batch_images[img_name] = img_array
        batch_masks[img_name] = mask_array
        valid_count += 1
        if valid_count % 100 == 0:
            logging.info(f"Processed {valid_count} image-mask pairs")

    if valid_count == 0:
        logging.error("No valid image-mask pairs processed")
        raise ValueError("No valid image-mask pairs processed")

    images_path = os.path.join(output_dir, "images.npz")
    masks_path = os.path.join(output_dir, "masks.npz")
    try:
        np.savez(images_path, **{k: v for k, v in batch_images.items() if isinstance(k, str)})
        np.savez(masks_path, **{k: v for k, v in batch_masks.items() if isinstance(k, str)})
        logging.info(f"Saved {len(batch_images)} images and {len(batch_masks)} masks")
    except Exception as e:
        logging.error(f"Error saving .npz files: {e}")
        raise

    # Verify saved files
    if verify_npz(images_path) and verify_npz(masks_path):
        logging.info(f"Verified {images_path} and {masks_path}")
        print(f"Saved and verified {len(batch_images)} images and {len(batch_masks)} masks")
    else:
        logging.error("Failed to verify .npz files")
        raise ValueError("Corrupted .npz files detected")