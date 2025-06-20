import os
import numpy as np
from PIL import Image
import pandas as pd
import nltk
import logging
import pickle

logging.basicConfig(
    filename=r"E:\ImageProject\preprocess_flickr8k.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def build_vocab(captions, threshold=4):  # Lowered threshold to match ~8920
    word_counts = {}
    for caption in captions:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        for word in tokens:
            word_counts[word] = word_counts.get(word, 0) + 1
    vocab = {word: idx + 1 for idx, (word, count) in enumerate(word_counts.items()) if count >= threshold}
    vocab['<pad>'] = 0
    vocab['<start>'] = len(vocab)
    vocab['<end>'] = len(vocab)
    logging.info(f"Vocabulary size: {len(vocab)}")
    return vocab

def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None

def tokenize_caption(caption, vocab, max_length=20):
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    indices = [vocab['<start>']] + [vocab.get(word, vocab['<pad>']) for word in tokens[:max_length-2]] + [vocab['<end>']]
    if len(indices) < max_length:
        indices += [vocab['<pad>']] * (max_length - len(indices))
    return indices[:max_length]

if __name__ == "__main__":
    nltk.download('punkt')
    data_dir = r"E:\ImageProject\datasets\Flickr8k"
    caption_file = os.path.join(data_dir, "captions.txt")
    image_dir = os.path.join(data_dir, "Images")
    output_dir = r"E:\ImageProject\datasets\processed\Flickr8k"
    os.makedirs(output_dir, exist_ok=True)

    try:
        captions_df = pd.read_csv(caption_file)
    except Exception as e:
        logging.error(f"Error reading {caption_file}: {e}")
        raise

    captions = captions_df['caption'].tolist()
    vocab = build_vocab(captions)
    max_length = 20

    batch_images = {}
    batch_captions = {}
    valid_count = 0
    for idx, row in captions_df.iterrows():
        img_name = row['image']
        caption = row['caption']
        img_path = os.path.join(image_dir, img_name)
        img_array = preprocess_image(img_path)
        if img_array is None:
            continue
        caption_indices = tokenize_caption(caption, vocab, max_length)
        if img_name not in batch_images:
            batch_images[img_name] = img_array
            batch_captions[img_name] = []
        batch_captions[img_name].append(caption_indices)
        valid_count += 1
        if valid_count % 100 == 0:
            logging.info(f"Processed {valid_count} captions")

    for img_name in batch_captions:
        captions = batch_captions[img_name]
        if len(captions) < 5:
            captions += [[vocab['<pad>']] * max_length] * (5 - len(captions))
        batch_captions[img_name] = captions[:5]

    np.savez(os.path.join(output_dir, "images.npz"), **batch_images)
    with open(os.path.join(output_dir, "captions.pkl"), 'wb') as f:
        pickle.dump(batch_captions, f)
    with open(os.path.join(output_dir, "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    logging.info(f"Saved {len(batch_images)} images and {valid_count} captions")
    print(f"Saved {len(batch_images)} images and {valid_count} captions")