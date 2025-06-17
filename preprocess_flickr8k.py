import os
import numpy as np
from PIL import Image
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import pickle

# Download NLTK data
nltk.download('punkt')

# Paths
base_dir = r"E:\ImageProject\datasets\Flickr8k"
output_dir = r"E:\ImageProject\datasets\processed\Flickr8k"
os.makedirs(output_dir, exist_ok=True)

# Load captions, skipping header
captions_file = os.path.join(base_dir, "captions.txt")
captions_df = pd.read_csv(captions_file, sep=',', names=['image', 'caption'], skiprows=1)

# Validate image filenames
captions_df = captions_df[captions_df['image'].str.endswith('.jpg')]
print(f"Total captions: {len(captions_df)}, Unique images: {len(captions_df['image'].unique())}")

# Preprocess images
def preprocess_image(img_path, size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size, Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    return img_array

# Process images in batches
unique_images = captions_df['image'].unique()
batch_size = 1000
image_arrays = {}
image_output_file = os.path.join(output_dir, "images.npz")
missing_images = []

for i in range(0, len(unique_images), batch_size):
    batch_images = unique_images[i:i + batch_size]
    batch_arrays = {}
    for img_name in batch_images:
        img_path = os.path.join(base_dir, "Images", img_name)
        if os.path.exists(img_path):
            batch_arrays[img_name] = preprocess_image(img_path)
        else:
            missing_images.append(img_path)
    image_arrays.update(batch_arrays)
    print(f"Processed batch {i // batch_size + 1}/{(len(unique_images) // batch_size) + 1}, Images in batch: {len(batch_arrays)}")
if missing_images:
    print(f"Missing images: {len(missing_images)}")
    with open(os.path.join(output_dir, "missing_images.txt"), 'w') as f:
        f.write('\n'.join(missing_images))

# Save all images
np.savez(image_output_file, **image_arrays, allow_pickle=True)
print(f"Saved {len(image_arrays)} images to {image_output_file}")

# Tokenize captions
def tokenize_caption(caption):
    if isinstance(caption, str):
        tokens = word_tokenize(caption.lower())
        return ['<start>'] + tokens + ['<end>']
    return ['<start>', '<end>']

captions_df['tokens'] = captions_df['caption'].apply(tokenize_caption)

# Build vocabulary
all_tokens = [token for tokens in captions_df['tokens'] for token in tokens]
vocab = {word: idx + 1 for idx, word in enumerate(set(all_tokens))}
vocab['<pad>'] = 0

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

captions_df['indices'] = captions_df['tokens'].apply(lambda x: tokens_to_indices(x, vocab))

# Save captions and vocabulary
captions_df.to_pickle(os.path.join(output_dir, "captions.pkl"))
with open(os.path.join(output_dir, "vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f)

print("Flickr8k preprocessing complete.")