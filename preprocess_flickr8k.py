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

# Load captions
captions_file = os.path.join(base_dir, "captions.txt")
captions_df = pd.read_csv(captions_file, sep=',', names=['image', 'caption'])

# Preprocess images
def preprocess_image(img_path, size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array

image_arrays = {}
for img_name in captions_df['image'].unique():
    img_path = os.path.join(base_dir, "Images", img_name)
    if os.path.exists(img_path):
        image_arrays[img_name] = preprocess_image(img_path)
    else:
        print(f"Image not found: {img_path}")

# Save preprocessed images
with open(os.path.join(output_dir, "images.pkl"), 'wb') as f:
    pickle.dump(image_arrays, f)

# Tokenize captions
def tokenize_caption(caption):
    tokens = word_tokenize(caption.lower())
    return ['<start>'] + tokens + ['<end>']

captions_df['tokens'] = captions_df['caption'].apply(tokenize_caption)

# Build vocabulary
all_tokens = [token for tokens in captions_df['tokens'] for token in tokens]
vocab = {word: idx + 1 for idx, word in enumerate(set(all_tokens))}
vocab['<pad>'] = 0  # Padding token

# Convert tokens to indices
def tokens_to_indices(tokens, vocab):
    return [vocab[token] for token in tokens]

captions_df['indices'] = captions_df['tokens'].apply(lambda x: tokens_to_indices(x, vocab))

# Save preprocessed captions and vocabulary
captions_df.to_pickle(os.path.join(output_dir, "captions.pkl"))
with open(os.path.join(output_dir, "vocab.pkl"), 'wb') as f:
    pickle.dump(vocab, f)

print("Flickr8k preprocessing complete.")