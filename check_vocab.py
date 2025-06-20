# check_vocab.py
import pickle

with open(r"E:\ImageProject\datasets\processed\Flickr8k\vocab.pkl", 'rb') as f:
    vocab = pickle.load(f)
print(f"Vocabulary size: {len(vocab)}")