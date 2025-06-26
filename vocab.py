import nltk
from collections import Counter
nltk.download('punkt')

class Vocabulary:
    def __init__(self, caption_file, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.pad_idx = self.stoi["<PAD>"]
        self.build_vocab(caption_file)

    def tokenizer(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def build_vocab(self, caption_file):
        counter = Counter()
        with open(caption_file, 'r', encoding='utf-8') as f:
            for line in f:
                caption = line.strip().split('\t')[-1]
                tokens = self.tokenizer(caption)
                counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.stoi[word] = idx
                self.itos[idx] = word

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
    
    def __len__(self):
        return len(self.itos)
