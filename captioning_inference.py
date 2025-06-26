import os
import torch
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from vocab import Vocabulary
from model.captioning_model import CaptioningModel

# === Load & Preprocess a Single Image ===
def load_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)  # PIL image + tensor

# === Caption Generation ===
def generate_caption(model, image_tensor, vocab, max_len=30):
    model.eval()
    with torch.no_grad():
        # Encode image to feature
        features = model.encoder(image_tensor)        # (1, C, H, W)
        features = torch.flatten(features, start_dim=1)  # (1, C*H*W)
        features = model.enc_fc(features)             # (1, embed_size)
        features = features.unsqueeze(1)              # (1, 1, embed_size)


        caption = [vocab.stoi["<SOS>"]]
        inputs = features  # initial input to LSTM (1, 1, embed_size)

        for _ in range(max_len):
            last_word = torch.tensor([caption[-1]]).to(image_tensor.device)
            embed = model.embed(last_word).unsqueeze(1)  # (1, 1, embed_size)

            # Append the word embedding to inputs
            inputs = torch.cat((inputs, embed), dim=1)  # (1, t+1, embed_size)

            # Forward pass through LSTM
            hiddens, _ = model.lstm(inputs)
            output = model.fc(hiddens[:, -1, :])  # (1, vocab_size)

            predicted = output.argmax(1).item()
            if predicted == vocab.stoi["<EOS>"]:
                break
            caption.append(predicted)

        decoded = [vocab.itos[idx] for idx in caption[1:]]  # Skip <SOS>
        return ' '.join(decoded)


# === Draw Caption on Image ===
def save_captioned_image(pil_image, caption, save_path):
    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    margin = 5
    draw.rectangle([(0, 0), (pil_image.width, 30)], fill=(0, 0, 0, 128))
    draw.text((margin, 5), caption, font=font, fill="white")
    pil_image.save(save_path)

# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='captioning_model_final.pth')
    parser.add_argument('--captions_file', type=str, default='datasets/Flickr8k/captions.txt')
    parser.add_argument('--image_dir', type=str, default='datasets/Flickr8k/Images')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default='captioning_outputs')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[🖥️] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("[📖] Loading vocabulary...")
    vocab = Vocabulary(args.captions_file, freq_threshold=5)
    print(f"[✅] Vocabulary size: {len(vocab)}")

    print("[🧠] Loading model...")
    embed_size = 256
    hidden_size = 512
    model = CaptioningModel(embed_size, hidden_size, len(vocab)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print("[✅] Model loaded successfully.")

    all_images = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith('.jpg')])
    selected_images = all_images[args.start:args.end]
    print(f"[📸] Captioning images {args.start} to {args.end}...")

    for idx, img_name in enumerate(selected_images):
        print(f"\n🔄 [{idx+1}/{len(selected_images)}] {img_name}")
        image_path = os.path.join(args.image_dir, img_name)
        pil_img, image_tensor = load_image(image_path, transform)
        image_tensor = image_tensor.to(device)

        caption = generate_caption(model, image_tensor, vocab)
        print(f"📝 Caption: {caption}")

        save_path = os.path.join(args.output_dir, img_name)
        save_captioned_image(pil_img, caption, save_path)
        print(f"💾 Saved: {save_path}")

    print(f"\n✅ All {len(selected_images)} images captioned and saved to: {args.output_dir}")

if __name__ == "__main__":
    main()