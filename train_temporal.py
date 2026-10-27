import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from temporal_model import get_temporal_model

class VideoSequenceDataset(Dataset):
    """
    Dataset yielding sequences of T consecutive face crops from video folders.
    Organized as:
      root_dir/
        real/
          video_id_seq0/ [frame_0.jpg, frame_1.jpg, ... frame_T.jpg]
        fake/
          video_id_seq0/ [frame_0.jpg, ...]
    """
    def __init__(self, samples, transform=None, seq_len=8):
        self.samples = samples # list of (sequence_dir, label)
        self.transform = transform
        self.seq_len = seq_len
        self.classes = ['fake', 'real']
        self.class_to_idx = {'fake': 0, 'real': 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, label = self.samples[idx]
        files = sorted([os.path.join(seq_dir, f) for f in os.listdir(seq_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # If fewer frames than seq_len, repeat last frame; if more, uniform sample
        if len(files) >= self.seq_len:
            indices = np.linspace(0, len(files) - 1, self.seq_len).astype(int)
            selected_files = [files[i] for i in indices]
        else:
            selected_files = files + [files[-1]] * (self.seq_len - len(files))

        frames = []
        for fpath in selected_files:
            img = Image.open(fpath).convert('RGB')
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Shape: (T, C, H, W)
        seq_tensor = torch.stack(frames, dim=0)
        return seq_tensor, label

def create_synthetic_sequence_dataset(base_dir="temporal_dummy_dataset", num_sequences=80, seq_len=8):
    """Generates a mock sequence dataset for quick testing and verification."""
    print(f"Creating mock sequence dataset at {base_dir}...")
    for split in ["train", "val"]:
        for label in ["fake", "real"]:
            for seq_idx in range(num_sequences // 4):
                s_dir = os.path.join(base_dir, split, label, f"seq_{seq_idx:03d}")
                os.makedirs(s_dir, exist_ok=True)
                for f_idx in range(seq_len):
                    # Random image with consistent color per sequence to simulate video
                    color = np.random.randint(50, 200, 3, dtype=np.uint8)
                    arr = np.ones((224, 224, 3), dtype=np.uint8) * color
                    Image.fromarray(arr).save(os.path.join(s_dir, f"f_{f_idx:02d}.jpg"))
    print("Mock sequence dataset ready.")

def validate_temporal(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    epoch_loss = running_loss / len(all_labels)
    overall_acc = (all_preds == all_labels).float().mean().item()

    fake_rec = ((all_preds == 0) & (all_labels == 0)).sum().float() / max(1, (all_labels == 0).sum())
    real_rec = ((all_preds == 1) & (all_labels == 1)).sum().float() / max(1, (all_labels == 1).sum())
    bal_acc = ((fake_rec + real_rec) / 2).item()

    return epoch_loss, overall_acc, bal_acc, fake_rec.item(), real_rec.item()

def main():
    parser = argparse.ArgumentParser(description="Train Temporal CNN-LSTM Deepfake Classifier")
    parser.add_argument("--data_dir", type=str, default="temporal_dummy_dataset")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4, help="Keep 2-4 on 4GB VRAM for video sequences")
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=8, help="Number of consecutive frames per sequence")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--save_path", type=str, default="best_temporal_model.pth")
    args = parser.parse_args()

    if args.data_dir == "temporal_dummy_dataset" and not os.path.exists(args.data_dir):
        create_synthetic_sequence_dataset(args.data_dir, seq_len=args.seq_len)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")

    def gather_samples(root):
        samples = []
        for label in ["fake", "real"]:
            ldir = os.path.join(root, label)
            if not os.path.exists(ldir): continue
            label_idx = 0 if label == "fake" else 1
            for sname in os.listdir(ldir):
                s_path = os.path.join(ldir, sname)
                if os.path.isdir(s_path):
                    samples.append((s_path, label_idx))
        return samples

    train_samples = gather_samples(train_dir)
    val_samples   = gather_samples(val_dir)

    print(f"Found {len(train_samples)} training sequences, {len(val_samples)} validation sequences.")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = VideoSequenceDataset(train_samples, transform=train_tf, seq_len=args.seq_len)
    val_ds   = VideoSequenceDataset(val_samples,   transform=val_tf,   seq_len=args.seq_len)

    # Class balance sampling
    labels = [s[1] for s in train_samples]
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / np.maximum(class_counts[labels], 1)
    sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(train_samples), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Temporal CNN-LSTM on device: {device}")

    model = get_temporal_model(num_classes=2, pretrained_spatial=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\\n--- Temporal Epoch {epoch+1}/{args.epochs} ---")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad()

        for step, (seqs, lbls) in enumerate(tqdm(train_dl, desc="Training")):
            seqs, lbls = seqs.to(device), lbls.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, lbls) / args.accum_steps
            loss.backward()

            running_loss += loss.item() * seqs.size(0) * args.accum_steps
            _, preds = torch.max(outputs, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_dl):
                optimizer.step()
                optimizer.zero_grad()

        val_loss, val_acc, bal_acc, fake_rec, real_rec = validate_temporal(model, val_dl, criterion, device)
        scheduler.step()

        print(f"Train Loss: {running_loss/total:.4f} | Train Acc: {correct/total:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Balanced Acc: {bal_acc:.4f}")
        print(f"Fake Recall: {fake_rec*100:.1f}% | Real Recall: {real_rec*100:.1f}%")

        if bal_acc > best_acc:
            best_acc = bal_acc
            torch.save({
                'epoch': epoch,
                'arch': 'temporal_cnn_lstm',
                'model_state_dict': model.state_dict(),
                'classes': ['fake', 'real'],
                'accuracy': bal_acc,
                'fake_recall': fake_rec,
                'real_recall': real_rec
            }, args.save_path)
            print(f"*** New best temporal balanced acc ({bal_acc:.4f}) saved to {args.save_path} ***")

if __name__ == "__main__":
    main()
