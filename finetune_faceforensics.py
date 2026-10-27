import os
import sys
import random
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, r"D:\college\Sem 3\Macro Project-I\project df")
from train_classifier import get_model, compute_fft_channel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_FF = r"C:\Users\asus\Downloads\archive\FaceForensics++_C23"
OUTPUT_DIR = r"D:\college\Sem 3\Macro Project-I\project df\ff_finetune_dataset"
PROJECT_DIR = r"D:\college\Sem 3\Macro Project-I\project df"

# 1. Extraction function
def extract_balanced_ff(num_train_vids=40, num_val_vids=10, frames_per_vid=6):
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "fake"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val", "real"), exist_ok=True)

    fake_categories = ["FaceShifter", "NeuralTextures", "Face2Face", "FaceSwap", "Deepfakes"]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Reserve videos >= 900 for validation (includes user test videos)
    print("Extracting balanced FaceForensics++ faces...")

    def get_crops(vpath, max_frames=6):
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            return []
        step = max(1, total // (max_frames + 2))
        crops = []
        frame_idx = 0
        while cap.isOpened() and len(crops) < max_frames and frame_idx < total:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
                    x, y, w, h = faces[0]
                    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                    H, W = frame.shape[:2]
                    crop = frame[max(0, y-pad_y):min(H, y+h+pad_y), max(0, x-pad_x):min(W, x+w+pad_x)]
                    if crop.size > 0:
                        crops.append(cv2.resize(crop, (224, 224)))
            frame_idx += 1
        cap.release()
        return crops

    # Process fake categories
    for cat in fake_categories:
        cat_dir = os.path.join(BASE_FF, cat)
        if not os.path.exists(cat_dir):
            continue
        all_vids = [f for f in os.listdir(cat_dir) if f.endswith(".mp4")]
        train_vids = [f for f in all_vids if not f.startswith("9")][:num_train_vids]
        val_vids = [f for f in all_vids if f.startswith("9")][:num_val_vids]

        for split, vids in [("train", train_vids), ("val", val_vids)]:
            out_cat_dir = os.path.join(OUTPUT_DIR, split, "fake")
            for vf in tqdm(vids, desc=f"{cat} ({split})"):
                vpath = os.path.join(cat_dir, vf)
                crops = get_crops(vpath, frames_per_vid)
                base = f"{cat}_{os.path.splitext(vf)[0]}"
                for idx, c in enumerate(crops):
                    cv2.imwrite(os.path.join(out_cat_dir, f"{base}_{idx:02d}.jpg"), c)

    # Process real (original)
    real_dir = os.path.join(BASE_FF, "original")
    if os.path.exists(real_dir):
        all_vids = [f for f in os.listdir(real_dir) if f.endswith(".mp4")]
        train_vids = [f for f in all_vids if not f.startswith("9")][:num_train_vids * len(fake_categories)]
        val_vids = [f for f in all_vids if f.startswith("9")][:num_val_vids * len(fake_categories)]

        for split, vids in [("train", train_vids), ("val", val_vids)]:
            out_cat_dir = os.path.join(OUTPUT_DIR, split, "real")
            for vf in tqdm(vids, desc=f"original ({split})"):
                vpath = os.path.join(real_dir, vf)
                crops = get_crops(vpath, frames_per_vid)
                base = f"real_{os.path.splitext(vf)[0]}"
                for idx, c in enumerate(crops):
                    cv2.imwrite(os.path.join(out_cat_dir, f"{base}_{idx:02d}.jpg"), c)

class SimpleFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        for label_name, label_idx in [("fake", 0), ("real", 1)]:
            d = os.path.join(root, label_name)
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith((".jpg", ".png")):
                        self.samples.append((os.path.join(d, f), label_idx))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def finetune():
    extract_balanced_ff(num_train_vids=40, num_val_vids=10, frames_per_vid=6)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = SimpleFolderDataset(os.path.join(OUTPUT_DIR, "train"), transform=train_tf)
    val_ds = SimpleFolderDataset(os.path.join(OUTPUT_DIR, "val"), transform=val_tf)

    print(f"Dataset ready: {len(train_ds)} training faces, {len(val_ds)} validation faces.")
    if len(train_ds) < 100:
        print("Not enough images extracted. Check directory paths.")
        return

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    # Load existing best_modern_model.pth
    ckpt_path = os.path.join(PROJECT_DIR, "best_modern_model.pth")
    backup_path = os.path.join(PROJECT_DIR, "best_modern_model_preff.pth")
    if not os.path.exists(backup_path) and os.path.exists(ckpt_path):
        shutil.copy2(ckpt_path, backup_path)
        print(f"Backed up current model to {backup_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = get_model(num_classes=2, arch="dual_fft", pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(DEVICE)

    # Finetune with low learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    epochs = 4
    best_acc = 0.0

    print("Starting rapid fine-tuning on FaceForensics++...")
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            fft = compute_fft_channel(imgs)
            x = torch.cat([imgs, fft], dim=1)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(x)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        # Validation
        model.eval()
        val_correct = 0
        val_fake_correct = 0
        val_fake_total = 0
        val_real_correct = 0
        val_real_total = 0
        total_val = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                fft = compute_fft_channel(imgs)
                x = torch.cat([imgs, fft], dim=1)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)

                val_correct += (preds == labels).sum().item()
                val_fake_correct += ((preds == 0) & (labels == 0)).sum().item()
                val_fake_total += (labels == 0).sum().item()
                val_real_correct += ((preds == 1) & (labels == 1)).sum().item()
                val_real_total += (labels == 1).sum().item()
                total_val += labels.size(0)

        train_acc = train_correct / total_train
        val_acc = val_correct / total_val
        fake_rec = val_fake_correct / max(1, val_fake_total)
        real_rec = val_real_correct / max(1, val_real_total)
        balanced_acc = (fake_rec + real_rec) / 2.0

        print(f"Epoch {ep+1}: Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.1f}% | Balanced: {balanced_acc*100:.1f}% | Fake Recall: {fake_rec*100:.1f}% | Real Recall: {real_rec*100:.1f}%")

        if balanced_acc > best_acc:
            best_acc = balanced_acc
            torch.save({
                'epoch': ep,
                'arch': 'dual_fft',
                'model_state_dict': model.state_dict(),
                'classes': ['fake', 'real'],
                'class_to_idx': {'fake': 0, 'real': 1},
                'accuracy': val_acc,
                'balanced_acc': balanced_acc
            }, ckpt_path)
            print(f"Saved new best model checkpoint to {ckpt_path}")

    # Re-export ONNX
    print("\nRe-exporting updated model to ONNX...")
    import subprocess
    subprocess.run([sys.executable, "export_onnx.py", "--checkpoint", "best_modern_model.pth", "--output", "mobilenetv3_deepfake.onnx"], cwd=PROJECT_DIR)
    print("Fine-tuning and export complete!")

if __name__ == "__main__":
    finetune()
