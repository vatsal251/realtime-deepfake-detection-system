import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, r"D:\college\Sem 3\Macro Project-I\project df")
from train_classifier import get_model, compute_fft_channel, RandomBlur, RandomJPEGCompression, RandomLowRes

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = r"D:\college\Sem 3\Macro Project-I\project df\unified_dataset"
PROJECT_DIR = r"D:\college\Sem 3\Macro Project-I\project df"

def train_epoch(model, loader, criterion, optimizer, scaler, is_fft=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        if is_fft:
            fft_c = compute_fft_channel(inputs)
            inputs = torch.cat([inputs, fft_c], dim=1)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, is_fft=True):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Val", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            if is_fft:
                fft_c = compute_fft_channel(inputs)
                inputs = torch.cat([inputs, fft_c], dim=1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    fake_mask = (all_labels == 0)
    real_mask = (all_labels == 1)
    fake_recall = (all_preds[fake_mask] == 0).mean() if fake_mask.sum() > 0 else 0.0
    real_recall = (all_preds[real_mask] == 1).mean() if real_mask.sum() > 0 else 0.0
    balanced_acc = (fake_recall + real_recall) / 2.0
    return running_loss / len(all_labels), acc, balanced_acc, fake_recall, real_recall

def main():
    parser = argparse.ArgumentParser(description="Unified Multi-Domain Deepfake Training")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--init_ckpt", type=str, default="best_unified_model.pth" if os.path.exists("best_unified_model.pth") else "best_modern_model.pth")
    parser.add_argument("--save_path", type=str, default="best_unified_model.pth")
    args = parser.parse_args()

    print(f"============================================================")
    print(f"  UNIFIED MULTI-DOMAIN CLASSIFIER TRAINING")
    print(f"  Device: {DEVICE} | Arch: DUAL_FFT MobileNetV3 (4 Channels)")
    print(f"  Initial Checkpoint: {args.init_ckpt}")
    print(f"  Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"============================================================")

    # 1. Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        RandomJPEGCompression(p=0.4, quality_min=40, quality_max=85),
        RandomBlur(p=0.3, radius_min=0.4, radius_max=1.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10))
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    print(f"Classes: {train_ds.classes} -> {train_ds.class_to_idx}")
    targets = np.array(train_ds.targets)
    counts = np.bincount(targets)
    print(f"Train counts: Fake={counts[0]}, Real={counts[1]} (Total: {len(train_ds)})")
    
    val_targets = np.array(val_ds.targets)
    val_counts = np.bincount(val_targets)
    print(f"Val counts:   Fake={val_counts[0]}, Real={val_counts[1]} (Total: {len(val_ds)})")

    # Balanced Sampler
    weights = 1.0 / counts[targets]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 2. Model Initialization
    model = get_model(num_classes=2, arch="dual_fft")
    init_path = os.path.join(PROJECT_DIR, args.init_ckpt)
    best_balanced_acc = 0.0
    start_epoch = 0
    if os.path.exists(init_path):
        print(f"Loading warm starting weights from {init_path}...")
        ckpt = torch.load(init_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt)
        best_balanced_acc = float(ckpt.get('balanced_acc', 0.0))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        model.load_state_dict(state_dict, strict=False)
        print(f"Warm weights loaded successfully! Resuming from epoch {start_epoch}, previous best balanced acc = {best_balanced_acc*100:.2f}%")

    model.to(DEVICE)

    # Freeze stem for early epochs
    if start_epoch < 2:
        print("Freezing early stem layers (model.spatial_features[:3]) for warm adaptation...")
        for p in model.spatial_features[:3].parameters():
            p.requires_grad = False
    else:
        print("Resuming in unfreezed state...")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - start_epoch), eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    for epoch in range(start_epoch, args.epochs):
        # Unfreeze all layers at epoch 2
        if epoch == 2:
            print("\n>>> Unfreezing all backbone layers for joint fine-tuning...")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.7, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 2), eta_min=1e-6)

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        val_loss, val_acc, bal_acc, f_rec, r_rec = eval_epoch(model, val_loader, criterion)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% BalAcc: {bal_acc*100:.1f}% | "
              f"Fake Recall: {f_rec*100:.1f}% Real Recall: {r_rec*100:.1f}%")

        if bal_acc > best_balanced_acc:
            best_balanced_acc = bal_acc
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'balanced_acc': bal_acc,
                'accuracy': val_acc,
                'fake_recall': f_rec,
                'real_recall': r_rec,
                'classes': ['fake', 'real'],
                'arch': 'dual_fft'
            }
            torch.save(save_dict, os.path.join(PROJECT_DIR, args.save_path))
            print(f"  --> Saved new best checkpoint to {args.save_path} (Balanced Acc: {bal_acc*100:.2f}%)")

    print(f"\n============================================================")
    print(f"Training Complete! Best Balanced Accuracy: {best_balanced_acc*100:.2f}%")
    print(f"Saved weights: {args.save_path}")

    # Auto-export to ONNX
    print("Exporting best checkpoint to mobilenetv3_deepfake.onnx...")
    from export_onnx import export_to_onnx
    onnx_out = os.path.join(PROJECT_DIR, "mobilenetv3_deepfake.onnx")
    export_to_onnx(os.path.join(PROJECT_DIR, args.save_path), onnx_out)
    print(f"SUCCESS: Exported {onnx_out}")
    print(f"============================================================")

if __name__ == "__main__":
    main()
