import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
from social_media_augmentations import MotionBlur, VideoCodecCompression
from sbi_generator import SelfBlendedImageGenerator


class AddGaussianNoise:
    def __init__(self, mean=0., std=0.05, p=0.4):
        self.std = std
        self.mean = mean
        self.p = p
    def __call__(self, tensor):
        if np.random.rand() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

class RandomBlur:
    def __init__(self, p=0.5, radius_min=0.5, radius_max=1.5):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
    def __call__(self, img):
        if np.random.rand() < self.p:
            radius = np.random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomJPEGCompression:
    def __init__(self, p=0.5, quality_min=30, quality_max=85):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max
    def __call__(self, img):
        if np.random.rand() < self.p:
            q = np.random.randint(self.quality_min, self.quality_max)
            output = BytesIO()
            img.save(output, format="JPEG", quality=q)
            output.seek(0)
            return Image.open(output)
        return img

class RandomLowRes:
    """Simulate cheap/low-resolution webcam by downscaling then upscaling."""
    def __init__(self, p=0.4, scale_min=0.4, scale_max=0.85):
        self.p = p
        self.scale_min = scale_min
        self.scale_max = scale_max
    def __call__(self, img):
        if np.random.rand() < self.p:
            w, h = img.size
            scale = np.random.uniform(self.scale_min, self.scale_max)
            small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return img

def generate_dummy_dataset(base_dir="dummy_dataset", num_samples=100):
    """Generates a dummy dataset of random images for testing training execution."""
    print(f"Creating dummy dataset at {base_dir} to verify training functionality...")
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            path = os.path.join(base_dir, split, label)
            os.makedirs(path, exist_ok=True)
            for i in range(num_samples // 4):
                # Make random RGB image
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(os.path.join(path, f"sample_{i}.jpg"))
    print("Dummy dataset creation complete.")

def compute_fft_channel(batch_tensors):
    """
    Computes 2D FFT log-magnitude on RGB tensor batch (B, 3, H, W).
    Returns (B, 1, H, W) normalized FFT magnitude map.
    """
    # Grayscale approximation: 0.2989*R + 0.5870*G + 0.1140*B
    gray = 0.2989 * batch_tensors[:, 0:1] + 0.5870 * batch_tensors[:, 1:2] + 0.1140 * batch_tensors[:, 2:3]
    fft = torch.fft.fft2(gray)
    fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
    mag = torch.log(torch.abs(fft_shift) + 1e-6)
    m_min = mag.amin(dim=(-2, -1), keepdim=True)
    m_max = mag.amax(dim=(-2, -1), keepdim=True)
    mag_norm = (mag - m_min) / (m_max - m_min + 1e-8)
    return mag_norm

class FrequencyBranch(nn.Module):
    """Lightweight CNN branch to extract features from 2D FFT magnitude spectrum."""
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DualStreamDetector(nn.Module):
    """
    Dual-stream network combining:
    1. Spatial stream: Pretrained MobileNetV3-Large (RGB image)
    2. Frequency stream: FrequencyBranch CNN (2D FFT magnitude spectrum)
    Fused via concatenation and joint classification head.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        self.spatial_features = base.features
        self.spatial_pool = base.avgpool
        self.freq_branch = FrequencyBranch(out_features=128)
        self.classifier = nn.Sequential(
            nn.Linear(960 + 128, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x is (B, 4, 224, 224): channels 0..2 are RGB, channel 3 is FFT magnitude
        rgb = x[:, :3, :, :]
        fft = x[:, 3:, :, :]
        s_feat = self.spatial_pool(self.spatial_features(rgb))
        s_feat = torch.flatten(s_feat, 1)
        f_feat = self.freq_branch(fft)
        combined = torch.cat([s_feat, f_feat], dim=1)
        return self.classifier(combined)

class EfficientNetDualStream(nn.Module):
    """
    Dual-stream architecture combining EfficientNet-B4 spatial feature extractor
    with FrequencyBranch 2D FFT feature extractor.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b4(weights=weights)
        self.spatial_features = base.features
        self.spatial_pool = base.avgpool
        self.freq_branch = FrequencyBranch(out_features=128)
        self.classifier = nn.Sequential(
            nn.Linear(1792 + 128, 256),
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        rgb = x[:, :3, :, :]
        fft = x[:, 3:, :, :]
        s_feat = self.spatial_pool(self.spatial_features(rgb))
        s_feat = torch.flatten(s_feat, 1)
        f_feat = self.freq_branch(fft)
        combined = torch.cat([s_feat, f_feat], dim=1)
        return self.classifier(combined)

class SBIEnhancedDataset(torch.utils.data.Dataset):
    """
    Self-Blended Image (SBI) dataset wrapper.
    For real images (label=1), with probability p_sbi, synthesizes a modern pseudo-deepfake
    with realistic facial seam blending, and switches the training label to 0 (fake).
    """
    def __init__(self, base_dataset, p_sbi=0.35, transform=None):
        self.base_dataset = base_dataset
        self.p_sbi = p_sbi
        self.transform = transform
        self.sbi_gen = SelfBlendedImageGenerator(p=1.0)
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx
        self.targets = base_dataset.targets

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        path, label = self.base_dataset.samples[idx]
        img = Image.open(path).convert('RGB')
        
        # If image is real (label == 1), dynamically generate synthetic face swap with probability p_sbi
        if label == 1 and np.random.rand() < self.p_sbi:
            img, _ = self.sbi_gen(img)
            label = 0 # Swapped face is now FAKE

        if self.transform:
            img = self.transform(img)
        return img, label

def get_model(num_classes=2, arch="dual_fft", pretrained=True):
    """Returns requested model architecture (dual_fft, mobilenet, efficientnet, efficientnet_dual)."""
    if arch == "dual_fft":
        return DualStreamDetector(num_classes=num_classes, pretrained=pretrained)
    elif arch == "efficientnet_dual":
        return EfficientNetDualStream(num_classes=num_classes, pretrained=pretrained)
    elif arch == "efficientnet":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    else: # mobilenet
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model



def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, arch="dual_fft"):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    is_fft_arch = arch in ["dual_fft", "efficientnet_dual"]

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            if is_fft_arch:
                fft_c = compute_fft_channel(inputs)
                inputs = torch.cat([inputs, fft_c], dim=1)
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

    # Per-class recall (fake=0, real=1)
    fake_recall = ((all_preds == 0) & (all_labels == 0)).sum().float() / (all_labels == 0).sum()
    real_recall = ((all_preds == 1) & (all_labels == 1)).sum().float() / (all_labels == 1).sum()
    balanced_acc = ((fake_recall + real_recall) / 2).item()

    return epoch_loss, overall_acc, balanced_acc, fake_recall.item(), real_recall.item()

def main():
    parser = argparse.ArgumentParser(description="Train Deepfake Classifier")
    parser.add_argument("--data_dir", type=str, default="dummy_dataset")
    parser.add_argument("--arch", type=str,
                        choices=["dual_fft", "mobilenet", "efficientnet", "efficientnet_dual"],
                        default="dual_fft",
                        help="Model architecture: dual_fft, mobilenet, efficientnet, or efficientnet_dual")
    parser.add_argument("--use_sbi", action="store_true",
                        help="Enable on-the-fly Self-Blended Images (SBI) modern synthetic face-swap generation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8, help="Keep small (8) on 4GB VRAM")
    parser.add_argument("--accum_steps", type=int, default=4, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset_best", action="store_true", help="Reset best_acc to 0 when resuming (use when changing save criterion)")
    
    args = parser.parse_args()
    
    # If the specified path is 'dummy_dataset' and doesn't exist, create it
    if args.data_dir == "dummy_dataset" and not os.path.exists(args.data_dir):
        generate_dummy_dataset(args.data_dir)
        
    # Check if directories exist
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    if not os.path.exists(val_dir) and os.path.exists(os.path.join(args.data_dir, "valid")):
        val_dir = os.path.join(args.data_dir, "valid")
        
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: The dataset directory must contain 'train' and 'val' (or 'valid') subfolders. Paths not found:\n- {train_dir}\n- {val_dir}")
        return
        
    # Data transformations with Social Media Robustness (MotionBlur, VideoCodecCompression)
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.3, 1.2)),
        MotionBlur(p=0.35),                                     # camera shake & motion blur
        VideoCodecCompression(p=0.35),                         # WhatsApp/Telegram H.264 blockiness
        RandomLowRes(p=0.4),                                   # cheap webcam resolution
        RandomBlur(p=0.3),
        RandomJPEGCompression(p=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        AddGaussianNoise(p=0.35),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.12), value='random')
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets and Loaders
    if args.use_sbi:
        print("Self-Blended Images (SBI) synthetic artifact generator ENABLED for real training images.")
        raw_train = datasets.ImageFolder(train_dir, transform=None)
        train_dataset = SBIEnhancedDataset(raw_train, p_sbi=0.35, transform=train_transforms)
    else:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)

    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    print(f"Classes found: {train_dataset.classes}")
    print(f"Class to index mapping: {train_dataset.class_to_idx}")

    # ── Balanced sampler: each batch sees equal real/fake regardless of disk counts ──
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    print(f"Raw class counts — " + ", ".join(f"{train_dataset.classes[i]}: {class_counts[i]}" for i in range(len(class_counts))))
    sample_weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                    num_samples=len(train_dataset),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    
    # Device selection (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected Architecture: {args.arch.upper()} ({'Spatial + Frequency (FFT)' if args.arch == 'dual_fft' else 'Spatial Only'})")
    
    model = get_model(num_classes=len(train_dataset.classes), arch=args.arch)
    model.to(device)

    # Optionally resume from a checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_acc = ckpt.get('accuracy', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.4f}")
    
    # With WeightedRandomSampler the batches are already balanced,
    # so we use equal weights in the loss (or mild rebalancing if desired)
    class_weights = np.ones(len(train_dataset.classes))  # equal weight — sampler handles balance
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    print(f"Using balanced WeightedRandomSampler — loss weights all set to 1.0")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if not (args.resume and os.path.exists(args.resume)):
        best_acc = 0.0
        start_epoch = 0
    if args.reset_best:
        best_acc = 0.0
        print("best_acc reset to 0.0 (new saving criterion)")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # ── Training with gradient accumulation ──
        model.train()
        running_loss = 0.0; correct = 0; total = 0
        optimizer.zero_grad()
        for step, (inputs, labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs, labels = inputs.to(device), labels.to(device)
            if args.arch in ["dual_fft", "efficientnet_dual"]:
                fft_c = compute_fft_channel(inputs)
                inputs = torch.cat([inputs, fft_c], dim=1)
            outputs = model(inputs)
            loss = criterion(outputs, labels) / args.accum_steps
            loss.backward()
            running_loss += loss.item() * inputs.size(0) * args.accum_steps
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            # Save mid-epoch checkpoint every 500 steps so crashes don't waste all work
            if (step + 1) % 500 == 0:
                torch.save({
                    'epoch': epoch, 'step': step,
                    'arch': args.arch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'accuracy': best_acc,
                    'classes': train_dataset.classes,
                }, 'last_checkpoint.pth')
        train_loss = running_loss / total
        train_acc  = correct / total

        val_loss, val_acc, bal_acc, fake_rec, real_rec = validate(model, val_loader, criterion, device, arch=args.arch)
        scheduler.step()
        torch.cuda.empty_cache()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Balanced Acc: {bal_acc:.4f}")
        print(f"Fake Recall: {fake_rec*100:.1f}%  |  Real Recall: {real_rec*100:.1f}%")

        # Save on BALANCED accuracy so real-biased models don't win
        if bal_acc > best_acc:
            best_acc = bal_acc
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'classes': train_dataset.classes,
                'class_to_idx': train_dataset.class_to_idx,
                'accuracy': bal_acc,
                'fake_recall': fake_rec,
                'real_recall': real_rec,
            }, args.save_path)
            print(f"*** New best balanced acc ({bal_acc:.4f}) — fake={fake_rec*100:.1f}% real={real_rec*100:.1f}% — saved to {args.save_path} ***")


if __name__ == "__main__":
    main()
