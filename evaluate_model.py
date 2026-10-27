"""
evaluate_model.py - Formal Benchmark Evaluation of Deepfake Classifiers.
Supports: MobileNetV3, DualStream (RGB + 2D FFT), EfficientNet-B4, and EfficientNet Dual.
Produces: Per-class Precision, Recall, F1, Balanced Accuracy, Confusion Matrix,
          ROC Curve, AUC Score, and saves visual plots.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_classifier import get_model, compute_fft_channel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def print_divider(char="-", width=65):
    print(char * width)

def confusion_matrix_manual(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm

def per_class_metrics(cm, class_names):
    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)
        support   = cm[i, :].sum()
        rows.append((name, precision, recall, f1, support))
    return rows

def compute_roc_auc(y_true, y_scores):
    """
    Computes False Positive Rate, True Positive Rate, and AUC using trapezoidal integration.
    y_true: binary 0 (fake) or 1 (real).
    y_scores: probability of being real.
    """
    thresholds = np.linspace(0.0, 1.0, 201)
    tprs, fprs = [], []
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        tprs.append(tp / pos)
        fprs.append(fp / neg)

    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    fprs_sorted = fprs[order]
    tprs_sorted = tprs[order]
    auc = np.trapezoid(tprs_sorted, fprs_sorted) if hasattr(np, 'trapezoid') else np.trapz(tprs_sorted, fprs_sorted)
    return fprs_sorted, tprs_sorted, abs(float(auc))

def save_plots(cm, class_names, fprs, tprs, auc, cm_path="eval_results.png", roc_path="eval_roc.png"):
    if not HAS_MPL:
        return
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=[c.upper() for c in class_names],
           yticklabels=[c.upper() for c in class_names],
           ylabel="Ground Truth",
           xlabel="Predicted Label",
           title="Deepfake Classifier Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nConfusion Matrix plot saved -> {cm_path}")

    # 2. ROC Curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fprs, tprs, color="#1e88e5", lw=2.5, label=f"ROC Curve (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#757575", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"ROC Curve plot saved -> {roc_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Classifier Checkpoint")
    parser.add_argument("--data_dir", default="general_dataset", help="Dataset directory containing val/valid/test")
    parser.add_argument("--split", default="val", help="Split name to evaluate (val, valid, or test)")
    parser.add_argument("--checkpoint", default="best_model.pth", help="Checkpoint (.pth) file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cm_path", default="eval_results.png", help="Path to save confusion matrix image")
    parser.add_argument("--roc_path", default="eval_roc.png", help="Path to save ROC curve image")
    args = parser.parse_args()

    # Find evaluation directory
    eval_dir = os.path.join(args.data_dir, args.split)
    if not os.path.exists(eval_dir) and args.split == "val" and os.path.exists(os.path.join(args.data_dir, "valid")):
        eval_dir = os.path.join(args.data_dir, "valid")

    if not os.path.exists(eval_dir):
        print(f"ERROR: Evaluation directory not found at {eval_dir}")
        return

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        return

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt.get("classes", ["fake", "real"])
    num_classes = len(class_names)
    state_dict = ckpt["model_state_dict"]

    # Detect architecture
    has_fft = "freq_branch.conv.0.weight" in state_dict
    has_eff = "spatial_features.7.2.block.3.0.weight" in state_dict or "features.7.2.block.3.0.weight" in state_dict
    saved_arch = ckpt.get("arch", None)

    if saved_arch == "efficientnet_dual" or (has_eff and has_fft):
        arch = "efficientnet_dual"
        in_channels = 4
    elif saved_arch == "efficientnet" or (has_eff and not has_fft):
        arch = "efficientnet"
        in_channels = 3
    elif saved_arch == "dual_fft" or has_fft:
        arch = "dual_fft"
        in_channels = 4
    else:
        arch = "mobilenet"
        in_channels = 3

    print_divider()
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Architecture : {arch.upper()} (Inputs: {in_channels} channels)")
    print(f"  Classes      : {class_names}")
    print(f"  Eval Split   : {eval_dir}")
    print_divider()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device       : {device}")
    print_divider()

    model = get_model(num_classes=num_classes, arch=arch, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_dataset = datasets.ImageFolder(eval_dir, transform=val_transforms)
    eval_loader  = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"  Total samples: {len(eval_dataset)}")
    print(f"  Folder classes: {eval_dataset.classes}")
    print_divider()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in tqdm(eval_loader, desc=f"Evaluating [{arch.upper()}]"):
            imgs = imgs.to(device)
            if in_channels == 4:
                fft_mag = compute_fft_channel(imgs)
                model_inputs = torch.cat([imgs, fft_mag], dim=1)
            else:
                model_inputs = imgs

            logits = model(model_inputs)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    overall_acc = (all_preds == all_labels).mean()

    # Per-class metrics
    cm = confusion_matrix_manual(all_labels, all_preds, num_classes)
    metrics = per_class_metrics(cm, class_names)

    fake_rec = metrics[0][2] if len(metrics) > 0 else 0.0
    real_rec = metrics[1][2] if len(metrics) > 1 else 0.0
    bal_acc  = (fake_rec + real_rec) / 2.0

    # ROC / AUC (real class prob is index 1)
    real_probs = all_probs[:, 1]
    fprs, tprs, auc_score = compute_roc_auc(all_labels, real_probs)

    print_divider("=")
    print(f"  OVERALL ACCURACY  : {overall_acc*100:.2f}%  ({(all_preds==all_labels).sum()}/{len(all_labels)})")
    print(f"  BALANCED ACCURACY : {bal_acc*100:.2f}%")
    print(f"  ROC-AUC SCORE     : {auc_score:.4f}")
    print_divider("=")

    print(f"\n  {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print_divider()
    for name, p, r, f1, sup in metrics:
        print(f"  {name:<10} {p*100:>9.2f}% {r*100:>9.2f}% {f1*100:>9.2f}% {sup:>10}")
    print_divider()

    macro_p  = np.mean([m[1] for m in metrics])
    macro_r  = np.mean([m[2] for m in metrics])
    macro_f1 = np.mean([m[3] for m in metrics])
    print(f"  {'MACRO AVG':<10} {macro_p*100:>9.2f}% {macro_r*100:>9.2f}% {macro_f1*100:>9.2f}%")
    print_divider()

    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = f"  {'':10}" + "".join(f"{c.upper():>10}" for c in class_names)
    print(header)
    for i, row_name in enumerate(class_names):
        row_str = f"  {row_name.upper():<10}" + "".join(f"{cm[i,j]:>10}" for j in range(num_classes))
        print(row_str)
    print_divider("=")

    save_plots(cm, class_names, fprs, tprs, auc_score, args.cm_path, args.roc_path)

if __name__ == "__main__":
    main()
