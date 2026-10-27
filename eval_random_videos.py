import os
import sys
import random
import cv2
import numpy as np
from collections import Counter
import time

sys.path.insert(0, r"D:\college\Sem 3\Macro Project-I\project df")
from infer_rt import DeepfakeClassifier

BASE_FF = r"C:\Users\asus\Downloads\archive\FaceForensics++_C23"
BASE_CELEB = r"C:\Users\asus\Downloads\archive (2)"

print("Initializing Hybrid Deepfake Classifier...")
clf = DeepfakeClassifier(backend="hybrid")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Set random seed for reproducible random selection
random.seed(42)

# Pick 5 random videos from each category
categories = [
    # Fakes
    (os.path.join(BASE_FF, "NeuralTextures"), "FAKE", "FF++ NeuralTextures", 5),
    (os.path.join(BASE_FF, "Face2Face"), "FAKE", "FF++ Face2Face", 5),
    (os.path.join(BASE_FF, "FaceSwap"), "FAKE", "FF++ FaceSwap", 5),
    (os.path.join(BASE_FF, "FaceShifter"), "FAKE", "FF++ FaceShifter", 5),
    (os.path.join(BASE_FF, "Deepfakes"), "FAKE", "FF++ Deepfakes", 5),
    (os.path.join(BASE_CELEB, "Celeb-synthesis"), "FAKE", "Celeb-DF Synthesis", 5),
    # Reals
    (os.path.join(BASE_CELEB, "YouTube-real"), "REAL", "YouTube Real (Interviews)", 5),
    (os.path.join(BASE_FF, "original"), "REAL", "FF++ Original Real", 5),
    (os.path.join(BASE_CELEB, "Celeb-real"), "REAL", "Celeb-DF Celeb-Real", 5),
]

random_test_suite = []
for folder, gt, cat_name, count in categories:
    if os.path.exists(folder):
        all_vids = [f for f in os.listdir(folder) if f.endswith(('.mp4', '.avi'))]
        # Choose randomly
        sampled = random.sample(all_vids, min(count, len(all_vids)))
        for v in sampled:
            random_test_suite.append((os.path.join(folder, v), gt, cat_name))

print(f"Selected {len(random_test_suite)} randomly sampled test videos.")
print("=" * 115)
print(f"{'Category':26s} | {'Video Name':20s} | {'GT':4s} | {'Spatial':7s} | {'Temporal':8s} | {'Verdict':7s} | {'Confidence':10s} | {'Status':8s}")
print("=" * 115)

results = []

for vpath, gt, cat in random_test_suite:
    cap = cv2.VideoCapture(vpath)
    crops = []
    frames_read = 0
    while cap.isOpened() and len(crops) < 25 and frames_read < 80:
        ret, frame = cap.read()
        frames_read += 1
        if not ret: break
        if frames_read % 2 != 0: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        if len(faces) == 0: continue
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        pad = int(0.1 * max(w, h))
        H, W = frame.shape[:2]
        crop = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if crop.size > 0:
            crops.append(crop)
    cap.release()

    if not crops:
        print(f"{cat:26s} | {os.path.basename(vpath):20s} | {gt:4s} | NO FACE DETECTED")
        continue

    s_list, t_list, fused_list = [], [], []
    for i in range(max(1, len(crops) - 8), len(crops)):
        window = crops[max(0, i - 7):i + 1]
        lbl, conf, s_f, t_f = clf.classify_hybrid(crops[i], window)
        s_list.append(s_f)
        t_list.append(t_f)
        raw_fake = conf if lbl == "fake" else (1.0 - conf)
        fused_list.append(raw_fake)

    avg_s = float(np.mean(s_list))
    avg_t = float(np.mean(t_list))
    avg_fused = float(np.mean(fused_list))
    
    final_verdict = "FAKE" if avg_fused >= 0.50 else "REAL"
    final_conf = avg_fused if final_verdict == "FAKE" else (1.0 - avg_fused)
    
    is_correct = (final_verdict == gt)
    status_str = "MATCH" if is_correct else "FAIL"
    
    results.append({
        "category": cat,
        "video": os.path.basename(vpath),
        "gt": gt,
        "spatial": avg_s,
        "temporal": avg_t,
        "fused": avg_fused,
        "verdict": final_verdict,
        "conf": final_conf,
        "correct": is_correct
    })
    
    print(f"{cat:26s} | {os.path.basename(vpath):20s} | {gt:4s} | {avg_s*100:6.1f}% | {avg_t*100:6.1f}%  | {final_verdict:7s} | {final_conf*100:6.1f}%     | {status_str:8s}")

print("=" * 115)

# Metrics calculation
total = len(results)
correct = sum(1 for r in results if r["correct"])
real_cases = [r for r in results if r["gt"] == "REAL"]
fake_cases = [r for r in results if r["gt"] == "FAKE"]

TP = sum(1 for r in fake_cases if r["verdict"] == "FAKE") # True Positive
FN = sum(1 for r in fake_cases if r["verdict"] == "REAL") # False Negative
TN = sum(1 for r in real_cases if r["verdict"] == "REAL") # True Negative
FP = sum(1 for r in real_cases if r["verdict"] == "FAKE") # False Positive

acc = (correct / total * 100) if total else 0
sensitivity = (TP / (TP + FN) * 100) if (TP + FN) else 0 # Fake Recall
specificity = (TN / (TN + FP) * 100) if (TN + FP) else 0 # Real Recall
precision = (TP / (TP + FP) * 100) if (TP + FP) else 0
f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0

print(f"\n=======================================================")
print(f"       RANDOMIZED MULTI-DATASET EVALUATION REPORT      ")
print(f"=======================================================")
print(f"Total Random Videos Evaluated: {total}")
print(f"Overall Accuracy:              {correct}/{total} ({acc:.2f}%)")
print(f"Deepfake Sensitivity (Recall): {TP}/{TP+FN} ({sensitivity:.2f}%)")
print(f"Real Video Specificity:        {TN}/{TN+FP} ({specificity:.2f}%)")
print(f"Precision:                     {precision:.2f}%")
print(f"F1-Score:                      {f1:.2f}%")
print(f"\nConfusion Matrix:")
print(f"                 Predicted FAKE   Predicted REAL")
print(f"  Actual FAKE          {TP:4d}             {FN:4d}")
print(f"  Actual REAL          {FP:4d}             {TN:4d}")
print(f"=======================================================")

print(f"\nPer-Category Breakdown (5 Random Videos Each):")
cat_groups = {}
for r in results:
    cat_groups.setdefault(r["category"], []).append(r)

for cat, items in cat_groups.items():
    c_tot = len(items)
    c_cor = sum(1 for i in items if i["correct"])
    c_acc = (c_cor / c_tot * 100) if c_tot else 0
    gt_type = items[0]["gt"]
    print(f"  {cat:28s} (GT: {gt_type:4s}): {c_cor}/{c_tot} correct ({c_acc:.1f}%)")
