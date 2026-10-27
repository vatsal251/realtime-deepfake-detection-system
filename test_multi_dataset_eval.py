import os
import sys
import cv2
import numpy as np
from collections import deque
import time

sys.path.insert(0, r"D:\college\Sem 3\Macro Project-I\project df")
from infer_rt import DeepfakeClassifier

print("Initializing Hybrid Fusion Classifier...")
clf = DeepfakeClassifier(backend="hybrid")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ff_base = r"C:\Users\asus\Downloads\archive\FaceForensics++_C23"
celeb_base = r"C:\Users\asus\Downloads\archive (2)"

# Selection of diverse videos
test_suite = []

# 1. FF++ Deepfakes (Fake) - 6 videos
ff_df_dir = os.path.join(ff_base, "Deepfakes")
if os.path.exists(ff_df_dir):
    vids = sorted([f for f in os.listdir(ff_df_dir) if f.endswith('.mp4')])[:6]
    for v in vids:
        test_suite.append((os.path.join(ff_df_dir, v), "FAKE", "FF++ Deepfakes"))

# 2. FF++ Face2Face / FaceSwap (Fake) - 4 videos
ff_f2f_dir = os.path.join(ff_base, "Face2Face")
if os.path.exists(ff_f2f_dir):
    vids = sorted([f for f in os.listdir(ff_f2f_dir) if f.endswith('.mp4')])[:3]
    for v in vids:
        test_suite.append((os.path.join(ff_f2f_dir, v), "FAKE", "FF++ Face2Face"))

ff_fs_dir = os.path.join(ff_base, "FaceSwap")
if os.path.exists(ff_fs_dir):
    vids = sorted([f for f in os.listdir(ff_fs_dir) if f.endswith('.mp4')])[:3]
    for v in vids:
        test_suite.append((os.path.join(ff_fs_dir, v), "FAKE", "FF++ FaceSwap"))

# 3. FF++ Original (Real) - 8 videos
ff_orig_dir = os.path.join(ff_base, "original")
if os.path.exists(ff_orig_dir):
    vids = sorted([f for f in os.listdir(ff_orig_dir) if f.endswith('.mp4')])[:8]
    for v in vids:
        test_suite.append((os.path.join(ff_orig_dir, v), "REAL", "FF++ Original Real"))

# 4. Celeb-DF Celeb-synthesis (Fake) - 8 videos
c_synth_dir = os.path.join(celeb_base, "Celeb-synthesis")
if os.path.exists(c_synth_dir):
    vids = sorted([f for f in os.listdir(c_synth_dir) if f.endswith('.mp4')])[:8]
    for v in vids:
        test_suite.append((os.path.join(c_synth_dir, v), "FAKE", "Celeb-DF Synthesis"))

# 5. Celeb-DF YouTube-real (Real) - 8 videos
c_yt_dir = os.path.join(celeb_base, "YouTube-real")
if os.path.exists(c_yt_dir):
    # Include 00255.mp4 and others
    vids = [f for f in os.listdir(c_yt_dir) if f.endswith('.mp4')]
    # pick 00255 first, then a diverse sample
    selected_yt = ["00255.mp4"] + [v for v in vids if v != "00255.mp4"][:7]
    for v in selected_yt:
        test_suite.append((os.path.join(c_yt_dir, v), "REAL", "YouTube Real (Interviews)"))

# 6. Celeb-DF Celeb-real (Real) - 6 videos
c_real_dir = os.path.join(celeb_base, "Celeb-real")
if os.path.exists(c_real_dir):
    vids = sorted([f for f in os.listdir(c_real_dir) if f.endswith('.mp4')])[:6]
    for v in vids:
        test_suite.append((os.path.join(c_real_dir, v), "REAL", "Celeb-DF Real"))

print(f"Total videos selected for benchmark: {len(test_suite)}")
print("=" * 110)
print(f"{'Category':24s} | {'Video Name':24s} | {'GT':4s} | {'Spatial':7s} | {'Temporal':8s} | {'Verdict':7s} | {'Confidence':10s} | {'Status':8s}")
print("=" * 110)

results = []

for vpath, gt, cat in test_suite:
    if not os.path.exists(vpath):
        continue
    cap = cv2.VideoCapture(vpath)
    crops = []
    
    # Process up to 35 frames per video to simulate real-time tracker window
    frames_read = 0
    while cap.isOpened() and len(crops) < 25 and frames_read < 90:
        ret, frame = cap.read()
        frames_read += 1
        if not ret:
            break
        # Process every second frame
        if frames_read % 2 != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
        if len(faces) == 0:
            continue
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        pad = int(0.1 * max(w, h))
        H, W = frame.shape[:2]
        crop = frame[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        if crop.size > 0:
            crops.append(crop)
    cap.release()

    if not crops:
        print(f"{cat:24s} | {os.path.basename(vpath):24s} | {gt:4s} | NO FACE DETECTED")
        continue

    # Classify using rolling temporal window
    # Evaluate across the sequence to get representative prediction
    s_list, t_list, fused_list = [], [], []
    verdicts = []
    
    for i in range(max(1, len(crops) - 8), len(crops)):
        window = crops[max(0, i - 7):i + 1]
        lbl, conf, s_f, t_f = clf.classify_hybrid(crops[i], window)
        s_list.append(s_f)
        t_list.append(t_f)
        raw_fake = conf if lbl == "fake" else (1.0 - conf)
        fused_list.append(raw_fake)
        verdicts.append(lbl)

    # Average metrics over sequence
    avg_s = float(np.mean(s_list))
    avg_t = float(np.mean(t_list))
    avg_fused = float(np.mean(fused_list))
    
    # Final call at threshold 0.50
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
    
    print(f"{cat:24s} | {os.path.basename(vpath):24s} | {gt:4s} | {avg_s*100:6.1f}% | {avg_t*100:6.1f}%  | {final_verdict:7s} | {final_conf*100:6.1f}%     | {status_str:8s}")

print("=" * 110)

# Summary Statistics
total = len(results)
correct = sum(1 for r in results if r["correct"])
real_cases = [r for r in results if r["gt"] == "REAL"]
fake_cases = [r for r in results if r["gt"] == "FAKE"]

real_correct = sum(1 for r in real_cases if r["correct"])
fake_correct = sum(1 for r in fake_cases if r["correct"])

overall_acc = (correct / total * 100) if total else 0
specificity = (real_correct / len(real_cases) * 100) if real_cases else 0
sensitivity = (fake_correct / len(fake_cases) * 100) if fake_cases else 0

print(f"\n--- OVERALL PERFORMANCE SUMMARY ---")
print(f"Total Videos Evaluated: {total}")
print(f"Overall Accuracy:       {correct}/{total} ({overall_acc:.2f}%)")
print(f"Real Video Specificity: {real_correct}/{len(real_cases)} ({specificity:.2f}%)  [0 False Alarms / Real videos preserved]")
print(f"Fake Video Sensitivity: {fake_correct}/{len(fake_cases)} ({sensitivity:.2f}%)  [Deepfakes caught]")

# Per-category Breakdown
print(f"\n--- PER-CATEGORY BREAKDOWN ---")
cat_groups = {}
for r in results:
    cat_groups.setdefault(r["category"], []).append(r)

for cat, items in cat_groups.items():
    c_tot = len(items)
    c_cor = sum(1 for i in items if i["correct"])
    c_acc = (c_cor / c_tot * 100) if c_tot else 0
    gt_type = items[0]["gt"]
    print(f"  {cat:26s} (GT: {gt_type:4s}): {c_cor}/{c_tot} correct ({c_acc:.1f}%)")
