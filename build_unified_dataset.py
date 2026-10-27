import os
import sys
import cv2
import random
import numpy as np
from tqdm import tqdm
import time

OUTPUT_DIR = r"D:\college\Sem 3\Macro Project-I\project df\unified_dataset"
BASE_FF = r"C:\Users\asus\Downloads\archive\FaceForensics++_C23"
BASE_CELEB = r"C:\Users\asus\Downloads\archive (2)"

print(f"Target dataset directory: {OUTPUT_DIR}")
for split in ["train", "val"]:
    for cls in ["fake", "real"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_faces_from_video(video_path, max_frames=6):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 10:
        cap.release()
        return []
    
    step = max(1, total // (max_frames + 2))
    crops = []
    frame_idx = 0
    while cap.isOpened() and len(crops) < max_frames and frame_idx < total:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                H, W = frame.shape[:2]
                crop = frame[max(0, y - pad_y):min(H, y + h + pad_y), max(0, x - pad_x):min(W, x + w + pad_x)]
                if crop.size > 0:
                    crops.append(cv2.resize(crop, (224, 224)))
        frame_idx += 1
    cap.release()
    return crops

def process_category(vids, category_name, target_class, max_vids_train, max_vids_val, frames_per_vid=5):
    print(f"\nProcessing [{category_name}] -> {target_class.upper()}...")
    # Stratify by video ID: videos starting with '9' or ending of list go to val
    val_candidates = [v for v in vids if os.path.basename(v).startswith("9") or os.path.basename(v).startswith("id0_")]
    train_candidates = [v for v in vids if v not in val_candidates]
    
    if len(val_candidates) < max_vids_val:
        # take tail of train
        needed = max_vids_val - len(val_candidates)
        val_candidates += train_candidates[-needed:]
        train_candidates = train_candidates[:-needed]

    train_selected = train_candidates[:max_vids_train]
    val_selected = val_candidates[:max_vids_val]

    t_count, v_count = 0, 0
    # Train
    for vpath in tqdm(train_selected, desc=f"  train/{target_class} ({category_name})"):
        crops = extract_faces_from_video(vpath, max_frames=frames_per_vid)
        base = os.path.splitext(os.path.basename(vpath))[0]
        for idx, c in enumerate(crops):
            out_name = f"{category_name}_{base}_f{idx}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "train", target_class, out_name), c)
            t_count += 1

    # Val
    for vpath in tqdm(val_selected, desc=f"  val/{target_class} ({category_name})"):
        crops = extract_faces_from_video(vpath, max_frames=frames_per_vid)
        base = os.path.splitext(os.path.basename(vpath))[0]
        for idx, c in enumerate(crops):
            out_name = f"{category_name}_{base}_f{idx}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, "val", target_class, out_name), c)
            v_count += 1

    print(f"  -> Extracted {t_count} train images, {v_count} val images for {category_name}")

def main():
    t_start = time.time()
    
    # --- FAKE CATEGORIES ---
    # 1. NeuralTextures (FF++)
    nt_dir = os.path.join(BASE_FF, "NeuralTextures")
    if os.path.exists(nt_dir):
        vids = [os.path.join(nt_dir, f) for f in os.listdir(nt_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_NeuralTextures", "fake", max_vids_train=250, max_vids_val=40, frames_per_vid=6)

    # 2. Face2Face (FF++)
    f2f_dir = os.path.join(BASE_FF, "Face2Face")
    if os.path.exists(f2f_dir):
        vids = [os.path.join(f2f_dir, f) for f in os.listdir(f2f_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_Face2Face", "fake", max_vids_train=250, max_vids_val=40, frames_per_vid=6)

    # 3. FaceSwap (FF++)
    fs_dir = os.path.join(BASE_FF, "FaceSwap")
    if os.path.exists(fs_dir):
        vids = [os.path.join(fs_dir, f) for f in os.listdir(fs_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_FaceSwap", "fake", max_vids_train=250, max_vids_val=40, frames_per_vid=6)

    # 4. FaceShifter (FF++)
    fsh_dir = os.path.join(BASE_FF, "FaceShifter")
    if os.path.exists(fsh_dir):
        vids = [os.path.join(fsh_dir, f) for f in os.listdir(fsh_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_FaceShifter", "fake", max_vids_train=200, max_vids_val=30, frames_per_vid=6)

    # 5. Deepfakes (FF++)
    df_dir = os.path.join(BASE_FF, "Deepfakes")
    if os.path.exists(df_dir):
        vids = [os.path.join(df_dir, f) for f in os.listdir(df_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_Deepfakes", "fake", max_vids_train=250, max_vids_val=40, frames_per_vid=6)

    # 6. Celeb-DF Synthesis (Celeb-DF)
    cs_dir = os.path.join(BASE_CELEB, "Celeb-synthesis")
    if os.path.exists(cs_dir):
        vids = [os.path.join(cs_dir, f) for f in os.listdir(cs_dir) if f.endswith('.mp4')]
        process_category(vids, "Celeb_Synthesis", "fake", max_vids_train=300, max_vids_val=50, frames_per_vid=6)

    # --- REAL CATEGORIES ---
    # 7. YouTube Real (Critical for video compression & interview preservation)
    yt_dir = os.path.join(BASE_CELEB, "YouTube-real")
    if os.path.exists(yt_dir):
        vids = [os.path.join(yt_dir, f) for f in os.listdir(yt_dir) if f.endswith('.mp4')]
        process_category(vids, "Celeb_YouTubeReal", "real", max_vids_train=240, max_vids_val=50, frames_per_vid=8)

    # 8. FaceForensics++ Original Real
    orig_dir = os.path.join(BASE_FF, "original")
    if os.path.exists(orig_dir):
        vids = [os.path.join(orig_dir, f) for f in os.listdir(orig_dir) if f.endswith('.mp4')]
        process_category(vids, "FF_OriginalReal", "real", max_vids_train=350, max_vids_val=50, frames_per_vid=6)

    # 9. Celeb-DF Celeb-Real
    cr_dir = os.path.join(BASE_CELEB, "Celeb-real")
    if os.path.exists(cr_dir):
        vids = [os.path.join(cr_dir, f) for f in os.listdir(cr_dir) if f.endswith('.mp4')]
        process_category(vids, "Celeb_CelebReal", "real", max_vids_train=350, max_vids_val=50, frames_per_vid=6)

    duration = time.time() - t_start
    print("\n=======================================================")
    print(f"Unified Dataset Extraction Complete in {duration:.1f}s!")
    for split in ["train", "val"]:
        n_fake = len(os.listdir(os.path.join(OUTPUT_DIR, split, "fake")))
        n_real = len(os.listdir(os.path.join(OUTPUT_DIR, split, "real")))
        print(f"  {split.upper()}: {n_fake} Fakes, {n_real} Reals (Total: {n_fake + n_real})")
    print("=======================================================")

if __name__ == "__main__":
    main()
