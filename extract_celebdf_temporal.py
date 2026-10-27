import os
import argparse
import cv2
import random
import numpy as np
from tqdm import tqdm
from infer_rt import FaceDetector

def extract_consecutive_sequences(video_path, detector, seq_len=8, max_seqs_per_video=2):
    """
    Extracts consecutive sequences of `seq_len` face crops from a video.
    Ensures that frames in a sequence are truly consecutive in time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < seq_len + 5:
        cap.release()
        return []

    sequences = [] # list of lists: [[crop_0, crop_1, ... crop_T-1], ...]
    
    # Try at most 3 random start points to find a clean sequence
    start_candidates = sorted(random.sample(range(0, max(1, total_frames - seq_len - 2)), 
                                            min(max_seqs_per_video * 2, total_frames - seq_len)))
    
    for start_f in start_candidates:
        if len(sequences) >= max_seqs_per_video:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        current_seq = []
        consecutive_valid = True
        
        for _ in range(seq_len):
            ret, frame = cap.read()
            if not ret:
                consecutive_valid = False
                break
                
            faces = detector.detect(frame)
            if not faces:
                consecutive_valid = False
                break
                
            # Pick largest face
            faces.sort(key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = faces[0]
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            h_orig, w_orig = frame.shape[:2]
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w_orig, x + w + pad_x), min(h_orig, y + h + pad_y)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                consecutive_valid = False
                break
                
            current_seq.append(cv2.resize(crop, (224, 224)))
            
        if consecutive_valid and len(current_seq) == seq_len:
            sequences.append(current_seq)
            
    cap.release()
    return sequences

def main():
    parser = argparse.ArgumentParser(description="Extract consecutive face sequences from Celeb-DF v2 for Temporal LSTM")
    parser.add_argument("--celebdf_dir", type=str, default="C:\\Users\\asus\\Downloads\\archive (2)", 
                        help="Path to Celeb-DF v2 root")
    parser.add_argument("--output_dir", type=str, default="C:\\Users\\asus\\Downloads\\celebdf_sequences",
                        help="Output directory (saved to C: drive to preserve D: space)")
    parser.add_argument("--num_videos", type=int, default=250, 
                        help="Number of videos per class to process for fast temporal dataset creation")
    parser.add_argument("--seq_len", type=int, default=8, help="Number of consecutive frames per sequence")
    args = parser.parse_args()

    detector = FaceDetector(mode="mediapipe")
    
    real_dirs = [
        os.path.join(args.celebdf_dir, "Celeb-real"),
        os.path.join(args.celebdf_dir, "YouTube-real")
    ]
    fake_dirs = [
        os.path.join(args.celebdf_dir, "Celeb-synthesis")
    ]
    
    real_videos = []
    for rdir in real_dirs:
        if os.path.exists(rdir):
            for f in os.listdir(rdir):
                if f.lower().endswith(".mp4"):
                    real_videos.append(os.path.join(rdir, f))
                    
    fake_videos = []
    for fdir in fake_dirs:
        if os.path.exists(fdir):
            for f in os.listdir(fdir):
                if f.lower().endswith(".mp4"):
                    fake_videos.append(os.path.join(fdir, f))
                    
    print(f"Discovered: {len(real_videos)} Real videos, {len(fake_videos)} Fake videos.")
    
    random.seed(42)
    random.shuffle(real_videos)
    random.shuffle(fake_videos)
    
    selected_real = real_videos[:args.num_videos]
    selected_fake = fake_videos[:args.num_videos]
    
    for split in ["train", "val"]:
        for label in ["real", "fake"]:
            os.makedirs(os.path.join(args.output_dir, split, label), exist_ok=True)
            
    def process_group(video_list, label):
        val_cutoff = int(len(video_list) * 0.2)
        total_seqs = 0
        for idx, vpath in enumerate(tqdm(video_list, desc=f"Extracting {label.upper()} sequences")):
            split = "val" if idx < val_cutoff else "train"
            vname = os.path.splitext(os.path.basename(vpath))[0]
            seqs = extract_consecutive_sequences(vpath, detector, seq_len=args.seq_len, max_seqs_per_video=2)
            
            for s_idx, seq in enumerate(seqs):
                seq_dir = os.path.join(args.output_dir, split, label, f"{vname}_seq{s_idx:02d}")
                os.makedirs(seq_dir, exist_ok=True)
                for f_idx, crop in enumerate(seq):
                    cv2.imwrite(os.path.join(seq_dir, f"frame_{f_idx:02d}.jpg"), crop)
                total_seqs += 1
        return total_seqs

    print("\nExtracting consecutive sequences for Real class...")
    real_count = process_group(selected_real, "real")
    
    print("\nExtracting consecutive sequences for Fake class...")
    fake_count = process_group(selected_fake, "fake")
    
    print("\nSequence extraction complete!")
    print(f"Total extracted: {real_count} Real sequences, {fake_count} Fake sequences.")
    print(f"Saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
