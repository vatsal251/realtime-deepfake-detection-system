import os
import argparse
import cv2
import random
from tqdm import tqdm
from infer_rt import FaceDetector

def extract_faces_from_video(video_path, detector, max_frames=25):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    indices = sorted(random.sample(range(total_frames), min(total_frames, max_frames)))
    crops = []
    frame_idx = 0
    
    while cap.isOpened() and frame_idx <= max(indices):
        ret, frame = cap.read()
        if not ret: break
        if frame_idx in indices:
            faces = detector.detect(frame)
            h_orig, w_orig = frame.shape[:2]
            for (x, y, w, h) in faces:
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w_orig, x + w + pad_x), min(h_orig, y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(cv2.resize(crop, (224, 224)))
        frame_idx += 1
    cap.release()
    return crops

def process_celebdf_or_wild(input_dir, output_dir, dataset_type="celebdf", val_split=0.2, max_frames=20):
    detector = FaceDetector(mode="mediapipe")
    os.makedirs(os.path.join(output_dir, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "fake"), exist_ok=True)

    print(f"Scanning {dataset_type} directory: {input_dir}")
    video_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov')):
                full_path = os.path.join(root, f)
                # Infer label from folder or filename
                lower_path = full_path.lower()
                if "real" in lower_path or "celeb-real" in lower_path or "youtube-real" in lower_path:
                    label = "real"
                elif "fake" in lower_path or "synthesis" in lower_path or "celeb-synthesis" in lower_path:
                    label = "fake"
                else:
                    label = "fake" # default
                video_files.append((full_path, label))

    print(f"Found {len(video_files)} videos in {input_dir}.")
    random.shuffle(video_files)
    num_val = int(len(video_files) * val_split)
    
    count_saved = 0
    for idx, (v_path, label) in enumerate(tqdm(video_files, desc="Processing")):
        split = "val" if idx < num_val else "train"
        target_dir = os.path.join(output_dir, split, label)
        base_name = os.path.splitext(os.path.basename(v_path))[0]
        
        crops = extract_faces_from_video(v_path, detector, max_frames=max_frames)
        for c_idx, crop in enumerate(crops):
            save_name = f"{base_name}_crop{c_idx:02d}.jpg"
            cv2.imwrite(os.path.join(target_dir, save_name), crop)
            count_saved += 1

    print(f"Extraction complete! Saved {count_saved} face crops to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to downloaded dataset root")
    parser.add_argument("--output_dir", type=str, default="modern_dataset")
    parser.add_argument("--dataset_type", type=str, choices=["celebdf", "wilddeepfake", "general"], default="celebdf")
    parser.add_argument("--max_frames", type=int, default=20)
    args = parser.parse_args()
    process_celebdf_or_wild(args.input_dir, args.output_dir, args.dataset_type, max_frames=args.max_frames)
