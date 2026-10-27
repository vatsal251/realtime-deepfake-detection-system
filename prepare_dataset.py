import os
import argparse
import cv2
import random
from tqdm import tqdm
from infer_rt import FaceDetector

def process_video_folder(video_dir, output_dir, label, detector, max_frames_per_video=20, val_split=0.2, max_videos=None):
    """
    Reads videos from video_dir, extracts faces from selected frames,
    and saves them to output_dir/train/label and output_dir/val/label.
    """
    os.makedirs(os.path.join(output_dir, "train", label), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", label), exist_ok=True)
    
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print(f"No video files found in {video_dir}")
        return
        
    # Shuffle and split into train and validation sets
    random.shuffle(video_files)
    if max_videos is not None and max_videos > 0:
        video_files = video_files[:max_videos]
        
    print(f"\nProcessing {len(video_files)} videos in '{video_dir}' for label '{label}'...")
    
    num_val = int(len(video_files) * val_split)
    val_videos = set(video_files[:num_val])
    
    img_counter = 0
    
    for video_name in tqdm(video_files):
        video_path = os.path.join(video_dir, video_name)
        split = "val" if video_name in val_videos else "train"
        save_dir = os.path.join(output_dir, split, label)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            continue
            
        # Select evenly spaced frames throughout the video (avoiding just the beginning/end)
        frame_indices = sorted(random.sample(range(total_frames), min(total_frames, max_frames_per_video)))
        
        frame_idx = 0
        extracted_from_this_video = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx in frame_indices:
                faces = detector.detect(frame)
                h_orig, w_orig = frame.shape[:2]
                
                for idx, (x, y, w, h) in enumerate(faces):
                    # Pad the face box slightly to capture context (ears, haircut, etc.)
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(w_orig, x + w + pad_x)
                    y2 = min(h_orig, y + h + pad_y)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        img_filename = f"{os.path.splitext(video_name)[0]}_f{frame_idx}_face{idx}.jpg"
                        cv2.imwrite(os.path.join(save_dir, img_filename), face_crop)
                        img_counter += 1
                        extracted_from_this_video += 1
                        
            frame_idx += 1
            if frame_idx > max(frame_indices):
                break
                
        cap.release()

    print(f"Extracted {img_counter} faces for label '{label}'.")

def main():
    parser = argparse.ArgumentParser(description="Extract faces from FaceForensics++ video directories")
    parser.add_argument("--kag_root", type=str, required=True, help="Root folder of the downloaded Kaggle dataset containing 'original' and 'Deepfakes' directories")
    parser.add_argument("--out_dir", type=str, default="cropped_dataset", help="Output directory where the image folder structure will be generated")
    parser.add_argument("--max_frames", type=int, default=20, help="Maximum frames to extract per video")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum videos to process per video folder")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set split ratio (0.0 to 1.0)")
    parser.add_argument("--detector", type=str, choices=["yolo", "cascade"], default="yolo", help="Face detection method to use for cropping")
    
    args = parser.parse_args()
    
    real_video_dir = os.path.join(args.kag_root, "original")
    
    # FaceForensics++ contains multiple fake folders. We can read from Deepfakes or consolidate them.
    # We will search the root directory for potential fake folders
    potential_fake_dirs = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "DeepFakeDetection", "FaceShifter"]
    fake_video_dirs = []
    
    for fd in potential_fake_dirs:
        path = os.path.join(args.kag_root, fd)
        if os.path.exists(path):
            fake_video_dirs.append(path)
            
    if not os.path.exists(real_video_dir):
        print(f"Error: Could not find 'original' video directory at: {real_video_dir}")
        print("Please check that you entered the correct path to the downloaded Kaggle files.")
        return
        
    if not fake_video_dirs:
        print(f"Error: Could not find any deepfake directories at: {args.kag_root}")
        print(f"Expected to find at least one of these: {potential_fake_dirs}")
        return

    # Initialize Face Detector (Haar cascade as a fast processor or YOLO)
    print("Initializing Face Detector...")
    detector = FaceDetector(mode=args.detector)
    
    # Process Real Videos
    process_video_folder(
        video_dir=real_video_dir,
        output_dir=args.out_dir,
        label="real",
        detector=detector,
        max_frames_per_video=args.max_frames,
        val_split=args.val_split,
        max_videos=args.max_videos
    )
    
    # Process Fake Videos (iterates and groups all present fake types into 'fake' label)
    for fake_dir in fake_video_dirs:
        process_video_folder(
            video_dir=fake_dir,
            output_dir=args.out_dir,
            label="fake",
            detector=detector,
            max_frames_per_video=args.max_frames,
            val_split=args.val_split,
            max_videos=args.max_videos
        )
        
    print(f"\nPreprocessing finished! Face crops dataset created at: {args.out_dir}")
    print("You can now start training with the command:")
    print(f"python train_classifier.py --data_dir {args.out_dir} --epochs 20")

if __name__ == "__main__":
    main()
