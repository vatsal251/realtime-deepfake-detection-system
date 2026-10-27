import os
import cv2
import time
from infer_rt import FaceDetector

def main():
    dataset_dir = "balanced_dataset"
    train_real_dir = os.path.join(dataset_dir, "train", "real")
    val_real_dir = os.path.join(dataset_dir, "val", "real")
    
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(val_real_dir, exist_ok=True)
    
    detector = FaceDetector(mode="cascade")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("\n" + "="*50)
    print("PERSONAL FACE CAPTURE INSTRUCTIONS:")
    print("1. Look at the camera.")
    print("2. Slowly tilt your head, smile, and change angles slightly.")
    print("3. The script will capture 150 face crops of you.")
    print("="*50 + "\n")
    
    print("Starting capture in 3 seconds... Get ready!")
    time.sleep(3)
    
    count = 0
    max_images = 150
    val_split = 0.2  # 20% to validation, 80% to training
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        faces = detector.detect(frame)
        h_orig, w_orig = frame.shape[:2]
        
        # Overlay instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Captured: {count}/{max_images}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Capturing Your Real Face Crops...", display_frame)
        
        for (x, y, w, h) in faces:
            pad_x = int(w * 0.1)
            pad_y = int(h * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w_orig, x + w + pad_x)
            y2 = min(h_orig, y + h + pad_y)
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                # Decide split
                if count % int(1/val_split) == 0:
                    save_path = os.path.join(val_real_dir, f"self_{count}.jpg")
                else:
                    save_path = os.path.join(train_real_dir, f"self_{count}.jpg")
                    
                cv2.imwrite(save_path, face_crop)
                count += 1
                break  # Process only one face per frame
                
        # Wait a small delay to avoid capturing identical contiguous frames
        if cv2.waitKey(80) & 0xFF == ord('q'):
            print("Capture cancelled by user.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCapture complete! Saved {count} face crops to:")
    print(f"- Training: {train_real_dir}")
    print(f"- Validation: {val_real_dir}")

if __name__ == "__main__":
    main()
