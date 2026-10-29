import os
import time
import argparse
import cv2
import numpy as np

# Enable Windows High-DPI awareness so OpenCV renders razor-sharp text and frames
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# Try importing ML frameworks; fall back gracefully if not fully installed yet
try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from train_classifier import get_model as get_pytorch_model
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


class ScreenCapture:
    """High-speed screen capture streamer for real-time meetings and desktop video."""
    def __init__(self, monitor_idx=1, bbox=None, target_fps=30):
        self.bbox = bbox
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.last_grab = 0
        self.is_opened = True
        self.use_mss = False
        self.sct = None
        
        try:
            import mss
            self.sct = mss.mss()
            monitors = self.sct.monitors
            if monitor_idx < len(monitors):
                self.monitor = monitors[monitor_idx]
            else:
                self.monitor = monitors[0]
            self.use_mss = True
            print(f"[ScreenCapture] High-speed MSS active ({self.monitor['width']}x{self.monitor['height']}).")
        except Exception as e:
            print(f"[ScreenCapture] MSS notice: {e}. Falling back to PIL ImageGrab.")
            self.use_mss = False

    def isOpened(self):
        return self.is_opened

    def set_roi(self, x, y, w, h):
        if w > 20 and h > 20:
            if self.use_mss and self.sct:
                base_left = self.monitor.get('left', 0)
                base_top = self.monitor.get('top', 0)
                self.bbox = {
                    "left": int(base_left + x),
                    "top": int(base_top + y),
                    "width": int(w),
                    "height": int(h)
                }
            else:
                self.bbox = (int(x), int(y), int(x + w), int(y + h))
            print(f"[ScreenCapture] Locked ROI: {w}x{h} at ({x}, {y})")

    def reset_roi(self):
        self.bbox = None
        print("[ScreenCapture] Reset to Full Screen.")

    def read(self):
        if not self.is_opened:
            return False, None
            
        now = time.time()
        elapsed = now - self.last_grab
        if elapsed < self.frame_delay:
            time.sleep(max(0.001, self.frame_delay - elapsed))
        self.last_grab = time.time()
        
        try:
            if self.use_mss and self.sct:
                target = self.bbox if self.bbox is not None else self.monitor
                img = np.array(self.sct.grab(target))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                return True, frame
            else:
                from PIL import ImageGrab
                img = ImageGrab.grab(bbox=self.bbox if isinstance(self.bbox, tuple) else None)
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                return True, frame
        except Exception:
            time.sleep(0.05)
            return False, None

    def release(self):
        self.is_opened = False
        if self.use_mss and self.sct:
            try:
                self.sct.close()
            except Exception:
                pass


class FaceDetector:
    """Face detection supporting MediaPipe (best), YOLOv8, and Haar Cascade (fallback)."""
    def __init__(self, mode="mediapipe", yolo_weights="yolov8n-face.pt"):
        self.mode = mode
        self.yolo_model = None
        self.cascade    = None
        self.mp_detector = None

        # ── MediaPipe ────────────────────────────────────────────────────────
        if mode == "mediapipe":
            if HAS_MEDIAPIPE:
                model_path = "blaze_face_full_range.tflite"
                if not os.path.exists(model_path):
                    print("Downloading BlazeFace model for MediaPipe (~800KB)...")
                    import urllib.request
                    url = ("https://storage.googleapis.com/mediapipe-models/"
                           "face_detector/blaze_face_full_range/float16/latest/"
                           "blaze_face_full_range.tflite")
                    try:
                        urllib.request.urlretrieve(url, model_path)
                        print("BlazeFace model downloaded.")
                    except Exception as e:
                        print(f"Download failed: {e}. Falling back to Haar Cascade.")
                        self.mode = "cascade"
                if self.mode == "mediapipe":
                    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
                    opts = mp_vision.FaceDetectorOptions(
                        base_options=base_opts,
                        min_detection_confidence=0.50
                    )
                    self.mp_detector = mp_vision.FaceDetector.create_from_options(opts)
                    print("Loaded MediaPipe BlazeFace detector (full-range model).")
            else:
                print("MediaPipe not installed. Falling back to Haar Cascade.")
                self.mode = "cascade"

        # ── YOLO ─────────────────────────────────────────────────────────────
        elif mode == "yolo" and HAS_YOLO:
            if not os.path.exists(yolo_weights):
                print(f"YOLO weights '{yolo_weights}' not found. Falling back to Haar Cascade.")
                self.mode = "cascade"
            else:
                try:
                    self.yolo_model = YOLO(yolo_weights)
                    print(f"Loaded YOLO face model from {yolo_weights}")
                except Exception as e:
                    print(f"Failed to load YOLO model: {e}. Falling back to Haar Cascade.")
                    self.mode = "cascade"

        # ── Haar Cascade (fallback) ───────────────────────────────────────────
        if self.mode == "cascade" or (mode not in ("mediapipe", "yolo")):
            self.mode = "cascade"
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            print("Loaded OpenCV Haar Cascade face detector.")

    def detect(self, frame):
        """Returns list of (x, y, w, h) for all faces found in frame."""
        faces = []

        # ── MediaPipe ────────────────────────────────────────────────────────
        if self.mode == "mediapipe" and self.mp_detector is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.mp_detector.detect(mp_image)
            if result.detections:
                for det in result.detections:
                    bb = det.bounding_box
                    x, y, w, h = bb.origin_x, bb.origin_y, bb.width, bb.height
                    if w >= 60 and h >= 60:   # ignore tiny background detections
                        faces.append((x, y, w, h))

        # ── YOLO ─────────────────────────────────────────────────────────────
        elif self.mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(frame, verbose=False)[0]
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                faces.append((x1, y1, x2 - x1, y2 - y1))

        # ── Haar Cascade ─────────────────────────────────────────────────────
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = self.cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=11,
                minSize=(120, 120)
            )
            for (x, y, w, h) in detected:
                faces.append((x, y, w, h))

        return faces




class DeepfakeClassifier:
    """Deepfake classification client supporting PyTorch (.pth) and ONNX (.onnx) backends."""
    def __init__(self, backend="onnx", model_path="mobilenetv3_deepfake.onnx"):
        self.backend = backend
        if not os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(script_dir, model_path)
            if os.path.exists(candidate):
                model_path = candidate
        self.model_path = model_path
        self.classes = ["fake", "real"]
        
        # Define image normalizations (must match training parameters)
        self.img_size = 224
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"Initializing Classifier with {backend.upper()} backend loading: {model_path}")
        
        self.is_temporal = False
        self.is_hybrid = False

        if backend == "hybrid":
            self.is_hybrid = True
            script_dir = os.path.dirname(os.path.abspath(__file__))
            sp_path = os.path.join(script_dir, "mobilenetv3_deepfake.onnx") if not os.path.exists("mobilenetv3_deepfake.onnx") else "mobilenetv3_deepfake.onnx"
            tp_path = os.path.join(script_dir, "best_temporal_model.pth") if not os.path.exists("best_temporal_model.pth") else "best_temporal_model.pth"
            
            # Spatial ONNX stream
            avail = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in avail]
            if not providers: providers = ['CPUExecutionProvider']
            self.session = ort.InferenceSession(sp_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.in_channels = 4
            
            # Temporal PyTorch stream
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            from temporal_model import get_temporal_model
            ckpt = torch.load(tp_path, map_location=self.device, weights_only=False)
            self.temp_model = get_temporal_model(num_classes=2, pretrained_spatial=False)
            self.temp_model.load_state_dict(ckpt['model_state_dict'])
            self.temp_model.to(self.device)
            self.temp_model.eval()
            print(f"HYBRID FUSION ENGINE ACTIVE: Spatial ONNX (4ch FFT) + Temporal Bi-LSTM on {self.device}.")

        elif backend == "onnx":
            if not HAS_ORT:
                raise ImportError("onnxruntime is not installed. Run 'pip install onnxruntime' or use --backend pytorch")
            avail = ort.get_available_providers()
            providers = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in avail]
            if not providers:
                providers = ['CPUExecutionProvider']
            print(f"ONNX Execution providers: {providers}")
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            in_shape = self.session.get_inputs()[0].shape
            
            # Check if 5D input: (B, T, C, H, W)
            if len(in_shape) == 5:
                self.is_temporal = True
                self.in_channels = in_shape[2] if isinstance(in_shape[2], int) else 3
                print(f"ONNX Model operates in TEMPORAL MODE (Sequence of frames, {self.in_channels} channels).")
            else:
                self.in_channels = in_shape[1] if (len(in_shape) > 1 and isinstance(in_shape[1], int)) else 3
                print(f"ONNX Model expects {self.in_channels} input channels ({'Spatial + Frequency (FFT)' if self.in_channels == 4 else 'Spatial Only'}).")
        
        elif backend == "pytorch":
            if not HAS_TORCH:
                raise ImportError("PyTorch is not installed. Use --backend onnx")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.device = device
            print(f"PyTorch Runtime Device: {device}")
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            self.classes = checkpoint.get("classes", ["fake", "real"])
            state_dict = checkpoint.get('model_state_dict', {})
            arch = checkpoint.get("arch", None)
            if arch is None:
                if "lstm.weight_ih_l0" in state_dict:
                    arch = "temporal_cnn_lstm"
                elif "freq_branch.conv.0.weight" in state_dict:
                    arch = "dual_fft"
                else:
                    arch = "mobilenet"
                    
            if arch == "temporal_cnn_lstm":
                self.is_temporal = True
                self.in_channels = 3
                from temporal_model import get_temporal_model
                print("PyTorch Model operates in TEMPORAL MODE (Sequence of frames).")
                self.model = get_temporal_model(num_classes=len(self.classes), pretrained_spatial=False)
                self.model.load_state_dict(state_dict)
                self.model.to(device)
                self.model.eval()
            else:
                self.in_channels = 4 if arch in ["dual_fft", "efficientnet_dual"] else 3
                print(f"PyTorch Model architecture: {arch.upper()} ({self.in_channels} channels).")
                self.model = get_pytorch_model(num_classes=len(self.classes), arch=arch, pretrained=False)
                self.model.load_state_dict(state_dict)
                self.model.to(device)
                self.model.eval()



    def compute_fft_map(self, img_normalized):
        """Computes normalized 2D FFT log-magnitude spectrum matching train_classifier.py exactly."""
        # img_normalized is (3, 224, 224) float32 with mean subtracted and std divided
        # Grayscale approximation: 0.2989*R + 0.5870*G + 0.1140*B
        gray = 0.2989 * img_normalized[0] + 0.5870 * img_normalized[1] + 0.1140 * img_normalized[2]
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        mag = np.log(np.abs(fft_shift) + 1e-6)
        m_min, m_max = mag.min(), mag.max()
        mag_norm = (mag - m_min) / (m_max - m_min + 1e-8)
        return mag_norm.astype(np.float32)

    def preprocess_face(self, face_crop):
        """Converts BGR face crop to normalized RGB (+ FFT if dual_fft model) format."""
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (self.img_size, self.img_size))
        
        # Normalized RGB
        img = face_resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1) # (3, 224, 224)
        
        # If 4-channel model, append 2D FFT frequency map computed on normalized RGB
        if self.in_channels == 4:
            fft_map = self.compute_fft_map(img) # (224, 224)
            img = np.concatenate([img, fft_map[np.newaxis, :, :]], axis=0) # (4, 224, 224)
            
        if self.backend in ["onnx", "hybrid"]:
            return img
        else:
            return torch.from_numpy(img)


    def classify_batch(self, face_crops):
        """Runs batch inference on list of cropped faces."""
        if not face_crops:
            return []
            
        preprocessed = [self.preprocess_face(crop) for crop in face_crops]
        
        if self.backend in ["onnx", "hybrid"]:
            # Stack images along the batch dimension
            batch_input = np.stack(preprocessed, axis=0).astype(np.float32)
            outputs = self.session.run(None, {self.input_name: batch_input})[0]
            
            # Apply Softmax to raw scores (logits)
            exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            results = []
            for prob in probs:
                label_idx = np.argmax(prob)
                label = self.classes[label_idx]
                confidence = prob[label_idx]
                results.append((label, confidence))
            return results
            
        else: # PyTorch backend
            with torch.no_grad():
                # Stack to batched tensor
                batch_tensor = torch.stack(preprocessed).to(self.device)
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                results = []
                for prob in probs:
                    label_idx = np.argmax(prob)
                    label = self.classes[label_idx]
                    confidence = prob[label_idx]
                    results.append((label, confidence))
                return results

    def classify_sequence(self, sequence_crops, target_seq_len=8):
        """Classifies a temporal sequence of face crops across time."""
        if not sequence_crops:
            return "fake", 0.5
        
        # If fewer than target_seq_len, repeat last crop; if more, uniform sample
        if len(sequence_crops) >= target_seq_len:
            indices = np.linspace(0, len(sequence_crops) - 1, target_seq_len).astype(int)
            sampled = [sequence_crops[i] for i in indices]
        else:
            sampled = list(sequence_crops) + [sequence_crops[-1]] * (target_seq_len - len(sequence_crops))

        preprocessed = [self.preprocess_face(c) for c in sampled]

        if self.backend == "onnx":
            # Shape: (1, T, C, H, W)
            seq_np = np.stack(preprocessed, axis=0)[np.newaxis, ...].astype(np.float32)
            outputs = self.session.run(None, {self.input_name: seq_np})[0]
            exp_logits = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            prob = probs[0]
            idx = np.argmax(prob)
            return self.classes[idx], prob[idx]
        else:
            with torch.no_grad():
                seq_t = torch.stack(preprocessed, dim=0).unsqueeze(0).to(self.device)
                outputs = self.model(seq_t)
                prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                idx = np.argmax(prob)
                return self.classes[idx], prob[idx]

    def preprocess_temporal_crop(self, crop):
        """Preprocesses RGB face crop for PyTorch Temporal model."""
        face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (self.img_size, self.img_size))
        img = face_resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)

    def classify_hybrid(self, face_crop, history_crops, target_seq_len=8):
        """Unified Spatial-Frequency (ONNX) + Temporal Sequence (Bi-LSTM) Adaptive Fusion."""
        # 1. Spatial stream on current face crop
        p_sp = self.classify_batch([face_crop])[0]
        s_fake = float(p_sp[1] if p_sp[0] == 'fake' else (1.0 - p_sp[1]))

        # 2. Temporal stream on rolling sequence
        if not history_crops:
            sampled = [face_crop] * target_seq_len
        elif len(history_crops) >= target_seq_len:
            indices = np.linspace(0, len(history_crops) - 1, target_seq_len).astype(int)
            sampled = [history_crops[i] for i in indices]
        else:
            sampled = list(history_crops) + [history_crops[-1]] * (target_seq_len - len(history_crops))

        preprocessed = [self.preprocess_temporal_crop(c) for c in sampled]
        with torch.no_grad():
            seq_t = torch.stack(preprocessed, dim=0).unsqueeze(0).to(self.device)
            out_t = self.temp_model(seq_t)
            prob_t = torch.softmax(out_t, dim=1).cpu().numpy()[0]
            t_fake = float(prob_t[0])

        # 3. Multi-Modal Adaptive Evidence Fusion
        if s_fake >= 0.75:
            # Overwhelming spatial/FFT artifact evidence (NeuralTextures, Face2Face, FaceSwap, Deepfakes)
            fused_score = s_fake
            label = "fake"
        elif t_fake >= 0.65:
            # High-confidence temporal jitter / warping (Celeb-DF, Wav2Lip)
            fused_score = t_fake
            label = "fake"
        elif s_fake >= 0.50 and t_fake >= 0.25:
            # Concordant evidence: spatial detects anomaly AND temporal does NOT contradict
            fused_score = 0.65 * s_fake + 0.35 * t_fake
            label = "fake" if fused_score >= 0.50 else "real"
        else:
            # Real video baseline: temporal stream strongly refutes manipulation (t_fake < 0.25)
            # or spatial confirms authentic textures (s_fake < 0.50)
            fused_score = 0.60 * s_fake + 0.40 * t_fake
            label = "real"

        confidence = fused_score if label == "fake" else (1.0 - fused_score)
        return label, confidence, s_fake, t_fake




def get_iou(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def match_boxes(boxA, boxB):
    """Robust tracker matching combining IoU and normalized center distance."""
    iou = get_iou(boxA, boxB)
    if iou > 0.25:
        return iou
    # Proximity matching: prevents track resets during head tilts and subtle movement
    cA_x, cA_y = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
    cB_x, cB_y = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
    wA, hA = (boxA[2] - boxA[0]), (boxA[3] - boxA[1])
    wB, hB = (boxB[2] - boxB[0]), (boxB[3] - boxB[1])
    avg_dim = max(10.0, (wA + hA + wB + hB) / 4.0)
    dist = np.hypot(cA_x - cB_x, cA_y - cB_y) / avg_dim
    if dist < 0.60:
        return max(0.26, 1.0 - dist)
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Real-Time Deepfake Detection Inference")
    parser.add_argument("--input", type=str, default="0", help="Camera index (e.g. 0), path to input video file, or 'screen' for live screen/meeting capture")
    parser.add_argument("--select_roi", action="store_true", help="Interactively select a meeting window / screen region to monitor")
    parser.add_argument("--detector", type=str, choices=["mediapipe", "yolo", "cascade"], default="mediapipe", help="Face detector: mediapipe (best), yolo, or cascade (fallback)")
    parser.add_argument("--backend", type=str, choices=["hybrid", "pytorch", "onnx"], default="hybrid", help="Inference engine backend: hybrid (Spatial+Temporal auto-fusion), onnx, or pytorch")
    parser.add_argument("--model", type=str, default="mobilenetv3_deepfake.onnx", help="Path to classifier weights file")
    parser.add_argument("--real_threshold", type=float, default=None,
                        help="Legacy parameter for classification threshold.")
    parser.add_argument("--fake_threshold", type=float, default=None,
                        help="Threshold on fake probability to flag face as FAKE (default 0.50). Lower = more sensitive to fakes.")
    
    args = parser.parse_args()
    
    # Adjust default weights if backend is PyTorch
    if args.backend == "pytorch" and args.model in ("mobilenetv3_deepfake.onnx", "best_model.pth"):
        for candidate in ["best_modern_model.pth", "best_temporal_model.pth", "best_model.pth"]:
            if os.path.exists(candidate):
                args.model = candidate
                break
        
    if args.backend == "hybrid":
        if not os.path.exists("mobilenetv3_deepfake.onnx") or not os.path.exists("best_temporal_model.pth"):
            print("Warning: Required weights for hybrid mode missing (mobilenetv3_deepfake.onnx or best_temporal_model.pth).")
            return
    elif not os.path.exists(args.model):
        print(f"Warning: Classifier weight file not found: {args.model}")
        print("Please train a model or ensure steps have run to create dummy placeholder files first.")
        # If both are missing, let's create dynamic placeholders
        if args.backend == "onnx":
            print("To generate a dummy ONNX model, you can run: python export_onnx.py")
        else:
            print("To generate a dummy PyTorch model, you can run: python train_classifier.py")
        return

    # Check frame sources
    is_screen = (args.input.lower() == "screen")
    is_webcam = args.input.isdigit() if not is_screen else False
    
    if is_screen:
        cap = ScreenCapture()
    else:
        cap = cv2.VideoCapture(int(args.input) if is_webcam else args.input)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.input}")
        return

    if is_screen:
        print("\n" + "="*65)
        print("🖥️  LIVE SCREEN & MEETING DETECTION ACTIVE")
        print("• Monitoring live desktop / Zoom / Teams / Google Meet / YouTube")
        print("• Controls in HUD:")
        print("   - Press 's' to drag & select a specific Meeting window ROI")
        print("   - Press 'f' to return to Full Screen monitoring")
        print("   - Press 'd' to toggle the bottom diagnostics dock")
        print("   - Press 'q' to quit")
        print("="*65 + "\n")
        if args.select_roi:
            time.sleep(0.3)  # Brief pause for background windows to stabilize
            ret_init, init_frame = cap.read()
            if ret_init:
                sel_frame = init_frame.copy()
                h_s, w_s = sel_frame.shape[:2]
                banner = sel_frame.copy()
                cv2.rectangle(banner, (0, 0), (w_s, 60), (20, 20, 25), -1)
                cv2.addWeighted(banner, 0.75, sel_frame, 0.25, 0, sel_frame)
                cv2.putText(sel_frame, "DRAG A BOX OVER YOUR MEETING / YOUTUBE VIDEO, THEN PRESS ENTER OR SPACE",
                            (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
                
                win_roi_title = "Meeting Area Selector - Drag box & press ENTER (or 'c' for Full Screen)"
                cv2.namedWindow(win_roi_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_roi_title, min(1280, w_s), min(720, h_s))
                roi = cv2.selectROI(win_roi_title, sel_frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(win_roi_title)
                if roi and roi[2] > 20 and roi[3] > 20:
                    cap.set_roi(roi[0], roi[1], roi[2], roi[3])
        
    print("Initializing components...")
    detector = FaceDetector(mode=args.detector)
    classifier = DeepfakeClassifier(backend=args.backend, model_path=args.model)
    
    print("\nStarting video stream. Press 'q' to quit.")
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    # Resolve fake detection threshold
    if args.fake_threshold is not None:
        fake_threshold = args.fake_threshold
    elif args.real_threshold is not None:
        fake_threshold = args.real_threshold
    else:
        fake_threshold = 0.50
    print(f"Fake Detection Threshold: {fake_threshold:.2f}  (faces with fake prob >= {fake_threshold:.2f} flagged as FAKE)")
    
    # Temporal smoothing tracker initialization
    active_tracks = {}
    next_track_id = 0
    smoothing_factor = 0.18  # Balanced EMA: steady transitions without abrupt jumps
    hysteresis = 0.06        # Robust anti-flicker deadband margin
    show_diagnostics = True
    is_paused = False
    is_topmost = False
    
    window_name = "Real-Time Deepfake Detection HUD (Screen / Meeting)" if is_screen else "Real-Time Deepfake Detection HUD"
    if is_screen:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Prevent window from capturing itself using Windows 10/11 Display Affinity
        try:
            import ctypes
            user32 = ctypes.windll.user32
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            hwnd = user32.FindWindowW(None, window_name)
            if hwnd:
                user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass
    else:
        # Native 1:1 razor-sharp auto-size for video files and webcam
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    display_frame = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or error reading frame.")
                break
                
            frame_count += 1
            h_orig, w_orig = frame.shape[:2]
            
            # High-resolution screen optimization: downscale frame for detection to ensure 45+ FPS
            detection_start = time.time()
            if w_orig > 1080:
                det_scale = 960.0 / w_orig
                small_frame = cv2.resize(frame, (960, int(h_orig * det_scale)))
                raw_faces = detector.detect(small_frame)
                faces = [(int(fx / det_scale), int(fy / det_scale), int(fw / det_scale), int(fh / det_scale))
                         for (fx, fy, fw, fh) in raw_faces]
            else:
                faces = detector.detect(frame)
            detection_latency = (time.time() - detection_start) * 1000.0
            
            face_crops = []
            valid_boxes = []
            
            for (x, y, w, h) in faces:
                # Add 10% padding to face crop for boundary seam analysis
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_orig, x + w + pad_x)
                y2 = min(h_orig, y + h + pad_y)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    face_crops.append(crop)
                    valid_boxes.append((x1, y1, x2, y2))
                    
            # Classification
            classification_latency = 0.0
            classes_predicted = []
            
            if face_crops and not classifier.is_temporal and not classifier.is_hybrid:
                class_start = time.time()
                classes_predicted = classifier.classify_batch(face_crops)
                classification_latency = (time.time() - class_start) * 1000.0
                
            # Match detections with existing tracks (IoU + Centroid proximity)
            matched_track_ids = {}
            for det_idx, det_box in enumerate(valid_boxes):
                best_score = 0.0
                best_track_id = None
                for track_id, track in active_tracks.items():
                    score = match_boxes(det_box, track['box'])
                    if score > best_score:
                        best_score = score
                        best_track_id = track_id
                if best_score > 0.25:
                    matched_track_ids[det_idx] = best_track_id
                    
            # Update tracks / Create new tracks
            from collections import deque
            for det_idx, det_box in enumerate(valid_boxes):
                crop = face_crops[det_idx]
                
                if det_idx in matched_track_ids:
                    track_id = matched_track_ids[det_idx]
                    track = active_tracks[track_id]
                    old_b = track['box']
                    # Smooth box coordinates (70% current, 30% previous) to eliminate jitter
                    track['box'] = (
                        int(0.70 * det_box[0] + 0.30 * old_b[0]),
                        int(0.70 * det_box[1] + 0.30 * old_b[1]),
                        int(0.70 * det_box[2] + 0.30 * old_b[2]),
                        int(0.70 * det_box[3] + 0.30 * old_b[3])
                    )
                    track['missed_frames'] = 0
                    track['history'].append(crop)
                    
                    if classifier.is_hybrid:
                        class_start = time.time()
                        h_label, h_conf, s_prob, t_prob = classifier.classify_hybrid(crop, list(track['history']))
                        classification_latency = (time.time() - class_start) * 1000.0
                        raw_fake_prob = h_conf if h_label == "fake" else (1.0 - h_conf)
                        track['s_fake'] = s_prob
                        track['t_fake'] = t_prob
                    elif classifier.is_temporal:
                        class_start = time.time()
                        t_label, t_conf = classifier.classify_sequence(list(track['history']))
                        classification_latency = (time.time() - class_start) * 1000.0
                        raw_fake_prob = t_conf if t_label == "fake" else (1.0 - t_conf)
                    else:
                        label, confidence = classes_predicted[det_idx]
                        raw_fake_prob = confidence if label == "fake" else (1.0 - confidence)

                    track['raw_fake_prob'] = raw_fake_prob
                    if 'pred_history' not in track:
                        track['pred_history'] = deque(maxlen=10)
                    track['pred_history'].append(raw_fake_prob)
                    
                    # Median filter across rolling window eliminates random single-frame outliers
                    median_prob = float(np.median(list(track['pred_history'])))
                    prev_smoothed = track.get('smoothed_fake_prob', median_prob)
                    
                    # Symmetric, robust EMA on median-filtered evidence:
                    # Preserves clean real baselines while eliminating neural texture flickers
                    track['smoothed_fake_prob'] = 0.20 * median_prob + 0.80 * prev_smoothed
                else:
                    # Initialize new track
                    track_id = next_track_id
                    next_track_id += 1
                    hist = deque(maxlen=8)
                    hist.append(crop)
                    pred_hist = deque(maxlen=10)
                    
                    s_init, t_init = None, None
                    if classifier.is_hybrid:
                        class_start = time.time()
                        h_label, h_conf, s_prob, t_prob = classifier.classify_hybrid(crop, list(hist))
                        classification_latency = (time.time() - class_start) * 1000.0
                        raw_fake_prob = h_conf if h_label == "fake" else (1.0 - h_conf)
                        s_init, t_init = s_prob, t_prob
                    elif classifier.is_temporal:
                        t_label, t_conf = classifier.classify_sequence(list(hist))
                        raw_fake_prob = t_conf if t_label == "fake" else (1.0 - t_conf)
                    else:
                        label, confidence = classes_predicted[det_idx]
                        raw_fake_prob = confidence if label == "fake" else (1.0 - confidence)

                    pred_hist.append(raw_fake_prob)
                    init_label = 'fake' if raw_fake_prob >= fake_threshold else 'real'
                    new_track = {
                        'box': det_box,
                        'raw_fake_prob': raw_fake_prob,
                        'smoothed_fake_prob': raw_fake_prob,
                        'pred_history': pred_hist,
                        'missed_frames': 0,
                        'label': init_label,
                        'dwell_counter': 0,
                        'history': hist
                    }
                    if s_init is not None:
                        new_track['s_fake'] = s_init
                        new_track['t_fake'] = t_init
                    active_tracks[track_id] = new_track
                    
            # Handle missed tracks
            active_track_ids = list(active_tracks.keys())
            matched_ids_set = set(matched_track_ids.values())
            for track_id in active_track_ids:
                if track_id not in matched_ids_set:
                    active_tracks[track_id]['missed_frames'] += 1
                    # If face is lost for more than 15 frames, remove it
                    if active_tracks[track_id]['missed_frames'] > 15:
                        del active_tracks[track_id]
                        
            # Draw Overlays using temporally stabilized tracks
            for track_id, track in active_tracks.items():
                if track['missed_frames'] > 0:
                    continue
                    
                x1, y1, x2, y2 = track['box']
                smoothed_fake = track['smoothed_fake_prob']
                prev_label = track.get('label', 'real')
                
                # Balanced Hysteresis Deadband (±0.06 around fake_threshold)
                hyst = 0.06
                if prev_label == 'fake':
                    target_label = 'fake' if smoothed_fake >= (fake_threshold - hyst) else 'real'
                else:
                    target_label = 'fake' if smoothed_fake >= (fake_threshold + hyst) else 'real'
                    
                # Dwell time confirmation: requires 4 consecutive agreeing frames to switch state
                if target_label != prev_label:
                    track['dwell_counter'] = track.get('dwell_counter', 0) + 1
                    if track['dwell_counter'] >= 4:
                        track['label'] = target_label
                        track['dwell_counter'] = 0
                else:
                    track['dwell_counter'] = 0
                    
                is_fake = (track['label'] == 'fake')
                smoothed_conf = smoothed_fake if is_fake else (1.0 - smoothed_fake)
                
                # Cyber/Forensic Bounding Box styling
                color = (30, 45, 245) if is_fake else (40, 220, 80) # Vivid Coral Red vs Neon Green
                
                # Dark drop-shadow for high contrast over any background
                cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (12, 12, 18), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Cyber corner accent brackets
                c_len = min(16, max(6, int((x2 - x1) * 0.15)))
                cv2.line(frame, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2)
                cv2.line(frame, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2)
                cv2.line(frame, (x2, y1), (x2 - c_len, y1), (255, 255, 255), 2)
                cv2.line(frame, (x2, y1), (x2, y1 + c_len), (255, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x1 + c_len, y2), (255, 255, 255), 2)
                cv2.line(frame, (x1, y2), (x1, y2 - c_len), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2)
                cv2.line(frame, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2)
                
                # Adaptive bold crisp font scaling based on face bounding box width:
                box_w = max(40, x2 - x1)
                font_scale = max(0.62, min(1.05, box_w / 220.0))
                font_thick = 2
                font = cv2.FONT_HERSHEY_DUPLEX
                tag = "FAKE" if is_fake else "REAL"
                text = f"[{tag}] {smoothed_conf*100:.1f}%"
                text_size = cv2.getTextSize(text, font, font_scale, font_thick)[0]
                banner_h = text_size[1] + 12
                
                if y1 - banner_h - 4 >= 0:
                    b_y1, b_y2, t_y = y1 - banner_h - 4, y1 - 2, y1 - 8
                else:
                    b_y1, b_y2, t_y = y1 + 2, y1 + banner_h + 2, y1 + banner_h - 4
                    
                b_x2 = min(x1 + text_size[0] + 18, frame.shape[1] - 1)
                cv2.rectangle(frame, (x1 - 1, b_y1 - 1), (b_x2 + 1, b_y2 + 1), (15, 15, 20), -1)
                cv2.rectangle(frame, (x1, b_y1), (b_x2, b_y2), color, -1)
                cv2.putText(frame, text, (x1 + 8, t_y), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
                
            # Calculate Frame-rate
            if frame_count % 10 == 0:
                end_time = time.time()
                fps = 10.0 / max(end_time - fps_start_time, 1e-5)
                fps_start_time = end_time
                
            # Display Diagnostics OUTSIDE the video (in a bottom telemetry dock bar)
            if show_diagnostics:
                h_f, w_f = frame.shape[:2]
                bar_height = 42
                display_frame = cv2.copyMakeBorder(
                    frame, 0, bar_height, 0, 0,
                    borderType=cv2.BORDER_CONSTANT,
                    value=(20, 20, 26)
                )
                cv2.line(display_frame, (0, h_f), (w_f, h_f), (55, 55, 68), 1)
                
                # Formatted telemetry metrics
                fps_color = (80, 220, 100) if fps >= 25 else (80, 180, 240)
                if classifier.is_hybrid:
                    engine_str = "Hybrid"
                    last_s, last_t = None, None
                    for t in active_tracks.values():
                        if 's_fake' in t:
                            last_s = t['s_fake']
                            last_t = t['t_fake']
                            break
                    items = [
                        (f"FPS: {fps:.1f}", fps_color),
                        (f"Spatial: {last_s*100:.0f}%" if last_s is not None else "Spatial: --", (220, 220, 225)),
                        (f"Temporal: {last_t*100:.0f}%" if last_t is not None else "Temporal: --", (220, 220, 225)),
                        (f"Latency: {classification_latency:.1f}ms", (220, 220, 225)),
                        (f"Faces: {len(faces)}", (220, 220, 225)),
                        (f"Engine: {engine_str}", (80, 205, 255))
                    ]
                else:
                    engine_str = "ONNX" if args.backend == "onnx" else "PyTorch"
                    items = [
                        (f"FPS: {fps:.1f}", fps_color),
                        (f"Detect: {detection_latency:.1f}ms", (220, 220, 225)),
                        (f"Classifier: {classification_latency:.1f}ms", (220, 220, 225)),
                        (f"Faces: {len(faces)}", (220, 220, 225)),
                        (f"Engine: {engine_str}", (235, 185, 105))
                    ]
                
                pip_str = "PIP: ON ('t')" if is_topmost else "PIP: OFF ('t')"
                items.append((pip_str, (130, 225, 255) if is_topmost else (165, 165, 175)))
                
                if is_screen:
                    items.append(("Screen ('s':Crop 'f':Full)", (255, 185, 55)))
                
                cur_x = 18
                text_y = h_f + 28
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 0.50
                
                for idx, (item_txt, item_col) in enumerate(items):
                    t_w = cv2.getTextSize(item_txt, font, font_scale, 1)[0][0]
                    if cur_x + t_w > w_f - 10:
                        break
                    cv2.putText(display_frame, item_txt, (cur_x, text_y), font, font_scale, item_col, 1, cv2.LINE_AA)
                    cur_x += t_w + 16
                    if idx < len(items) - 1 and cur_x < w_f - 30:
                        cv2.putText(display_frame, "|", (cur_x - 8, text_y), font, font_scale, (70, 70, 80), 1, cv2.LINE_AA)
                        cur_x += 8
            else:
                display_frame = frame
        
        # Render clean view
        render_img = display_frame if display_frame is not None else frame
        if is_paused:
            # Draw Freeze-Frame Banner
            pause_banner = render_img.copy()
            cv2.rectangle(pause_banner, (0, 0), (render_img.shape[1], 32), (0, 0, 180), -1)
            cv2.addWeighted(pause_banner, 0.70, render_img, 0.30, 0, render_img)
            cv2.putText(render_img, "FREEZE-FRAME PAUSED | PRESS SPACE TO RESUME",
                        (16, 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, render_img)
        
        key = cv2.waitKey(1 if not is_paused else 25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_diagnostics = not show_diagnostics
        elif key == 32:  # SPACE bar
            is_paused = not is_paused
            print(f"[HUD] Video {'PAUSED' if is_paused else 'RESUMED'}")
        elif key == ord('t'):  # Toggle Picture-in-Picture Always-On-Top
            is_topmost = not is_topmost
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1 if is_topmost else 0)
            print(f"[HUD] Always-On-Top Picture-in-Picture: {'ON' if is_topmost else 'OFF'}")
        elif is_screen and key == ord('s'):
            # Allow interactive meeting region crop
            cap.reset_roi()
            ret_f, full_f = cap.read()
            if ret_f:
                sel_frame = full_f.copy()
                h_s, w_s = sel_frame.shape[:2]
                banner = sel_frame.copy()
                cv2.rectangle(banner, (0, 0), (w_s, 60), (20, 20, 25), -1)
                cv2.addWeighted(banner, 0.75, sel_frame, 0.25, 0, sel_frame)
                cv2.putText(sel_frame, "DRAG A BOX OVER YOUR MEETING / YOUTUBE VIDEO, THEN PRESS ENTER OR SPACE",
                            (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
                win_roi_title = "Meeting Area Selector - Drag box & press ENTER (or 'c' for Full Screen)"
                cv2.namedWindow(win_roi_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_roi_title, min(1280, w_s), min(720, h_s))
                roi = cv2.selectROI(win_roi_title, sel_frame, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(win_roi_title)
                if roi and roi[2] > 20 and roi[3] > 20:
                    cap.set_roi(roi[0], roi[1], roi[2], roi[3])
        elif is_screen and key == ord('f'):
            cap.reset_roi()
            
    cap.release()
    cv2.destroyAllWindows()
    print("Stream terminated. Windows closed.")

if __name__ == "__main__":
    main()
