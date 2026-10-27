# Real-Time Deepfake Detection System: Comprehensive Evaluator & Viva Defense Guide

---

## 1. Executive Summary & Project Identity

- **Project Title:** Real-Time Multi-Modal Deepfake Detection and Forensic Analysis System
- **Core Problem:** Standard deepfake detectors are either slow offline batch scripts or expensive cloud APIs that overfit to one specific dataset (e.g. catching full face swaps but completely failing on localized lip/expression reenactments like NeuralTextures, or falsely flagging compressed real YouTube interviews as fake).
- **Our Innovation:** A hybrid on-device system combining **2D Fourier Transform (FFT) Frequency Analysis (MobileNetV3 4-Channel)** with **Temporal Motion Continuity (Bi-LSTM on 8-frame sequences)**, running locally in real-time at **40+ FPS (~22ms latency)** on consumer hardware without any cloud dependency.

---

## 2. Key Architecture & Engineering Highlights

```
                          ┌─────────────────────────────────────┐
                          │     Input Video / Live Webcam       │
                          └──────────────────┬──────────────────┘
                                             │
                                ┌────────────┴────────────┐
                                │ MediaPipe BlazeFace ROI │
                                └────────────┬────────────┘
                                             │ (Cropped Faces + 10% Padding)
                    ┌────────────────────────┴────────────────────────┐
                    ▼                                                 ▼
       ┌───────────────────────────┐                    ┌───────────────────────────┐
       │   Stream 1: Spatial-FFT   │                    │ Stream 2: Temporal Motion │
       │     (ONNX Runtime)        │                    │    (PyTorch GPU / CUDA)   │
       ├───────────────────────────┤                    ├───────────────────────────┤
       │ • 4 Channels (RGB + FFT)  │                    │ • 8-Frame Rolling Window  │
       │ • MobileNetV3-Large Dual  │                    │ • CNN Feature Extractor   │
       │ • Latency: ~4ms (CPU/GPU) │                    │ • Bidirectional LSTM      │
       │ • Catches boundary seams  │                    │ • Latency: ~18ms (CUDA)   │
       │   & high-freq noise grids │                    │ • Catches motion warping  │
       └─────────────┬─────────────┘                    └─────────────┬─────────────┘
                     │ Spatial Fake Prob (S)                          │ Temporal Fake Prob (T)
                     └───────────────────────┬────────────────────────┘
                                             ▼
                             ┌───────────────────────────────┐
                             │    Adaptive Fusion Engine     │
                             │ (EMA Tracking + Anti-Flicker) │
                             └───────────────┬───────────────┘
                                             ▼
                       ┌───────────────────────────────────────────┐
                       │   Real-Time HUD (30–50 FPS, Bounding Box) │
                       │    + One-Click Forensic Audit Report      │
                       └───────────────────────────────────────────┘
```

### Stream 1: Dual-Branch Spatial-Frequency Classifier (4 Channels)
- **Input:** $224 \times 224$ pixels, 4 channels:
  - Channels 1–3: Normalized RGB face image.
  - Channel 4: **2D Fast Fourier Transform (FFT) Log-Magnitude Spectrum**, shifted to center zero-frequency:
    $$\text{FFT}_{\text{map}} = \log(1 + |\text{fftshift}(\mathcal{F}\{\text{Grayscale}(I)\})|)$$
- **Why Frequency Analysis?** Convolutional neural networks and GAN upsampling layers (e.g. transposed convolutions) inevitably leave subtle high-frequency grid periodicities and spectral decay mismatches that are invisible to the naked human eye in the spatial domain, but glow brightly in the 2D FFT spectrum.
- **Backbone:** Modified MobileNetV3-Large with a parallel frequency branch fusing into a 1088-dimensional unified embedding before the classification head.
- **Runtime:** Exported to optimized **ONNX Runtime**, executing in **~4 milliseconds**.

### Stream 2: Temporal Sequence Stream (Bi-LSTM on Consecutive Frames)
- **Input:** 8 consecutive cropped faces sampled over a rolling sliding window.
- **Architecture:** MobileNet feature encoder feeding into a **Bidirectional Long Short-Term Memory (Bi-LSTM)** network.
- **Purpose:** Analyzes human temporal consistency (blinking dynamics, micro-expressions, heartbeat pulse color variations) vs. inter-frame synthesis jitter.

### Stream 3: Real-Time Tracker & Anti-Flicker Stabilization
- **Matching:** Intersection over Union (IoU $\ge 0.3$) face tracker maintaining persistent identity tracks across missed frames.
- **Smoothing:** Exponential Moving Average (EMA) with $\alpha = 0.25$:
  $$P_{\text{smooth}}^{(t)} = \alpha P_{\text{raw}}^{(t)} + (1 - \alpha) P_{\text{smooth}}^{(t-1)}$$
- **Hysteresis:** A $\pm 0.05$ decision margin preventing flickering when a score hovers near the threshold.

---

## 3. The 7 Critical Viva Questions & Perfect Answers

### Q1: "Why did you use 4 channels instead of standard RGB 3-channel input?"
> **Answer:** Standard RGB images hide manipulation artifacts because GANs and neural textures blend pixel colors smoothly. However, generative networks use upsampling operators that introduce unnatural periodic grid patterns in the frequency domain. By appending the **normalized 2D Discrete Fourier Transform (FFT) log-magnitude spectrum as the 4th channel**, the convolutional filters can inspect both spatial blending boundaries and high-frequency spectral anomalies simultaneously.

---

### Q2: "Why MobileNetV3-Large instead of ResNet-50, EfficientNet, or a Vision Transformer?"
> **Answer:** The primary objective was **real-time on-device operation**. While ResNet-50 requires ~4.1 GFLOPs and takes 25–40ms per frame, MobileNetV3-Large utilizes depthwise separable convolutions, hard-swish activation, and lightweight Squeeze-and-Excitation (SE) blocks requiring only **~0.22 GFLOPs** (~18x lighter). Exported via ONNX Runtime, it evaluates in **4ms**, allowing a full pipeline (detection, FFT computation, inference, telemetry) to run smoothly at **40+ FPS** on consumer laptops.

---

### Q3: "What is the difference between Deepfakes and NeuralTextures, and why did standard models fail on NeuralTextures?"
> **Answer:**
> - In **Deepfakes / FaceSwap**, the entire face identity is swapped, leaving boundary seams and whole-face frequency mismatches.
> - In **NeuralTextures (and Face2Face)**, only facial reenactment occurs — 90% to 95% of the head (eyes, forehead, skin, hair, and background) is **100% genuine real video**, with only a small patch inside the mouth/lips modified.
> - Standard models evaluate the entire face crop, so 95% genuine pixels dilute the 5% manipulated pixels, causing the model to output "Real". We solved this by extracting localized high-frequency mouth anomalies during multi-domain joint training, raising NeuralTextures detection sensitivity from 2% to over 66%.

---

### Q4: "What is Catastrophic Forgetting, and how did you prevent it when fine-tuning?"
> **Answer:** When a pre-trained neural network is fine-tuned *only* on new data (e.g. FaceForensics++), its gradient updates overwrite previously learned feature weights, causing it to "forget" older domains (which previously caused real YouTube interviews like Linda Raynier to be misclassified as 94% fake). We prevented this using:
> 1. **Balanced Replay Buffer:** Training simultaneously on a 50/50 balanced dataset (17,481 faces) containing all 5 fake types alongside real YouTube interviews, Celeb-Real, and FF++ original videos.
> 2. **Stem Layer Freezing:** Locking the early convolutional stem (`spatial_features[:3]`) for the first two epochs to safeguard low-level edge/texture filters.
> 3. **Cosine Annealing Learning Rate:** Using a low learning rate ($\eta = 5 \times 10^{-5}$) to prevent drastic weight displacement.

---

### Q5: "What datasets were used to train and validate this system?"
> **Answer:** The system was trained and evaluated on two primary international benchmarks:
> 1. **FaceForensics++ (C23 lightly compressed):** Containing 1,000 videos each of *Deepfakes*, *Face2Face*, *FaceSwap*, *FaceShifter*, *NeuralTextures*, and *Original Real*.
> 2. **Celeb-DF v2:** Containing 5,639 high-quality synthesis deepfakes and hundreds of real celebrity and YouTube interview videos.
> 3. **Self-Blended Images (SBI):** Synthetic self-blending masks generating realistic boundary artifacts on genuine faces during data augmentation.

---

### Q6: "What are your benchmark performance metrics?"
> **Answer:** Evaluated across a standardized 42-video multi-source benchmark at threshold `0.50`:
> - **Overall Accuracy:** **90.48% (38/42 videos)**
> - **Fake Video Sensitivity (Catch Rate):** **100.0% (20/20 fakes caught)**
>   - *FF++ Deepfakes:* 100.0% (6/6)
>   - *FF++ Face2Face:* 100.0% (3/3)
>   - *FF++ FaceSwap:* 100.0% (3/3)
>   - *Celeb-DF Synthesis:* 100.0% (8/8)
> - **Real Video Specificity:** **81.82% (18/22 real videos preserved)**
>   - *YouTube Real Interviews (e.g. Linda Raynier):* **87.5% preserved** (Fake prob down to 6.6%)
>   - *FF++ Original Real:* **87.5% preserved**

---

### Q7: "How is this practically useful in the real world compared to existing tools?"
> **Answer:**
> 1. **Live Remote Call Protection (Zoom, Meet, Teams):** Existing academic tools are offline batch scripts taking minutes per video. Our system runs locally at 40+ FPS, allowing real-time protection against live job interview impersonation and executive impersonation scams.
> 2. **Zero Cloud Costs & Absolute Privacy:** Cloud APIs (like Sensity or Reality Defender) charge per minute and require uploading private video calls to external servers. Our system runs 100% on the user's local GPU/CPU.
> 3. **Explainable AI (One-Click Forensic Audit Report):** Rather than giving a blind score, clicking "Generate Forensic Video Report" creates an interactive HTML audit trail with timestamped keyframes and millisecond markers suitable for HR fraud investigations.

---

## 4. Hardware & Software Specifications

- **Operating System:** Windows 11 64-bit
- **Acceleration:** NVIDIA GeForce RTX 2050 (4GB VRAM) running CUDA 12 + TensorRT / ONNX Runtime
- **Core Frameworks:** PyTorch 2.6, ONNX Runtime 1.20, OpenCV 4.10, MediaPipe, CustomTkinter (GUI)
- **Active Weights on Disk:**
  - Spatial Stream: [`mobilenetv3_deepfake.onnx`](file:///D:/college/Sem%203/Macro%20Project-I/project%20df/mobilenetv3_deepfake.onnx) (13.4 MB)
  - Temporal Stream: [`best_temporal_model.pth`](file:///D:/college/Sem%203/Macro%20Project-I/project%20df/best_temporal_model.pth) (22.4 MB)
  - PyTorch Master Checkpoint: [`best_unified_model.pth`](file:///D:/college/Sem%203/Macro%20Project-I/project%20df/best_unified_model.pth) (13.6 MB)
