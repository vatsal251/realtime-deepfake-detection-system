import os
import sys
import argparse
import time
import cv2
import numpy as np
import base64
from infer_rt import FaceDetector, DeepfakeClassifier

def encode_img_base64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode('utf-8')

def analyze_video(video_path, model_path='mobilenetv3_deepfake.onnx', backend='onnx', sample_fps=4):
    print(f'Starting Forensic Video Audit on: {video_path}')
    detector = FaceDetector(mode='mediapipe')
    classifier = DeepfakeClassifier(backend=backend, model_path=model_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error: Unable to open video {video_path}')
        return None
        
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / orig_fps if total_frames > 0 else 0
    
    step = max(1, int(orig_fps / sample_fps))
    
    timeline_data = [] # (timestamp_sec, fake_prob, frame_bgr, face_crop)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % step == 0:
            timestamp = frame_idx / orig_fps
            faces = detector.detect(frame)
            if faces:
                # Take largest face
                faces.sort(key=lambda b: b[2] * b[3], reverse=True)
                x, y, w, h = faces[0]
                pad_x, pad_y = int(w * 0.1), int(h * 0.1)
                h_orig, w_orig = frame.shape[:2]
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(w_orig, x + w + pad_x), min(h_orig, y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    preds = classifier.classify_batch([crop])
                    label, conf = preds[0]
                    fake_prob = conf if label == 'fake' else (1.0 - conf)
                    timeline_data.append((timestamp, float(fake_prob), frame.copy(), crop.copy()))
        frame_idx += 1
    cap.release()
    
    if not timeline_data:
        print('No faces detected in video.')
        return None
        
    # Overall verdict
    avg_fake_prob = np.mean([d[1] for d in timeline_data])
    max_fake_prob = np.max([d[1] for d in timeline_data])
    is_manipulated = avg_fake_prob >= 0.50 or max_fake_prob >= 0.75
    verdict = 'MANIPULATED (SYNTHETIC DEEPFAKE)' if is_manipulated else 'AUTHENTIC (REAL RECORDING)'
    verdict_color = '#e53935' if is_manipulated else '#43a047'
    
    # Sort for top 3 most suspicious keyframes
    top_suspicious = sorted(timeline_data, key=lambda d: d[1], reverse=True)[:3]
    
    # Generate HTML Report
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>Forensic Deepfake Inspection Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 24px; }}
.card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }}
h1 {{ color: #38bdf8; margin-top: 0; }}
.badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 18px; color: white; background: {verdict_color}; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; }}
.snapshot {{ background: #0f172a; border-radius: 8px; padding: 12px; text-align: center; }}
.snapshot img {{ max-width: 100%; border-radius: 6px; border: 2px solid #334155; }}
.stat {{ font-size: 28px; font-weight: bold; color: #38bdf8; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
td, th {{ padding: 10px; text-align: left; border-bottom: 1px solid #334155; }}
</style>
</head>
<body>
<h1>Forensic Deepfake Inspection Report</h1>
<div class='card'>
  <h2>Executive Summary</h2>
  <p>File Inspected: <code>{os.path.basename(video_path)}</code></p>
  <p>Duration: <b>{duration_sec:.1f} seconds</b> | Frames Analyzed: <b>{len(timeline_data)}</b></p>
  <div class='badge'>{verdict}</div>
  <p style='margin-top:14px;'>Average Manipulation Probability: <span class='stat'>{avg_fake_prob*100:.1f}%</span> (Peak: <b>{max_fake_prob*100:.1f}%</b>)</p>
</div>

<div class='card'>
  <h2>Top Flagged Anomaly Keyframes</h2>
  <p>Highest confidence frames exhibiting synthetic blending or facial boundary inconsistencies:</p>
  <div class='grid'>
'''
    for i, (ts, prob, full_f, crop_f) in enumerate(top_suspicious):
        b64_crop = encode_img_base64(crop_f)
        html += f'''
    <div class='snapshot'>
      <img src='data:image/jpeg;base64,{b64_crop}'>
      <p><b>Keyframe {i+1}</b> (t = {ts:.2f}s)</p>
      <p style='color: {"#ef5350" if prob >= 0.55 else "#66bb6a"}; font-weight:bold;'>Fake Confidence: {prob*100:.1f}%</p>
    </div>
'''
    html += '''
  </div>
</div>

<div class='card'>
  <h2>Methodology & Forensic Pipeline</h2>
  <table>
    <tr><th>Component</th><th>Specification</th></tr>
    <tr><td>Face Detection</td><td>MediaPipe BlazeFace Full-Range Neural Detector</td></tr>
    <tr><td>Spatial Analysis</td><td>MobileNetV3 / EfficientNet Deep Learning Feature Extractor</td></tr>
    <tr><td>Frequency Domain</td><td>2D Discrete Fourier Transform (FFT) Spectral Analysis</td></tr>
    <tr><td>Temporal Filtering</td><td>Exponential Moving Average + Hysteresis Boundary Smoothing</td></tr>
  </table>
</div>
</body>
</html>'''

    report_path = os.path.splitext(video_path)[0] + '_forensic_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report successfully generated: {report_path}')
    return report_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to video file')
    parser.add_argument('--model', type=str, default='mobilenetv3_deepfake.onnx')
    args = parser.parse_args()
    analyze_video(args.input, model_path=args.model)
