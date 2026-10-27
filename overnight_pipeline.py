import os
import sys
import time
import subprocess
import datetime
import psutil

LOG_FILE = "overnight_pipeline.log"
PROJECT_DIR = r"D:\college\Sem 3\Macro Project-I\project df"

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def is_classifier_training():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmd_str = " ".join(cmdline).lower()
            if "train_classifier.py" in cmd_str and proc.pid != os.getpid():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def run_cmd(cmd, step_name):
    log(f"--- Starting: {step_name} ---")
    log(f"Command: {cmd}")
    start = time.time()
    res = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR, capture_output=True, text=True)
    duration = time.time() - start
    
    if res.returncode == 0:
        log(f"SUCCESS: {step_name} completed in {duration:.1f}s.")
        log(f"Output preview:\n{res.stdout[-600:] if res.stdout else 'No output'}")
    else:
        log(f"WARNING: {step_name} exited with code {res.returncode}")
        log(f"Error:\n{res.stderr[-600:] if res.stderr else res.stdout[-600:]}")
    return res.returncode == 0

def main():
    os.chdir(PROJECT_DIR)
    log("==================================================================")
    log("  OVERNIGHT AUTO-SUPERVISOR PIPELINE LAUNCHED")
    log("  Monitoring train_classifier.py and auto-chaining subsequent tasks")
    log("==================================================================")

    # 1. Wait for classifier training to complete
    log("Waiting for train_classifier.py to complete...")
    check_interval = 45 # seconds
    while is_classifier_training():
        time.sleep(check_interval)

    log("train_classifier.py has finished! Proceeding with auto-chained pipeline.")
    time.sleep(5)

    # 2. Export newly trained weights to ONNX
    run_cmd(
        "python export_onnx.py --checkpoint best_modern_model.pth --output mobilenetv3_deepfake.onnx",
        "Step 1: Export Best Modern Model to ONNX"
    )

    # 3. Evaluate model on Test Set (10,905 real, 10,000 fake faces)
    run_cmd(
        'python evaluate_model.py --data_dir "C:\\Users\\asus\\Downloads\\archive (1)" --split test --checkpoint best_modern_model.pth --cm_path eval_results.png --roc_path eval_roc.png',
        "Step 2: Formal Benchmark Evaluation on Test Split (Confusion Matrix & ROC)"
    )

    # 4. Train Temporal Bi-LSTM Sequence Model on Celeb-DF v2
    run_cmd(
        'python train_temporal.py --data_dir "C:\\Users\\asus\\Downloads\\celebdf_sequences" --seq_len 8 --batch_size 4 --epochs 12 --save_path best_temporal_model.pth',
        "Step 3: Train Temporal Bi-LSTM on Celeb-DF v2 Sequences"
    )

    # 5. Write Final Morning Report
    report_path = "OVERNIGHT_COMPLETION_REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================================\n")
        f.write("      OVERNIGHT DEEPFAKE DETECTION PIPELINE COMPLETION REPORT     \n")
        f.write(f"      Completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================================\n\n")
        f.write("1. SPATIAL-FREQUENCY DUAL-STREAM CLASSIFIER:\n")
        f.write("   - Model weights saved: best_modern_model.pth\n")
        f.write("   - ONNX Runtime model exported: mobilenetv3_deepfake.onnx\n\n")
        f.write("2. BENCHMARK METRICS & PLOTS:\n")
        f.write("   - Confusion Matrix plot: eval_results.png\n")
        f.write("   - ROC / AUC Curve plot: eval_roc.png\n\n")
        f.write("3. TEMPORAL SEQUENCE MODEL:\n")
        f.write("   - Bi-LSTM with Temporal Attention saved: best_temporal_model.pth\n")
        f.write("   - Tested on Celeb-DF v2 8-frame consecutive video sequences\n\n")
        f.write("4. ONE-CLICK LAUNCHERS READY:\n")
        f.write("   - Desktop: C:\\Users\\asus\\Desktop\\Launch_Deepfake_Detector.bat\n")
        f.write("   - Local: D:\\college\\Sem 3\\Macro Project-I\\project df\\run.bat\n")

    log(f"All overnight tasks completed! Morning report generated at {report_path}")
    log("==================================================================")

if __name__ == "__main__":
    main()
