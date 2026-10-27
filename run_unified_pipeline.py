import os
import sys
import time
import subprocess
import psutil

PROJECT_DIR = r"D:\college\Sem 3\Macro Project-I\project df"
LOG_FILE = os.path.join(PROJECT_DIR, "unified_pipeline.log")

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def is_extraction_running():
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            cmd = " ".join(p.info.get('cmdline') or []).lower()
            if "build_unified_dataset.py" in cmd:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False

def main():
    os.chdir(PROJECT_DIR)
    log("==================================================================")
    log("  UNIFIED MULTI-DOMAIN DEEPFAKE PIPELINE DRIVER")
    log("==================================================================")

    # 1. Wait for extraction
    log("Checking dataset extraction progress...")
    while is_extraction_running():
        time.sleep(15)

    log("Extraction complete! Starting unified model training...")
    time.sleep(3)

    # 2. Train unified model
    train_cmd = [sys.executable, "train_unified_classifier.py", "--epochs", "5", "--batch_size", "16", "--lr", "3e-5"]
    log(f"Running command: {' '.join(train_cmd)}")
    t0 = time.time()
    res = subprocess.run(train_cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if res.returncode == 0:
        log(f"Training SUCCESS in {elapsed:.1f}s!")
        log("Output preview:\n" + res.stdout[-1000:])
    else:
        log(f"Training FAILED (code {res.returncode}):\n" + (res.stderr or res.stdout)[-1000:])
        return

    # 3. Re-run benchmark evaluation
    log("Running multi-dataset benchmark test...")
    eval_cmd = [sys.executable, "test_multi_dataset_eval.py"]
    res_eval = subprocess.run(eval_cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    log("Evaluation Results:\n" + res_eval.stdout[-1500:])

    log("==================================================================")
    log("  PIPELINE COMPLETE - READY FOR REAL-TIME INFERENCE")
    log("==================================================================")

if __name__ == "__main__":
    main()
