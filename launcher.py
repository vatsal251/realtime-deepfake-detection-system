import customtkinter as ctk
import subprocess
import sys
import os
import json
from tkinter import filedialog

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

RECENT_FILE = os.path.join(os.path.dirname(__file__), ".recent_videos.json")
MAX_RECENT  = 8

def load_recent():
    try:
        with open(RECENT_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def save_recent(paths):
    with open(RECENT_FILE, "w") as f:
        json.dump(paths[:MAX_RECENT], f)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Detector")
        self.geometry("560x680")
        self.resizable(False, False)
        self.process = None
        self.recent = load_recent()
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 20, "pady": 6}

        header = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=12)
        header.pack(fill="x", padx=16, pady=(16, 4))
        ctk.CTkLabel(header, text="Deepfake Detection System",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").pack(pady=14)
        ctk.CTkLabel(header, text="MobileNetV3-Large  |  ONNX Backend  |  Real-Time",
                     font=ctk.CTkFont(size=11), text_color="#78909c").pack(pady=(0, 14))

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", **pad)
        self.webcam_btn = ctk.CTkButton(
            btn_frame, text="Open Webcam", height=52,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#1565c0", hover_color="#1976d2",
            command=self.launch_webcam)
        self.webcam_btn.pack(fill="x", pady=(0, 8))
        self.video_btn = ctk.CTkButton(
            btn_frame, text="Browse Video File", height=52,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#2e7d32", hover_color="#388e3c",
            command=self.browse_video)
        self.video_btn.pack(fill="x", pady=(0, 8))

        self.report_btn = ctk.CTkButton(
            btn_frame, text="📑  Generate Forensic Video Report", height=46,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#6a1b9a", hover_color="#8e24aa",
            command=self.generate_report)
        self.report_btn.pack(fill="x")


        settings = ctk.CTkFrame(self, corner_radius=10)
        settings.pack(fill="x", **pad)
        ctk.CTkLabel(settings, text="Settings",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))

        thresh_row = ctk.CTkFrame(settings, fg_color="transparent")
        thresh_row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(thresh_row, text="Fake Sensitivity Threshold:", width=175,
                     font=ctk.CTkFont(size=12)).pack(side="left")
        self.thresh_val = ctk.CTkLabel(thresh_row, text="0.50",
                                       font=ctk.CTkFont(size=12, weight="bold"),
                                       text_color="#4fc3f7", width=40)
        self.thresh_val.pack(side="right")
        self.thresh_slider = ctk.CTkSlider(thresh_row, from_=0.40, to=0.85,
                                           number_of_steps=45,
                                           command=self._on_thresh)
        self.thresh_slider.set(0.50)
        self.thresh_slider.pack(side="left", fill="x", expand=True, padx=8)
        ctk.CTkLabel(settings, text="  0.50 = Standard balanced threshold (optimal for Hybrid Fusion across all datasets)",
                     font=ctk.CTkFont(size=10), text_color="#78909c").pack(anchor="w", padx=14)

        back_row = ctk.CTkFrame(settings, fg_color="transparent")
        back_row.pack(fill="x", padx=14, pady=(8, 4))
        ctk.CTkLabel(back_row, text="Model Architecture:", width=140,
                     font=ctk.CTkFont(size=12)).pack(side="left")
        self.backend_var = ctk.StringVar(value="hybrid")
        ctk.CTkRadioButton(back_row, text="Hybrid Fusion (Auto)",
                           variable=self.backend_var, value="hybrid").pack(side="left", padx=4)
        ctk.CTkRadioButton(back_row, text="ONNX Fast (~4ms)",
                           variable=self.backend_var, value="onnx").pack(side="left", padx=4)
        ctk.CTkRadioButton(back_row, text="Temporal Bi-LSTM (~25ms)",
                           variable=self.backend_var, value="pytorch_temporal").pack(side="left", padx=4)
        ctk.CTkRadioButton(back_row, text="PyTorch Spatial",
                           variable=self.backend_var, value="pytorch_spatial").pack(side="left", padx=4)
        ctk.CTkLabel(settings, text="  Tip: 'Hybrid Fusion' automatically combines Spatial FFT (FaceForensics++) and Temporal Motion (Celeb-DF & YouTube) for dataset-independent accuracy.",
                     font=ctk.CTkFont(size=10), text_color="#81c784").pack(anchor="w", padx=14, pady=(0, 6))

        status_frame = ctk.CTkFrame(self, corner_radius=10)
        status_frame.pack(fill="x", **pad)
        status_top = ctk.CTkFrame(status_frame, fg_color="transparent")
        status_top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(status_top, text="Status:",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        self.stop_btn = ctk.CTkButton(
            status_top, text="Stop", width=80, height=28,
            fg_color="#c62828", hover_color="#d32f2f",
            font=ctk.CTkFont(size=12),
            state="disabled", command=self.stop_detection)
        self.stop_btn.pack(side="right")
        self.status_lbl = ctk.CTkLabel(
            status_frame, text="Idle - ready to detect",
            font=ctk.CTkFont(size=12), text_color="#78909c",
            wraplength=480, justify="left")
        self.status_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        recent_frame = ctk.CTkFrame(self, corner_radius=10)
        recent_frame.pack(fill="both", expand=True, **pad)
        recent_top = ctk.CTkFrame(recent_frame, fg_color="transparent")
        recent_top.pack(fill="x", padx=14, pady=(10, 4))
        ctk.CTkLabel(recent_top, text="Recent Videos",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(side="left")
        ctk.CTkButton(recent_top, text="Clear", width=54, height=24,
                      fg_color="#424242", hover_color="#616161",
                      font=ctk.CTkFont(size=11),
                      command=self._clear_recent).pack(side="right")
        self.recent_list = ctk.CTkScrollableFrame(recent_frame, height=160)
        self.recent_list.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._refresh_recent()

        ctk.CTkLabel(self, text="Press Q inside the detection window to stop",
                     font=ctk.CTkFont(size=11), text_color="#546e7a").pack(pady=(0, 10))

    def _on_thresh(self, val):
        self.thresh_val.configure(text=f"{float(val):.2f}")

    def _set_status(self, msg, color="#4fc3f7"):
        self.status_lbl.configure(text=msg, text_color=color)

    def _refresh_recent(self):
        for w in self.recent_list.winfo_children():
            w.destroy()
        if not self.recent:
            ctk.CTkLabel(self.recent_list, text="No recent videos yet.",
                         text_color="#546e7a").pack(anchor="w", padx=6)
            return
        for path in self.recent:
            name = os.path.basename(path)
            row = ctk.CTkFrame(self.recent_list, fg_color="#2b2b3b", corner_radius=6)
            row.pack(fill="x", pady=2, padx=2)
            ctk.CTkLabel(row, text=name, font=ctk.CTkFont(size=12),
                         anchor="w", wraplength=380).pack(side="left", padx=8, pady=6)
            ctk.CTkButton(row, text="Play", width=60, height=26,
                          fg_color="#1565c0", hover_color="#1976d2",
                          font=ctk.CTkFont(size=11),
                          command=lambda p=path: self.launch_video(p)).pack(side="right", padx=6, pady=4)

    def _add_recent(self, path):
        if path in self.recent:
            self.recent.remove(path)
        self.recent.insert(0, path)
        self.recent = self.recent[:MAX_RECENT]
        save_recent(self.recent)
        self._refresh_recent()

    def _clear_recent(self):
        self.recent = []
        save_recent(self.recent)
        self._refresh_recent()

    def _build_cmd(self, video_path=None):
        threshold = f"{self.thresh_slider.get():.2f}"
        backend   = self.backend_var.get()
        if backend == "hybrid":
            model = "mobilenetv3_deepfake.onnx"
            backend_arg = "hybrid"
        elif backend == "onnx":
            model = "mobilenetv3_deepfake.onnx"
            backend_arg = "onnx"
        elif backend == "pytorch_temporal":
            model = "best_temporal_model.pth"
            backend_arg = "pytorch"
        else:
            model = "best_modern_model.pth"
            backend_arg = "pytorch"

        cmd = [sys.executable, "infer_rt.py",
               "--backend", backend_arg,
               "--model",   model,
               "--fake_threshold", threshold]
        if video_path:
            cmd += ["--input", video_path]
        return cmd

    def _launch(self, cmd, label):
        if self.process and self.process.poll() is None:
            self._set_status("Already running - click Stop first.", "#ef9a9a")
            return
        self._set_status(f"Running: {label}", "#a5d6a7")
        self.stop_btn.configure(state="normal")
        self.process = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        import threading
        def _watch():
            self.process.wait()
            self.after(0, self._on_stopped)
        threading.Thread(target=_watch, daemon=True).start()

    def _on_stopped(self):
        self._set_status("Idle - ready to detect", "#78909c")
        self.stop_btn.configure(state="disabled")

    def launch_webcam(self):
        self._launch(self._build_cmd(), "Webcam (live)")

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if path:
            self.launch_video(path)

    def launch_video(self, path):
        self._add_recent(path)
        self._launch(self._build_cmd(path), os.path.basename(path))

    def stop_detection(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._set_status("Stopped.", "#ef9a9a")
            self.stop_btn.configure(state="disabled")

    def generate_report(self):
        path = filedialog.askopenfilename(
            title="Select a video file to audit",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if path:
            self._set_status(f"Generating Forensic Report for {os.path.basename(path)}...", "#ba68c8")
            import threading
            def _run_audit():
                try:
                    from generate_report import analyze_video
                    report_path = analyze_video(path)
                    if report_path and os.path.exists(report_path):
                        import webbrowser
                        webbrowser.open(report_path)
                        self.after(0, lambda: self._set_status(f"Report ready! Opened in browser.", "#a5d6a7"))
                    else:
                        self.after(0, lambda: self._set_status("No faces found in video.", "#ef9a9a"))
                except Exception as e:
                    self.after(0, lambda: self._set_status(f"Report error: {e}", "#ef9a9a"))
            threading.Thread(target=_run_audit, daemon=True).start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
