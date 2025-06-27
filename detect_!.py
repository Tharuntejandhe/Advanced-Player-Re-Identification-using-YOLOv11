# Full updated code with advanced ResNet50-based feature extraction
# and improved Re-ID matching system

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import threading
import os
import torchvision.transforms as transforms
from torchvision.models import resnet50
import torch.nn as nn

class PlayerReIDSystem:
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0

        self.next_id = 1
        self.active_tracks = {}
        self.disappeared_tracks = {}
        self.max_disappeared = 30
        self.similarity_threshold = 0.6
        self.colors = self.generate_colors(100)

        # Load YOLO model
        if self.model_path:
            self.load_model_from_path()

        # Load ResNet50 for feature extraction
        self.feature_extractor = resnet50(pretrained=True)
        self.feature_extractor.fc = nn.Identity()
        self.feature_extractor.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.setup_gui()

    def generate_colors(self, num):
        np.random.seed(42)
        return [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(num)]

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Advanced Player Re-ID System")
        self.root.geometry("800x600")

        model_frame = ttk.LabelFrame(self.root, text="Model Setup", padding=10)
        model_frame.pack(fill="x", padx=10, pady=5)

        if self.model_path:
            model_info = f"Model: {os.path.basename(self.model_path)}"
            model_info += " ✓ Loaded" if self.model else " ✗ Failed to load"
            self.model_status = ttk.Label(model_frame, text=model_info)
            self.model_status.pack(side="left", padx=10)
            ttk.Button(model_frame, text="Change Model", command=self.load_model).pack(side="left", padx=5)
        else:
            ttk.Button(model_frame, text="Load YOLOv11 Model", command=self.load_model).pack(side="left", padx=5)
            self.model_status = ttk.Label(model_frame, text="No model loaded")
            self.model_status.pack(side="left", padx=10)

        video_frame = ttk.LabelFrame(self.root, text="Video Input", padding=10)
        video_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(video_frame, text="Select Video", command=self.select_video).pack(side="left", padx=5)
        self.video_status = ttk.Label(video_frame, text="No video selected")
        self.video_status.pack(side="left", padx=10)

        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x", padx=10, pady=10)

        self.start_btn = ttk.Button(control_frame, text="Start Processing", command=self.start_processing, state="disabled")
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        self.stats_text = tk.Text(self.root, height=10)
        self.stats_text.pack(fill="both", expand=True, padx=10, pady=5)

        self.output_label = ttk.Label(self.root, text="Output saved as 'output_with_reid.mp4'")
        self.output_label.pack()

    def load_model_from_path(self):
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Model load failed: {e}")

    def load_model(self):
        path = filedialog.askopenfilename(title="Select YOLOv11 Model", filetypes=[("PT files", "*.pt")])
        if path:
            self.model_path = path
            self.load_model_from_path()
            self.model_status.config(text=f"Model loaded: {os.path.basename(path)}")
            self.update_start_button_state()

    def select_video(self):
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi")])
        if path:
            self.video_path = path
            self.video_status.config(text=f"Video: {os.path.basename(path)}")
            cap = cv2.VideoCapture(path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.update_start_button_state()

    def update_start_button_state(self):
        if self.model and self.video_path:
            self.start_btn.config(state="normal")

    def extract_features(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(2048)
        try:
            tensor = self.transform(crop).unsqueeze(0)
            with torch.no_grad():
                feature = self.feature_extractor(tensor).squeeze().numpy()
            return feature / (np.linalg.norm(feature) + 1e-6)
        except:
            return np.zeros(2048)

    def calculate_similarity(self, feat1, feat2):
        return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-6)

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def start_processing(self):
        self.is_playing = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.process_video).start()

    def stop_processing(self):
        self.is_playing = False
        self.stop_btn.config(state="disabled")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter("output_with_reid.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_idx = 0
        while cap.isOpened() and self.is_playing:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        if conf > 0.5:
                            feat = self.extract_features(frame, box)
                            detections.append((box, conf, feat))

            matched_ids = self.match_detections(detections, frame_idx)

            for i, (box, _, _) in enumerate(detections):
                if i < len(matched_ids):
                    tid = matched_ids[i]
                    color = self.active_tracks[tid]['color']
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            out.write(frame)
            frame_idx += 1
            self.progress['value'] = (frame_idx / self.total_frames) * 100
            self.root.update()

        cap.release()
        out.release()
        self.stop_processing()
        messagebox.showinfo("Done", "Video processed and saved as output_with_reid.mp4")

    def match_detections(self, detections, frame_idx):
        matched_ids = []
        unmatched = []

        for box, conf, feat in detections:
            best_id = None
            best_sim = 0

            for tid, info in self.active_tracks.items():
                sim = self.calculate_similarity(feat, info['features'])
                iou = self.calculate_iou(box, info['bbox'])
                score = 0.7 * sim + 0.3 * iou
                if score > best_sim and score > self.similarity_threshold:
                    best_sim = score
                    best_id = tid

            if best_id is not None:
                self.active_tracks[best_id]['bbox'] = box
                self.active_tracks[best_id]['features'] = 0.8 * self.active_tracks[best_id]['features'] + 0.2 * feat
                matched_ids.append(best_id)
            else:
                unmatched.append((box, feat))

        for box, feat in unmatched:
            tid = self.next_id
            self.next_id += 1
            self.active_tracks[tid] = {'bbox': box, 'features': feat, 'color': self.colors[tid % len(self.colors)]}
            matched_ids.append(tid)

        return matched_ids

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    MODEL_PATH = "C:/object_detection_soccer/best_1.pt"
    app = PlayerReIDSystem(model_path=MODEL_PATH)
    app.run()
