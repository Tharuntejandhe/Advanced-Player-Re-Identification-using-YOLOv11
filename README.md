# Advanced Player Re-Identification System using YOLLOv11 🎯

Welcome to the official repository of the **Player Re-Identification System**, a computer vision-based project that combines real-time object detection with deep appearance-based tracking to uniquely identify and persistently track each player in a football video.
Here is the [link for the model](https://drive.google.com/file/d/11wp2Y2-y2Qw2zOLBW1acwEVypcPHAGCW/view?usp=drive_link).

---

## 🔍 Project Overview

The **Player Re-ID System** is built using a fine-tuned **Ultralytics YOLOv11 model** for player detection and a feature-based matching algorithm using **ResNet50** for re-identification. It ensures that each player is consistently assigned the same ID across frames — even after occlusion, disappearance, or re-entry.
## PLease click on the select video button on GUI (Select video --> Start )
##  The processing starts.

> ⚽️ Imagine you're watching a football match and want to analyze player movement. This system annotates each player with a consistent ID, color, and bounding box, helping automate tactical analysis and behavior tracking.

---

## 🚀 Features

- 🎥 Input: Custom video footage (soccer match, sports video)  
- 📦 Model: YOLOv11 fine-tuned for person detection  
- 🔍 Re-ID: ResNet50 for feature vector generation  
- 🧠 Matching: Custom cosine similarity + IOU logic  
- 🖥 GUI: Tkinter interface for model/video selection and tracking visualization  
- 🎯 Output: Processed video with bounding boxes and consistent IDs saved as `output_with_reid.mp4`

---

## 🧭 Full Approach and Methodology

### 1. **Detection**
Used a fine-tuned Ultralytics YOLOv11 model to detect players with high accuracy across match footage.

### 2. **Feature Extraction**
Extracted a 2048-dimensional feature vector from each player's cropped bounding box using ResNet50 (last classification layer removed).

### 3. **ID Assignment**
Cosine similarity and Intersection over Union (IoU) were used to match new detections with existing tracks. Unmatched detections were assigned new IDs with unique color tags.

### 4. **GUI System**
Built with Tkinter to allow users to:
- Load the YOLOv11 model
- Select input video
- Start/Stop the tracking process
- View live progress

### 5. **Tracking Management**
A custom identity management system maintains consistency over time and handles occlusion recovery.

---

## 📊 Techniques Explored

| Technique          | Outcome                                                                 |
|-------------------|-------------------------------------------------------------------------|
| ByteTrack & SORT  | High-speed but unstable ID switching during re-entry scenarios          |
| DeepSORT          | Required heavy tuning; complex for YOLOv11 integration                  |
| Manual ReID + IOU | Delivered most consistent and explainable results with higher accuracy  |

---

## ⚠️ Challenges Faced

- Identity drift on occlusions and player re-entry  
- Small detection boxes leading to poor feature vectors  
- High compute time for real-time scenarios  
- ID switching during sudden overlaps  

---
## 📁 Before & After Samples

| Sample | Video                 | Description                                           |
|--------|-----------------------|-------------------------------------------------------|
| Before | `15sec_input_720p`    | Raw video with untracked, unidentified players        |
| After  | `output_with_reid.mp4`| Annotated video with color-coded, uniquely identified players across frames |

> 📂 *Make sure to view the `output_with_reid.mp4` generated in your run folder after execution.*

---

## 📎 Documentation Attached

A detailed **PDF documentation** (`Player_ReID_Documentation.pdf`) is included, containing:

- Full technical breakdown 📘  
- Approaches compared ✅  
- Challenges analyzed ❗  
- Proposed improvements 🔧

Make sure to go through the document to understand the complete lifecycle of this project.

---

## 🛠 Tech Stack

- Python  
- OpenCV  
- PyTorch  
- Ultralytics YOLOv11  
- TorchVision  
- Tkinter  

---

## 🧑‍💻 Author

**Tharun Tej**  
AI Developer | UIUX Developer | Video Editor

---

## 📬 Let's Connect

For collaborations, improvements, or any queries:
- ✉️ Email: tharuntejandhe@gmail.com  
- 🌐 Website: [www.vatrixai.com](https://nirvaagni-oeg1.onrender.com/) <!-- Replace if different -->

---

> "This project isn't just a tool — it's a gateway to smarter sports analysis."



**⭐ Don't forget to star the repo if you found it useful!**
