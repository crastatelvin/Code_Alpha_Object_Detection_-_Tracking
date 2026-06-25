<div align="center">

# 👁️ DETECT & TRACK

### Real-Time Object Detection & Multi-Object Tracking — YOLOv8 + SORT / Deep SORT Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-FF6F00?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Detect & Track** is a production-grade multi-object tracking system. Connect a local camera or supply a raw video file, and run real-time inference using YOLOv8 coupled with state-of-the-art tracking (custom motion-based SORT and appearance-based Deep SORT) and live crossing boundary counters — all configurable through a modular, object-oriented codebase.

<br/>

![YOLOv8](https://img.shields.io/badge/Inference-YOLOv8_Nano-f3b44f?style=for-the-badge) ![Tracking](https://img.shields.io/badge/State_Estimation-SORT_%2F_Deep_SORT-5aa6ff?style=for-the-badge) ![Counting](https://img.shields.io/badge/Analytics-Line_Crossing_Counts-b189ff?style=for-the-badge) ![OOP](https://img.shields.io/badge/Structure-Modular_OOP_Design-36cfc9?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [CLI Reference](#-cli-reference)
- [Configuration](#-configuration)
- [Testing \& Run Verification](#-testing--run-verification)
- [License](#-license)

---

## 🧠 Overview

This project provides a robust, production-style video processing pipeline for real-time computer vision tasks. The backend normalizes video input, applies deep learning object detection (YOLOv8), feeds detection coordinates and scores into a state estimator, and handles multi-object association frame-by-frame. 

Users can track common COCO classes (like pedestrians, vehicles, bicycles), define physical counting gates (virtual lines), monitor real-time processing FPS, and export annotated runs to disk.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Advanced Detection** | Runs YOLOv8 models (`yolov8n.pt` up to `yolov8x.pt`) to identify coordinates, classes, and confidence |
| 🔄 **SORT tracking** | Fast, motion-only tracking using Kalman Filters and Hungarian bounding-box association |
| 🧠 **Deep SORT tracking** | Appearance-based tracking via deep visual features (MobileNet) to maintain ID consistency |
| 🚷 **Virtual Line Crossing** | Custom boundary lines to count items entering/exiting with 2D cross-product math |
| ⚡ **Stable FPS Overlay** | Deque-based sliding average window for accurate, smooth frame-rate visualizations |
| 🎯 **Flexible Class Filters** | Track specific target classes (e.g. only persons/cars) or track all COCO classes |
| 📽️ **Output Recording** | Automatically saves processed runs to custom `.mp4` video files |
| 💻 **Headless Mode** | CLI flag to run without window GUI, enabling execution on servers or headless containers |

---

## 🏗️ Architecture

```
Webcam / Video File
       │
       ▼
┌───────────────────────────────────────────────────────────────┐
│                    OpenCV Video Capture                       │
│  Reads frame matrix ──► Checks resolution, FPS ──► Decodes    │
└──────────────────────┬────────────────────────────────────────┘
                       │ Frame Image
                       ▼
┌───────────────────────────────────────────────────────────────┐
│                      YOLOv8 Inference                         │
│  Detects objects ──► Extracts Bounding Boxes, Conf, Classes   │
└──────────────────────┬────────────────────────────────────────┘
                       │ Bbox (x1, y1, x2, y2) + Confidence
                       ▼
┌───────────────────────────────────────────────────────────────┐
│                     Tracker (SORT/DeepSORT)                   │
│  Kalman Filters predict state ──► Hungarian Match ──► Keep ID │
└──────────────────────┬────────────────────────────────────────┘
                       │ Active Tracks (x1, y1, x2, y2, ID)
                       ▼
┌───────────────────────────────────────────────────────────────┐
│                      Analytics Overlays                       │
│  Line Counter (2D Cross Product) ──► FPS Calculator (Deque)   │
└──────────────────────┬────────────────────────────────────────┘
                       │ Annotated Frame Output
                       ▼
            Screen UI / Saved Video File
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Core Language** | Python 3.10+ |
| **Object Detection** | Ultralytics YOLOv8 PyTorch SDK |
| **Multi-Object Tracking** | Kalman Filters (filterpy), Hungarian Assignment (scipy) |
| **Visual Embeddings** | Deep SORT (deep-sort-realtime) |
| **Image Processing** | OpenCV (opencv-python), NumPy |
| **Video Math** | SciPy, NumPy |

---

## 📁 Project Structure

```
Code_Alpha_Object_Detection_-_Tracking/
│
├── config.py              # Central project configuration parameters
├── main.py                # Command-line entry point & pipeline setup
├── requirements.txt       # Python package dependencies
├── videos/
│   └── test.mp4           # Test video file (downloads automatically)
│
└── src/
    ├── __init__.py        # Package initialization marker
    ├── detector.py        # YOLODetector wrapper class
    ├── tracker.py         # SORT & Deep SORT tracker wrapper classes
    ├── video_processor.py # Main loop for capture, tracking & writing
    ├── utils.py           # FPSCalculator & LineCounter classes
    ├── capture_test.py    # OpenCV video capture test tool
    └── detect_test.py     # YOLOv8 basic inference test tool
```

---

## 🚀 Installation

### 1) Clone
```bash
git clone https://github.com/crastatelvin/Code_Alpha_Object_Detection_-_Tracking.git
cd Code_Alpha_Object_Detection_-_Tracking
```

### 2) Environment & Setup
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 💻 Usage

Run the pipeline using the central entry point `main.py`.

### 1. Basic Run (Webcam source 0, default SORT)
```bash
python main.py --source 0
```

### 2. Run on a Video File (Downloads a sample traffic video on first run)
```bash
python main.py --source videos/test.mp4
```

### 3. Run with Deep SORT Tracker (Appearance-based)
```bash
python main.py --source videos/test.mp4 --tracker deepsort
```

### 4. Custom Filters (Track only cars and persons, confidence > 0.4)
```bash
python main.py --source videos/test.mp4 --classes car person --conf 0.40
```

### 5. Record Output to Disk
```bash
python main.py --source videos/test.mp4 --output output/tracking_result.mp4
```

### 6. Headless Server Run (Process 150 frames, skip UI window)
```bash
python main.py --source videos/test.mp4 --headless --max-frames 150
```

---

## 📡 CLI Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `videos/test.mp4` | Webcam index (e.g. `0`) or path to a local video file |
| `--tracker` | `str` | `sort` | Tracker engine choice: `sort` or `deepsort` |
| `--conf` | `float` | `0.35` | Confidence score threshold for keeping YOLO detections |
| `--classes` | `list` | `["person", "car"]` | List of target classes to track (space-separated) |
| `--output` | `str` | `None` | Path to save output video, e.g. `output/result.mp4` |
| `--headless` | `flag` | `False` | Run program without opening active window displays |
| `--max-frames`| `int` | `None` | Process up to this number of frames, then stop |

---

## ⚙️ Configuration

Central default parameters can be managed directly in [`config.py`](./config.py):

```python
# Central settings
MODEL_PATH = "yolov8n.pt"      # YOLOv8 weight scale
CONF_THRESHOLD = 0.35          # Keeping score
TRACKER_TYPE = "sort"          # Default tracker ('sort'/'deepsort')
FILTER_CLASSES = ["person", "car"] # Target labels

# Line-Crossing gate: coordinates represent the counting segment
COUNTING_LINE = [(100, 300), (540, 300)]
```

---

## 🧪 Testing & Run Verification

Run quick validation tests to verify that dependencies and hardware are properly configured.

### Video Input Capture Test
```bash
python src/capture_test.py --source 0 --headless
```

### Object Detection Test
```bash
python src/detect_test.py
```

### Pipeline Verification Test
```bash
python main.py --headless --max-frames 30 --tracker sort
python main.py --headless --max-frames 30 --tracker deepsort
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

<div align="center">
  Built by Telvin Crasta · Production-style CV · Live today
  <br/>
  ⭐ If this tracker helped you build MOT pipelines faster, star the repo.
</div>
