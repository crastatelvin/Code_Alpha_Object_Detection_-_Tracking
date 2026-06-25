# Object Detection & Tracking (YOLOv8 + SORT / Deep SORT)

This repository implements a production-grade real-time Object Detection and Multi-Object Tracking (MOT) system using **YOLOv8**, a custom **SORT** tracker, and **Deep SORT** (via `deep-sort-realtime`). It features custom class filtering, sliding-average FPS calculation, and virtual line-crossing object counting.

## 🚀 Key Features

* **Advanced Object Detection**: Uses **YOLOv8** (defaulting to the lightweight `yolov8n.pt` model) for fast real-time inference.
* **Dual Trackers Supported**:
  * **SORT**: High-speed movement-only tracking using Kalman Filters and Hungarian association.
  * **Deep SORT**: Accurate appearance-based tracking utilizing deep visual embeddings (MobileNet) to prevent ID switching under occlusions.
* **Flexible Class Filtering**: Customize which classes to track (e.g., track only `person` and `car` and ignore background classes).
* **Virtual Line Crossing (Object Counting)**: Define a custom 2D line segment inside the frame to count objects crossing **IN** (entry) or **OUT** (exit) dynamically.
* **Stable FPS Overlay**: Real-time FPS monitoring with moving average filtering for smooth UI display.
* **Output Recording**: Option to render and save the processing results directly to a video file.
* **Headless Execution**: CLI argument to run without standard GUI window displays (perfect for headless Linux/Windows servers, docker containers, and automated workflows).

---

## 🛠️ Installation

Ensure you have **Python 3.10+** installed, then follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/crastatelvin/Code_Alpha_Object_Detection_-_Tracking.git
   cd Code_Alpha_Object_Detection_-_Tracking
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: PyTorch and torchvision are required by the Ultralytics package. If they are not present, pip will automatically resolve and install them.*

---

## 💻 Usage

The system is designed with a CLI-argument-driven entry point. Run `python main.py --help` to see all available arguments.

### 1. Basic Run (Webcam source 0, default SORT)
```bash
python main.py --source 0
```

### 2. Run on a Video File (Downloads a sample video if not found)
```bash
python main.py --source videos/test.mp4
```

### 3. Run with Deep SORT Tracker
```bash
python main.py --source videos/test.mp4 --tracker deepsort
```

### 4. Headless Mode (CLI/Server execution - stops after 150 frames)
```bash
python main.py --source videos/test.mp4 --headless --max-frames 150
```

### 5. Save Processed Video to Output File
```bash
python main.py --source videos/test.mp4 --output output/result.mp4
```

### 6. Filter Detections (Track only cars and persons, confidence > 0.4)
```bash
python main.py --source videos/test.mp4 --classes car person --conf 0.40
```

---

## ⚙️ Configuration (`config.py`)

You can customize various parameters inside `config.py` directly:
* `MODEL_PATH`: YOLOv8 weight type (`yolov8n.pt`, `yolov8s.pt`, etc.)
* `CONF_THRESHOLD`: Detection confidence limit.
* `TRACKER_TYPE`: Pick between `"sort"` and `"deepsort"`.
* `FILTER_CLASSES`: Default classes to keep (e.g. `["person", "car"]`).
* `COUNTING_LINE`: Virtual counting line segment `[(x1, y1), (x2, y2)]`. Set to `None` to disable.

---

## 📂 Codebase Architecture

```text
.
├── config.py              # Central project configuration parameters
├── main.py                # Command-line entry point and runner setup
├── requirements.txt       # Python package dependencies
├── videos/
│   └── test.mp4           # Downloaded/local sample input video
└── src/
    ├── __init__.py        # Package initialization file
    ├── detector.py        # YOLOv8 class detector wrapper
    ├── tracker.py         # SORT & Deep SORT tracker classes
    ├── video_processor.py # Main frame capture and processing pipeline
    └── utils.py           # FPS moving average and line crossing counting utilities
```
