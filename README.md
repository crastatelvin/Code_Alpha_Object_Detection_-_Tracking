<div align="center">

# 👁️ DETECT & TRACK

### Real-Time Object Detection & Multi-Object Tracking — FastAPI + React Web Console

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.x-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-FF6F00?style=for-the-badge&logo=ultralytics&logoColor=white)](https://ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Detect & Track** is a production-grade multi-object tracking platform. Connect a camera feed or stream raw video, run real-time inference (YOLOv8 + SORT / Deep SORT), and watch live tracking telemetry on a glassmorphic dashboard. Adjust parameters like confidence thresholds, class filters, and tracker engines in real-time.

<br/>

![YOLOv8](https://img.shields.io/badge/Inference-YOLOv8_Nano-f3b44f?style=for-the-badge) ![Tracking](https://img.shields.io/badge/State_Estimation-SORT_%2F_Deep_SORT-5aa6ff?style=for-the-badge) ![Web](https://img.shields.io/badge/Console-React_%2B_Vite_Dashboard-b189ff?style=for-the-badge) ![AutoStream](https://img.shields.io/badge/Protocol-MJPEG_%2B_WebSockets-36cfc9?style=for-the-badge)

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
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Testing & Run Verification](#-testing--run-verification)
- [License](#-license)

---

## 🧠 Overview

This project decouples deep-learning computer vision from client rendering by building a high-performance, asynchronous FastAPI backend and a responsive React client web console. 

The backend runs camera processing on a background thread, updates trackers, counts line-crossings via 2D vector mathematics, and streams annotated frame buffers via Motion JPEG (MJPEG) alongside live WebSocket telemetry. The React client displays this video feed in a sci-fi viewport HUD and provides instant control sliders and class-filtering tags.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Advanced Detection** | Identifies and filters targets using YOLOv8 models (`yolov8n.pt` up to `yolov8x.pt`) |
| 🔄 **Dual Tracking Engines** | Choose **SORT** (motion-based state estimation) or **Deep SORT** (appearance-based embeddings) |
| 🚷 **Virtual Line Crossing** | Tracks and counts objects crossing entry/exit boundaries in real-time |
| 🧪 **Responsive WebSocket Telemetry** | Broadcasts current FPS, active track count, and crossed metrics at 10Hz |
| 🎛️ **Live Parameter Tuning** | Adjust confidence thresholds, class tags, and tracker engines on-the-fly without resets |
| 📽️ **MJPEG Stream Viewport** | Zero-overhead processed camera frame streams rendered directly in browser image viewport |
| 🎨 **Premium Neon Dashboard** | Glassmorphic cards, scifi HUD crosshairs, glowing speed dials, and responsive layouts |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       React Web Console                     │
│                                                             │
│   Viewport Stream (<img>)  ◄─── GET /api/video_feed         │
│   Telemetry Analytics      ◄─── WS /ws (10Hz push)          │
│   Parameter Tuning Controls ───► WS /ws (config JSON update)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                       │
│                                                             │
│   /api/video_feed ──► Serves MJPEG frame buffers            │
│   /ws             ──► Pushes telemetry / receives configs   │
└──────────────────────────────┬──────────────────────────────┘
                               │ Thread Decoupled Frame
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Background Processing Thread                 │
│                                                             │
│   OpenCV Capture ──► YOLOv8 ──► SORT/DeepSORT ──► Counter   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI, Uvicorn, Websockets, Python 3.10+ |
| **Object Detection** | Ultralytics YOLOv8 PyTorch SDK |
| **State Tracking** | Kalman Filters (filterpy), Deep SORT (deep-sort-realtime) |
| **Data Math** | NumPy, SciPy |
| **Frontend UI** | React 18, Vite, Lucide Icons, Custom CSS |
| **Communication** | REST (MJPEG) + WebSockets |

---

## 📁 Project Structure

```
Code_Alpha_Object_Detection_-_Tracking/
│
├── backend/               # FastAPI Web Server Code
│   ├── main.py            # API routes, WebSockets, background threads
│   ├── config.py          # Central tracking configuration settings
│   ├── requirements.txt   # Python server requirements
│   ├── videos/            # Sample videos (downloads automatically)
│   └── src/
│       ├── detector.py    # YOLOv8 detector class
│       ├── tracker.py     # SORT & Deep SORT tracker engines
│       └── utils.py       # LineCounter, FPSCalculator, Downloader
│
└── frontend/              # React Web Client Code
    ├── src/
    │   ├── App.jsx        # Dashboard layout, WebSocket sync
    │   ├── App.css        # Premium neon styling overlays & grid
    │   ├── index.css      # Reset styles & global fonts
    │   └── main.jsx       # React entry point
    ├── index.html         # HTML layout template with Outfit fonts
    ├── vite.config.js     # Dev proxy definitions for /api and /ws
    └── package.json       # Node package manager configurations
```

---

## 🚀 Installation

### 1) Clone the Repository
```bash
git clone https://github.com/crastatelvin/Code_Alpha_Object_Detection_-_Tracking.git
cd Code_Alpha_Object_Detection_-_Tracking
```

### 2) Setup the Backend
```bash
cd backend
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate
# Activate environment (Linux/macOS)
source .venv/bin/activate

pip install -r requirements.txt
```

### 3) Setup the Frontend
Open a new terminal window:
```bash
cd frontend
npm install
```

---

## 💻 Usage

### 1) Run the FastAPI Server
From the `backend/` directory (with virtual environment active):
```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
*The server will cache/download YOLOv8 weights and the sample video file automatically on startup.*

### 2) Run the React Web App
From the `frontend/` directory:
```bash
npm run dev
```
Open your browser and navigate to **`http://localhost:3000`** to view the live dashboard!

---

## 📡 API Reference

| Endpoint | Method | Protocol | Description |
|---|---|---|---|
| `/api/status` | `GET` | HTTP | Returns API server status and running configurations |
| `/api/config` | `POST`| HTTP | Dynamic parameter overrides via standard request payloads |
| `/api/video_feed`| `GET` | HTTP | Streams the processed bounding-box overlay feed using MJPEG |
| `/ws` | `GET` | WebSocket | Pushes live metrics (FPS, counters, tracks) & listens to parameter updates |

---

## ⚙️ Configuration

Tweak default settings directly in [`backend/config.py`](./backend/config.py):

```python
# Central settings
MODEL_PATH = "yolov8n.pt"      # YOLOv8 weight scale
CONF_THRESHOLD = 0.35          # Default score threshold
TRACKER_TYPE = "sort"          # Default tracker ('sort'/'deepsort')
FILTER_CLASSES = ["person", "car"] # Target classes

# Line-Crossing coordinates segment [(x1, y1), (x2, y2)]
COUNTING_LINE = [(100, 300), (540, 300)]
```

---

## 🧪 Testing & Run Verification

Before starting the web services, you can verify video frame decoding, YOLO inferences, and tracker modules:

### Test Video Capture
```bash
python backend/src/capture_test.py --source 0 --headless
```

### Test YOLOv8 Inference
```bash
python backend/src/detect_test.py
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

<div align="center">
  Built by Telvin Crasta · Production-style CV · Live today
  <br/>
  ⭐ If this tracker helped you build MOT pipelines faster, star the repo.
</div>
