# Code Alpha Object Detection & Tracking (YOLOv8 + SORT / Deep SORT)

This repository implements a production-grade real-time Object Detection and Multi-Object Tracking (MOT) system using YOLOv8, SORT, and Deep SORT.

## Features (In Progress)
- **Object Detection**: YOLOv8 (Ultra-fast, lightweight)
- **Multi-Object Tracking**: SORT (Simple Online Realtime Tracking) and Deep SORT (Visual appearance association)
- **Class Filtering**: Selectively track specific classes (e.g. only persons, cars, etc.)
- **FPS Counter**: Real-time frame-rate monitoring
- **Object Counting**: Line-crossing / boundary crossing entry-exit tracking

## Directory Structure
```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── config.py
└── src/
    ├── __init__.py
    ├── detector.py
    ├── tracker.py
    ├── video_processor.py
    └── utils.py
```
