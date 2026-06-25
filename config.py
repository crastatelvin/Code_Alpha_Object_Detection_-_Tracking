# Project Configurations for Object Detection & Tracking

# Model Configurations
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.35

# Tracking Configurations
TRACKER_TYPE = "sort"  # Options: 'sort', 'deepsort'
SORT_MAX_AGE = 5
SORT_MIN_HITS = 2
SORT_IOU_THRESHOLD = 0.3

# Class Filtering
# List of class names to detect/track. If empty list or None, all classes are tracked.
# Common COCO classes: 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'dog', 'cat', etc.
FILTER_CLASSES = ["person", "car"]

# Visual Overlays
SHOW_FPS = True
BOX_COLOR = (255, 0, 255)  # Magenta for active tracks
TEXT_COLOR = (255, 255, 255)
FPS_COLOR = (0, 255, 0)  # Green for FPS overlay

# Line-Crossing Counting Configuration
# Coordinates of the counting line: [(x1, y1), (x2, y2)]
# Set to None to disable line counting
COUNTING_LINE = [(100, 300), (540, 300)]
