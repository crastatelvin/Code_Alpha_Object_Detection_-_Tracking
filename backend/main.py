import os
import cv2
import json
import time
import threading
import numpy as np
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Load tracking modules
import config
from src.detector import YOLODetector
from src.tracker import Sort, DeepSortTracker
from src.utils import FPSCalculator, draw_fps_overlay, LineCounter, download_sample_video

app = FastAPI(title="Detect & Track Real-Time API")

# Enable CORS for frontend interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Application State
global_state = {
    "tracker_type": config.TRACKER_TYPE,
    "conf_threshold": config.CONF_THRESHOLD,
    "filter_classes": config.FILTER_CLASSES,
    "source": "videos/test.mp4",
    "is_running": False
}

# Real-Time Telemetry Metrics
telemetry = {
    "fps": 0.0,
    "in_count": 0,
    "out_count": 0,
    "active_tracks": 0
}

# Frame Buffer for MJPEG Stream
latest_frame_bytes = None
frame_lock = threading.Lock()

def calculate_iou(box1, box2):
    """Computes Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

class VideoProcessingThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.stop_event = threading.Event()
        self.detector = None
        self.tracker = None
        self.tracker_type = None
        self.track_id_to_class = {}  # Map track_id -> class label

    def run(self):
        global latest_frame_bytes, telemetry
        
        # 1. Setup FPS & Line Counter
        fps_calc = FPSCalculator()
        counter = LineCounter(config.COUNTING_LINE)
        
        # 2. Instantiate YOLOv8
        self.detector = YOLODetector(config.MODEL_PATH)
        
        active_source = global_state["source"]
        if active_source == "videos/test.mp4":
            download_sample_video(active_source)

        cap = cv2.VideoCapture(int(active_source) if active_source.isdigit() else active_source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source: {active_source}")
            global_state["is_running"] = False
            return

        is_webcam = active_source.isdigit()
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_delay = 1.0 / native_fps

        print("Video processing thread started successfully.")
        failed_frames = 0
        
        while not self.stop_event.is_set():
            # Check for dynamic video source switching
            if global_state["source"] != active_source:
                print(f"Switching video source from {active_source} to {global_state['source']}")
                active_source = global_state["source"]
                cap.release()
                if active_source == "videos/test.mp4":
                    download_sample_video(active_source)
                cap = cv2.VideoCapture(int(active_source) if active_source.isdigit() else active_source)
                is_webcam = active_source.isdigit()
                native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_delay = 1.0 / native_fps
                failed_frames = 0
                # Reset counters
                counter.in_count = 0
                counter.out_count = 0
                counter.counted_ids.clear()
                counter.track_history.clear()
                self.track_id_to_class.clear()

            t_start = time.time()

            # Read frame
            success, frame = cap.read()
            if not success:
                # Loop video file if it ends
                if not is_webcam:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # Reset counting stats on loop restart to keep metrics accurate
                    counter.in_count = 0
                    counter.out_count = 0
                    counter.counted_ids.clear()
                    counter.track_history.clear()
                    self.track_id_to_class.clear()
                    continue
                else:
                    failed_frames += 1
                    if failed_frames > 50:
                        print("Webcam disconnected (too many failed frames).")
                        break
                    time.sleep(0.1)
                    continue

            failed_frames = 0

            # 4. Check for Tracker Type changes
            current_tracker_type = global_state["tracker_type"]
            if self.tracker is None or self.tracker_type != current_tracker_type:
                print(f"Instantiating new tracker: {current_tracker_type.upper()}")
                self.tracker_type = current_tracker_type
                self.track_id_to_class.clear()
                if self.tracker_type == "deepsort":
                    self.tracker = DeepSortTracker(
                        max_age=config.SORT_MAX_AGE,
                        n_init=config.SORT_MIN_HITS,
                        max_cosine_distance=config.SORT_IOU_THRESHOLD
                    )
                else:
                    self.tracker = Sort(
                        max_age=config.SORT_MAX_AGE,
                        min_hits=config.SORT_MIN_HITS,
                        iou_threshold=config.SORT_IOU_THRESHOLD
                    )

            # 5. Detection & Class Filter
            conf_threshold = global_state["conf_threshold"]
            filter_classes = global_state["filter_classes"]
            
            detections, _ = self.detector.detect(frame, conf_threshold, filter_classes)
            
            # Format detections for tracker update
            sort_dets = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                score = det["confidence"]
                sort_dets.append([x1, y1, x2, y2, score])
            
            sort_dets = np.array(sort_dets) if len(sort_dets) > 0 else np.empty((0, 5))

            # 6. Tracker Update
            tracks = self.tracker.update(sort_dets, frame=frame)

            # 7. Draw Visual Tracking Annotations
            for track in tracks:
                tx1, ty1, tx2, ty2, track_id = track
                tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
                track_id = int(track_id)
                
                # Match track_id with detections to determine class label
                if track_id not in self.track_id_to_class:
                    best_iou = 0
                    best_label = "object"
                    for det in detections:
                        iou = calculate_iou([tx1, ty1, tx2, ty2], det["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_label = det["label"]
                    if best_iou > 0.3:
                        self.track_id_to_class[track_id] = best_label
                    else:
                        self.track_id_to_class[track_id] = "object"

                class_label = self.track_id_to_class.get(track_id, "object")
                
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)
                label = f"{class_label.upper()} #{track_id}"
                cv2.putText(frame, label, (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # 8. Run FPS and Line Counter Analytics
            current_fps = fps_calc.tick()
            draw_fps_overlay(frame, current_fps, config.FPS_COLOR)
            counter.update(frame, tracks)

            # Update live metrics telemetry
            telemetry["fps"] = round(current_fps, 1)
            telemetry["in_count"] = counter.in_count
            telemetry["out_count"] = counter.out_count
            telemetry["active_tracks"] = len(tracks)

            # 9. Compress Frame to JPEG Bytes
            success, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                global latest_frame_bytes
                with frame_lock:
                    latest_frame_bytes = jpeg.tobytes()

            # Pacing logic for video file inputs
            if not is_webcam:
                elapsed = time.time() - t_start
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        cap.release()
        print("Video processing thread stopped.")

    def stop(self):
        self.stop_event.set()

# Active Background Thread Instance
bg_thread = None

def start_pipeline():
    global bg_thread
    if bg_thread is None or not bg_thread.is_alive():
        global_state["is_running"] = True
        bg_thread = VideoProcessingThread()
        bg_thread.start()

def stop_pipeline():
    global bg_thread
    if bg_thread is not None:
        bg_thread.stop()
        bg_thread.join()
        bg_thread = None
    global_state["is_running"] = False

@app.on_event("startup")
def startup_event():
    start_pipeline()

@app.on_event("shutdown")
def shutdown_event():
    stop_pipeline()

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "pipeline_running": global_state["is_running"],
        "config": {
            "tracker_type": global_state["tracker_type"],
            "conf_threshold": global_state["conf_threshold"],
            "filter_classes": global_state["filter_classes"],
            "source": global_state["source"]
        }
    }

@app.post("/api/config")
def update_config_http(cfg: dict):
    """Updates config dynamically from standard HTTP requests."""
    if "tracker_type" in cfg:
        global_state["tracker_type"] = cfg["tracker_type"]
    if "conf_threshold" in cfg:
        global_state["conf_threshold"] = float(cfg["conf_threshold"])
    if "filter_classes" in cfg:
        global_state["filter_classes"] = cfg["filter_classes"]
    if "source" in cfg:
        global_state["source"] = str(cfg["source"])
    return {"status": "success", "config": global_state}

def frame_generator():
    """MJPEG Stream Frame Generator yielding multipart responses."""
    global latest_frame_bytes
    while global_state["is_running"]:
        if latest_frame_bytes is not None:
            with frame_lock:
                frame = latest_frame_bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)  # limit stream rate to ~30fps to avoid network overflow

@app.get("/api/video_feed")
def video_feed():
    """Streams the processed bounding-box overlay feed directly to browser."""
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time telemetry streaming and UI configuration controls."""
    await websocket.accept()
    print("WebSocket client connected.")
    
    # Task to periodically push telemetry metrics to client
    async def push_telemetry():
        try:
            while True:
                await websocket.send_json({
                    "type": "telemetry",
                    "data": {
                        "fps": telemetry["fps"],
                        "in_count": telemetry["in_count"],
                        "out_count": telemetry["out_count"],
                        "active_tracks": telemetry["active_tracks"]
                    }
                })
                await asyncio.sleep(0.1)  # Push at 10Hz
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Telemetry push error: {e}")

    import asyncio
    push_task = asyncio.create_task(push_telemetry())

    try:
        while True:
            # Listen for configuration updates from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "update_config":
                cfg = message.get("config", {})
                if "tracker_type" in cfg:
                    global_state["tracker_type"] = cfg["tracker_type"]
                if "conf_threshold" in cfg:
                    global_state["conf_threshold"] = float(cfg["conf_threshold"])
                if "filter_classes" in cfg:
                    global_state["filter_classes"] = cfg["filter_classes"]
                if "source" in cfg:
                    global_state["source"] = str(cfg["source"])
                
                # Send confirmation response
                await websocket.send_json({
                    "type": "config_updated",
                    "config": {
                        "tracker_type": global_state["tracker_type"],
                        "conf_threshold": global_state["conf_threshold"],
                        "filter_classes": global_state["filter_classes"],
                        "source": global_state["source"]
                    }
                })
                print(f"Config dynamically updated via WS: {global_state}")
                
    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    finally:
        push_task.cancel()
        await push_task
