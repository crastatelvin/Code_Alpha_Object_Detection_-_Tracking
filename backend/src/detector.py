import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    """
    A wrapper class for the YOLOv8 model to perform object detection
    with confidence filtering and class-specific filtering.
    """
    def __init__(self, model_path: str = "yolov8n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.names = self.model.names  # Dictionary of class ID to name
        print(f"Loaded {len(self.names)} classes successfully.")

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.3, filter_classes: list = None):
        """
        Runs inference on the frame and filters detections.
        
        Args:
            frame: Input BGR image (numpy array)
            conf_threshold: Minimum confidence score to keep a detection
            filter_classes: List of class names (strings) or IDs (ints) to keep. 
                            If None, all classes are kept.
                            
        Returns:
            detections: List of dicts, each with keys 'bbox' (x1, y1, x2, y2), 'confidence', 'class_id', 'label'
            raw_results: Raw Ultralytics Result object
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        # Parse filter classes if provided
        target_ids = None
        if filter_classes:
            target_ids = []
            for cls in filter_classes:
                if isinstance(cls, int):
                    target_ids.append(cls)
                elif isinstance(cls, str):
                    # Find class ID from label string
                    for cid, name in self.names.items():
                        if name.lower() == cls.lower():
                            target_ids.append(cid)
                            break

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract coordinates, confidence and class ID
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.names[cls_id]

                # Filter by confidence
                if conf < conf_threshold:
                    continue

                # Filter by class
                if target_ids is not None and cls_id not in target_ids:
                    continue

                detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "label": label
                })

        return detections, results[0]

    def get_class_name(self, class_id: int) -> str:
        """Helper to get class name from ID."""
        return self.names.get(class_id, "Unknown")
