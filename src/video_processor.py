import cv2
import numpy as np
import time
import sys

class VideoProcessor:
    """
    Manages the video pipeline: reads frames, runs detection,
    updates tracking, draws annotations, and writes output video.
    """
    def __init__(self, detector, tracker, source="0", output_path=None, headless=False):
        """
        Args:
            detector: Instance of YOLODetector
            tracker: Instance of SORT or Deep SORT tracker
            source: Path to video file or webcam index
            output_path: Path to save output video (optional)
            headless: Run without creating GUI windows
        """
        self.detector = detector
        self.tracker = tracker
        self.headless = headless
        
        # Parse source
        self.source = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source '{self.source}'")
            sys.exit(1)
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        self.writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"Video writer initialized. Saving output to: {output_path}")

        # Hook for downstream analytics (like counting)
        self.on_frame_processed_callbacks = []

    def register_callback(self, callback):
        """Register a callback function to run on each frame.
        Callback signature: callback(frame, tracks) -> None
        """
        self.on_frame_processed_callbacks.append(callback)

    def process(self, conf_threshold=0.3, filter_classes=None, max_frames=None):
        """
        Starts processing the video stream frame-by-frame.
        
        Args:
            conf_threshold: Confidence threshold for YOLO detection
            filter_classes: Classes to detect/track (names or IDs)
            max_frames: Stop after this number of frames (useful for tests)
        """
        print("Starting video processing pipeline...")
        frame_count = 0
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("Finished reading video stream.")
                    break
                    
                frame_count += 1
                if max_frames and frame_count > max_frames:
                    print(f"Reached max frames limit ({max_frames}). Stopping.")
                    break

                # 1. Detection
                detections, _ = self.detector.detect(frame, conf_threshold, filter_classes)
                
                # 2. Format detections for SORT tracker: [[x1, y1, x2, y2, score], ...]
                sort_dets = []
                for det in detections:
                    x1, y1, x2, y2 = det["bbox"]
                    score = det["confidence"]
                    sort_dets.append([x1, y1, x2, y2, score])
                
                if len(sort_dets) > 0:
                    sort_dets = np.array(sort_dets)
                else:
                    sort_dets = np.empty((0, 5))

                # 3. Tracking Update
                # tracker.update returns: [[x1, y1, x2, y2, id], ...]
                tracks = self.tracker.update(sort_dets, frame=frame)

                # 4. Draw tracks
                for track in tracks:
                    tx1, ty1, tx2, ty2, track_id = track
                    tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
                    track_id = int(track_id)

                    # Draw bounding box
                    cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 2)
                    
                    # Draw label with track ID
                    label = f"ID {track_id}"
                    cv2.putText(frame, label, (tx1, ty1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                # 5. Call registered analytics (e.g. FPS overlay, Counting)
                for callback in self.on_frame_processed_callbacks:
                    callback(frame, tracks)

                # 6. Save output frame
                if self.writer:
                    self.writer.write(frame)

                # 7. Display Frame
                if not self.headless:
                    cv2.imshow("Object Tracking & Detection", frame)
                    if cv2.waitKey(1) == 27:  # ESC key
                        print("ESC pressed, stopping process.")
                        break
                else:
                    if frame_count % 10 == 0 or frame_count == 1:
                        print(f"Processed frame {frame_count} | Active tracks: {len(tracks)}")

        except KeyboardInterrupt:
            print("Processing interrupted by user.")
        finally:
            self.release()

    def release(self):
        """Release resources."""
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("Resources released successfully.")
