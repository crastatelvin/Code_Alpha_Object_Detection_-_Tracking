import cv2
import time
from collections import deque

class FPSCalculator:
    """
    Computes running frame-per-second (FPS) value using a moving average
    over a sliding window of frame execution times.
    """
    def __init__(self, window_size: int = 30):
        self.times = deque(maxlen=window_size)
        self.prev_time = None

    def tick(self) -> float:
        """Call this on each frame to register timestamp and get current FPS."""
        current_time = time.time()
        
        if self.prev_time is not None:
            time_diff = current_time - self.prev_time
            if time_diff > 0:
                self.times.append(time_diff)
                
        self.prev_time = current_time
        
        if len(self.times) == 0:
            return 0.0
            
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

def draw_fps_overlay(frame, fps: float, color=(0, 255, 0)):
    """Draws a neat FPS overlay on the top-left of the frame."""
    fps_text = f"FPS: {int(fps)}"
    cv2.putText(
        frame,
        fps_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )
