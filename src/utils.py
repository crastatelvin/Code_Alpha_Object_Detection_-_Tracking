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

def ccw(A, B, C):
    """Checks if points A, B, C are in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    """Returns True if line segment AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

class LineCounter:
    """
    Counts tracked objects crossing a virtual line segment in the frame.
    Supports distinguishing entry ('In') and exit ('Out') directions.
    """
    def __init__(self, line_coords):
        """
        Args:
            line_coords: list of two tuples [(x1, y1), (x2, y2)] representing the counting line.
        """
        self.line = line_coords
        self.track_history = {}  # track_id -> last_known_center
        self.in_count = 0
        self.out_count = 0
        self.counted_ids = set()

    def update(self, frame, tracks):
        """
        Updates the counting state with the current frame's tracks and draws overlays.
        
        Args:
            frame: BGR frame to draw annotations on.
            tracks: list of tracks from the tracker: [[x1, y1, x2, y2, track_id], ...]
        """
        if self.line is None:
            return

        # Draw the counting line (in orange)
        cv2.line(frame, self.line[0], self.line[1], (0, 165, 255), 3, cv2.LINE_AA)
        
        # Label the counting line
        cv2.putText(
            frame, "Counting Line", (self.line[0][0], self.line[0][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
        )

        active_ids = set()
        
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            active_ids.add(track_id)

            # Calculate center point
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            curr_center = (cx, cy)

            # Draw center point on frame
            cv2.circle(frame, curr_center, 4, (0, 0, 255), -1)

            # Check if we have history for this track
            if track_id in self.track_history:
                prev_center = self.track_history[track_id]

                # Check if movement segment intersects counting line
                if intersect(prev_center, curr_center, self.line[0], self.line[1]):
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        
                        # Determine direction using 2D cross product
                        # Line vector (L)
                        lx = self.line[1][0] - self.line[0][0]
                        ly = self.line[1][1] - self.line[0][1]
                        
                        # Movement vector (M)
                        mx = cx - prev_center[0]
                        my = cy - prev_center[1]
                        
                        # Cross product: L x M
                        cross_product = lx * my - ly * mx
                        
                        if cross_product > 0:
                            self.in_count += 1
                            print(f"[Counter] Object {track_id} crossed IN. In: {self.in_count}, Out: {self.out_count}")
                        else:
                            self.out_count += 1
                            print(f"[Counter] Object {track_id} crossed OUT. In: {self.in_count}, Out: {self.out_count}")

            # Update history
            self.track_history[track_id] = curr_center

        # Clean up track history for inactive IDs to prevent memory growth
        inactive_ids = set(self.track_history.keys()) - active_ids
        for id_to_remove in inactive_ids:
            self.track_history.pop(id_to_remove, None)

        # Draw counts on frame (top-right overlay)
        h, w = frame.shape[:2]
        cv2.putText(frame, f"In: {self.in_count}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Out: {self.out_count}", (w - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

