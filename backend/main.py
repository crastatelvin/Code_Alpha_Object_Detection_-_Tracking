import argparse
import sys
import os

# Import modules
import config
from src.detector import YOLODetector
from src.tracker import Sort, DeepSortTracker
from src.video_processor import VideoProcessor
from src.utils import FPSCalculator, draw_fps_overlay, LineCounter, download_sample_video

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection and Tracking (YOLOv8 + SORT / Deep SORT)")
    
    parser.add_argument(
        "--source", 
        type=str, 
        default=None, 
        help="Video source: index of webcam (e.g. 0) or path to video file (e.g. videos/test.mp4). If None, defaults to config or downloads a sample."
    )
    parser.add_argument(
        "--tracker", 
        type=str, 
        choices=["sort", "deepsort"], 
        default=config.TRACKER_TYPE,
        help=f"Tracker type to use: 'sort' or 'deepsort' (default: {config.TRACKER_TYPE})"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=config.CONF_THRESHOLD,
        help=f"Confidence threshold for detection (default: {config.CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--classes", 
        type=str, 
        nargs="+", 
        default=config.FILTER_CLASSES,
        help=f"Space-separated list of COCO class names to track, e.g. person car (default: {config.FILTER_CLASSES})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Path to save output video (e.g. output/result.mp4). Set to save processing results."
    )
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run without displaying the OpenCV window (useful for servers / terminal verification)"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        default=None,
        help="Maximum frames to process before stopping (default: all)"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Determine and prepare video source
    source = args.source
    if source is None:
        # Check if config has COUNTING_LINE or default video
        # Let's default to a test video file for a complete demo
        source = "videos/test.mp4"
        
    # If the source is a file path and doesn't exist, try to download a sample video
    if not source.isdigit() and not os.path.exists(source) and source == "videos/test.mp4":
        download_sample_video(source)

    # 2. Instantiate YOLOv8 Detector
    detector = YOLODetector(config.MODEL_PATH)
    
    # 3. Instantiate Tracker
    print(f"Initializing tracker: {args.tracker.upper()}")
    if args.tracker.lower() == "deepsort":
        tracker = DeepSortTracker(
            max_age=config.SORT_MAX_AGE,
            n_init=config.SORT_MIN_HITS,
            max_cosine_distance=config.SORT_IOU_THRESHOLD
        )
    else:
        tracker = Sort(
            max_age=config.SORT_MAX_AGE,
            min_hits=config.SORT_MIN_HITS,
            iou_threshold=config.SORT_IOU_THRESHOLD
        )

    # 4. Prepare Output Directory if recording is enabled
    output_path = args.output
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    # 5. Create Video Processor
    processor = VideoProcessor(
        detector=detector,
        tracker=tracker,
        source=source,
        output_path=output_path,
        headless=args.headless
    )

    # 6. Setup FPS Calculator and register callback
    if config.SHOW_FPS:
        fps_calc = FPSCalculator()
        def fps_callback(frame, tracks):
            fps = fps_calc.tick()
            draw_fps_overlay(frame, fps, config.FPS_COLOR)
        processor.register_callback(fps_callback)

    # 7. Setup Line Counter and register callback
    if config.COUNTING_LINE is not None:
        print(f"Enabling Line Counter at segment: {config.COUNTING_LINE}")
        counter = LineCounter(config.COUNTING_LINE)
        def counting_callback(frame, tracks):
            counter.update(frame, tracks)
        processor.register_callback(counting_callback)

    # 8. Start Processing Loop
    print(f"Starting processing on source: {source}")
    print(f"Tracking classes: {args.classes}")
    print(f"Detection confidence threshold: {args.conf}")
    
    processor.process(
        conf_threshold=args.conf,
        filter_classes=args.classes,
        max_frames=args.max_frames
    )
    
    # Print final counting results
    if config.COUNTING_LINE is not None:
        print("\n--- Final Tracking Counts ---")
        print(f"Total Crossed IN:  {counter.in_count}")
        print(f"Total Crossed OUT: {counter.out_count}")
        print("-----------------------------")

if __name__ == "__main__":
    main()
