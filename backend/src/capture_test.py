import cv2
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="OpenCV Video Capture Test")
    parser.add_argument("--source", type=str, default="0", help="Camera index (e.g. 0) or path to video file")
    parser.add_argument("--headless", action="store_true", help="Run without showing GUI window")
    args = parser.parse_args()

    # Convert numeric string to int if it's a webcam source
    source = args.source
    if source.isdigit():
        source = int(source)

    print(f"Initializing video capture from source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: Resolution: {int(width)}x{int(height)}, FPS: {fps}")

    frame_count = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("End of video stream or failed to read frame.")
                break

            frame_count += 1
            if args.headless:
                if frame_count % 10 == 0 or frame_count == 1:
                    print(f"Headless: Read frame {frame_count}, shape: {frame.shape}")
                if frame_count >= 30:
                    print("Read 30 frames in headless mode. Exiting test.")
                    break
            else:
                cv2.imshow("Video Capture Test (Press ESC to exit)", frame)
                # Press 'ESC' key to exit
                if cv2.waitKey(1) == 27:
                    print("ESC pressed. Exiting.")
                    break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture released.")

if __name__ == "__main__":
    main()
