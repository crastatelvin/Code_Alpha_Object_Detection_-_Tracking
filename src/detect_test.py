import cv2
import sys
from ultralytics import YOLO

def main():
    print("Loading YOLOv8 nano model (yolov8n.pt)...")
    # This will automatically download yolov8n.pt to the current directory if it doesn't exist
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully.")

    # Initialize capture
    source = 0
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        sys.exit(1)

    success, frame = cap.read()
    cap.release()

    if not success:
        print("Error: Could not read frame from video source.")
        sys.exit(1)

    print(f"Frame captured. Shape: {frame.shape}. Running YOLOv8 inference...")
    results = model(frame)

    print("\n--- YOLOv8 Detections ---")
    detections_found = False
    for result in results:
        boxes = result.boxes
        for box in boxes:
            detections_found = True
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            print(f"Detected Class: {label} (ID: {cls_id}), Conf: {conf:.2f}, Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")

    if not detections_found:
        print("No objects detected in this frame.")

    # Save results to a file
    output_path = "output_test.jpg"
    print(f"\nSaving annotated image to {output_path}...")
    annotated_frame = results[0].plot()  # plot returns a numpy array with bounding boxes drawn
    cv2.imwrite(output_path, annotated_frame)
    print("Done!")

if __name__ == "__main__":
    main()
