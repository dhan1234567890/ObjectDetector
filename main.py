import cv2
from ultralytics import YOLO
import sys
import argparse
import time
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection using YOLOv8")
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="Path to YOLOv8 model file (e.g., yolov8n.pt, yolov8s.pt). Default is yolov8n.pt"
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for detections (0.0 to 1.0). Default is 0.5"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading YOLOv8 model: {args.model} ...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Initializing webcam...")
    # Initialize the webcam (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(1)

    print("Starting detection loop. Press 'q' to exit.")
    
    # Variable to track time for FPS calculation
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # Perform detection with confidence threshold
        results = model(frame, conf=args.conf, verbose=False)

        # Get the first (and only) result object
        result = results[0]
        
        # Visualize the results on the frame
        annotated_frame = result.plot()

        # Draw FPS on the frame
        cv2.putText(
            annotated_frame, 
            f"FPS: {int(fps)}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), # Green color
            2
        )

        # Object Counting logic
        # Extract class IDs from the boxes
        class_ids = result.boxes.cls.cpu().numpy() if result.boxes else []
        # Map class IDs to class names
        class_names = [model.names[int(cls_id)] for cls_id in class_ids]
        # Count occurrences of each object
        object_counts = Counter(class_names)
        
        # Draw the counts on the screen
        y_offset = 70
        for obj, count in object_counts.items():
            text = f"{obj}: {count}"
            cv2.putText(
                annotated_frame, 
                text, 
                (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 0), # Cyan color
                2
            )
            y_offset += 30

        # Display the output
        cv2.imshow("Real-Time Object Detection", annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
