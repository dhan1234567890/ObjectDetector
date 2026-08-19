import cv2
from ultralytics import YOLO
import sys

def main():
    print("Loading YOLOv8 model...")
    # Using the 'nano' model to ensure smooth real-time video without lag
    try:
        model = YOLO('yolov8n.pt')
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
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Perform detection with a confidence threshold (only keep detections with > 50% confidence)
        results = model(frame, conf=0.5, verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

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
