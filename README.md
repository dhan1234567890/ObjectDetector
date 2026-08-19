# Real-Time Object Detection

This project demonstrates a real-time object detection system built using Python, OpenCV, and the state-of-the-art **Ultralytics YOLOv8** neural network. It accesses your computer's webcam to capture live video, processes each frame through the AI model to detect 80 different common objects (people, cell phones, chairs, cups, etc.), and displays the live feed with bounding boxes and confidence scores drawn around the recognized objects.

This project is an excellent demonstration of applied computer vision, machine learning inference, and handling real-time video streams.

## Features
* **Real-Time Inference:** Uses the optimized YOLOv8 "nano" model (`yolov8n.pt`) to ensure high frame rates and smooth video playback on standard hardware.
* **Live Webcam Processing:** Uses OpenCV to interface directly with hardware cameras.
* **Confidence Filtering:** Automatically filters out low-probability guesses (ghost detections) to maintain high accuracy.
* **FPS Monitoring:** Calculates and displays a live Frames Per Second counter to measure performance.
* **Live Object Counting:** Parses neural network outputs to tally and display the total count of each detected object type in real time.
* **CLI Flexibility:** Configurable via command-line arguments to easily swap models or tweak confidence thresholds without editing code.

## Prerequisites
* Python 3.8 or higher
* A working webcam

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/dhan1234567890/ObjectDetector.git
   cd ObjectDetector
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Ensure your virtual environment is active, then run the main script:

```bash
python main.py
```

* The application will download the YOLOv8 weights file on the very first run.
* A window will open displaying your webcam feed with real-time detections, FPS, and object counts.
* Click on the video window and press the **'q'** key to quit the application.

### Command-Line Arguments

You can customize the execution using command-line arguments:

```bash
python main.py --model yolov8s.pt --conf 0.7
```

* `--model`: The YOLOv8 model to use (e.g., `yolov8n.pt` for speed, `yolov8s.pt` for accuracy). Default is `yolov8n.pt`.
* `--conf`: The confidence threshold to filter out weak detections (0.0 to 1.0). Default is `0.5`.

## Acknowledgements
* Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* Built with [OpenCV](https://opencv.org/)
