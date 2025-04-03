## Overview
This project is an AI-powered waste sorting system that utilizes the YOLOv8 object detection model to identify and classify waste items into appropriate bins. The system also includes text-to-speech (TTS) functionality to provide real-time audio guidance on where to dispose of detected items.

## Features
- **Real-Time Object Detection**: Uses YOLOv8 to classify waste items.
- **Text-to-Speech (TTS) Guidance**: Provides spoken instructions on correct waste disposal.
- **Category-Based Sorting**: Assigns detected items to three main bins:
  - **Recycle**
  - **Food Waste**
  - **Hazardous Waste**
- **Customizable Thresholds**: Adjustable confidence levels for object classification.
- **Sensor Simulation**: Mimics real-world bin activation for smart waste management.

## Requirements
### Dependencies
Ensure you have the following dependencies installed:
```bash
pip install opencv-python numpy ultralytics pyttsx3
```

### Hardware Requirements
- Webcam or camera for real-time video processing.
- Speaker for TTS functionality.

## Installation & Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-waste-sorting.git
cd smart-waste-sorting
```
2. Install required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the YOLOv8 model (modify the path if using a custom model):
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```
4. Run the system:
```bash
python main.py
```

## Configuration
Modify `MODEL_PATH`, `CONFIDENCE_THRESHOLD_SORT`, and `VIDEO_SOURCE` in the script as needed:
```python
MODEL_PATH = 'yolov8n.pt'  # Path to YOLO model
CONFIDENCE_THRESHOLD_SORT = 0.60  # Confidence threshold for sorting
VIDEO_SOURCE = 0  # Camera source (0 for default webcam)
```

## Usage
- Start the script and point the camera at objects.
- The system will detect and classify the objects.
- Audio guidance will inform users of the correct bin.
- Press 'q' to quit the program.

## Supported Object Categories
The system supports the following categories:
- **Recycle:** Bottles, books, cups, electronics, etc.
- **Food Waste:** Fruits, vegetables, pizza, sandwiches, etc.
- **Hazardous:** Batteries, knives, scissors, etc.

## Known Issues & Troubleshooting
- If no voice output is heard, check your system's available voices in `pyttsx3`.
- If YOLO does not detect objects correctly, try fine-tuning the confidence threshold.
- Ensure your camera is properly connected and accessible.

## Future Enhancements
- **Multi-Language Support**: Expanding TTS functionality beyond English and Thai.
- **Improved Object Classification**: Fine-tuning the model with a custom dataset.
- **Physical Bin Integration**: Connecting to IoT-enabled waste bins for automated sorting.

## License
This project is open-source under the MIT License.

## Acknowledgments
- Built with [YOLOv8](https://github.com/ultralytics/ultralytics)
- Inspired by AI-driven sustainability projects.

