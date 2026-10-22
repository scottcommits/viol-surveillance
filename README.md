
# AI CCTV Violation Detection

A Python-based CCTV video analysis project using YOLO26 and OpenCV to detect and track people and identify potential grass-zone violations.

## Features

- Person detection and tracking
- Custom grass-zone monitoring
- Potential violation alerts
- Video visualization with bounding boxes

## Requirements

- Python 3.10+
- YOLO26 model (`yolo26n.pt`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your CCTV video, `yolo26n.pt`, and `grass_polygon.json` in the project directory.
2. Run the detection script:

```bash
python detect_violations.py
```

3. Press `q` to exit the video window.

## Note

This project is a prototype. Detection results depend on the accuracy of the defined grass-zone coordinates and object tracking.

## License

For educational and research purposes.
