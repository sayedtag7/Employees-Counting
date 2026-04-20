# People Count using YOLOv8

Maintained by **Sayed Tag-Eldin** (ID: **222101484**) — https://github.com/sayedtag7

**MAY-x Team** — Robotics Projects, New Mansoura University

This project detects and counts people in a video file, laptop webcam, or IP camera stream using Ultralytics YOLOv8 with centroid-based tracking.

It also generates a **movement heatmap** from tracked centroids and overlays it on the output frames.

## Installation

1. Clone your fork (replace `<repo>` with your repository name):

```
git clone https://github.com/sayedtag7/<repo>.git
cd <repo>
```

2. (Recommended) Create and activate a virtual environment.
3. Install dependencies:

```
pip install -r requirements.txt
```

## Model

The YOLOv8 model used in this project is `yolov8x.pt`.

## Usage

Choose an input source with `--source`:

```
# Video file
python main.py --source "Input/input.mp4"

# Laptop webcam (usually 0)
python main.py --source 0

# IP camera / live stream (example)
python main.py --source "rtsp://<user>:<pass>@<ip>:554/stream"
```

The app opens a window showing the stream with bounding boxes and counters. Press **Esc** to exit.

Output video is saved as `Final_output.mp4` in the project directory (includes boxes/counters + heatmap overlay).

## Heatmap

The heatmap accumulates tracked object centroids over time, then blurs + normalizes + color-maps the result and blends it onto the same frame **before** both saving and displaying.

## Performance defaults (current code)

These are the current settings in [main.py](main.py) that trade accuracy for speed:

- Processing resolution: `640x360`
- Frame skipping: processes **1 out of every 5** frames
- YOLO inference: `imgsz=256`, `conf=0.35`
- Heatmap tuning: updates every **2 processed** frames, `radius=12`, `value=1.5`, blur sigma `3`

The display window is resizable and set to `1200x700` by default.

## COCO Classes

Edit `coco.txt` to match your class list (one class name per line).

## What changed in this fork

- Added `--source` CLI argument to run on a video file, webcam index, or stream URL.
- Improved webcam support on Windows by using the DirectShow backend.
- Added a clear error message if the input source cannot be opened.
- Added a centroid-based movement heatmap overlay.
- Tuned runtime performance (smaller processing size, more frame skipping, lighter inference/heatmap settings).
