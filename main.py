import argparse
import cv2
import pandas as pd
import numpy as np
import os
from ultralytics import YOLO
from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import FPS
import logging
import time
import threading


# execution start time
start_time = time.time()

# setup logger
logging.basicConfig(level = logging.INFO, format = "[INFO] %(message)s")
logger = logging.getLogger(__name__)

model = YOLO('yolov8x.pt')

# Dashboard UI settings (rendering only)
DASHBOARD_TITLE = "EMPLOYEES COUNTING"
UI_WIDTH = 1200
UI_HEIGHT = 700


def _parse_source(value: str):
    value = value.strip()
    if value.lstrip("-").isdigit():
        return int(value)
    return value


def _open_capture(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {source}")
    return cap


def update_heatmap(heatmap, x, y, radius=35, value=3.0):
    if heatmap is None:
        return

    h, w = heatmap.shape[:2]
    x = int(x)
    y = int(y)
    if x < 0 or y < 0 or x >= w or y >= h:
        return

    r = int(radius)
    if r <= 0:
        heatmap[y, x] += float(value)
        return

    x0 = max(0, x - r)
    x1 = min(w, x + r + 1)
    y0 = max(0, y - r)
    y1 = min(h, y + r + 1)

    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    sigma = max(1.0, r / 2.0)
    kernel = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    heatmap[y0:y1, x0:x1] += kernel * float(value)


def _draw_rounded_rect(img, x, y, w, h, r, color):
    r = max(0, int(r))
    if r == 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        return

    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
    cv2.circle(img, (x + r, y + r), r, color, -1)
    cv2.circle(img, (x + w - r, y + r), r, color, -1)
    cv2.circle(img, (x + r, y + h - r), r, color, -1)
    cv2.circle(img, (x + w - r, y + h - r), r, color, -1)


def compose_dashboard_frame(video_frame, in_count, out_count, status_text, source_type):
    bg = np.full((UI_HEIGHT, UI_WIDTH, 3), (20, 22, 28), dtype=np.uint8)

    title_h = 80
    pad = 20
    panel_w = 350

    title_bg = (28, 30, 38)
    panel_bg = (30, 33, 42)
    card_bg = (40, 44, 56)
    border = (55, 60, 75)
    text = (245, 245, 245)
    subtext = (185, 190, 205)
    accent = (0, 165, 255)
    green = (80, 210, 140)
    red = (70, 70, 230)

    cv2.rectangle(bg, (0, 0), (UI_WIDTH, title_h), title_bg, -1)
    cv2.rectangle(bg, (0, title_h - 4), (UI_WIDTH, title_h), accent, -1)

    (tw, th), _ = cv2.getTextSize(DASHBOARD_TITLE, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    tx = (UI_WIDTH - tw) // 2
    ty = (title_h + th) // 2 + 6
    cv2.putText(bg, DASHBOARD_TITLE, (tx + 2, ty + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(bg, DASHBOARD_TITLE, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.6, text, 4, cv2.LINE_AA)

    content_y = title_h + pad
    content_h = UI_HEIGHT - title_h - pad * 2
    video_x = pad
    video_y = content_y
    video_w = UI_WIDTH - panel_w - pad * 3
    video_h = content_h
    panel_x = video_x + video_w + pad
    panel_y = content_y

    _draw_rounded_rect(bg, video_x, video_y, video_w, video_h, 18, (26, 28, 36))
    cv2.rectangle(bg, (video_x, video_y), (video_x + video_w, video_y + video_h), border, 2)
    cv2.putText(bg, "LIVE FEED", (video_x + 16, video_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, subtext, 2, cv2.LINE_AA)

    if video_frame is not None:
        vh, vw = video_frame.shape[:2]
        scale = min(video_w / float(vw), video_h / float(vh))
        rw = max(1, int(vw * scale))
        rh = max(1, int(vh * scale))
        resized = cv2.resize(video_frame, (rw, rh))
        ox = video_x + (video_w - rw) // 2
        oy = video_y + (video_h - rh) // 2
        bg[oy:oy + rh, ox:ox + rw] = resized

    _draw_rounded_rect(bg, panel_x, panel_y, panel_w, content_h, 18, panel_bg)
    cv2.rectangle(bg, (panel_x, panel_y), (panel_x + panel_w, panel_y + content_h), border, 2)

    ix = panel_x + 14
    iw = panel_w - 28
    card_h = 150
    gap = 16

    # IN card
    _draw_rounded_rect(bg, ix, panel_y + 14, iw, card_h, 18, card_bg)
    cv2.rectangle(bg, (ix, panel_y + 14), (ix + 10, panel_y + 14 + card_h), green, -1)
    cv2.putText(bg, "IN", (ix + 24, panel_y + 14 + 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, subtext, 2, cv2.LINE_AA)
    in_text = f"{int(in_count):02d}"
    (cw, ch), _ = cv2.getTextSize(in_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 7)
    cv2.putText(bg, in_text, (ix + (iw - cw) // 2, panel_y + 14 + (card_h + ch) // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, text, 7, cv2.LINE_AA)

    # OUT card
    out_y = panel_y + 14 + card_h + gap
    _draw_rounded_rect(bg, ix, out_y, iw, card_h, 18, card_bg)
    cv2.rectangle(bg, (ix, out_y), (ix + 10, out_y + card_h), red, -1)
    cv2.putText(bg, "OUT", (ix + 24, out_y + 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, subtext, 2, cv2.LINE_AA)
    out_text = f"{int(out_count):02d}"
    (cw, ch), _ = cv2.getTextSize(out_text, cv2.FONT_HERSHEY_SIMPLEX, 3.0, 7)
    cv2.putText(bg, out_text, (ix + (iw - cw) // 2, out_y + (card_h + ch) // 2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, text, 7, cv2.LINE_AA)

    # Info rows
    small_h = 70
    small_gap = 12
    info_y = out_y + card_h + gap
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    def draw_info_row(y, label, value, dot_color=None):
        _draw_rounded_rect(bg, ix, y, iw, small_h, 16, card_bg)
        if dot_color is not None:
            cv2.circle(bg, (ix + 26, y + small_h // 2), 10, dot_color, -1)
        cv2.putText(bg, label, (ix + 50, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, subtext, 2, cv2.LINE_AA)
        cv2.putText(bg, value, (ix + 50, y + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text, 2, cv2.LINE_AA)

    draw_info_row(info_y, "STATUS", status_text, dot_color=green if status_text.upper() in ("RUNNING", "ACTIVE") else red)
    draw_info_row(info_y + small_h + small_gap, "SOURCE", source_type)
    draw_info_row(info_y + (small_h + small_gap) * 2, "TIME", now)

    return bg


with open("coco.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n")


#function for detect person coordinate
def get_person_coordinates(frame):
    """
    Extracts the coordinates of the person bounding boxes from the YOLO model predictions.

    Args:
        frame: Input frame for object detection.

    Returns:
        list: List of person bounding box coordinates in the format [x1, y1, x2, y2].
    """
    results = model.predict(frame, imgsz=256, conf=0.35, verbose=False)
    a = results[0].boxes.data.detach().cpu()
    px = pd.DataFrame(a).astype("float")

    list_corr = []
    for index, row in px.iterrows():
        x1 = row[0]
        y1 = row[1]
        x2 = row[2]
        y2 = row[3]
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            list_corr.append([x1, y1, x2, y2])
    return list_corr


def people_counter(source):
    """
    Counts the number of people entering and exiting based on object tracking.
    """
    count = 0

    cap = _open_capture(source)

    if isinstance(source, int):
        source_type = "WEBCAM"
    else:
        s = str(source).lower()
        if s.startswith("rtsp://") or s.startswith("rtsps://"):
            source_type = "RTSP"
        elif s.startswith("http://") or s.startswith("https://"):
            source_type = "URL"
        else:
            source_type = "VIDEO"

    writer = None
    heatmap_accum = None
    ct = CentroidTracker(maxDisappeared=40, maxDistance=40)
    trackableObjects = {}

    # Initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # Initialize empty lists to store the counting data
    total = []
    move_out = []
    move_in = []

    fps = FPS().start()

    cv2.namedWindow("People Count", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("People Count", 1200, 700)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 5 != 0:
            continue

        frame = cv2.resize(frame, (640, 360))

        if writer is None:
            (H, W) = frame.shape[:2]
            if heatmap_accum is None:
                heatmap_accum = np.zeros((H, W), dtype=np.float32)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter('Final_output.mp4', fourcc, 30, (UI_WIDTH, UI_HEIGHT), True)

        per_corr = get_person_coordinates(frame)

        rects = []
        for bbox in per_corr:
            x1, y1, x2, y2 = bbox
            rects.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)

        cv2.line(frame, (0, H // 2 - 10), (W, H // 2 - 10), (0, 0, 0), 2)

        objects = ct.update(rects)

        for (objectID, centroid) in objects.items():
            to = trackableObjects.get(objectID)

            if heatmap_accum is not None and (count % 2 == 0):
                update_heatmap(heatmap_accum, int(centroid[0]), int(centroid[1]), radius=12, value=1.5)

            if to is None:
                to = TrackableObject(objectID, centroid)
            else:
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                if not to.counted:
                    if direction < 0 and centroid[1] < H // 2 - 20:
                        totalUp += 1
                        move_out.append(totalUp)
                        to.counted = True
                    elif direction > 0 and centroid[1] > (H // 2 + 4):
                        totalDown += 1
                        move_in.append(totalDown)
                        to.counted = True

                        total = []
                        total.append(len(move_in) - len(move_out))

            trackableObjects[objectID] = to

            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

        if heatmap_accum is not None:
            heatmap_blur = cv2.GaussianBlur(heatmap_accum, (0, 0), 3)
            heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_norm.astype(np.uint8)
            _, heatmap_uint8 = cv2.threshold(heatmap_uint8, 10, 255, cv2.THRESH_TOZERO)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_color[heatmap_uint8 == 0] = 0
            frame = cv2.addWeighted(frame, 1.0, heatmap_color, 0.35, 0)

        frame = compose_dashboard_frame(
            video_frame=frame,
            in_count=totalUp,
            out_count=totalDown,
            status_text="RUNNING",
            source_type=source_type,
        )

        writer.write(frame)
        cv2.imshow("People Count", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        totalFrames += 1
        fps.update()

        end_time = time.time()
        num_seconds = (end_time - start_time)
        if num_seconds > 28800:
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    fps.stop()
    logger.info("Elapsed time: {:.2f}".format(fps.elapsed()))
    logger.info("Approx. FPS: {:.2f}".format(fps.fps()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 people counting (video file, webcam index, or IP stream URL)")
    parser.add_argument(
        "--source",
        "-s",
        default="Input/input.mp4",
        help="Video source: path to file, webcam index (e.g. 0), or RTSP/HTTP URL",
    )
    args = parser.parse_args()

    source = _parse_source(args.source)
    logger.info(f"Starting source: {source}")
    people_counter(source)
    

## Apply threading also

# def start_people_counter():
#     t1 = threading.Thread(target=people_counter)
#     t1.start()


# if __name__ == "__main__":
#     start_people_counter()




