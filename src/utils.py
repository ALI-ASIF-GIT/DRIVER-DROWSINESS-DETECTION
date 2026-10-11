"""Shared helpers for the driver drowsiness detection scripts."""
from pathlib import Path

import cv2
from ultralytics import YOLO

# BGR colour per class
COLORS = {
    "awake": (255, 0, 0),
    "calling": (0, 255, 0),
    "chatting": (0, 255, 255),
    "closed_eye": (255, 0, 255),
    "drink": (255, 255, 0),
    "drowsy": (0, 0, 255),
    "eating": (255, 140, 0),
    "no_yawn": (128, 0, 128),
    "open_eye": (0, 255, 255),
    "smoking": (0, 165, 255),
    "yawn": (0, 0, 0),
}
DEFAULT_COLOR = (255, 255, 255)


def draw_detections(frame, result, names):
    """Draw boxes and 'class conf' labels for one YOLO result on a frame."""
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = names[int(box.cls[0])]
        color = COLORS.get(name, DEFAULT_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {float(box.conf[0]):.2f}", (x1, max(y1 - 10, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def run_stream(source, weights, conf=0.25, output=None, window="Driver Drowsiness Detection"):
    """Run detection on a video file or webcam index. Press 'q' to quit."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open source: {source}")

    model = YOLO(weights)
    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 10
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for result in model(frame, conf=conf, verbose=False):
            draw_detections(frame, result, model.names)
        if writer:
            writer.write(frame)
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved: {output}")
    cv2.destroyAllWindows()
