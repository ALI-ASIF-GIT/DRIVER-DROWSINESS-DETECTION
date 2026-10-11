"""Run drowsiness detection on a single image.

Usage:
    python src/detect_image.py --source path/to/image.jpg --weights weights/best.pt
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_detections


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="path to input image")
    p.add_argument("--weights", default="weights/best.pt", help="path to trained weights")
    p.add_argument("--output", default="outputs/result.jpg", help="where to save the annotated image")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--show", action="store_true", help="display the result in a window")
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.source)
    if img is None:
        raise SystemExit(f"Could not read image: {args.source}")

    model = YOLO(args.weights)
    for result in model(img, conf=args.conf, verbose=False):
        draw_detections(img, result, model.names)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")

    if args.show:
        cv2.imshow("Driver Drowsiness Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
