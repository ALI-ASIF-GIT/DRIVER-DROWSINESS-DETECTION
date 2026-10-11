"""Run live drowsiness detection from a webcam (press 'q' to quit).

Usage:
    python src/detect_webcam.py --weights weights/best.pt
    python src/detect_webcam.py --weights weights/best.pt --output outputs/webcam.avi
"""
import argparse

from utils import run_stream


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--weights", default="weights/best.pt", help="path to trained weights")
    p.add_argument("--output", default=None, help="optional path to record the annotated stream")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_stream(a.source, a.weights, a.conf, a.output)
