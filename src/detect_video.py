"""Run drowsiness detection on a video file (press 'q' to quit).

Usage:
    python src/detect_video.py --source path/to/video.mp4 --weights weights/best.pt
"""
import argparse

from utils import run_stream


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="path to input video")
    p.add_argument("--weights", default="weights/best.pt", help="path to trained weights")
    p.add_argument("--output", default="outputs/result.avi", help="where to save the annotated video")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run_stream(a.source, a.weights, a.conf, a.output)
