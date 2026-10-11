<p align="center">
  <img src="assets/banner.svg" alt="Driver Drowsiness Detection" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-7C3AED" alt="YOLOv8">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/mAP%400.5-0.980-16A34A" alt="mAP">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<p align="center">
  Real-time detection of driver drowsiness and distraction cues from <b>images, videos and a live webcam</b>, using a fine-tuned <b>YOLOv8m</b> model.<br>
  B.Tech (CSE, AI) final-year project, Jamia Hamdard, 2024.
</p>

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#results">Results</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#limitations">Limitations</a> ·
  <a href="docs/Dissertation_Report.pdf">Full report (PDF)</a>
</p>

---

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="Live webcam detection demo" width="460">
</p>

## What it detects

| Group | Classes | Why it matters |
|---|---|---|
| **Drowsiness cues** | `awake`, `drowsy`, `open_eye`, `closed_eye`, `yawn`, `no_yawn` | Eye state, yawning and head posture are early signs of fatigue |
| **Distraction cues** | `calling`, `chatting`, `drink`, `eating`, `smoking` | Hands-off-wheel and attention-diverting behaviour |

> The validation split contains the six drowsiness classes only, so the metrics below cover those six.

## How it works

**Inference pipeline**

```mermaid
flowchart LR
    A(["Image / Video / Webcam"]) --> B["Frame capture<br/>OpenCV"]
    B --> C["YOLOv8m<br/>best.pt"]
    C --> D["Detections<br/>box + class + confidence"]
    D --> E{"confidence >= threshold?"}
    E -- yes --> F["Draw boxes and labels"]
    E -- no --> G["Discard"]
    F --> H(["Display / save output"])
```

**Training workflow**

```mermaid
flowchart LR
    A["Collect images<br/>(web sources)"] --> B["Annotate<br/>Roboflow"]
    B --> C["Export YOLOv8 format<br/>168 train / 36 val / 36 test"]
    C --> D["Fine-tune yolov8m.pt<br/>Colab GPU, up to 200 epochs"]
    D --> E["Evaluate<br/>mAP, P, R, F1"]
    E --> F[("best.pt")]
```

| Item | Value |
|---|---|
| Base model | `yolov8m.pt` (Ultralytics, COCO-pretrained) |
| Epochs / batch / image size | up to 200 / 32 / 640 |
| Dataset | 240 annotated images, 11 classes |
| Training hardware | Google Colab GPU |
| Local inference | Intel i5 10th gen, GTX 1650 Ti (4 GB) |

## Results

<p align="center">
  <img src="assets/metrics_overview.png" alt="Metrics overview" width="92%">
</p>

| Metric | Value | At confidence |
|---|---|---|
| mAP@0.5 (all classes) | **0.980** | n/a |
| Precision | **1.00** | 0.933 |
| Recall | **0.99** | 0.000 |
| F1-score | **0.95** | 0.459 |

*Precision, recall and F1 are peak values read from their respective confidence curves, so each is reported at a different threshold.*

<p align="center">
  <img src="assets/confusion_matrix_normalized.png" width="48%" alt="Normalized confusion matrix">
  <img src="assets/PR_curve.png" width="48%" alt="Precision-Recall curve">
</p>

<details>
<summary><b>More evaluation plots</b> (F1, precision, recall, training curves)</summary>
<br>
<p align="center">
  <img src="assets/F1_curve.png" width="48%" alt="F1 curve">
  <img src="assets/P_curve.png" width="48%" alt="Precision curve">
  <img src="assets/R_curve.png" width="48%" alt="Recall curve">
  <img src="assets/confusion_matrix.png" width="48%" alt="Confusion matrix">
</p>
<p align="center"><img src="assets/results.png" width="96%" alt="Training curves"></p>
</details>

## Limitations

- **Small dataset** (240 images, 36 for validation), so metrics may overstate real-world performance.
- **Frame-by-frame detection only**: no temporal logic (e.g. eyes closed for N seconds), so alerts depend on single-frame predictions.
- `closed_eye` is misclassified as `open_eye` in ~22% of cases (see normalized confusion matrix).
- Lower confidence on night / infrared video frames, and occasional background false positives (visible in the demo).

## Project Structure

```
driver-drowsiness-detection/
├── README.md
├── LICENSE
├── requirements.txt
├── data.yaml             # dataset config (11 classes)
├── src/
│   ├── detect_image.py
│   ├── detect_video.py
│   ├── detect_webcam.py
│   └── utils.py          # shared drawing / streaming helpers
├── notebooks/
│   └── YOLOv8_training.ipynb
├── weights/              # put best.pt here (download from Releases)
├── assets/               # banner, demo gif, curves, confusion matrices
└── docs/                 # dissertation report
```

## Getting Started

```bash
git clone https://github.com/ALI-ASIF-GIT/DRIVER-DROWSINESS-DETECTION.git
cd DRIVER-DROWSINESS-DETECTION
pip install -r requirements.txt
```

Download `best.pt` from the [Releases](../../releases) page and place it in `weights/`.

```bash
# Image
python src/detect_image.py --source path/to/image.jpg

# Video (press q to quit)
python src/detect_video.py --source path/to/video.mp4

# Live webcam (press q to quit)
python src/detect_webcam.py
```

Common options: `--weights weights/best.pt`, `--conf 0.25`, `--output <path>`. Results are saved to `outputs/` by default.

## Future Work

- [ ] Temporal drowsiness logic (blink duration / PERCLOS) with an audio alert
- [ ] Facial-landmark features (eye aspect ratio) to complement detection
- [ ] Web demo (Flask / Streamlit)
- [ ] Larger, more diverse dataset with proper night-time evaluation
- [ ] Export to ONNX / TFLite for edge deployment

## Author

**Asif Ali** | B.Tech CSE (AI), Jamia Hamdard
[LinkedIn](https://linkedin.com/in/asif-ali-083202208) · [GitHub](https://github.com/ALI-ASIF-GIT)

## License

MIT, see [LICENSE](LICENSE).
