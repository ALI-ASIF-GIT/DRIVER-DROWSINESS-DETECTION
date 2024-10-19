from ultralytics import YOLO
import cv2

def video_detection(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('C:/Users/ASIF ALI/1. FINAL SEM PROJECT/Outputs/Videos/Video_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    model = YOLO("C:/Users/ASIF ALI/1. FINAL SEM PROJECT/Drive Drowsiness Detection.v3i.yolov8/runs/detect/train/weights/best.pt")
    classNames = ["awake", "calling", "chatting", "closed_eye", "drink", 
                  "drowsy", "eating", "no_yawn", "open_eye", "smoking", "yawn"]

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                colors = {
                    "awake": (255, 0, 0),         # Blue
                    "calling": (0, 255, 0),       # Green
                    "chatting": (0, 255, 255),    # Yellow
                    "closed_eye": (255, 0, 255),  # Magenta
                    "drink": (255, 255, 0),       # Cyan
                    "drowsy": (0, 0, 255),        # Red
                    "eating": (255, 140, 0),      # Orange
                    "no_yawn": (128, 0, 128),     # Purple
                    "open_eye": (0, 255, 255),    # Light Blue
                    "smoking": (0, 165, 255),     # Light Orange
                    "yawn": (0, 0, 0)             # Black
                }
                color = colors.get(class_name, (255, 255, 255))

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out.write(img)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Specify the path to the directory containing the videos
video_path = r"C:\Users\ASIF ALI\1. FINAL SEM PROJECT\DATASET IMG\video\Drowsy Driving.mp4"

# Run the video detection
video_detection(video_path)