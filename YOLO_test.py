from ultralytics import YOLO
import cv2

# Load the image
img = cv2.imread(r'C:\Users\ASIF ALI\1. FINAL SEM PROJECT\DATASET IMG\d35.jpg')

# Initialize the model
model = YOLO("C:/Users/ASIF ALI/1. FINAL SEM PROJECT/Drive Drowsiness Detection.v3i.yolov8/runs/detect/train/weights/best.pt")
classNames = ["awake", "calling", "chatting", "closed_eye", "drink", 
              "drowsy", "eating", "no_yawn", "open_eye", "smoking", "yawn"]

# Perform detection
results = model(img, stream=True)

for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = round(float(box.conf[0]), 2)  # Convert tensor to float and round
        cls = int(box.cls[0])
        class_name = classNames[cls]
        label = f'{class_name} {conf}'
        
        # Define colors for different classes
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
        
        # Get color for current class
        color = colors.get(class_name, (255, 255, 255))  # Default to white if class color is not defined
        
        # Draw bounding box with color and reduced thickness
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

# Save the output image
cv2.imwrite('C:/Users/ASIF ALI/1. FINAL SEM PROJECT/Outputs/Images/IMGoutput.jpg', img)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
