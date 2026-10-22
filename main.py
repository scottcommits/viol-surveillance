import cv2
from ultralytics import YOLO

# Pre-trained object detection model
model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture("VIRAT_S_010204_05_000856_000890.mp4")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    # Detect objects in this frame
    results = model(frame)

    # Display detections
    annotated_frame = results[0].plot()
    cv2.imshow("Footage Analysis", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()