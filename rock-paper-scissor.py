import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture(0)
THRESHOLD=0.75

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=THRESHOLD)
    # print(results[0])
    annotated_frame = results[0].plot()

    cv2.imshow("Rock Paper Scissors - YOLOv8", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()