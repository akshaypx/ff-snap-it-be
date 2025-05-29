import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.6, stream=True)
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow("YOLOv8 Live Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
