import cv2
from ultralytics import YOLO
import csv 

model = YOLO("YOLOv10-License-Plate-Detection/runs/detect/train2/weights/best.pt")

cap = cv2.VideoCapture(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/YOLOv10-License-Plate-Detection/traffic7.mp4")

if not cap.isOpened():
    print("Video not opened")
    exit()

file=open("detected_plates.csv","w", newline="")
writer = csv.writer(file)
writer.writerow(["Frame", "Detections"])

frame_number = 0

while True:
    ret, frame = cap.read()

    frame_number += 1

    if not ret:
        break

    h,w, _ = frame.shape
    frame = frame[int(h*0.4):h, 0:w]

    results = model(frame, conf=0.5, imgsz=1920)

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:

        x1,y1,x2,y2 = map(int, box)

        if y1 < frame.shape[0] * 0.4:
            continue

        cv2.rectangle(frame,(x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(frame, "plate",(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0), 2)

        plate = frame[y1:y2, x1:x2]

        if plate.size > 0:
            
            cv2.imwrite(f"detected_plate_{frame_number}.jpg", plate)

    cv2.imshow("Detections",frame)

    detections = len(results[0].boxes)

    writer.writerow([frame_number, detections])

    print("Detections:", len(results[0].boxes))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

file.close()

cap.release()
cv2.destroyAllWindows()