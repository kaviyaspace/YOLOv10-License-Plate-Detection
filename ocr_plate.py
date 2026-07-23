import cv2
import easyocr
from ultralytics import YOLO

model = YOLO("YOLOv10-License-plate-Detection/runs/detect/train2/weights/best.pt")


img = cv2.imread("YOLOv10-License-Plate-Detection\car1.jpg")

reader = easyocr.Reader(['en'])

results = model(img, conf=0.2, imgsz=1280)

for result in results:

    boxes = result.boxes.xyxy.cpu().numpy()

if len(boxes) > 0:
    
    box = boxes[0]
    
    x1, y1, x2, y2 = map(int, box)

    plate = img[y1:y2, x1:x2]

    text = reader.readtext(plate)

    plate_text = "".join([t[1] for t in text])

    print("Detected Number Plate:", plate_text)

    cv2.rectangle(img,
                      (x1, y1),
                      (x2, y2),
                      (0,255,0),   
                      3)

    cv2.putText(img,
                    plate_text,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),   
                    2)

cv2.imshow("Number Plate Detection + OCR", img)

cv2.waitKey(0)

cv2.destroyAllWindows()