import cv2
import easyocr
import numpy as np
import re
from ultralytics import YOLO
from collections import Counter

# ✅ ADD HERE
valid_states = {
    "AP","AR","AS","BR","CG","CH","DD","DL","DN","GA","GJ","HP","HR",
    "JH","JK","KA","KL","LA","LD","MH","ML","MN","MP","MZ","NL","OD",
    "PB","PY","RJ","SK","TN","TR","TS","UK","UP","WB"
}

plate_votes = Counter()

def correct_plate(text):
    text = list(text)

    for i, c in enumerate(text):

        if i in [2, 3, len(text)-4, len(text)-3, len(text)-2, len(text)-1]:
            if c == 'O': text[i] = '0'
            if c == 'B': text[i] = '8'
            if c == 'I': text[i] = '1'
            if c == 'Z': text[i] = '2'

        else:
            if c == '0': text[i] = 'O'
            if c == '8': text[i] = 'B'
            if c == '1': text[i] = 'I'

    return "".join(text)
# Load YOLOv10 model
model = YOLO(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/YOLOv10-License-Plate-Detection/best (3).pt")

# Initialize OCR
reader = easyocr.Reader(['en'])

# Regex for Indian plates
plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

# Open video
cap = cv2.VideoCapture(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/YOLOv10-License-Plate-Detection/traffic7.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    results = model(frame)

    for result in results:

        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            # Crop number plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # ---------- PREPROCESSING ----------

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

            gray = cv2.bilateralFilter(gray, 11, 17, 17)

            edged = cv2.Canny(gray, 30, 200)

            # Resize for OCR
            gray = cv2.resize(gray, None, fx=4, fy=4)

            # ---------- OCR ----------

            result_ocr = reader.readtext(gray)

            for detection in result_ocr:
                plate_text = detection[1]

                # Clean text
                plate_text = plate_text.upper()
                plate_text = re.sub('[^A-Z0-9]', '', plate_text)
                plate_text = correct_plate(plate_text)

                # Check valid Indian plate
                if plate_pattern.match(plate_text):
                    state_code = plate_text[:2]
                    if state_code in valid_states:
                        plate_votes[plate_text] += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                    cv2.putText(
                        frame,
                        plate_text,
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2
                    )

    # Resize display
    frame = cv2.resize(frame, (1280,720))

    cv2.imshow("License Plate Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()

# ---------- FINAL OUTPUT ----------

print("\nFINAL DETECTED NUMBER PLATES:\n")

# Sort by frequency (most common first)
sorted_plates = sorted(plate_votes.items(), key=lambda x: x[1], reverse=True)

final_plates = []

for plate, count in sorted_plates:
    if count >= 7:  
        final_plates.append(plate)

for plate in final_plates:
    print(plate)

# Save
with open("detected_plates.txt", "w") as f:
    for plate in final_plates:
        f.write(plate + "\n")

print("\nPlates saved to detected_plates.txt")