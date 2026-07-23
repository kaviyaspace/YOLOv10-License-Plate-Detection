import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

detected_plates = set()


def preprocess_for_ocr(plate_crop):
    """
    Takes a raw BGR image crop of a license plate and applies
    transformations to optimize it for EasyOCR.
    """
    # 1. Resize (Scale up by 2x using Cubic Interpolation for smoothness)
    resized = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # 3. Apply Gaussian Blur to reduce background noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Binarization (Otsu's Thresholding)
    # The image becomes pure black and white.
    # cv2.THRESH_OTSU automatically finds the best threshold value.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Morphological Operations (Closing)
    # Uses a tiny 2x2 grid to close small gaps in the character strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed_plate = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Note: Sometimes license plates have dark text on a light background,
    # and sometimes light text on dark. EasyOCR handles both well, but
    # ensuring high contrast is key.

    return processed_plate


# 1. Initialize Models
print("Loading YOLO model...")
yolo_model = YOLO(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/YOLOv10-License-Plate-Detection/best (2).pt")

print("Loading OCR model...")
# Initialize EasyOCR (set gpu=True if you have a local NVIDIA GPU, otherwise False)
reader = easyocr.Reader(["en"], gpu=False)

# 2. Open Video Capture (Change to your video file name, or 0 for webcam)
cap = cv2.VideoCapture(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/YOLOv10-License-Plate-Detection/traffic7.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO detection on the frame
    results = yolo_model(frame)

    # 4. Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box around the license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 5. Crop the license plate from the frame
            plate_crop = frame[y1:y2, x1:x2]

            # Check if the crop is valid (prevents crashes on weird edge-case detections)
            if plate_crop.size == 0:
                continue

            # Apply our new pre-processing pipeline
            processed_plate = preprocess_for_ocr(plate_crop)

            # (Optional Debugging) Show the processed plate in a separate window
            # to see exactly what EasyOCR is looking at
            cv2.imshow("Processed Plate", processed_plate)

            # 6. Read text using EasyOCR (feed it the processed image)
            ocr_results = reader.readtext(processed_plate)

            # Extract and display the text
            for bbox, text, prob in ocr_results:
                    clean_text = "".join(e for e in text if e.isalnum())
                    clean_text = clean_text.upper()

                    print("OCR:", clean_text, "Confidence:", prob)

                    if clean_text:
                         detected_plates.add(clean_text)
                         
                         if prob > 0.3:
                             cv2.putText(
                                 frame,
                                 clean_text,
                                 (x1, y1 - 10),
                                 cv2.FONT_HERSHEY_SIMPLEX,
                                 0.9,
                                (0, 255, 0),
                                 2,
                                  )

                    print("OCR:", clean_text, "Confidence:", prob)

    # Display the output
    cv2.imshow("License Plate Detection & OCR", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFINAL DETECTED NUMBER PLATES:\n")

if len(detected_plates) == 0:
    print("No plates detected.")
else:
    for plate in detected_plates:
        print(plate)