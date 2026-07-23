import cv2
import easyocr
import re

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Load full image (NOT cropped plate)
img = cv2.imread(r"C:/Users/hplap/OneDrive/Documents/Desktop/yolov10-custom-object-detection/detected_plate_10.jpg")

if img is None:
    print("Image not found")
    exit()

# Resize image slightly for better OCR
img = cv2.resize(img, None, fx=1.5, fy=1.5)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Improve contrast
gray = cv2.bilateralFilter(gray, 11, 17, 17)

# Run OCR on the FULL image
results = reader.readtext(gray, detail=0)

plate_candidates = []

for text in results:
    # Clean text
    cleaned = re.sub('[^A-Z0-9]', '', text.upper())

    # Indian plates usually length 6–12
    if 6 <= len(cleaned) <= 12:
        plate_candidates.append(cleaned)

# Print results
if plate_candidates:
    for plate in plate_candidates:
        print("Detected Number Plate:", plate)
else:
    print("No plate detected")

# Show image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()