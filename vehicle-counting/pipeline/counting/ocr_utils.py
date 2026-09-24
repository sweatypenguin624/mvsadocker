import cv2
import pytesseract
import re
from datetime import datetime

def extract_time_from_frame(frame, roi=None):
    if roi and len(roi) == 4:
        y1, y2, x1, x2 = roi
        h, w = frame.shape[:2]
        # ensure within bounds
        y1, y2 = max(0, y1), min(h, y2)
        x1, x2 = max(0, x1), min(w, x2)
        if y2 > y1 and x2 > x1:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame
    else:
        crop = frame
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    custom_config = r"--oem 3 --psm 7"
    text = pytesseract.image_to_string(gray, config=custom_config)
    
    match = re.search(r"(\d{2}:\d{2}:\d{2})", text)
    if match:
        return match.group(1)
    return None
