import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from openid.capture.detector import detect_document
from openid.capture.quality import detect_document_contour

def validate_capture_strict(frame: np.ndarray, doc_type: str = "passport") -> Optional[Dict[str, str]]:
    """
    Run strict local validation on the captured frame.
    Returns a dictionary with error info if validation fails, otherwise None.
    """
    # 1. Detect Document Bounds to ensure it's fully in frame
    h, w = frame.shape[:2]
    bbox = None
    yolo_used = False

    # Try YOLO first
    yolo_box, conf, _ = detect_document(frame)
    if yolo_box is not None and conf > 0.45:
        # YOLO boxes are already shrunk ~8% inside the frame in detector.py,
        # so they will always appear slightly inside the border — no crop check needed.
        bbox = yolo_box
        yolo_used = True
    else:
        # Fallback to contour
        contour, _ = detect_document_contour(frame, camera_mode=True)
        if contour is not None:
            x, y, cw, ch = cv2.boundingRect(contour)
            bbox = np.array([x, y, x + cw, y + ch])

    if bbox is None:
        return {
            "error": "DOCUMENT_NOT_FOUND",
            "action": "Ensure the document is clearly visible and placed within the frame."
        }

    x1, y1, x2, y2 = map(int, bbox)

    # Check 1: Cropped check removed as it was causing false positives.

    # Extract Document Region of Interest (ROI)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {
            "error": "DOCUMENT_INVALID",
            "action": "Capture failed. Please try again."
        }

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_area = roi.shape[0] * roi.shape[1]

    # Check 2: Glare Detection
    # Glare is typically saturated white pixels. Relaxed from 2% to 10% (0.10)
    _, glare_mask = cv2.threshold(roi_gray, 240, 255, cv2.THRESH_BINARY)
    glare_ratio = cv2.countNonZero(glare_mask) / roi_area
    if glare_ratio > 0.10: # More than 10% pure white glare
        return {
            "error": "LIGHT_GLARE_DETECTED",
            "action": "Tilt the document slightly or change lighting to avoid reflections and glare."
        }

    # Check 3: Shadow Detection
    # Very dark patches. Relaxed from 15% to 50% (0.50)
    _, shadow_mask = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV)
    shadow_ratio = cv2.countNonZero(shadow_mask) / roi_area
    if shadow_ratio > 0.50: # More than 50% dark shadows
        return {
            "error": "SHADOW_DETECTED",
            "action": "Ensure even lighting across the document. Avoid casting shadows with your hand or phone."
        }

    # Check 4: Finger Detection
    # Basic skin color thresholds in HSV. Relaxed from 5% to 25% (0.25)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(roi_hsv, lower_skin, upper_skin)
    skin_ratio = cv2.countNonZero(skin_mask) / roi_area
    if skin_ratio > 0.25: # More than 25% skin color
        return {
            "error": "FINGERS_DETECTED",
            "action": "Do not cover the document with your fingers. Hold it by the extreme edges or place it on a flat surface."
        }

    # Check 5: MRZ Visibility (If passport)
    if doc_type.lower() == "passport":
        if not check_mrz_presence(roi_gray):
            return {
                "error": "MRZ_NOT_VISIBLE",
                "action": "Ensure the bottom machine-readable zone (MRZ) is fully visible, sharp, and not covered."
            }

    return None


def check_mrz_presence(roi_gray: np.ndarray) -> bool:
    """
    Use morphological operations to detect the presence of MRZ lines at the bottom of the document.
    """
    h, w = roi_gray.shape
    
    # MRZ is usually in the bottom 35% of the document
    bottom_y = int(h * 0.65)
    bottom_roi = roi_gray[bottom_y:h, 0:w]
    
    # Resize for consistent processing
    bottom_roi = cv2.resize(bottom_roi, (600, int(600 * bottom_roi.shape[0] / bottom_roi.shape[1])))
    
    # Blackhat to find dark text on light background
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(bottom_roi, cv2.MORPH_BLACKHAT, rectKernel)
    
    # Compute Scharr gradient along x-axis
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    if maxVal == 0:
        return False
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    
    # Close gaps between characters to form lines
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    _, thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Another closing operation to solidify the lines
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mrz_lines_found = 0
    for c in contours:
        (x, y, cw, ch) = cv2.boundingRect(c)
        ar = cw / float(ch)
        
        # MRZ lines are very horizontal. Relaxed aspect ratio and width requirements.
        if ar > 3.0 and cw > bottom_roi.shape[1] * 0.4:
            mrz_lines_found += 1

    # Passports usually have 2 lines, some IDs have 3. We look for at least 1 very strong candidate or 2.
    # We will be slightly lenient (>= 1) because lighting can cause the lines to merge into a single thick block in morphology.
    return mrz_lines_found >= 1
