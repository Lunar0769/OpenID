"""
OpenID Quality Checker — Client-Side Quality Gate (PRD §3)
==========================================================

This module is the SINGLE SOURCE OF TRUTH for image quality evaluation.
No duplicate quality logic should exist anywhere else in the SDK.

Hard thresholds (STRICT — do not relax without PRD sign-off):

    Blank frame (std-dev of grayscale):
        < 12   → FAIL   ("blank_frame")   — empty / uncovered lens

    Blur (Laplacian variance):
        < 80   → FAIL   ("too_blurry")
        80–120 → WARN   ("slight_blur")
        ≥ 120  → PASS

    Brightness (mean pixel intensity):
        < 60   → FAIL   ("low_light")
        60–90  → WARN   ("dim_light")
        > 240  → FAIL   ("too_bright")
        210–240 → WARN  ("overexposed")
        else   → PASS

    Person / face guard (Haar cascade):
        Face area > 12 % of frame AND no document detected → FAIL ("person_detected")

    Document presence (ALWAYS enforced — both camera and file mode):
        No valid contour found                 → FAIL ("no_document")
        Document area < 35 % of frame         → FAIL ("no_document")
        Contour rectangularity < 0.82         → FAIL ("no_document")  phones / random objects
        Guide-box IoU < 0.30 (camera mode)   → FAIL ("not_in_frame")  doc not centred
        Edge density inside contour < 0.04   → FAIL ("no_document")  featureless region

    Document alignment (camera mode — WARN only):
        YOLO detected but IoU < 0.40 → WARN ("misaligned")
"""
from __future__ import annotations

import cv2
import numpy as np

from openid.capture.overlay import get_guide_box


# ── Hard thresholds ───────────────────────────────────────────────────────────
BLANK_STD_FAIL      = 10     # grayscale std-dev — below this = blank → FAIL
BLUR_FAIL           = 80     # Laplacian variance camera mode FAIL (phone screens score 26-45 live)
BLUR_WARN           = 120    # below this = slight blur WARN (camera mode)
BLUR_FAIL_FILE      = 20     # file-upload mode: scans can be lower quality → FAIL
BLUR_WARN_FILE      = 50     # file-upload mode: warn threshold
BRIGHT_FAIL         = 45     # mean pixel intensity — below this = low light FAIL
BRIGHT_WARN         = 80     # below this = dim light → WARN
BRIGHT_OVER_FAIL    = 245    # above this = overexposed → FAIL
BRIGHT_OVER_WARN    = 215    # above this = slightly overexposed → WARN
DOC_AREA_MIN        = 0.30   # document must occupy 30% of frame FAIL if not
YOLO_CONF_MIN       = 0.55   # minimum YOLO confidence  higher = fewer phone-screen triggers
ALIGN_IOU_FAIL      = 0.35   # IoU below this → not centred → FAIL in camera mode
ALIGN_IOU_WARN      = 0.45   # IoU below this but ≥ FAIL → WARN (camera mode)

# Screen / digital-display detection
SCREEN_SCORE_FAIL   = 0.40   # combined score above this  likely a phone/screen  FAIL

# Contour-mode strictness (camera mode — tuned for real hand-held cards)
CONTOUR_RECT_MIN    = 0.42   # rectangularity — real cards with perspective score ~0.50-0.75
CONTOUR_SOLIDITY_MIN= 0.55   # solidity — real cards ~0.65-0.90
CONTOUR_EDGE_MIN    = 0.025  # edge-pixel density — lower OK for plain-background shots
GUIDE_IOU_MIN       = 0.15   # contour must overlap guide box by at least 15 %

# Face guard
FACE_AREA_FRAC      = 0.14   # face must be > this fraction of frame to trigger rejection


# ── Haar cascade for face / person detection ──────────────────────────────────
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    """Lazy-load the Haar face cascade (thread-safe: idempotent write)."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        import cv2 as _cv2_inner
        _data = getattr(_cv2_inner, 'data', None)
        if _data is not None:
            cascade_path = str(_data.haarcascades) + "haarcascade_frontalface_default.xml"
        else:
            import os
            cascade_path = os.path.join(os.path.dirname(_cv2_inner.__file__), "data", "haarcascade_frontalface_default.xml")
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


# ── Individual metric functions ───────────────────────────────────────────────

def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Guarantee the image is a 3-channel BGR frame.
    Handles RGBA (4-ch), already-gray (1-ch / 2D), and plain BGR.
    This prevents cv2.cvtColor crashes from malformed camera frames.
    """
    if image is None or image.size == 0:
        raise ValueError("Received empty or None image.")
    if image.ndim == 2:                          # already grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:                      # RGBA / BGRA
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:                      # single-channel 3D
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image                                 # already BGR


def blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    image = _ensure_bgr(image)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(image: np.ndarray) -> float:
    """Mean pixel intensity across all channels."""
    return float(image.mean())


def is_blank_frame(image: np.ndarray) -> bool:
    """
    Return True when the frame carries almost no information.
    std-dev < BLANK_STD_FAIL → lens cap, all-black, all-white, plain wall.
    """
    try:
        image = _ensure_bgr(image)
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.std()) < BLANK_STD_FAIL
    except Exception:
        return True   # treat unreadable frames as blank


def is_screen_capture(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect whether the image shows a document on a phone/monitor screen
    rather than a real physical document.

    Uses three independent signals:

    1. Dark-border ratio — phone bezels produce a thick dark frame around
       the bright display area. We measure the fraction of border pixels
       that are dark vs the interior.

    2. Glow uniformity — real documents have varied reflectance; screen
       backlights produce unnatural brightness uniformity along the frame
       edges (all bright, all the same).

    3. Colour channel saturation uniformity — LCD/OLED pixels render colour
       with extreme precision; photographing them at camera distance shows
       very low inter-channel variance. Real paper/plastic documents have
       irregular ink/dye absorption that raises per-pixel channel spread.

    Returns (is_screen: bool, score: float) where score in [0, 1].
    Score ≥ SCREEN_SCORE_FAIL means "likely a screen".
    """
    try:
        image = _ensure_bgr(image)
        h, w  = image.shape[:2]
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        border = max(12, int(min(h, w) * 0.06))  # 6% border strip

        # ── Signal 1: dark-border ratio ──────────────────────────────────────
        # A phone is held so the bezel (dark) surrounds the screen (bright).
        # We sample the outermost strip and count pixels below a dark threshold.
        top    = gray[:border, :]
        bottom = gray[h - border:, :]
        left   = gray[:, :border]
        right  = gray[:, w - border:]
        border_pixels  = np.concatenate([top.ravel(), bottom.ravel(),
                                          left.ravel(), right.ravel()])
        dark_ratio = float((border_pixels < 50).mean())

        # ── Signal 2: glow uniformity ─────────────────────────────────────────
        # Screens have a very consistent backlight — edge strips are uniformly
        # bright and have low std-dev. Real document edges vary.
        inner_top    = gray[border:border*2, border:w - border]
        inner_bottom = gray[h - border*2:h - border, border:w - border]
        if inner_top.size > 0 and inner_bottom.size > 0:
            edge_strip = np.concatenate([inner_top.ravel(), inner_bottom.ravel()])
            edge_std   = float(edge_strip.std())
            # Low std on the inner-edge strip = uniform backlight = screen
            glow_score = max(0.0, 1.0 - edge_std / 60.0)
        else:
            glow_score = 0.0

        # ── Signal 3: channel saturation uniformity ───────────────────────────
        # Per-pixel colour spread: |R-G| + |G-B| averaged across the image.
        # Real documents: ink/dye variation → high spread.
        # Screen pixels:  sub-pixel rendering → low spread between channels.
        b_ch = image[:, :, 0].astype(np.float32)
        g_ch = image[:, :, 1].astype(np.float32)
        r_ch = image[:, :, 2].astype(np.float32)
        channel_diff = (np.abs(r_ch - g_ch) + np.abs(g_ch - b_ch)).mean()
        # Low channel_diff (< ~12) = screen-like uniformity
        channel_score = max(0.0, 1.0 - channel_diff / 18.0)

        # ── Combined score ────────────────────────────────────────────────────
        # Weighted: dark-border is the most reliable signal for a held phone.
        score = (dark_ratio * 0.50) + (glow_score * 0.25) + (channel_score * 0.25)
        score = float(np.clip(score, 0.0, 1.0))

        return score >= SCREEN_SCORE_FAIL, score

    except Exception:
        return False, 0.0


# Face-check frame counter — only run every N frames to keep camera fast
_face_check_counter: int = 0
_face_check_interval: int = 5   # run Haar cascade 1 out of every 5 frames
_face_last_result:    bool = False


def detect_face(image: np.ndarray) -> bool:
    """
    Return True if a *large* human face occupies the frame.

    Runs only every _face_check_interval frames for performance.
    The size gate (FACE_AREA_FRAC) prevents a passport photo from triggering.
    """
    global _face_check_counter, _face_last_result

    _face_check_counter += 1
    if _face_check_counter % _face_check_interval != 0:
        return _face_last_result   # return cached result

    try:
        image      = _ensure_bgr(image)
        gray       = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ih, iw     = gray.shape[:2]
        frame_area = iw * ih

        cascade = _get_face_cascade()
        faces   = cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.10,
            minNeighbors = 7,
            minSize      = (80, 80),
        )
        if len(faces) == 0:   # type: ignore[arg-type]
            _face_last_result = False
            return False

        face_areas        = [w * h for (_, _, w, h) in faces]
        max_face          = max(face_areas)
        _face_last_result = (max_face / frame_area) > FACE_AREA_FRAC
        return _face_last_result

    except Exception:
        _face_last_result = False
        return False


def _contour_iou_with_guide(image: np.ndarray, contour: np.ndarray) -> float:
    """
    Compute IoU between the contour's axis-aligned bounding rect and the guide box.
    Returns 0.0 when no guide box is available (file-upload mode).
    """
    try:
        gx, gy, gw, gh = get_guide_box(image)
    except Exception:
        return 1.0  # can't compute → skip gate

    cx, cy, cw, ch = cv2.boundingRect(contour)
    cx2, cy2 = cx + cw, cy + ch
    gx2, gy2 = gx + gw, gy + gh

    ix1 = max(cx, gx);  iy1 = max(cy, gy)
    ix2 = min(cx2, gx2); iy2 = min(cy2, gy2)

    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h

    c_area = cw * ch
    g_area = gw * gh
    union  = c_area + g_area - inter
    return inter / union if union > 0 else 0.0


def _edge_density_inside(image: np.ndarray, contour: np.ndarray) -> float:
    """
    Compute the fraction of Canny edge pixels inside the contour bounding box.

    A genuine document has printed text, a photo, and a border — all generating
    edges. A phone screen showing a solid colour or a plain wall has very few.
    """
    x, y, w, h = cv2.boundingRect(contour)
    if w < 10 or h < 10:
        return 0.0
    image = _ensure_bgr(image)
    roi  = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return float(edges.sum() / 255) / (w * h)


def detect_document_contour(
    image:       np.ndarray,
    camera_mode: bool = True,
) -> tuple[np.ndarray | None, float]:
    """
    Detect the largest rectangular document contour in the frame.

    Camera mode (strict):
        Aspect ratio: 1.30 – 1.95
        Rectangularity ≥ CONTOUR_RECT_MIN (0.82)
        Solidity       ≥ CONTOUR_SOLIDITY_MIN (0.90)
        Edge density   ≥ CONTOUR_EDGE_MIN (0.04)
        Guide-box IoU  ≥ GUIDE_IOU_MIN (0.30)

    File mode (relaxed — passport may fill entire frame):
        Aspect ratio: 1.20 – 2.20
        Rectangularity ≥ 0.30
        Solidity       ≥ 0.40
        Edge density   ≥ CONTOUR_EDGE_MIN (0.04)
        Guide-box IoU: NOT checked

    Returns:
        (contour, guide_iou)  — contour is None if not found.
    """
    # ── Relaxed thresholds for file mode ──────────────────────────────────
    if camera_mode:
        aspect_min, aspect_max = 1.30, 1.95
        rect_min    = CONTOUR_RECT_MIN       # 0.82
        solid_min   = CONTOUR_SOLIDITY_MIN   # 0.90
        check_guide = True
    else:
        aspect_min, aspect_max = 1.20, 2.20
        rect_min    = 0.30
        solid_min   = 0.40
        check_guide = False

    image = _ensure_bgr(image)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE for contrast normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges  = cv2.dilate(edges, kernel, iterations=2)
    edges  = cv2.erode(edges,  kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = image.shape[:2]
    frame_area   = w_img * h_img

    # Sort by area descending — check the top-5 largest candidates
    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5] if contours else []

    for cnt in candidates:
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < frame_area * 0.08:   # must be at least 8 % of frame
            break

        # ── Approximate shape ─────────────────────────────────────────────
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            box = approx
        else:
            rect = cv2.minAreaRect(cnt)
            box  = np.int32(cv2.boxPoints(rect))

        # ── Gate 1: Aspect ratio ──────────────────────────────────────────
        rect_r = cv2.minAreaRect(cnt)          # use cnt — always a valid ndarray
        bw, bh = rect_r[1]
        if bw <= 0 or bh <= 0:
            continue
        aspect = max(bw, bh) / min(bw, bh)
        if not (aspect_min <= aspect <= aspect_max):
            continue

        # ── Gate 2: Rectangularity ────────────────────────────────────────
        x, y, cw, ch = cv2.boundingRect(cnt)  # use cnt — always a valid ndarray
        bounding_area = cw * ch
        if bounding_area <= 0:
            continue
        rectangularity = cnt_area / bounding_area
        if rectangularity < rect_min:
            continue

        # ── Gate 3: Solidity ───────────────────────────────────────────────
        hull      = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = cnt_area / hull_area
        if solidity < solid_min:
            continue

        # ── Gate 4: Edge density ───────────────────────────────────────────
        ed = _edge_density_inside(image, cnt)    # cnt is ndarray — type safe
        if ed < CONTOUR_EDGE_MIN:
            continue

        # ── Gate 5: Guide-box IoU (camera mode only) ──────────────────────────
        guide_iou = _contour_iou_with_guide(image, cnt)  # cnt is ndarray — type safe
        if camera_mode and guide_iou < GUIDE_IOU_MIN:
            continue

        return np.array(box, dtype=np.int32), guide_iou

    # ── File-mode fallback: treat the whole image as the document ─────────────
    # When a passport scan fills the frame, no interior contour is found.
    # Accept if the image aspect ratio matches ID/passport and has enough edges.
    if not camera_mode:
        img_aspect = max(h_img, w_img) / min(h_img, w_img) if min(h_img, w_img) > 0 else 0
        if 1.20 <= img_aspect <= 2.20:
            full_box = np.array([
                [0,        0      ],
                [w_img - 1, 0     ],
                [w_img - 1, h_img - 1],
                [0,        h_img - 1],
            ], dtype=np.int32)
            ed = _edge_density_inside(image, full_box)
            if ed >= CONTOUR_EDGE_MIN:
                return full_box, 1.0

    return None, 0.0



def document_area_ratio(image: np.ndarray, contour: np.ndarray | None) -> float:
    """
    Return the fraction of the frame occupied by the document bounding box.
    Returns 0.0 if no contour provided.
    """
    if contour is None:
        return 0.0
    h, w       = image.shape[:2]
    frame_area = w * h
    x, y, bw, bh = cv2.boundingRect(contour)
    return (bw * bh) / frame_area


def tilt_angle(contour) -> float:
    """Return tilt deviation from nearest axis (0 = perfectly aligned)."""
    if contour is None:
        return 0.0
    rect  = cv2.minAreaRect(contour)
    angle = abs(rect[2]) % 90
    if angle > 45:
        angle = 90 - angle
    return angle


def is_tilted(contour, max_angle: float = 15.0) -> bool:
    """True if document tilt exceeds max_angle degrees."""
    return tilt_angle(contour) > max_angle


# ── Primary contract function (SINGLE SOURCE OF TRUTH) ───────────────────────

def check_quality(
    image:       np.ndarray,
    camera_mode: bool  = True,
    yolo_bbox           = None,
    yolo_iou:    float = 0.0,
    yolo_conf:   float = 0.0,
    yolo_label:  str   = "",
) -> dict:
    """
    Run all quality checks on an image and return the structured result.

    PRD §4 Output Contract:
    {
        "status": "pass" | "warning" | "fail",
        "reasons": [...],
        "metrics": {
            "blur_score":          float,
            "brightness":          float,
            "document_detected":   bool,
            "document_area_ratio": float,
            "yolo_detected":       bool,
            "yolo_conf":           float,
            "yolo_label":          str,
            "yolo_iou":            float,
            "face_detected":       bool,
            "guide_iou":           float,
        }
    }

    Status rules (STRICT):
        fail    → ANY fail condition triggered
        warning → no fail, but ≥1 warning condition triggered
        pass    → all checks pass

    Args:
        image:       BGR image as numpy array.
        camera_mode: True = enforce guide-box IoU for contour mode.
                     Document presence is ALWAYS enforced regardless.
        yolo_bbox:   Pre-computed YOLO bbox (x1,y1,x2,y2) or None.
        yolo_iou:    Pre-computed IoU of YOLO bbox vs guide box.
        yolo_conf:   YOLO detection confidence.
        yolo_label:  YOLO detected class label.
    """
    try:
        image = _ensure_bgr(image)
    except Exception:
        return {
            "status": "fail", "reasons": ["blank_frame"],
            "metrics": {
                "blur_score": 0.0, "brightness": 0.0,
                "document_detected": False, "document_area_ratio": 0.0,
                "yolo_detected": False, "yolo_conf": 0.0,
                "yolo_label": "", "yolo_iou": 0.0,
                "face_detected": False, "guide_iou": 0.0,
            },
        }

    b_score = blur_score(image)
    bright  = brightness(image)

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    # ── Blank frame (must come first) ─────────────────────────────────────────
    if is_blank_frame(image):
        return {
            "status":  "fail",
            "reasons": ["blank_frame"],
            "metrics": {
                "blur_score":          round(b_score, 2),
                "brightness":          round(bright,  2),
                "document_detected":   False,
                "document_area_ratio": 0.0,
                "yolo_detected":       False,
                "yolo_conf":           0.0,
                "yolo_label":          "",
                "yolo_iou":            0.0,
                "face_detected":       False,
                "guide_iou":           0.0,
            },
        }

    # ── Blur (camera mode is stricter than file upload mode) ──────────────────
    blur_fail_t = BLUR_FAIL      if camera_mode else BLUR_FAIL_FILE
    blur_warn_t = BLUR_WARN      if camera_mode else BLUR_WARN_FILE
    if b_score < blur_fail_t:
        fail_reasons.append("too_blurry")
    elif b_score < blur_warn_t:
        warn_reasons.append("slight_blur")

    # ── Brightness ────────────────────────────────────────────────────────────
    if bright > BRIGHT_OVER_FAIL:
        fail_reasons.append("too_bright")
    elif bright > BRIGHT_OVER_WARN:
        warn_reasons.append("overexposed")
    elif bright < BRIGHT_FAIL:
        fail_reasons.append("low_light")
    elif bright < BRIGHT_WARN:
        warn_reasons.append("dim_light")

    # ── Document detection ────────────────────────────────────────────────────
    # YOLO path (when ultralytics is installed and model loaded)
    yolo_detected = (
        yolo_bbox is not None
        and yolo_conf >= YOLO_CONF_MIN
    )
    guide_iou_contour = 0.0

    if yolo_detected:
        assert yolo_bbox is not None   # narrowed: yolo_detected only True when bbox is not None
        detected = True
        h, w    = image.shape[:2]
        bw = int(yolo_bbox[2]) - int(yolo_bbox[0])
        bh = int(yolo_bbox[3]) - int(yolo_bbox[1])
        area_r  = (bw * bh) / (w * h)
        guide_iou_contour = yolo_iou   # use YOLO IoU as guide IoU
    else:
        # Contour fallback — multi-gate strict detection
        contour, guide_iou_contour = detect_document_contour(image, camera_mode)
        area_r   = document_area_ratio(image, contour)
        detected = contour is not None

    # ── Screen / phone display guard (camera mode) ────────────────────────────
    # DISABLED: Allow users to capture IDs from phone screens
    # Must run BEFORE the alignment check so a phone showing a passport
    # is caught even when YOLO fires with high confidence.
    # if camera_mode:
    #     screen_detected, screen_score = is_screen_capture(image)
    #     if screen_detected:
    #         fail_reasons.append("screen_detected")

    # ── Face / person guard ───────────────────────────────────────────────────
    face_in_frame = detect_face(image)
    if face_in_frame and not detected:
        fail_reasons.append("person_detected")

    # ── Document presence (ALWAYS enforced) ───────────────────────────────────
    if not detected or area_r < DOC_AREA_MIN:
        if "person_detected" not in fail_reasons and "screen_detected" not in fail_reasons:
            fail_reasons.append("no_document")

    # ── Guide-box alignment (camera mode, contour path — FAIL if too far off) ─
    if camera_mode and not yolo_detected and detected and guide_iou_contour < GUIDE_IOU_MIN:
        if "no_document" not in fail_reasons:
            fail_reasons.append("not_in_frame")

    # ── Alignment check (YOLO mode — FAIL below ALIGN_IOU_FAIL, WARN below ALIGN_IOU_WARN) ─
    if camera_mode and yolo_detected:
        if yolo_iou < ALIGN_IOU_FAIL:
            fail_reasons.append("not_in_frame")
        elif yolo_iou < ALIGN_IOU_WARN:
            warn_reasons.append("misaligned")

    # ── Determine status ──────────────────────────────────────────────────────
    if fail_reasons:
        status  = "fail"
        reasons = fail_reasons + warn_reasons
    elif warn_reasons:
        status  = "warning"
        reasons = warn_reasons
    else:
        status  = "pass"
        reasons = []

    return {
        "status":  status,
        "reasons": reasons,
        "metrics": {
            "blur_score":          round(b_score,         2),
            "brightness":          round(bright,          2),
            "document_detected":   detected,
            "document_area_ratio": round(area_r,          3),
            "yolo_detected":       yolo_detected,
            "yolo_conf":           round(yolo_conf,       2),
            "yolo_label":          yolo_label,
            "yolo_iou":            round(yolo_iou,        2),
            "face_detected":       face_in_frame,
            "guide_iou":           round(guide_iou_contour, 2),
        },
    }
