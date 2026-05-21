"""
OpenID Overlay — Camera UI Drawing Utilities
============================================

Draws:
  - Predefined guide box (fixed, NOT contour-based)
  - YOLOv8 detection box with confidence label
  - Status banner (top-left)
  - Metrics panel (top-right): Blur, Light, Doc %, YOLO conf, Alignment IoU
  - Stability progress bar (bottom)
"""
import cv2
import numpy as np


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
RED    = (0,   0,   255)
YELLOW = (0,   200, 255)
GREEN  = (0,   255, 0)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
GREY   = (80,  80,  80)
CYAN   = (200, 200, 0)   # golden-yellow for YOLO box

# ── Guide box geometry ────────────────────────────────────────────────────────
# Standard ID/passport: 85.6mm × 54mm ≈ 1.586:1
GUIDE_ASPECT       = 1.586
GUIDE_WIDTH_RATIO  = 0.80


def get_guide_box(frame: np.ndarray) -> tuple:
    """
    Return (x, y, w, h) of the centred predefined guide box.
    """
    fh, fw = frame.shape[:2]
    gw = int(fw * GUIDE_WIDTH_RATIO)
    gh = int(gw / GUIDE_ASPECT)
    gx = (fw - gw) // 2
    gy = (fh - gh) // 2
    return gx, gy, gw, gh


def draw_guide_box(frame: np.ndarray, status: str) -> None:
    """
    Draw the FIXED predefined document guide box (in-place).

    Colour: fail→red, warning→yellow, pass→green
    Outside the box is dimmed to focus attention on the guide area.
    """
    color = {"pass": GREEN, "warning": YELLOW, "fail": RED}.get(status, RED)
    x, y, w, h = get_guide_box(frame)

    # Dim region outside guide box
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), BLACK, -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    # Restore guide area to full brightness
    frame[y:y+h, x:x+w] = frame.copy()[y:y+h, x:x+w]

    # Corner brackets
    corner_len = min(w, h) // 6
    thickness  = 3
    corners = [
        [(x, y + corner_len), (x, y), (x + corner_len, y)],
        [(x + w - corner_len, y), (x + w, y), (x + w, y + corner_len)],
        [(x + w, y + h - corner_len), (x + w, y + h), (x + w - corner_len, y + h)],
        [(x + corner_len, y + h), (x, y + h), (x, y + h - corner_len)],
    ]
    for pts in corners:
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], color, thickness, cv2.LINE_AA)

    # Thin full outline
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)


def draw_yolo_box(
    frame:  np.ndarray,
    bbox,
    conf:   float,
    label:  str,
    iou:    float,
) -> None:
    """
    Draw the YOLOv8 detection bounding box with a confidence/label pill.

    Box colour:
        IoU >= 0.50 → green   (well aligned)
        IoU >= 0.25 → yellow  (close)
        IoU <  0.25 → cyan    (detected but misaligned)

    Args:
        frame: BGR frame to draw on (in-place).
        bbox:  (x1, y1, x2, y2) numpy array, or None.
        conf:  Detection confidence (0–1).
        label: Detected class label e.g. "Passport".
        iou:   Alignment IoU vs guide box.
    """
    if bbox is None or conf == 0:
        return

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    box_color = GREEN if iou >= 0.50 else (YELLOW if iou >= 0.25 else CYAN)

    # Corner-bracket style (matches guide box aesthetic)
    seg = max(8, min(x2 - x1, y2 - y1) // 5)
    for (px, py), (qx, qy) in [
        ((x1, y1 + seg), (x1, y1)),   ((x1, y1), (x1 + seg, y1)),
        ((x2 - seg, y1), (x2, y1)),   ((x2, y1), (x2, y1 + seg)),
        ((x2, y2 - seg), (x2, y2)),   ((x2, y2), (x2 - seg, y2)),
        ((x1 + seg, y2), (x1, y2)),   ((x1, y2), (x1, y2 - seg)),
    ]:
        cv2.line(frame, (px, py), (qx, qy), box_color, 2, cv2.LINE_AA)

    # Label pill
    font = cv2.FONT_HERSHEY_SIMPLEX
    tag  = f"{label}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(tag, font, 0.50, 1)
    pad = 4
    cv2.rectangle(frame, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), box_color, -1)
    cv2.putText(frame, tag, (x1 + pad, y1 - pad), font, 0.50, BLACK, 1, cv2.LINE_AA)


def draw_status_banner(
    frame:       np.ndarray,
    status_text: str,
    status:      str,
    metrics:     dict,
) -> None:
    """
    Draw status text (top-left) and metric readouts (top-right).

    Metrics shown:
        Blur, Light, Doc %, YOLO conf (if detected), Alignment IoU (if detected)
    """
    color  = {"pass": GREEN, "warning": YELLOW, "fail": RED}.get(status, RED)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    margin = 12

    # ── Status text (top-left) ────────────────────────────────────────────────
    scale, thick = 0.70, 2
    (tw, th), _  = cv2.getTextSize(status_text, font, scale, thick)
    cv2.putText(frame, status_text, (margin + 1, margin + th + 1),
                font, scale, BLACK, thick + 1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (margin,     margin + th),
                font, scale, color, thick, cv2.LINE_AA)

    # ── Metric readouts (top-right) ───────────────────────────────────────────
    blur_val  = metrics.get('blur_score', 0)
    bright_val = metrics.get('brightness', 0)
    doc_pct   = metrics.get('document_area_ratio', 0) * 100
    doc_yes   = metrics.get('document_detected', False)
    yolo_conf = metrics.get('yolo_conf', 0.0)
    yolo_iou  = metrics.get('yolo_iou', 0.0)
    yolo_lbl  = metrics.get('yolo_label', '')

    lines = [
        f"Blur   {blur_val:.0f}",
        f"Light  {bright_val:.0f}",
        f"Doc    {'Yes' if doc_yes else 'No'} ({doc_pct:.0f}%)",
    ]
    if yolo_lbl:
        lines.append(f"YOLO   {yolo_lbl} {yolo_conf*100:.0f}%")
        lines.append(f"Align  {yolo_iou*100:.0f}%")
    else:
        guide_iou_val = metrics.get('guide_iou', 0.0)
        lines.append(f"GBox   {guide_iou_val*100:.0f}%")

    fh, fw = frame.shape[:2]
    small_s = 0.50
    line_h  = 20
    base_y  = margin + th + 12 + line_h

    for i, line in enumerate(lines):
        (lw, _), _ = cv2.getTextSize(line, font, small_s, 1)
        x = fw - lw - margin
        y = base_y + i * line_h
        cv2.putText(frame, line, (x + 1, y + 1), font, small_s, BLACK, 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x,     y),     font, small_s, WHITE, 1, cv2.LINE_AA)


def draw_stability_bar(
    frame:             np.ndarray,
    stable_count:      int,
    capture_threshold: int,
    color,
) -> None:
    """
    Draw bottom-of-frame stability progress bar.
    """
    fh, fw = frame.shape[:2]
    bar_h  = 14
    margin = 16
    bar_y  = fh - margin - bar_h
    bar_x  = margin
    bar_w  = fw - 2 * margin

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), GREY, -1)

    fill = int(bar_w * min(stable_count, capture_threshold) / max(capture_threshold, 1))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    label = f"Stability  {stable_count}/{capture_threshold}"
    cv2.putText(frame, label, (bar_x, bar_y - 4), font, 0.45, WHITE, 1, cv2.LINE_AA)


def draw_loading_screen(frame: np.ndarray, message: str = "Loading YOLO model...") -> None:
    """
    Full-frame loading overlay displayed while YOLO model is being downloaded/loaded.
    """
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), BLACK, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (w1, h1), _ = cv2.getTextSize(message, font, 0.9, 2)
    cv2.putText(frame, message, ((fw - w1) // 2, fh // 2),
                font, 0.9, WHITE, 2, cv2.LINE_AA)
    sub = "This happens only once"
    (w2, h2), _ = cv2.getTextSize(sub, font, 0.55, 1)
    cv2.putText(frame, sub, ((fw - w2) // 2, fh // 2 + 30),
                font, 0.55, GREY, 1, cv2.LINE_AA)


def draw_side_prompt(frame: np.ndarray, side: str) -> None:
    """Full-frame prompt asking user to flip the card."""
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (fw, fh), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    msg1 = f"Now capture the {side}"
    msg2 = "Press SPACE to continue  |  ESC to cancel"
    (w1, _), _ = cv2.getTextSize(msg1, font, 1.1, 2)
    (w2, _), _ = cv2.getTextSize(msg2, font, 0.65, 1)
    cv2.putText(frame, msg1, ((fw - w1) // 2, fh // 2 - 20),
                font, 1.1, GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, msg2, ((fw - w2) // 2, fh // 2 + 28),
                font, 0.65, WHITE, 1, cv2.LINE_AA)
