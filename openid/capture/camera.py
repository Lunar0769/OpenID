"""
OpenID Camera — YOLOv8-Enhanced Live Capture
=============================================

Capture flow:
  1. Pre-load YOLOv8 document detector (shows loading screen if needed).
  2. On every frame:
       a. Run YOLO detection → get bbox, conf, label.
       b. Compute alignment IoU vs guide box.
       c. Run check_quality() with YOLO data as the single source of truth.
       d. Draw: guide box, YOLO box, status banner, metric panel, stability bar.
  3. Increment stable_count ONLY when status == "pass".
  4. Auto-capture after 25 consecutive pass frames.
  5. Crop saved image to guide box region.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from openid.capture.quality import check_quality
from openid.capture.overlay import (
    draw_guide_box,
    draw_yolo_box,
    draw_status_banner,
    draw_stability_bar,
    draw_loading_screen,
    draw_side_prompt,
    get_guide_box,
    GREEN, YELLOW, RED,
)
from openid.capture import detector as doc_detector


# ── Constants ─────────────────────────────────────────────────────────────────
FRAME_W           = 640
FRAME_H           = 480
CAPTURE_THRESHOLD = 15       # consecutive PASS frames for auto-capture (~0.5 s @ 30 fps)
DOC_LOCK_FRAMES   = 5        # document must be detected for this many frames before counting
SAVE_DIR          = "captures"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "No camera device found. Connect a webcam or use --file to provide an image."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


def _preload_yolo(cap: cv2.VideoCapture, window_title: str) -> bool:
    """
    Show a loading screen while the YOLO model is downloaded / loaded.
    Returns True when the model is ready.
    """
    # Start loading in background by calling load_model once
    loading_done = [False]

    import threading
    def _load():
        doc_detector.load_model()
        loading_done[0] = True

    t = threading.Thread(target=_load, daemon=True)
    t.start()

    while not loading_done[0]:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            draw_loading_screen(frame, "Loading YOLO model...")
            cv2.imshow(window_title, frame)
        if cv2.waitKey(30) & 0xFF == 27:
            return False    # User pressed ESC during load

    return True


# ── Main capture function ─────────────────────────────────────────────────────

def capture_image(label: str = "Document") -> tuple[np.ndarray | None, str | None]:
    """
    Open webcam with YOLOv8-enhanced quality overlay and auto-capture when ready.

    Auto-capture rules (STRICT):
      - YOLO must detect a document (or contour fallback)
      - status must be "pass" (not warning, not fail)
      - 25 consecutive pass frames required
      - Captured image is cropped to the guide box region

    Args:
        label: Display label for the window title and saved filename.

    Returns:
        (cropped_frame, full_frame, filepath) — or (None, None, None) if cancelled via ESC.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    window_title = f"OpenID -- {label}"

    cap = _open_camera()

    # Pre-load YOLO with a live loading screen
    if not _preload_yolo(cap, window_title):
        cap.release()
        cv2.destroyAllWindows()
        return None, None, None

    stable_count   = 0
    doc_streak     = 0    # consecutive frames with a valid document detected
    captured_frame = None
    full_frame     = None   # full (uncropped) frame — used for validation
    captured_path  = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # ── YOLO detection ────────────────────────────────────────────────────
        bbox, yolo_conf, yolo_label = doc_detector.detect_document(frame)
        yolo_iou = doc_detector.alignment_iou(frame, bbox)

        # ── Quality check (single source of truth) ────────────────────────────
        result  = check_quality(
            frame,
            camera_mode = True,
            yolo_bbox   = bbox,
            yolo_iou    = yolo_iou,
            yolo_conf   = yolo_conf,
            yolo_label  = yolo_label,
        )
        status  = result["status"]
        reasons = result["reasons"]
        metrics = result["metrics"]
        color   = {"pass": GREEN, "warning": YELLOW, "fail": RED}[status]

        # ── Document lock: track how long a document has been continuously seen ──
        doc_seen = result["metrics"].get("document_detected", False)
        if doc_seen and status != "fail":
            doc_streak += 1
        else:
            doc_streak     = 0
            stable_count   = 0  # hard reset — must re-establish document lock

        # ── Stability (STRICT: only pass frames count AND doc must be locked) ──
        if status == "pass" and doc_streak >= DOC_LOCK_FRAMES:
            stable_count += 1
        elif status != "pass":
            stable_count = 0

        # ── Auto-capture ──────────────────────────────────────────────────────
        if stable_count >= CAPTURE_THRESHOLD:
            x, y, w, h     = get_guide_box(frame)
            full_frame     = frame.copy()                      # keep full frame for validation
            captured_frame = frame[y:y + h, x:x + w].copy()  # crop for saving / display
            ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_label     = label.lower().replace(" ", "_")
            captured_path  = f"{SAVE_DIR}/{safe_label}_{ts}.jpg"
            cv2.imwrite(captured_path, captured_frame)
            print(f"  📸 Captured (cropped to guide box): {captured_path}")
            break

        # ── Draw UI ───────────────────────────────────────────────────────────
        draw_guide_box(frame, status)
        draw_yolo_box(frame, bbox, yolo_conf, yolo_label, yolo_iou)

        # Build status message
        if status == "pass" and doc_streak >= DOC_LOCK_FRAMES:
            remaining   = CAPTURE_THRESHOLD - stable_count
            status_text = f"HOLD STILL... {remaining}" if remaining > 0 else "CAPTURING!"
        elif status in ("pass", "warning") and doc_streak < DOC_LOCK_FRAMES and doc_streak > 0:
            status_text = f"Locking document... ({doc_streak}/{DOC_LOCK_FRAMES})"
        elif reasons:
            status_text = " | ".join(r.replace("_", " ").title() for r in reasons)
        else:
            status_text = f"Place document in box  ({stable_count}/{CAPTURE_THRESHOLD})"

        draw_status_banner(frame, status_text, status, metrics)
        draw_stability_bar(frame, stable_count, CAPTURE_THRESHOLD, color)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            print("  ⚠️  Capture cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame, full_frame, captured_path


# ── File mode ─────────────────────────────────────────────────────────────────

def capture_from_file(path: str) -> np.ndarray:
    """
    Load image from disk, run quality check, return if acceptable.

    Quality gate:
      - "fail"    → raises ImageQualityError (hard block)
      - "warning" → allowed, logs to stderr
      - "pass"    → silent pass

    Raises:
        ImageQualityError: Image failed quality checks.
        FileNotFoundError: File does not exist.
    """
    import sys
    from openid.exceptions import ImageQualityError

    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")

    result  = check_quality(image, camera_mode=False)
    status  = result["status"]
    metrics = result["metrics"]

    if status == "fail":
        reasons = result["reasons"]
        _msgs = {
            "blank_frame":      "Image appears blank — ensure the document is visible",
            "too_blurry":       "Image is too blurry — hold the camera steady",
            "slight_blur":      "Image has slight blur",
            "low_light":        "Image has insufficient lighting",
            "dim_light":        "Image lighting is dim",
            "too_bright":       "Image is overexposed — reduce glare or ambient light",
            "no_document":      "No identity document detected in image",
            "person_detected":  "A person was detected instead of a document — submit a flat scan of the ID",
        }
        primary = reasons[0] if reasons else "unknown"
        raise ImageQualityError(
            message = _msgs.get(primary, "Image failed quality checks"),
            reason  = primary,
            metrics = metrics,
        )

    if status == "warning":
        print(
            f"Warning: image quality suboptimal — {', '.join(result['reasons'])}. "
            f"Metrics: blur={metrics['blur_score']}, brightness={metrics['brightness']}",
            file=sys.stderr,
        )

    return image


# ── Flip prompt ───────────────────────────────────────────────────────────────

def wait_for_flip(cap_device: int = 0) -> bool:
    """Show 'flip the card' prompt. SPACE=continue, ESC=cancel."""
    cap = cv2.VideoCapture(cap_device)
    if not cap.isOpened():
        return True

    print("\n  🔄  Flip the card to the BACK side, then press SPACE...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        draw_side_prompt(frame, "BACK")
        cv2.imshow("OpenID — Flip Card", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            cap.release()
            cv2.destroyAllWindows()
            return True
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return False

    cap.release()
    cv2.destroyAllWindows()
    return True
