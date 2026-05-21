"""
OpenID Document Detector
========================

Detects rectangular documents (passports, ID cards) in a camera frame using
a locally trained YOLOv8 model.

Strategy:
  1. Resolve model path relative to the project root (works on any machine).
  2. Load model on first use (lazy, threaded).
  3. Run YOLO inference on frames.
  4. Return the highest-confidence bounding box.

Alignment:
  Computes IoU between detected bbox and the predefined guide box.

Falls back gracefully (returns None) when no document is found.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Any

import cv2
import numpy as np

# ── Model path (resolved relative to the project root) ───────────────────────
_HERE      = Path(__file__).resolve()          # .../openid/capture/detector.py
_PROJECT   = _HERE.parent.parent.parent        # .../OpenID/
MODEL_PATH = str(_PROJECT / "runs" / "detect" / "id_card_model" / "weights" / "best.pt")

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.45   # must match quality.YOLO_CONF_MIN — do not lower independently
ALIGN_GOOD     = 0.45   # IoU ≥ this → box is well-aligned (green)
ALIGN_WARN     = 0.25   # IoU ≥ this → box is close (yellow)

# ── Internal state ────────────────────────────────────────────────────────────
_model:       Optional[Any] = None   # ultralytics YOLO instance once loaded
_load_failed: bool          = False  # set True after first failure — suppress spam


def load_model() -> bool:
    """
    Load the locally trained YOLOv8 model.

    Returns True on success, False on failure.
    Errors are logged ONCE then suppressed to avoid console spam.
    """
    global _model, _load_failed

    if _model is not None:
        return True

    if _load_failed:
        return False   # already failed — skip silently

    try:
        import importlib
        _yolo_mod = importlib.import_module("ultralytics")  # type: ignore[import-untyped]
        _YOLO = _yolo_mod.YOLO

        # Silence ultralytics' own verbose output
        logging.getLogger("ultralytics").setLevel(logging.ERROR)

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

        _model = _YOLO(MODEL_PATH)

        # Warmup run so first real frame isn't slow
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _model.predict(dummy, verbose=False)

        print(f"[detector] YOLO model loaded: {MODEL_PATH}")
        return True

    except Exception as e:
        print(f"[detector] YOLO load failed: {e}")
        print(f"[detector] Falling back to contour-only detection.")
        _load_failed = True
        return False


def detect_document(frame: np.ndarray) -> tuple[np.ndarray | None, float, str]:
    """
    Detect a document using YOLOv8.

    Returns:
        (bbox, confidence, label)
        bbox       - (x1, y1, x2, y2) numpy array, or None
        confidence - float 0-1
        label      - "ID Card"
    """
    global _model

    if _model is None:
        if not load_model():
            return None, 0.0, ""

    assert _model is not None   # narrowed after successful load_model()
    # Ultralytics prediction
    results   = _model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    best_conf = 0.0
    best_box  = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box  = box.xyxy[0].cpu().numpy()

    if best_box is not None:
        # Shrink box slightly so the UI tightly hugs the physical card
        x1, y1, x2, y2 = best_box
        w  = x2 - x1
        h  = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        scale  = 0.92
        new_w  = (w * scale) - 6
        new_h  = (h * scale) - 6
        tight  = np.array([
            cx - new_w / 2,
            cy - new_h / 2,
            cx + new_w / 2,
            cy + new_h / 2,
        ])
        return tight, round(best_conf, 2), "ID Card"

    return None, 0.0, ""


def alignment_iou(frame: np.ndarray, bbox: np.ndarray | None) -> float:
    """
    Compute IoU between detected bbox and the predefined guide box.
    Returns float 0-1 (1.0 = perfect overlap).
    """
    from openid.capture.overlay import get_guide_box

    if bbox is None:
        return 0.0

    gx, gy, gw, gh = get_guide_box(frame)
    gx2, gy2 = gx + gw, gy + gh

    ix1 = max(int(bbox[0]), gx)
    iy1 = max(int(bbox[1]), gy)
    ix2 = min(int(bbox[2]), gx2)
    iy2 = min(int(bbox[3]), gy2)

    inter_w    = max(0, ix2 - ix1)
    inter_h    = max(0, iy2 - iy1)
    inter_area = inter_w * inter_h

    bbox_area  = (int(bbox[2]) - int(bbox[0])) * (int(bbox[3]) - int(bbox[1]))
    guide_area = gw * gh
    union_area = bbox_area + guide_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def alignment_status(iou: float) -> str:
    """Returns 'good', 'close', or 'bad' based on IoU."""
    if iou >= ALIGN_GOOD:
        return "good"
    if iou >= ALIGN_WARN:
        return "close"
    return "bad"
