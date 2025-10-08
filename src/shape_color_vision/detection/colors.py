import cv2
import numpy as np
from typing import Dict, Tuple, Optional

# HSV range typed as (Hmin, Smin, Vmin, Hmax, Smax, Vmax)
HSVRange = Tuple[int, int, int, int, int, int]

def _mask_for_range(hsv: np.ndarray, r: HSVRange) -> np.ndarray:
    low  = np.array(r[:3], dtype=np.uint8)
    high = np.array(r[3:], dtype=np.uint8)
    return cv2.inRange(hsv, low, high)

def classify_color(bgr_roi: np.ndarray, hsv_cfg: Dict[str, list]) -> Tuple[str, float]:
    """
    Return (color_name, confidence 0..1) for a BGR patch using provided HSV ranges.
    Handles red wrap-around with two ranges (red1 + red2).
    """
    if bgr_roi.size == 0:
        return ("unknown", 0.0)

    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # Build masks per color
    masks: Dict[str, np.ndarray] = {}

    # Red is split into two ranges
    red1 = tuple(hsv_cfg["red1"])  # type: ignore
    red2 = tuple(hsv_cfg["red2"])  # type: ignore
    masks["red"] = cv2.bitwise_or(_mask_for_range(hsv, red1), _mask_for_range(hsv, red2))

    for name in ["green", "blue", "yellow", "violet"]:
        masks[name] = _mask_for_range(hsv, tuple(hsv_cfg[name]))  # type: ignore

    scores = {k: float(cv2.countNonZero(v)) for k, v in masks.items()}
    total = float(sum(scores.values())) or 1.0
    color = max(scores, key=scores.get)
    conf = scores[color] / total
    return (color, conf)
