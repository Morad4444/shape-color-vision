import cv2
import numpy as np
from typing import Tuple, Optional, Dict

def classify_shape(contour: np.ndarray) -> Tuple[str, float]:
    """
    Classify a contour as circle, triangle, square, or rectangle.
    Returns (name, confidence 0..1).
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return ("unknown", 0.0)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)

    if v == 3:
        return ("Triangle", 1.0)
    elif v == 4:
        # square vs rectangle by aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h) if h else 0
        name = "Square" if 0.90 <= ar <= 1.10 else "Rectangle"
        conf = 1.0 - abs(1 - ar) if name == "Square" else min(abs(1 - ar), 1.0)
        return (name, float(max(0.5, min(1.0, conf))))
    else:
        # circle-ish: compare area vs. minEnclosingCircle area
        (x, y), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        ratio = area / circle_area if circle_area > 0 else 0
        if 0.7 <= ratio <= 1.2:
            return ("Circle", float(1.0 - abs(1 - ratio)))
        return ("Unknown", 0.4)
