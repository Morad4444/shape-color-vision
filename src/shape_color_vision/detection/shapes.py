import cv2
import numpy as np
from typing import Tuple

# ---------- filters (used by pipeline) ----------

def _solidity(cnt) -> float:
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    harea = cv2.contourArea(hull)
    if harea <= 0:
        return 0.0
    return float(area / harea)

def contour_is_valid(cnt, min_area: int = 1200, min_solidity: float = 0.85) -> bool:
    """Reject tiny or thin/noisy contours (text, outlines)."""
    if cv2.contourArea(cnt) < float(min_area):
        return False
    if _solidity(cnt) < float(min_solidity):
        return False
    return True

# ---------- geometry helpers ----------

def _circularity(cnt) -> float:
    """4πA/P² ; 1.0 for a perfect circle."""
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    if a <= 0 or p <= 0:
        return 0.0
    return float(4.0 * np.pi * a / (p * p))

def _ellipse_axis_ratio(cnt) -> float:
    """Major/minor axis ratio (>=1). If not available, +inf."""
    if len(cnt) < 5:
        return float("inf")
    (_, (MA, ma), _) = cv2.fitEllipse(cnt)
    if MA <= 0 or ma <= 0:
        return float("inf")
    return float(max(MA, ma) / min(MA, ma))

def _rot_rect_aspect_angle(cnt) -> Tuple[float, float]:
    """Return (aspect_ratio>=1, angle_deg in [0,90))."""
    (_, (w, h), ang) = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return (float("inf"), 0.0)
    ar = max(w, h) / min(w, h)
    ang = abs((ang + 90) if ang < -45 else ang)
    return (float(ar), float(ang))

# ---------- main classifier ----------

def classify_shape(contour) -> Tuple[str, float]:
    """
    Classify as one of: Circle, Triangle, Square, Rectangle, Unknown.
    Rules:
      * Circle: high circularity and ellipse axis ratio ~ 1
      * Triangle: polygon approx has 3 vertices
      * Square/Rectangle: only when polygon approx has 4 vertices
        - near-square but rotated (diamond) -> Unknown
      * Everything else -> Unknown
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return ("Unknown", 0.0)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)

    # Circle check first (reject ellipses)
    circ = _circularity(contour)
    if 5 <= v <= 7:
        return ("Unknown", 0.7)

    # True circles: many vertices, very high circularity, near-isotropic ellipse
    if v >= 8 and circ >= 0.86:
        axis_ratio = _ellipse_axis_ratio(contour)
        if axis_ratio <= 1.12:
            return ("Circle", float(min(1.0, circ)))

    # Triangle
    if v == 3:
        return ("Triangle", 1.0)

    if v == 4:
        # Rotated rectangle info
        ar, ang = _rot_rect_aspect_angle(contour)  # ar>=1, ang in [0,90)
        near_square = 0.90 <= ar <= 1.10
        axis_aligned = (ang < 10) or (ang > 80)

        # angles from approximated polygon
        pts = approx.reshape(-1, 2)
        def angle(p0, p1, p2):
            v1 = p0 - p1; v2 = p2 - p1
            cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
            cos = np.clip(cos, -1.0, 1.0)
            return np.degrees(np.arccos(cos))
        angs = [angle(pts[i-1], pts[i], pts[(i+1)%4]) for i in range(4)]
        rightish = all(75 <= a <= 105 for a in angs)  # ~90° +-15°

        # Equal-sided check (diamond)
        sides = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
        equal_sided = (max(sides)/max(1e-6, min(sides))) < 1.15

        # If it’s a rotated near-square (diamond) -> Unknown
        if near_square and not axis_aligned and equal_sided:
            return ("Unknown", 0.75)

        # Only call Rectangle/Square when corners are ~right angles
        if rightish:
            if near_square and axis_aligned:
                return ("Square", 0.9)
            # clearly rectangular aspect ratio
            conf = float(np.clip(1.0 - abs(1.0 - ar), 0.5, 1.0))
            return ("Rectangle", conf)

        # not right-angled -> treat as Unknown (e.g., trapezoid)
    return ("Unknown", 0.6)
