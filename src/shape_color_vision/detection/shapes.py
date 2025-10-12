import cv2
import numpy as np
from typing import Tuple

# ---------- filters (used by pipeline) ----------

def _radial_uniformity(cnt) -> float:
    """
    Return std(radius)/mean(radius) for all contour points measured
    from the minEnclosingCircle center. Circles ~ 0.000.06, polygons higher.
    """
    if len(cnt) < 5:
        return 1.0
    (cx, cy), _ = cv2.minEnclosingCircle(cnt)
    pts = cnt.reshape(-1, 2).astype(np.float32)
    r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    m = float(np.mean(r) + 1e-9)
    s = float(np.std(r))
    return s / m


def _solidity(cnt) -> float:
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    harea = cv2.contourArea(hull)
    if harea <= 0:
        return 0.0
    return float(area / harea)

def contour_is_valid(
    cnt,
    min_area: int = 800,
    min_solidity: float = 0.65,
    min_bbox_area_ratio: float = 0.0015,
    min_extent: float = 0.58,
    image_area: int | None = None,
    min_width: int = 18,
    min_height: int = 18,
    **_ignored
) -> bool:
    """Reject contours that are too small, thin, or likely to be text/noise."""
    area = cv2.contourArea(cnt)
    if area < float(min_area):
        return False

    x, y, w, h = cv2.boundingRect(cnt)
    if w < min_width or h < min_height:
        return False

    if image_area is not None and image_area > 0:
        if (w * h) / float(image_area) < float(min_bbox_area_ratio):
            return False

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return False
    solidity = area / float(hull_area)
    if solidity < float(min_solidity):
        return False

    extent = area / float(w * h)
    if extent < float(min_extent):
        return False

    aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
    if aspect_ratio > 6.0:
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
      * Circle: very high circularity, near-isotropic ellipse, and good fit to enclosing circle
      * Triangle: polygon approx has 3 vertices
      * Square/Rectangle: polygon approx has 4 vertices with ~right angles
        - near-square but rotated (diamond) -> Unknown
      * Everything else -> Unknown
    """
    area = cv2.contourArea(contour)
    if area <= 0:
        return ("Unknown", 0.0)

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)

    # Circle metrics
    circ = _circularity(contour)            # 4πA/P²
    axis_ratio = _ellipse_axis_ratio(contour)
    r_u = _radial_uniformity(contour)       # std(radius)/mean(radius)

    # Fit to enclosing circle
    (cx, cy), r = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * (r * r)
    a = float(cv2.contourArea(contour))
    fit_ratio = a / (circle_area + 1e-6)

    # ── Guard: regular 5–7-gons should NOT be circles ─────────────────────
    # Only allow 5–7 vertices to be Circle if the outline is *extremely* round.
    if 5 <= v <= 7 and not (r_u <= 0.055 and axis_ratio <= 1.06 and circ >= 0.90 and fit_ratio >= 0.90):
        return ("Unknown", 0.7)

    # ── Circle decision (robust, vertex-count independent) ─────────────────
    # Tightened to block hexagons but keep real disks.
    if (r_u <= 0.070 and axis_ratio <= 1.08 and fit_ratio >= 0.90) \
       or (v >= 8 and circ >= 0.90 and axis_ratio <= 1.10):
        return ("Circle", float(min(1.0, circ)))


    # Triangle
    if v == 3:
        return ("Triangle", 1.0)

    if v == 4:
        ar, ang = _rot_rect_aspect_angle(contour)  # ar>=1, ang in [0,90)
        near_square = 0.90 <= ar <= 1.10
        axis_aligned = (ang < 10) or (ang > 80)

        pts = approx.reshape(-1, 2)
        def angle(p0, p1, p2):
            v1 = p0 - p1; v2 = p2 - p1
            cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
            cos = np.clip(cos, -1.0, 1.0)
            return np.degrees(np.arccos(cos))
        angs = [angle(pts[i-1], pts[i], pts[(i+1)%4]) for i in range(4)]
        rightish = all(75 <= a <= 105 for a in angs)

        sides = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
        equal_sided = (max(sides)/max(1e-6, min(sides))) < 1.15

        if near_square and not axis_aligned and equal_sided:
            return ("Unknown", 0.75)

        if rightish:
            if near_square and axis_aligned:
                return ("Square", 0.9)
            conf = float(np.clip(1.0 - abs(1.0 - ar), 0.5, 1.0))
            return ("Rectangle", conf)

    return ("Unknown", 0.6)
