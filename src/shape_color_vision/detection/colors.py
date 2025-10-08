# src/shape_color_vision/detection/colors.py
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

# Ignore highlights/shadows
PIX_MIN_S, PIX_MIN_V = 40, 40  # tweak if needed

def _masked_hsv_pixels(bgr_roi: np.ndarray, roi_mask: Optional[np.ndarray]) -> np.ndarray:
    """Return HSV pixels inside mask with sufficient S and V."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    if roi_mask is None:
        roi_mask = np.ones(hsv.shape[:2], np.uint8) * 255
    m = roi_mask.astype(bool)
    m &= (hsv[..., 1] >= PIX_MIN_S) & (hsv[..., 2] >= PIX_MIN_V)
    return hsv[m]

def _masked_bgr_means(bgr_roi: np.ndarray, roi_mask: Optional[np.ndarray]) -> Tuple[float, float, float]:
    """Mean B, G, R inside mask (float)."""
    if roi_mask is None:
        mask = np.ones(bgr_roi.shape[:2], np.uint8) * 255
    else:
        mask = (roi_mask > 0).astype(np.uint8) * 255
    # compute means with mask
    b = float(cv2.mean(bgr_roi[:, :, 0], mask=mask)[0])
    g = float(cv2.mean(bgr_roi[:, :, 1], mask=mask)[0])
    r = float(cv2.mean(bgr_roi[:, :, 2], mask=mask)[0])
    return b, g, r

def classify_color(
    bgr_roi: np.ndarray,
    hsv_cfg: Dict[str, list],          # kept for API compatibility
    pastel_s_thresh: int = 0,          # unused
    roi_mask: Optional[np.ndarray] = None
) -> Tuple[str, float]:
    """
    Robust color classification:
      - uses masked pixels only
      - median HSV to resist shading
      - banded hue with pink guard
      - blue/violet boundary at 135 with an ambiguous-zone tie-breaker using BGR means
    """
    if bgr_roi is None or bgr_roi.size == 0:
        return ("unknown", 0.0)

    px = _masked_hsv_pixels(bgr_roi, roi_mask)
    if px.size == 0:
        return ("unknown", 0.0)

    h = float(np.median(px[:, 0]))  # 0..179
    s = float(np.median(px[:, 1]))
    v = float(np.median(px[:, 2]))

    # Low-saturation overall -> unknown
    if s < 60:
        return ("unknown", 0.0)

    # Pink guard (light magenta): hue near 150–175 but not saturated enough
    if 150 <= h < 175 and s < 120:
        return ("unknown", 0.0)
    
    # Orange sits between our red and yellow bands (~10..20 on OpenCV scale).
    if 10 <= h < 22:
        return ("unknown", 0.0)
    
    #--- NEW: Hot pink / magenta veto ---
    # Strong pinks live around 155–175 (high S,V). We don't want to call them violet.
    if 155 <= h < 175:
        return ("unknown", 0.0)

    # --- Hue bands ---
     # Distance to the red axis (handles wrap-around near 0/179)
    red_dist = min(h, 180.0 - h)  # e.g., h=178 → 2; h=7 → 7

    # If we're near red but not truly red, that's "orange" for our purposes → Unknown
    # (treat 5..22 degrees from red as orange; tweak 22 to 20 if you want tighter)
    if 5.0 <= red_dist < 22.0:
        return ("unknown", 0.0)

    # --- Hue bands (keep as-is, but change RED to use red_dist) ---
    color = "unknown"
    band_lo, band_hi = h, h

    # RED exactly (very close to the red axis)
    if red_dist < 5.0:
        color = "red" if s >= 110 else "unknown"
        # pick band edges consistent with whichever side we are on
        if h < 90:   # around 0°
            band_lo, band_hi = -5.0, 5.0
        else:        # around 180°
            band_lo, band_hi = 175.0, 185.0

    elif 22 <= h < 40:
        color, band_lo, band_hi = "yellow", 22, 40
    elif 40 <= h < 95:
        color, band_lo, band_hi = "green", 40, 95
    elif 95 <= h < 135:
        color, band_lo, band_hi = "blue", 95, 135
    elif 135 <= h < 155:
        color, band_lo, band_hi = "violet", 135, 155

    # --- Ambiguous zone tie-breaker (purple vs blue) ---
    # If hue is near the boundary, check if red is present alongside blue.
    if 128 <= h < 142:
        b, g, r = _masked_bgr_means(bgr_roi, roi_mask)
        if (r / (b + 1e-6)) >= 0.70:
            color = "violet"
            band_lo, band_hi = 135, 175
        else:
            color = "blue"
            band_lo, band_hi = 95, 135

    if color == "unknown":
        return ("unknown", 0.0)

    # Confidence: distance from band edges normalized to [0..1]
    edge_dist = min(abs(h - band_lo), abs(band_hi - h))
    band_half = max(1e-6, (band_hi - band_lo) / 2.0)
    conf = max(0.0, min(1.0, edge_dist / band_half))
    return (color, float(conf))
