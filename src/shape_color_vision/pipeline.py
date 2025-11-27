"""
pipeline.py — Central pipeline coordinating the full vision system.

Responsibilities:
• Connect all system components: preprocessing → detection → classification → output.
• Manage dependencies between modules (shapes, colors, viz, logging).
• Ensure each frame passes through a consistent ordered workflow.
• Serve as the main integration layer while keeping each task isolated
  (Integration/Operation Segregation Principle).

This module does NOT:
• Perform shape or color detection itself.
• Handle visualization details (delegated to io.viz).
• Load raw configuration files (delegated to utils.config).
"""


from __future__ import annotations

import os
import glob
from typing import Iterable, Tuple

import cv2
import numpy as np

from .detection.shapes import contour_is_valid, classify_shape
from .detection.colors import classify_color
from .io.logger_csv import CSVLogger

# ───────────────────────── helpers: read settings from cfg ─────────────────────────

def _detect_obj(cfg, for_camera: bool):
    if for_camera and getattr(cfg, "camera_detect", None):
        return cfg.camera_detect
    if not for_camera and getattr(cfg, "image_detect", None):
        return cfg.image_detect
    return cfg.detect

def _detect_kwargs(cfg, for_camera: bool) -> dict:
    d = _detect_obj(cfg, for_camera)
    return dict(
        min_area=int(getattr(d, "min_area", 800)),
        min_solidity=float(getattr(d, "min_solidity", 0.65)),
        min_bbox_area_ratio=float(getattr(d, "min_bbox_area_ratio", 0.0015)),
        min_extent=float(getattr(d, "min_extent", 0.50)),
        min_width=int(getattr(d, "min_width", 16)),
        min_height=int(getattr(d, "min_height", 16)),
    )

def _mask_sv(cfg, for_camera: bool) -> tuple[int, int]:
    mask = None
    if for_camera and getattr(cfg, "camera_mask", None):
        mask = cfg.camera_mask
    elif not for_camera and getattr(cfg, "image_mask", None):
        mask = cfg.image_mask
    if mask is not None:
        return int(getattr(mask, "s_min", 40)), int(getattr(mask, "v_min", 40))
    return (35, 45) if for_camera else (40, 40)

def _pastel_s_thresh(cfg) -> int:
    return int(getattr(getattr(cfg, "detect", None), "pastel_s_thresh", 90))

# ───────────────────────── helpers: image ops ─────────────────────────

def _color_mask(img: np.ndarray, s_min: int, v_min: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask

def _filled_roi_mask(contour: np.ndarray, w: int, h: int, x: int, y: int) -> np.ndarray:
    """Binary ROI mask filled with the contour, then eroded to ignore bright outlines."""
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    c_shift = contour.copy()
    c_shift[:, 0, 0] -= x
    c_shift[:, 0, 1] -= y
    cv2.drawContours(roi_mask, [c_shift], -1, 255, thickness=-1)
    # shrink slightly so color sampling avoids the neon-green border
    roi_mask = cv2.erode(roi_mask, np.ones((3, 3), np.uint8), iterations=1)
    return roi_mask

def _draw_label(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.7) -> None:
    x, y = org
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)

# ───────────────────────── public API ─────────────────────────

def analyze_image(path: str, cfg) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    h_img, w_img = img.shape[:2]
    img_area = int(h_img * w_img)

    s_min, v_min = _mask_sv(cfg, for_camera=False)
    mask = _color_mask(img, s_min, v_min)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger = CSVLogger(cfg.paths.log_csv)
    detect_kwargs = _detect_kwargs(cfg, for_camera=False)

    for c in cnts:
        if not contour_is_valid(c, **detect_kwargs, image_area=img_area):
            continue

        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y + h, x:x + w]
        roi_mask = _filled_roi_mask(c, w, h, x, y)

        color, p_color = classify_color(roi, cfg.colors_hsv.__dict__, _pastel_s_thresh(cfg), roi_mask=roi_mask)
        shape, p_shape = classify_shape(c)

        label = "UNKNOWN Unknown" if (shape == "Unknown" and color == "unknown") else f"{color.upper()} {shape}"
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        _draw_label(img, label, (x, max(20, y - 6)))

        logger.log(shape, color, float(max(p_color, p_shape)), "IMAGE", os.path.basename(path))

    return img

def analyze_dir(dir_path: str, cfg) -> None:
    paths: Iterable[str] = sorted(
        p for p in glob.glob(os.path.join(dir_path, "*"))
        if os.path.isfile(p)
    )
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    for p in paths:
        out = analyze_image(p, cfg)

        if getattr(cfg.video, "show_window", True):
            cv2.imshow("Result", out)
            cv2.waitKey(200)

        if getattr(cfg.video, "save_output", False):
            out_path = os.path.join(cfg.paths.output_dir, os.path.basename(p))
            cv2.imwrite(out_path, out)

    if getattr(cfg.video, "show_window", True):
        cv2.destroyAllWindows()

def analyze_frame(img: np.ndarray, cfg, logger: CSVLogger | None = None) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    img_area = int(h * w)

    s_min, v_min = _mask_sv(cfg, for_camera=True)
    mask = _color_mask(img, s_min, v_min)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if logger is None:
        logger = CSVLogger(cfg.paths.log_csv)

    detect_kwargs = _detect_kwargs(cfg, for_camera=True)

    for c in cnts:
        if not contour_is_valid(c, **detect_kwargs, image_area=img_area):
            continue

        x, y, ww, hh = cv2.boundingRect(c)
        roi = img[y:y + hh, x:x + ww]
        roi_mask = _filled_roi_mask(c, ww, hh, x, y)

        color, p_color = classify_color(roi, cfg.colors_hsv.__dict__, _pastel_s_thresh(cfg), roi_mask=roi_mask)
        shape, p_shape = classify_shape(c)

        label = "UNKNOWN Unknown" if (shape == "Unknown" and color == "unknown") else f"{color.upper()} {shape}"
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        _draw_label(img, label, (x, max(20, y - 6)))

        logger.log(shape, color, float(max(p_color, p_shape)), "CAMERA", "webcam")

    return img
