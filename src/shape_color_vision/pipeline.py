import os, glob
import cv2
import numpy as np
from typing import List
from .detection.colors import classify_color
from .detection.shapes import classify_shape, contour_is_valid
from .io.viz import draw_label, draw_contour
from .io.logger_csv import CSVLogger

def preprocess(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

def analyze_image(path: str, cfg) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    edged = preprocess(img)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger = CSVLogger(cfg.paths.log_csv)

    for c in cnts:
        if not contour_is_valid(c, cfg.detect.min_area, cfg.detect.min_solidity):
            continue

        x, y, w, h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]

        # make a filled contour mask in ROI coordinates
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        c_shift = c.copy()
        c_shift[:, 0, 0] -= x
        c_shift[:, 0, 1] -= y
        cv2.drawContours(roi_mask, [c_shift], -1, 255, thickness=-1)

        color, p_color = classify_color(
            roi,
            cfg.colors_hsv.__dict__,
            cfg.detect.pastel_s_thresh,
            roi_mask=roi_mask
        )
        shape, p_shape = classify_shape(c)

        if shape == "Unknown" and color == "unknown":
            label = "UNKNOWN Unknown"
        else:
            label = f"{color.upper()} {shape}"

        draw_contour(img, c)
        draw_label(img, label, (x, max(15, y - 5)))
        logger.log(shape, color, float((p_color + p_shape) / 2.0), "IMAGE", os.path.basename(path))

    return img

def analyze_dir(image_dir: str, cfg) -> List[str]:
    paths: List[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    return paths
