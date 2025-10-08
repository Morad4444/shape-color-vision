import cv2
import numpy as np
from typing import List, Dict, Tuple
from .detection.colors import classify_color
from .detection.shapes import classify_shape
from .io.logger_csv import CSVLogger
from .io.viz import draw_label, draw_bbox

def preprocess(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edged = cv2.dilate(edged, kernel, iterations=1)
    return edged

def analyze_image(path: str, cfg) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    edged = preprocess(img)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    logger = CSVLogger(cfg.paths.log_csv)
    for c in cnts:
        if cv2.contourArea(c) < cfg.detect.min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        roi = img[y:y+h, x:x+w]
        color, p_color = classify_color(roi, cfg.colors_hsv.__dict__)
        shape, p_shape = classify_shape(c)
        label = f"{color.upper()} {shape}"
        draw_bbox(img, x,y,w,h)
        draw_label(img, label, (x, max(15, y-5)))
        logger.log(shape, color, float((p_color+p_shape)/2), "IMAGE", path.split("/")[-1])

    return img

def analyze_dir(image_dir: str, cfg) -> List[str]:
    import glob, os
    paths = []
    for ext in ("*.png","*.jpg","*.jpeg","*.bmp"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths.sort()
    return paths
