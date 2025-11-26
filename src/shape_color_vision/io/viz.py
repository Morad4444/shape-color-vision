"""
viz.py — Visualization utilities for the Shape-Color Vision system.

Responsibilities:
• Draw detected shapes, colors, labels, and bounding boxes.
• Display intermediate stages (mask, contours, annotated frame).
• Provide optional UI elements that can be swapped out without changing
  detector logic (Open/Closed Principle).

Notes:
• No detection or classification should happen here.
• Keep it purely presentation-related.
"""



import cv2
import numpy as np
from typing import Tuple

def draw_label(img, text: str, org: Tuple[int, int]):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

def draw_contour(img, contour: np.ndarray):
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
