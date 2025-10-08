import cv2
from typing import Tuple

def draw_label(img, text: str, org: Tuple[int, int]):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def draw_bbox(img, x, y, w, h):
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
