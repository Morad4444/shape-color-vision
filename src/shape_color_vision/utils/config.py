"""
config.py — YAML configuration loader and structured access helper.

Responsibilities:
• Load and parse configuration files (e.g., default.yaml).
• Provide typed access (dict-like or attribute-like) to config values.
• Validate required fields and apply defaults.
• Isolate system parameters so detection code stays flexible (OCP).

Notes:
• Contains no detection or visualization logic.
• Changing system behavior should happen in YAML, not in code.
"""



from pathlib import Path

# src/shape_color_vision/utils/config.py
from dataclasses import dataclass
import yaml

@dataclass
class Paths:
    image_dir: str = "data/samples"
    output_dir: str = "data/output"
    log_csv: str = "logs/detections.csv"

@dataclass
class Video:
    camera_index: int = 0
    show_window: bool = True
    save_output: bool = False

@dataclass
class Detect:
    min_area: int = 800
    min_solidity: float = 0.65
    pastel_s_thresh: int = 90
    min_circularity: float = 0.75
    min_bbox_area_ratio: float = 0.0015
    min_extent: float = 0.50
    min_width: int = 16
    min_height: int = 16

@dataclass
class HSVRanges:
    red1: list | None = None
    red2: list | None = None
    green: list | None = None
    blue: list | None = None
    yellow: list | None = None
    violet: list | None = None

@dataclass
class Mask:
    s_min: int = 40
    v_min: int = 40

@dataclass
class AppConfig:
    paths: Paths
    video: Video
    detect: Detect                # default (used if per-mode not provided)
    colors_hsv: HSVRanges
    # Optional per-mode overrides & masks
    image_detect: Detect | None = None
    camera_detect: Detect | None = None
    image_mask: Mask | None = None
    camera_mask: Mask | None = None

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = Paths(**cfg.get("paths", {}))
    video = Video(**cfg.get("video", {}))
    detect = Detect(**cfg.get("detect", {}))
    colors = HSVRanges(**cfg.get("colors_hsv", {}))

    # Optional sections; fall back to defaults if absent
    image_detect = Detect(**cfg["image_detect"]) if "image_detect" in cfg else None
    camera_detect = Detect(**cfg["camera_detect"]) if "camera_detect" in cfg else None
    image_mask = Mask(**cfg["image_mask"]) if "image_mask" in cfg else None
    camera_mask = Mask(**cfg["camera_mask"]) if "camera_mask" in cfg else None

    return AppConfig(
        paths=paths,
        video=video,
        detect=detect,
        colors_hsv=colors,
        image_detect=image_detect,
        camera_detect=camera_detect,
        image_mask=image_mask,
        camera_mask=camera_mask,
    )
