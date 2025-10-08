from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Paths:
    image_dir: str
    output_dir: str
    log_csv: str

@dataclass
class Video:
    camera_index: int = 0
    show_window: bool = True
    save_output: bool = False

@dataclass
class Detect:
    min_area: int = 1200
    min_solidity: float = 0.85
    pastel_s_thresh: int = 120
    min_circularity: float = 0.78  # currently used in shapes.py constants

@dataclass
class HSVRanges:
    red1: list
    red2: list
    green: list
    blue: list
    yellow: list
    violet: list

@dataclass
class AppConfig:
    paths: Paths
    video: Video
    detect: Detect
    colors_hsv: HSVRanges

def load_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    return AppConfig(
        paths=Paths(**cfg["paths"]),
        video=Video(**cfg["video"]),
        detect=Detect(**cfg["detect"]),
        colors_hsv=HSVRanges(**cfg["colors_hsv"]),
    )
