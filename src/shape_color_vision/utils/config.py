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
    min_area: int = 800

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
    """Load YAML config and map it to dataclasses."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r") as f:
        cfg = yaml.safe_load(f) or {}

    # Expect the same keys as in configs/default.yaml
    paths = Paths(**cfg["paths"])
    video = Video(**cfg["video"])
    detect = Detect(**cfg["detect"])
    colors = HSVRanges(**cfg["colors_hsv"])

    return AppConfig(
        paths=paths,
        video=video,
        detect=detect,
        colors_hsv=colors,
    )
