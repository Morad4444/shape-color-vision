# tests/test_reference_outputs.py
import csv
import sys
from pathlib import Path
from collections import Counter

# --- make 'src' importable regardless of where pytest runs ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shape_color_vision.utils.config import load_config
from shape_color_vision.pipeline import analyze_image

# ---------- Expected reference outputs (exactly as in your screenshots) ----------
EXPECTED = {
    "shapes_test1.png": {
        ("BLUE", "Circle"),
        ("UNKNOWN", "Unknown"),     # pink oval
        ("RED", "Triangle"),
        ("YELLOW", "Rectangle"),
        ("GREEN", "Square"),
        ("VIOLET", "Unknown"),      # violet diamond
    },
    "shapes_test2.png": {
        ("RED", "Circle"),
        ("GREEN", "Triangle"),
        ("BLUE", "Square"),
        ("UNKNOWN", "Rectangle"),   # orange rectangle
        ("GREEN", "Unknown"),       # green oval
        ("VIOLET", "Unknown"),      # violet oval
        ("RED", "Unknown"),         # red parallelogram
    },
    "shapes_test3.png": {
        ("BLUE", "Square"),
        ("VIOLET", "Rectangle"),
        ("YELLOW", "Triangle"),
        ("UNKNOWN", "Unknown"),     # brown parallelogram
        ("GREEN", "Circle"),
        ("UNKNOWN", "Unknown"),     # orange oval
        ("RED", "Unknown"),         # red diamond
        ("GREEN", "Unknown"),       # green triangle
        ("BLUE", "Unknown"),        # blue hexagon
        ("UNKNOWN", "Unknown"),     # pink pentagon
        ("GREEN", "Unknown"),       # green star
        ("UNKNOWN", "Unknown"),     # orange star
    },
}

# If your images are named a bit differently locally, list aliases here:
ALIASES = {
    "shapes_test1.png": ["shapes_test1.png", "shapes_test.png"],
    "shapes_test2.png": ["shapes_test2.png"],
    "shapes_test3.png": ["shapes_test3.png"],
}

def _find_sample(basenames) -> Path:
    for n in basenames:
        p = ROOT / "data" / "samples" / n
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {basenames} found in data/samples/")

def _load_cfg(tmp_path):
    cfg = load_config(str(ROOT / "configs" / "default.yaml"))
    # keep test artifacts isolated
    cfg.paths.output_dir = str(tmp_path / "out")
    cfg.paths.log_csv    = str(tmp_path / "det.csv")
    cfg.video.save_output = False  # tests compare CSV, not images
    return cfg

def _read_csv_rows(csv_path: Path, image_name: str):
    pairs = []
    if not csv_path.exists():
        return pairs
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("source") == "IMAGE" and row.get("name") == image_name:
                color = (row.get("color") or "").strip().upper()
                shape = (row.get("shape") or "").strip().capitalize()
                # normalize "UNKNOWN" shape casing
                if shape.lower() == "unknown":
                    shape = "Unknown"
                pairs.append((color, shape))
    return pairs

def _format_table(expected_set, detected_set):
    def fmt(title, items):
        if not items:
            return f"{title}: (none)"
        return f"{title}:\n  - " + "\n  - ".join([f"{c} {s}" for c, s in sorted(items)])
    missing   = expected_set - detected_set
    unexpected= detected_set - expected_set
    lines = [
        f"Expected total: {len(expected_set)} | Detected total: {len(detected_set)}",
        fmt("Missing", missing),
        fmt("Unexpected", unexpected),
    ]
    return "\n".join(lines)

def _run_one(image_key: str, tmp_path):
    img_path = _find_sample(ALIASES[image_key])
    cfg = _load_cfg(tmp_path)

    # fresh CSV each run
    csv_path = Path(cfg.paths.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        csv_path.unlink()

    analyze_image(str(img_path), cfg)
    detected = set(_read_csv_rows(csv_path, img_path.name))
    expected = EXPECTED[image_key]

    # Quick sanity: all shapes/cols are uppercase/valid
    assert all(isinstance(x, tuple) and len(x) == 2 for x in detected)

    # Main check: exact set match (these are stable screenshots)
    if detected != expected:
        print("\n" + _format_table(expected, detected))  # shown in pytest output
    assert detected == expected, f"{image_key} detections differ (see table above)."

# -------------------- Tests --------------------

def test_sample1(tmp_path):
    _run_one("shapes_test1.png", tmp_path)

def test_sample2(tmp_path):
    _run_one("shapes_test2.png", tmp_path)

def test_sample3(tmp_path):
    _run_one("shapes_test3.png", tmp_path)
