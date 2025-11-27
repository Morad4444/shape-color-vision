# Shape & Color Vision
A lightweight and modular shape & color recognition pipeline using OpenCV and Python.

This project uses **Computer Vision** methods to detect basic geometric shapes  
(e.g., circle, square, triangle, rectangle) and **colors** (red, green, blue, yellow, violet, etc.) automatically.  
It can be executed on both **single images** and **live camera streams** (e.g., a webcam).

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Installation & Execution](#installation--execution)
4. [How It Works](#how-it-works)
5. [Module Overview](#module-overview)
6. [Example Output](#example-output)
7. [Developer Notes](#developer-notes)
8. [Learning Objectives](#learning-objectives)
9. [License](#license)

## Project Overview

The system analyzes an image or video frame, detects objects based on their **shape and color**,  
visually labels them in the output image, and logs the detection results to a **CSV file**.

### Processing Pipeline
1. **Mask generation** in HSV color space based on saturation and brightness thresholds.
2. **Contour detection** and geometric filtering.
3. For each detected contour:
   - Color analysis using median HSV values (`colors.py`)
   - Shape classification based on contour metrics (`shapes.py`)
4. Rendering labeled results into the output image.
5. Logging every detection into a CSV file (`logs/detections.csv`).

## Project Structure

```
shape_color_vision/
│
├── main.py               # CLI (Command Line Interface)
├── pipeline.py           # Processing Pipeline
│
├── colors.py             # Color classification (HSV-based)
├── shapes.py             # Shape classification (contour-based)
├── logger_csv.py         # CSV logger for detection results
├── viz.py                # Visualization (labeling, contour drawing)
├── config.py             # Load YAML configuration
├── __init__.py           # Package info (__version__, __app_name__)
│
├── data/
│   ├── samples/          # Sample images
│   └── output/           # Annotated output images
│
└── logs/
    └── detections.csv    # CSV log
```

## Installation & Execution

### Install dependencies
```bash
pip install -r requirements.txt
```

**Minimal packages:**
```bash
opencv-python
numpy
typer
pyyaml
```

## Quick Start

Run shape & color detection on sample images:
```bash
python -m shape_color_vision.main image --config configs/default.yaml --save-output
```

Run detection with a webcam:
```bash
python -m shape_color_vision.main camera --config configs/default.yaml
```

Press **q** to quit the live camera view.

## CLI Help

```bash
python -m shape_color_vision.main --help
```

```
Usage: main [OPTIONS] COMMAND [ARGS]...

Commands:
  image   Run detection on a directory of images
  camera  Run detection on a webcam stream
```

## How It Works

1. Load parameters from the configuration file (`default.yaml`)
2. Convert the image into **HSV color space**
3. Use morphology operations to mask regions of interest
4. Process each contour:
   - `classify_color()` → color
   - `classify_shape()` → geometric shape
5. Display results and log them into the CSV file

## Module Overview

| Module | Description |
|-------|--------------|
| **`colors.py`** | Robust color classification using median hue values. Ambiguous segments (e.g., orange, magenta) are labeled as *unknown*. |
| **`shapes.py`** | Identifies geometric shapes using contour perimeter, vertex count, aspect ratio, and circularity. |
| **`logger_csv.py`** | Writes each detection with timestamp, shape, color, and confidence into the CSV log. |
| **`viz.py`** | Draws labels and contours onto the image. |
| **`config.py`** | Loads YAML configuration files and creates an `AppConfig` object. |
| **`pipeline.py`** | Core logic: detection, classification, and logging. |
| **`main.py`** | CLI commands `image` and `camera` via the `typer` framework. |

## Example Output (CSV Log)

| timestamp | shape | color | confidence | source | name |
|-----------|--------|--------|-------------|---------|--------|
| 2025-10-24T14:30:01 | Circle | red | 0.95 | IMAGE | samples.png |
| 2025-10-24T14:31:11 | Square | blue | 0.88 | CAMERA | webcam |

## Developer Notes

- **Color space:** All computations are performed in HSV (`cv2.cvtColor(BGR → HSV)`).
- **Shape metrics:** Combination of *Circularity*, *Solidity*, and *Aspect Ratio* for reliable classification.
- **"Unknown" class:** Used for low saturation or ambiguous color tones.
- **Code quality:** Written according to **PEP8**, with type hints and clean modular structure.
- **Optimization:** Can be easily adapted for Raspberry Pi or embedded systems.
- **Logging:** CSV files are automatically created if they do not exist.

## Learning Objectives

- Understanding the HSV color space  
- Extracting and analyzing contours  
- Applying geometric shape metrics  
- Building a simple image processing pipeline  
- Logging and visualizing detections  

---

**Academic Note:**  
This project was created as part of a course assignment.  
The goal is to demonstrate fundamental computer vision techniques  
for shape and color recognition — not to provide a production-grade system.

## License

This project was developed for educational and research purposes.  
Use is permitted as long as the original authors are credited.

---

**Authors:** Morad Younis & Emrah Tekin  
**Version:** 0.1.0  
**FHGR – Photonics Engineering / Software Engineering Module**
