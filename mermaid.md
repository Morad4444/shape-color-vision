## C4 Model 

## 1. Context Diagramm

``` mermaid
graph LR
    User([User])
    subgraph External
        Camera[[Webcam]]
        ImgFolder[[Image Folder]]
        FileSystem[(File System / CSV Logs)]
        Libs[(OpenCV, NumPy, Typer, PyYAML)]
    end

    App[Shape & Color Vision Application]

    User -->|CLI commands| App
    Camera -->|Live video stream| App
    ImgFolder -->|Image files| App
    App -->|detections.csv| FileSystem
    App -.->|uses| Libs

```
## 2. Container Diagramm

``` mermaid
graph TB
    subgraph "CLI Application (Python 3.x, Typer)"
        Main["main.py (CLI Entry Point)"]
        Pipeline["pipeline.py (Main Pipeline)"]
        Color["colors.py (Color Detection)"]
        Shape["shapes.py (Shape Detection)"]
        Viz["viz.py (Visualization)"]
        Log["logger_csv.py (CSV Logger)"]
        Cfg["config.py (YAML Loader)"]
    end

    Cam[[Webcam]]
    Img[[Image Folder]]
    CSV[(logs/detections.csv)]
    YAML[[configs/*.yaml]]

    Main --> Pipeline
    Main --> Cfg
    Cfg --> YAML

    Pipeline --> Color
    Pipeline --> Shape
    Pipeline --> Viz
    Pipeline --> Log
    Pipeline --> Cam
    Pipeline --> Img
    Log --> CSV

```

## 3. Component Diagramm
``` mermaid
graph LR
    Cfg["AppConfig"]
    Src["Frame Source (Webcam / Image)"]
    HSV["HSV Conversion & Morphology"]
    Cnt["Contour Detection & Filtering"]
    CC["ColorClassifier"]
    SC["ShapeClassifier"]
    Ann["Annotator (viz.py)"]
    Log["CsvLogger"]
    CSV["detections.csv"]

    Cfg --> HSV
    Src --> HSV
    HSV --> Cnt
    Cnt --> CC
    Cnt --> SC
    CC --> Ann
    SC --> Ann
    Ann --> Log
    Log --> CSV
```

## 4. Code Diagramm

``` mermaid
classDiagram
    class Pipeline {
        +run(mode, config)
        -process_frame(frame): Detection[]
    }

    class ColorClassifier {
        +classify(hsvRegion): ColorLabel
    }

    class ShapeClassifier {
        +classify(contour): ShapeLabel
    }

    class CsvLogger {
        +append(timestamp, shape, color, confidence, source, filename)
    }

    class Annotator {
        +draw(frame, detections): frame
    }

    Pipeline --> ColorClassifier
    Pipeline --> ShapeClassifier
    Pipeline --> CsvLogger
    Pipeline --> Annotator

```

