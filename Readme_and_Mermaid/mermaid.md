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

```` 
