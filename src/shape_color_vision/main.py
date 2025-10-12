import cv2
from pathlib import Path
import typer
from typing import Optional
from .utils.config import load_config, AppConfig
from .pipeline import analyze_image, analyze_dir, analyze_frame
from .io.logger_csv import CSVLogger


app = typer.Typer(help="Shape & Color Vision")

def load(cfg_path: str) -> AppConfig:
    cfg = load_config(cfg_path)
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.image_dir).mkdir(parents=True, exist_ok=True)
    return cfg

@app.command()
def image(
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    image_dir: Optional[str] = typer.Option(None),
    save_output: bool = typer.Option(False),
    log_file: Optional[str] = typer.Option(None),
):
    cfg = load(config)
    if image_dir: cfg.paths.image_dir = image_dir
    if log_file:  cfg.paths.log_csv = log_file
    if save_output: cfg.video.save_output = True

    paths = analyze_dir(cfg.paths.image_dir, cfg)
    if not paths:
        typer.echo(f"No images found in {cfg.paths.image_dir}")
        raise typer.Exit(code=1)

    for p in paths:
        out = analyze_image(p, cfg)
        if cfg.video.show_window:
            cv2.imshow("Result", out); cv2.waitKey(500)
        if cfg.video.save_output:
            out_path = Path(cfg.paths.output_dir) / f"annotated_{Path(p).name}"
            cv2.imwrite(str(out_path), out)
    if cfg.video.show_window:
        cv2.destroyAllWindows()
    typer.echo(f"Processed {len(paths)} image(s). Results logged to {cfg.paths.log_csv}")

@app.command()
def camera(
    config: str = typer.Option("configs/default.yaml"),
    index: int = typer.Option(0, help="Webcam index"),
    save_output: bool = typer.Option(False, help="Save annotated video frames"),
):
    cfg = load_config(config)
    if save_output:
        cfg.video.save_output = True
        Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        typer.secho("Could not open camera", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    logger = CSVLogger(cfg.paths.log_csv)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out = analyze_frame(frame, cfg, logger)
            cv2.imshow("Shape & Color Vision (press q to quit)", out)

            if save_output:
                # optional: write individual PNGs, or wire a VideoWriter if you prefer
                pass

            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    app()

if __name__ == "__main__":
    main()


# --- NEW: console-script entrypoints ---
def image_main():
    """Console script wrapper for the `image` command."""
    import typer
    typer.run(image)

def camera_main():
    """Console script wrapper for the `camera` command."""
    import typer
    typer.run(camera)
