from pathlib import Path
import typer
from typing import Optional
from .utils.config import load_config, AppConfig

app = typer.Typer(help="Shape & Color Vision")

def load(cfg_path: str) -> AppConfig:
    cfg = load_config(cfg_path)
    # ensure folders exist
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.image_dir).mkdir(parents=True, exist_ok=True)
    return cfg

@app.command()
def image(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to YAML config"),
    image_dir: Optional[str] = typer.Option(None, help="Override image directory"),
    save_output: bool = typer.Option(False, help="Save annotated images"),
    log_file: Optional[str] = typer.Option(None, help="Override CSV log file"),
):
    """
    Analyze all images in a folder and log detections.
    (Detection logic will be added in the next step.)
    """
    cfg = load(config)
    if image_dir: cfg.paths.image_dir = image_dir
    if log_file:  cfg.paths.log_csv = log_file
    if save_output: cfg.video.save_output = True

    typer.echo(f"[IMAGE] dir={cfg.paths.image_dir} -> log={cfg.paths.log_csv} save={cfg.video.save_output}")
    typer.echo("Pipeline not implemented yet — coming next step.")

@app.command()
def camera(
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="Path to YAML config"),
    index: Optional[int] = typer.Option(None, help="Camera index"),
    show_window: bool = typer.Option(True, help="Show preview window"),
    log_file: Optional[str] = typer.Option(None, help="Override CSV log file"),
):
    """
    Run live detection on webcam and log detections.
    (Detection logic will be added in the next step.)
    """
    cfg = load(config)
    if index is not None: cfg.video.camera_index = index
    cfg.video.show_window = show_window
    if log_file:  cfg.paths.log_csv = log_file

    typer.echo(f"[CAMERA] index={cfg.video.camera_index} show={cfg.video.show_window} -> log={cfg.paths.log_csv}")
    typer.echo("Pipeline not implemented yet — coming next step.")

def main():
    app()

if __name__ == "__main__":
    main()
