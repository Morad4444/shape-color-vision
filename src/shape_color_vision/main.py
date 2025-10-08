import cv2
from pathlib import Path
import typer
from typing import Optional
from .utils.config import load_config, AppConfig
from .pipeline import analyze_image, analyze_dir

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
    config: str = typer.Option("configs/default.yaml", "--config", "-c"),
    index: Optional[int] = typer.Option(None),
    show_window: bool = typer.Option(True),
    log_file: Optional[str] = typer.Option(None),
):
    # (We’ll implement live mode after image mode is verified)
    typer.echo("Camera mode will be implemented after image mode verification.")

def main():
    app()

if __name__ == "__main__":
    main()
