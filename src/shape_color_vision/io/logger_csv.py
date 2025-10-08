from pathlib import Path
import csv
from datetime import datetime

HEADER = ["timestamp", "shape", "color", "confidence", "source", "name"]

class CSVLogger:
    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def log(self, shape: str, color: str, confidence: float, source: str, name: str):
        """Append a detection entry to the CSV log file."""
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                shape,
                color,
                f"{confidence:.2f}",
                source,
                name,
            ])
