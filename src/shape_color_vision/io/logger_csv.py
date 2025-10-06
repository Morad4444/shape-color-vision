from pathlib import Path
import csv
from datetime import datetime
HEADER = ["timestamp","pattern","color","confidence","source","name"]
class CSVLogger:
    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def log(self, pattern: str, color: str, confidence: float, source: str, name: str):
        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                pattern, color, f"{confidence:.2f}", source, name
            ])
