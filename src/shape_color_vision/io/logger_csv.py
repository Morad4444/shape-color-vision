from pathlib import Path
import csv
from datetime import datetime
from typing import Set, Tuple


HEADER = ["timestamp", "shape", "color", "confidence", "source", "name"]


class CSVLogger:
    """
    CSVLogger — append detection events to a CSV file.

    Responsibilities:
    • Ensure the CSV file exists with the correct header.
    • Append new rows in a simple, append-only fashion.
    • Optionally suppress duplicate events to avoid spamming the log.

    Duplicate suppression:
    • Each unique (shape, color, source, name) is only written ONCE per logger
      instance. If the same combination is detected again (e.g. same object
      in front of the camera for many frames), it will NOT be logged again.
    """

    def __init__(self, csv_path: str):
        self.path = Path(csv_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Track which detections we've already written
        self._seen: Set[Tuple[str, str, str, str]] = set()

        # Create file with header if it does not exist yet
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(HEADER)

    def log(self, shape: str, color: str, confidence: float, source: str, name: str):
        """
        Append a detection entry to the CSV log file.

        Duplicate policy:
        • If (shape, color, source, name) has already been logged by this
          logger instance, the event is ignored.
        """
        key = (shape, color, source, name)
        if key in self._seen:
            # Already logged this combination once → skip
            return

        self._seen.add(key)

        with self.path.open("a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(timespec="seconds"),
                shape,
                color,
                f"{confidence:.2f}",
                source,
                name,
            ])
