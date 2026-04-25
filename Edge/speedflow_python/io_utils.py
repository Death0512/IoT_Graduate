# speedflow/io_utils.py
import csv, os
from pathlib import Path

class CSVLogger:
    def __init__(self, path, header=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        if header: self.w.writerow(header)
    def write(self, row):
        self.w.writerow(row); self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass
