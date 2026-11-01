"""Jalankan semua modul uji dan cetak ringkasan lulus / gagal / dilewati per modul.

Dari folder _laya_eda_v2:  python -m tests.semua
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["test_steps", "test_tools", "test_requirements", "test_extra", "test_terima", "test_app"]


def main() -> None:
    env = {**os.environ, "LAYA_OUTPUT": tempfile.mkdtemp(prefix="laya_output_")}
    total, rows = {"lulus": 0, "gagal": 0, "dilewati": 0}, []
    for module in MODULES:
        run = subprocess.run([sys.executable, "-m", f"tests.{module}"], cwd=ROOT, capture_output=True, text=True,
                             env=env)
        found = re.findall(r"lulus (\d+), gagal (\d+), dilewati (\d+)", run.stdout)
        counts = dict(zip(total, map(int, found[-1]))) if found else {"lulus": 0, "gagal": 1, "dilewati": 0}
        for key in total:
            total[key] += counts[key]
        skipped = [line for line in run.stdout.splitlines() if line.startswith(("DILEWATI", "GAGAL"))]
        rows.append((module, counts, skipped, run.returncode))
    for module, counts, skipped, code in rows:
        print(f"{module}: lulus {counts['lulus']}, gagal {counts['gagal']}, dilewati {counts['dilewati']}"
              + ("" if code == 0 or counts["gagal"] else f" (exit {code})"))
        for line in skipped:
            print("   ", line[:220])
    print(f"TOTAL: lulus {total['lulus']}, gagal {total['gagal']}, dilewati {total['dilewati']}")
    sys.exit(1 if total["gagal"] else 0)


if __name__ == "__main__":
    main()
