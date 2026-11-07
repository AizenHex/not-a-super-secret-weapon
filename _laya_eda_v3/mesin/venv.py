"""Python venv untuk tugas model (G1). Laya.exe sendiri tidak membawa scikit-learn/cleanlab, jadi tugas model
dijalankan python venv (`setup.bat`) di proses terpisah."""
import os
import sys
from pathlib import Path


def skrip_tugas():
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "sumber" / "tugas.py"
    return Path(__file__).parent / "tugas.py"


def cari_python(*awal, pilihan=None):
    """Urutan: pilihan pengguna, venv aktif, python yang menjalankan Laya (bukan exe), lalu `.venv`/`venv` di tiap
    folder awal dan semua folder di atasnya (cara `cari_python.bat`). Tidak ketemu = None."""
    calon = [Path(pilihan)] if pilihan else []
    if os.environ.get("VIRTUAL_ENV"):
        calon.append(Path(os.environ["VIRTUAL_ENV"]) / "Scripts" / "python.exe")
    if not getattr(sys, "frozen", False):
        calon.append(Path(sys.executable))
    for a in awal:
        if a:
            p = Path(a).resolve()
            for d in (p, *p.parents):
                calon += [d / ".venv" / "Scripts" / "python.exe", d / "venv" / "Scripts" / "python.exe"]
    return next((str(c) for c in calon if c.is_file()), None)
