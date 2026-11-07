"""Laya v3: aplikasi desktop persiapan data. Visi dan fitur: VISI.md.

    python laya.py                 # buka proyek terakhir
    python laya.py <folder>        # buka folder data
    python laya.py --uji <folder>  # buka semua tabel, catat waktu ke laya_log.txt, keluar (cek exe/PC)
"""
import time

T0 = time.perf_counter()

import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402


def folder_aplikasi():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def folder_kerja():
    """Proyek + log di samping aplikasi; kalau tidak bisa ditulis (mis. Program Files), di LOCALAPPDATA/Laya."""
    for f in (folder_aplikasi() / "proyek", Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Laya"):
        try:
            f.mkdir(parents=True, exist_ok=True)
            (f / ".cek_tulis").write_text("ok")
            (f / ".cek_tulis").unlink()
            return f
        except OSError:
            continue
    raise SystemExit("Tidak ada folder yang bisa ditulis untuk proyek dan log.")


def main():
    kerja = folder_kerja()
    logging.basicConfig(filename=kerja / "laya_log.txt", level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s %(levelname)s %(message)s")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from tampilan.jendela import Jendela

    app = QApplication(sys.argv)
    app.setApplicationName("Laya")

    def kait(tipe, nilai, tb):  # galat tak tertangkap: catat + tampilkan, aplikasi jalan terus
        logging.error("galat tak tertangkap", exc_info=(tipe, nilai, tb))
        QMessageBox.critical(None, "Laya: galat", f"{tipe.__name__}: {nilai}\n\nRincian di {kerja / 'laya_log.txt'}")

    sys.excepthook = kait
    w = Jendela(kerja)
    w.show()
    logging.info("jendela tampil %.2f s setelah start (frozen=%s)", time.perf_counter() - T0,
                 getattr(sys, "frozen", False))
    arg = sys.argv[1:]
    if arg[:1] == ["--uji"] and len(arg) > 1:
        QTimer.singleShot(0, lambda: w.uji_otomatis(arg[1]))
    elif arg:
        w.buka_folder(arg[0])
    else:
        w.buka_terakhir()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
