"""Bantuan uji: lokasi data (riil bila ada, kalau tidak sintetis), status 'dilewati', dan runner sederhana.

Runner tanpa pytest (pytest tidak ada di requirements.txt): setiap fungsi `test_*` di modul dijalankan, hasilnya
dihitung lulus / gagal / dilewati. Uji yang butuh data riil memanggil `perlu_data_riil()`; tanpa data riil uji
itu DILEWATI dengan pesan, tidak dihapus.
"""
from __future__ import annotations

import inspect
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests import sintetis  # noqa: E402

LABEL = sintetis.LABEL
TRAIN_RIIL = ROOT.parent / "_laya_eda" / "data" / "train.csv"
ADA_DATA_RIIL = TRAIN_RIIL.exists()


class Lewati(Exception):
    """Uji tidak bisa dijalankan di lingkungan ini (mis. data riil atau library tidak ada)."""


def perlu_data_riil() -> None:
    if not ADA_DATA_RIIL:
        raise Lewati(f"butuh data riil {TRAIN_RIIL} (tidak di-commit; unggah terpisah untuk uji terima)")


def perlu_modul(*names: str) -> None:
    import importlib.util

    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise Lewati(f"library belum terpasang: {', '.join(missing)}")


def folder_sintetis(n: int = 3000) -> dict:
    """train/test/sample_submission sintetis di folder temp (dibuat sekali per ukuran)."""
    folder = Path(tempfile.gettempdir()) / f"laya_sintetis_{n}"
    paths = {"train": folder / "train.csv", "test": folder / "test.csv", "sample": folder / "sample_submission.csv"}
    if not all(path.exists() for path in paths.values()):
        paths = sintetis.tulis(folder, n)
    return paths


def train_csv() -> Path:
    """Data riil bila ada; kalau tidak, train sintetis (3.000 baris)."""
    return TRAIN_RIIL if ADA_DATA_RIIL else folder_sintetis()["train"]


def sumber_data() -> str:
    return "data riil" if ADA_DATA_RIIL else "data sintetis (tests/sintetis.py)"


def jalankan(namespace: dict, only: list[str] | None = None) -> dict:
    """Jalankan semua `test_*` di namespace modul. Kembalikan hitungan; exit code 1 bila ada yang gagal."""
    counts = {"lulus": 0, "gagal": 0, "dilewati": 0}
    print(f"[{namespace.get('__name__')}] sumber data: {sumber_data()}")
    for name, check in list(namespace.items()):
        if not name.startswith("test_") or not callable(check) or (only and name not in only):
            continue
        start = time.time()
        try:
            if "tmp_path" in inspect.signature(check).parameters:
                with tempfile.TemporaryDirectory() as folder:
                    check(Path(folder))
            else:
                check()
        except Lewati as reason:
            counts["dilewati"] += 1
            print(f"DILEWATI {name}: {reason}")
            continue
        except Exception:
            counts["gagal"] += 1
            print(f"GAGAL {name} ({time.time() - start:.1f} s)")
            traceback.print_exc()
            continue
        counts["lulus"] += 1
        print(f"lulus {name} ({time.time() - start:.1f} s)")
    print(f"[{namespace.get('__name__')}] lulus {counts['lulus']}, gagal {counts['gagal']}, "
          f"dilewati {counts['dilewati']}")
    return counts


def main(namespace: dict) -> None:
    only = sys.argv[1:] or None
    counts = jalankan(namespace, only)
    sys.exit(1 if counts["gagal"] else 0)


__all__ = ["ROOT", "LABEL", "TRAIN_RIIL", "ADA_DATA_RIIL", "Lewati", "perlu_data_riil", "perlu_modul",
           "folder_sintetis", "train_csv", "sumber_data", "jalankan", "main"]
