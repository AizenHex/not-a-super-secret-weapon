"""10 dataset tabular umum untuk uji G1–G7 (VISI.md bagian 0b). Disimpan di `_laya_eda/data/umum/` (diabaikan git).

Diunduh dari cermin GitHub (Hugging Face / OpenML / UCI diblokir di sesi cloud 2026-11-02). Fitness = unggahan
pengguna (tidak ada URL publik): bila filenya tidak ada, uji dataset itu dilewati.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

FOLDER = Path(__file__).resolve().parents[2] / "_laya_eda" / "data" / "umum"
RAW = "https://raw.githubusercontent.com/"
# nama: (url, label, temuan yang WAJIB muncul: jenis temuan core/findings.py)
DATASETS = {
    "titanic": (RAW + "datasciencedojo/datasets/master/titanic.csv", "Survived",
                {"kosong", "id", "kardinalitas_tinggi"}),
    "penguins": (RAW + "mwaskom/seaborn-data/master/penguins.csv", "species", {"kosong", "kategori_sedikit"}),
    "diamonds": (RAW + "mwaskom/seaborn-data/master/diamonds.csv", "price", {"duplikat", "kategori_sedikit"}),
    "taxis": (RAW + "mwaskom/seaborn-data/master/taxis.csv", "tip", {"tanggal", "kosong"}),
    "mpg": (RAW + "mwaskom/seaborn-data/master/mpg.csv", "mpg", {"kosong", "kardinalitas_tinggi"}),
    "adult": (RAW + "jbrownlee/Datasets/master/adult-all.csv", None, {"tanpa_judul"}),
    "ames": (RAW + "wblakecannon/ames/master/data/housing.csv", "SalePrice", {"kosong", "id", "kosong_bermakna"}),
    "heart": (RAW + "kb22/Heart-Disease-Prediction/master/dataset.csv", "target", {"kategori_angka"}),
    "telco": (RAW + "IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv", "Churn",
              {"angka_teks", "id"}),
    "fitness": (None, "Fitness Type", {"bocor", "teks_panjang", "id"}),
}


def path(name: str) -> Path | None:
    """Path CSV dataset; diunduh sekali bila belum ada. None = tidak tersedia (tanpa jaringan / tanpa URL)."""
    target = FOLDER / f"{name}.csv"
    if target.exists():
        return target
    url = DATASETS[name][0]
    if url is None:
        return None
    FOLDER.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, target)
    except OSError:
        target.unlink(missing_ok=True)
        return None
    return target
