"""Resep: semua keputusan pengguna dalam satu dict yang bisa disimpan sebagai JSON.

Resep adalah satu-satunya sumber kebenaran. Tab-tab hanya mengubah resep; kode pipeline
dibuat dari resep (core/codegen.py), lalu kode itulah yang dijalankan aplikasi.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

MISSING_MARKERS = ["", "-", "--", "?", "nan", "null", "none", "n/a", "na", "tidak tahu"]
SENTINEL_FLOOR = 2_000_000_000          # 2^31 - 1 = 2.147.483.647 dan variannya
MATERIAL_SEPARATORS = r"\s*(?:,|&|/|\+|;|\bdan\b|\batau\b|\(|\))\s*"

OUTLIER_METHODS = {
    "iqr": "IQR (Tukey): Q1 − k·IQR sampai Q3 + k·IQR, skala asli",
    "iqr_log": "IQR di skala log10 (harga, luas); nol tidak dihitung",
    "mad": "MAD (modified z-score): |0,6745·(x − median)/MAD| > k",
    "mad_log": "MAD di skala log10; nol tidak dihitung",
}
OUTLIER_ACTIONS = {
    "flag": "hanya tandai (data tidak berubah)",
    "clip": "potong ke batas (train + test)",
    "nan": "jadikan kosong (lalu diisi di langkah isi kosong)",
    "drop": "buang baris (train saja)",
}
IMPUTE_STRATEGIES = {
    "none": "biarkan kosong",
    "median": "median train",
    "mean": "rata-rata train",
    "most_frequent": "nilai tersering train (modus)",
    "constant": "konstanta",
}
ENCODERS = {
    "onehot": "one-hot (sklearn OneHotEncoder): satu kolom 0/1 per nilai",
    "ordinal": "kode angka (sklearn OrdinalEncoder): urut abjad, nilai baru = -1",
    "ordered": "berurutan: kode angka mengikuti urutan yang kamu tentukan",
    "materials": "one-hot per bahan (isian bebas): 'genteng, asbes' → 2 kolom bernilai 1",
    "string": "skrub StringEncoder: TF-IDF n-gram huruf + SVD (kolom angka tanpa nama)",
    "minhash": "skrub MinHashEncoder: MinHash n-gram huruf (kolom angka tanpa nama)",
    "gap": "skrub GapEncoder: topik laten dari n-gram (kolom angka per topik)",
}
SPLITS = {
    "none": "tidak di-split",
    "random": "acak (train_test_split)",
    "stratified": "stratified (proporsi label sama)",
    "kfold": "k-fold (StratifiedKFold / KFold), nomor fold disimpan terpisah",
}


def new_recipe(label: str | None = None) -> dict:
    """Resep kosong: tidak ada yang mengubah data sampai pengguna memilih (prinsip P6)."""
    return {
        "label": label,
        "drop_columns": [],
        "number_columns": [],
        "text_columns": [],
        "indonesian_numbers": True,
        "missing_markers": list(MISSING_MARKERS),
        "sentinels": {},              # kolom -> batas bawah nilai penjaga
        "zero_as_missing": [],
        "blank_as": {},               # kolom -> isi untuk sel kosong ("Tidak")
        "category_maps": {},          # kolom -> {ejaan asli: ejaan baku}
        "free_text": {},              # kolom -> {kata salah: kata baku}
        "outliers": {},               # kolom -> {"method", "factor", "action"}
        "drop_duplicates": False,
        "max_missing_per_row": None,
        "drop_queries": [],           # aturan pandas query: baris cocok dibuang (train)
        "impute": {},                 # kolom -> {"strategy", "value"}
        "encoding": {},               # kolom -> {"method", "order", "n_components", "min_share"}
        "split": {"method": "none", "valid_size": 0.2, "folds": 5, "seed": 42},
    }


def merge_defaults(recipe: dict) -> dict:
    """Resep lama yang kurang kunci dilengkapi dari resep kosong."""
    full = new_recipe(recipe.get("label"))
    full.update(copy.deepcopy(recipe))
    return full


def save(recipe: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: str | Path) -> dict:
    return merge_defaults(json.loads(Path(path).read_text(encoding="utf-8")))
