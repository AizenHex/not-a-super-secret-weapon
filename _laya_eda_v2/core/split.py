"""Split data: train → bagian latih + valid, SEBELUM langkah EDA apa pun.

Urutan buku teks: baca data → split → langkah EDA (statistik dari bagian latih SAJA, dipakai ke valid
dan test) → ekspor. Valid berperan seperti test yang labelnya kamu punya: tidak pernah dipelajari, tidak
pernah dibuang barisnya, jadi skor di valid = perkiraan jujur skor di test.

Kode split dihasilkan sebagai teks (tanpa def) dan dijalankan APA ADANYA oleh aplikasi (sama seperti
langkah), jadi baris valid di aplikasi = baris valid di skrip yang diunduh.

config = {"method": "acak" | "kelompok" | "waktu", "share": 0.2, "seed": 42,
          "stratify": bool (acak), "column": str (kelompok/waktu; ROW_GROUP = baris kembar), "numeric": bool (waktu)}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROW_GROUP = "(baris kembar)"     # kelompok = isi seluruh fitur: baris yang sama persis tidak terpisah
NO_SPLIT = "valid = train.iloc[:0]   # tanpa split: valid kosong (pilih metode di halaman Split)"


def describe(config: dict | None) -> str:
    """Teks pendek untuk sidebar dan judul blok kode."""
    if not config:
        return "tanpa split"
    share = f"{config['share']:.0%} ke valid"
    if config["method"] == "acak":
        return f"acak, {share}" + (", stratified" if config.get("stratify") else "")
    if config["method"] == "kelompok":
        return f"kelompok {config['column']}, {share}"
    return f"waktu {config['column']}, {share} terbaru"


def split_code(config: dict | None) -> str:
    """Blok kode pandas/sklearn lurus. Butuh `train`, `LABEL`, `pd` sudah ada; menghasilkan `train` dan `valid`."""
    if not config:
        return NO_SPLIT
    method, share, seed = config["method"], float(config["share"]), int(config.get("seed", 42))
    head = f"# split {describe(config)} (random_state={seed}); statistik langkah sesudah ini dari train saja\n"
    if method == "acak":
        stratify = "train[LABEL]" if config.get("stratify") else "None"
        return (head +
                "from sklearn.model_selection import train_test_split\n"
                f"BAGIAN_VALID = {share}\n"
                f"train, valid = train_test_split(train, test_size=BAGIAN_VALID, random_state={seed}, stratify={stratify})\n"
                "train, valid = train.sort_index().copy(), valid.sort_index().copy()   # kembalikan urutan baris asli\n"
                'print("train", train.shape, "valid", valid.shape)')
    if method == "kelompok":
        if config["column"] == ROW_GROUP:
            groups = ("kelompok = pd.util.hash_pandas_object(train.drop(columns=[LABEL], errors=\"ignore\"), index=False)"
                      "   # baris kembar = satu kelompok\n")
        else:
            groups = (f"KOLOM_KELOMPOK = {config['column']!r}\n"
                      "kelompok = train[KOLOM_KELOMPOK].astype(str)   # kosong = satu kelompok 'nan'\n")
        return (head +
                "# satu kelompok selalu utuh di train ATAU di valid; BAGIAN_VALID = porsi KELOMPOK, bukan baris\n"
                "from sklearn.model_selection import GroupShuffleSplit\n"
                f"BAGIAN_VALID = {share}\n" + groups +
                f"pemisah = GroupShuffleSplit(n_splits=1, test_size=BAGIAN_VALID, random_state={seed})\n"
                "baris_train, baris_valid = next(pemisah.split(train, groups=kelompok))\n"
                "train, valid = train.iloc[np.sort(baris_train)].copy(), train.iloc[np.sort(baris_valid)].copy()\n"
                'print("train", train.shape, "valid", valid.shape)')
    to_time = ("waktu = train[KOLOM_WAKTU]" if config.get("numeric")
               else 'waktu = pd.to_datetime(train[KOLOM_WAKTU], errors="coerce")')
    return (head +
            "# baris SESUDAH tanggal batas -> valid (masa depan); waktu kosong tetap di train\n"
            f"BAGIAN_VALID = {share}\n"
            f"KOLOM_WAKTU = {config['column']!r}\n"
            f"{to_time}\n"
            "BATAS_WAKTU = waktu.quantile(1 - BAGIAN_VALID)\n"
            "ke_valid = waktu > BATAS_WAKTU\n"
            "train, valid = train[~ke_valid].copy(), train[ke_valid].copy()\n"
            'print("batas waktu:", BATAS_WAKTU, "| train", train.shape, "valid", valid.shape)')


def apply(config: dict | None, label: str | None, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jalankan split_code apa adanya (satu sumber kebenaran dengan skrip). Kembalikan (train, valid)."""
    ns = {"pd": pd, "np": np, "LABEL": label, "train": train.copy()}
    exec(split_code(config), ns)
    return ns["train"], ns["valid"]


def stratify_problem(train: pd.DataFrame, label: str | None) -> str | None:
    """Kenapa stratified tidak bisa dipakai (None = bisa)."""
    if not label or label not in train:
        return "belum ada kolom label"
    values = train[label]
    if not is_classification(train, label):
        return "label angka dengan banyak nilai (regresi): tidak ada kelas untuk dijaga proporsinya"
    if values.isna().any():
        return f"label punya {int(values.isna().sum()):,} baris kosong"
    if values.value_counts().min() < 2:
        return "ada kelas yang hanya punya 1 baris"
    return None


def is_classification(train: pd.DataFrame, label: str | None) -> bool:
    """Label teks, atau angka dengan ≤ 20 nilai berbeda (0/1, 1–5) = kelas. Angka dengan banyak nilai = regresi."""
    if not label or label not in train:
        return False
    values = train[label]
    return not pd.api.types.is_numeric_dtype(values) or values.nunique() <= 20


def twins_across(train: pd.DataFrame, valid: pd.DataFrame, label: str | None) -> dict:
    """Baris valid yang fiturnya (tanpa label) sama persis dengan baris train: bocoran jawaban di valid."""
    if valid.empty:
        return {"kembar": 0, "beda label": 0}
    drop = [label] if label in train else []
    left = pd.util.hash_pandas_object(train.drop(columns=drop), index=False)
    right = pd.util.hash_pandas_object(valid.drop(columns=drop), index=False)
    twin = right.isin(set(left))
    different = 0
    if drop and twin.any():
        labels = pd.Series(train[label].to_numpy(), index=left.to_numpy()).groupby(level=0).agg(set)
        for key, value in zip(right[twin], valid.loc[twin.to_numpy(), label]):
            different += value not in labels[key]
    return {"kembar": int(twin.sum()), "beda label": int(different)}


__all__ = ["ROW_GROUP", "describe", "split_code", "apply", "stratify_problem", "is_classification", "twins_across"]
