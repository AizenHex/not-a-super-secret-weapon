"""Alat teks: usulan peta ejaan. Semua fungsi hanya MENGUSULKAN; pengguna menyetujui per baris.

Tiga cara, dari yang paling aman:
1. fingerprint (key collision, OpenRefine): beda huruf besar/spasi/tanda baca/urutan kata saja.
2. peta kata rapidfuzz: salah ketik per kata ('gentng' -> 'genting') ke kosakata yang sering muncul.
3. skrub.deduplicate (opsional): klaster n-gram huruf seluruh nilai. Kuat untuk variasi panjang,
   tapi bisa salah gabung ('Kab. Semarang' -> 'Kota Semarang'). Wajib diperiksa.
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz, process

from core.recipe import MATERIAL_SEPARATORS

ABBREVIATIONS = {"kab": "kabupaten", "kab.": "kabupaten", "kec": "kecamatan", "kel": "kelurahan",
                 "prov": "provinsi", "sd/mi": "sd", "sltp": "smp", "slta": "sma"}
# kata pendek/umum yang mirip ejaan dengan kata lain tapi beda makna: jangan pernah dipetakan
STOPWORDS = {"tanpa", "tanah", "tidak", "dak", "bagian", "sebagian", "dan", "atau", "dengan", "yang",
             "ada", "belum", "sudah", "lain", "lainnya", "sengon", "seng", "dari", "di", "ke", "pakai"}


def fingerprint(value: str) -> str:
    """OpenRefine fingerprint: huruf kecil, buang tanda baca, kata unik terurut."""
    words = re.sub(r"[^\w\s]", " ", str(value).lower()).split()
    words = [ABBREVIATIONS.get(word, word) for word in words]
    return " ".join(sorted(set(words)))


def fingerprint_groups(series: pd.Series) -> pd.DataFrame:
    """Kelompok ejaan yang fingerprint-nya sama. Usulan baku = ejaan tersering di kelompok."""
    counts = series.dropna().astype(str).value_counts()
    table = pd.DataFrame({"asli": counts.index, "jumlah": counts.to_numpy()})
    table["kunci"] = table["asli"].map(fingerprint)
    sizes = table.groupby("kunci")["asli"].transform("size")
    table = table[sizes > 1].copy()
    table["usulan baku"] = table.groupby("kunci")["asli"].transform("first").str.strip()
    table = table[table["asli"] != table["usulan baku"]]
    return table[["asli", "usulan baku", "jumlah", "kunci"]].reset_index(drop=True)


def words_of(series: pd.Series) -> pd.Series:
    parts = series.dropna().astype(str).str.lower().str.replace(".", " ", regex=False)
    parts = parts.str.split(MATERIAL_SEPARATORS, regex=True).explode().str.strip()
    return parts[parts != ""].str.split().explode()


def word_map(series: pd.Series, min_count: int = 30, cutoff: float = 75,
             related: list[pd.Series] | None = None) -> pd.DataFrame:
    """Kata jarang -> kata baku terdekat (rapidfuzz fuzz.ratio).

    Kosakata baku = kata yang muncul >= min_count kali di kolom ini DAN kolom terkait (related),
    supaya kata sah di kolom lain ('batu' di Bahan Tembok) tidak dianggap salah ketik ('bata').
    Kata pendek (< 4 huruf) butuh skor >= 80. Kata di STOPWORDS tidak pernah dipetakan.
    """
    counts = words_of(series).value_counts()
    total = pd.concat([words_of(s) for s in [series, *(related or [])]]).value_counts()
    vocabulary = [w for w in total[total >= min_count].index if w not in STOPWORDS and not w.isdigit()]
    rows = []
    for word, count in counts.items():
        if total[word] >= min_count or word in STOPWORDS or word.isdigit() or len(word) < 3:
            continue
        limit = cutoff if len(word) >= 4 else max(cutoff, 80)
        best = process.extractOne(word, vocabulary, scorer=fuzz.ratio, score_cutoff=limit)
        if best:
            rows.append({"kata asli": word, "usulan": best[0], "skor": round(best[1]),
                         "jumlah": int(count), "setuju": best[1] >= 85})
    return pd.DataFrame(rows, columns=["kata asli", "usulan", "skor", "jumlah", "setuju"])


def similar_vocabulary(series: pd.Series, min_count: int = 30, cutoff: float = 80) -> pd.DataFrame:
    """Pasangan kata baku yang mirip (genting/genteng): kandidat sinonim, tidak pernah otomatis."""
    counts = words_of(series).value_counts()
    vocabulary = [w for w in counts[counts >= min_count].index if w not in STOPWORDS]
    rows = []
    for i, first in enumerate(vocabulary):
        for second in vocabulary[i + 1:]:
            score = fuzz.ratio(first, second)
            if score >= cutoff:
                rows.append({"kata 1": first, "jumlah 1": int(counts[first]), "kata 2": second,
                             "jumlah 2": int(counts[second]), "skor": round(score)})
    return pd.DataFrame(rows, columns=["kata 1", "jumlah 1", "kata 2", "jumlah 2", "skor"])


def material_preview(series: pd.Series, words: dict[str, str], top: int = 20) -> pd.DataFrame:
    """Hasil pecah + peta kata, untuk dilihat sebelum disetujui: bahan baku dan jumlah barisnya."""
    def fix(text):
        parts = re.split(MATERIAL_SEPARATORS, str(text).lower().replace(".", " "))
        items = {" ".join(words.get(w, w) for w in part.split()) for part in parts}
        items.discard("")
        return sorted(items)

    items = series.dropna().map(fix).explode().dropna()
    share = items.value_counts().head(top)
    return pd.DataFrame({"bahan": share.index, "baris": share.to_numpy(),
                         "% baris": (share / series.notna().sum() * 100).round(1).to_numpy()})


def deduplicate_map(series: pd.Series, n_clusters: int | None = None) -> pd.DataFrame:
    """skrub.deduplicate (opsional, bisa lambat untuk ribuan nilai unik). Hanya baris yang berubah."""
    import skrub

    values = series.dropna().astype(str)
    unique = pd.Series(values.unique())
    counts = values.value_counts()
    mapped = skrub.deduplicate(unique, n_clusters=n_clusters)
    table = pd.DataFrame({"asli": unique.to_numpy(), "usulan baku": list(mapped)})
    table["jumlah"] = table["asli"].map(counts)
    table = table[table["asli"] != table["usulan baku"]].sort_values("jumlah", ascending=False)
    table["peringatan"] = [
        "beda kata depan, periksa" if fingerprint(a).split()[:1] != fingerprint(b).split()[:1] else ""
        for a, b in zip(table["asli"], table["usulan baku"])
    ]
    return table.reset_index(drop=True)
