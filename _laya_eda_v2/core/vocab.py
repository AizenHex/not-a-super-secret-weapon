"""Kamus proyek (VISI.md G4): semua kata domain yang dulu tertanam di kode, sekarang data yang bisa diedit.

Disimpan di proyek (`project["kosakata"]`) bersama langkah. `use(kamus)` dipanggil aplikasi setiap rerun;
alat analisis (core/text_tools.py, core/findings.py) membaca kamus aktif lewat `get`. Langkah yang dibuat menyalin
nilai kamus ke parameternya, jadi kode langkah tidak berubah walau kamus diedit sesudahnya.

UMUM = default proyek baru (tanpa kata domain). PRESET_UKT = kamus data `Golongan UKT` (Indonesia): singkatan
wilayah, jenjang pendidikan, kata 'tidak', bahan bangunan, kata pengisi, pemisah 'dan/atau'.
"""
from __future__ import annotations

import copy
import re

UMUM = {
    "penanda_kosong": ["", "-", "--", "?", "nan", "null", "none", "n/a", "na", "#n/a", "missing", "unknown"],
    "singkatan": {},          # regex kata → kata panjang, dipakai kunci ejaan ('kab' → 'kabupaten')
    "kata_tidak": ["no", "none", "tidak", "tidak ada"],   # jawaban 'tidak' yang digabung di tab Ejaan
    "pemisah": [",", "&", "+", ";", "/", "|"],             # pemisah isian bebas multi-nilai (simbol atau kata)
    "kata_umum": [],          # kata yang TIDAK pernah dipetakan sebagai salah ketik
    "kata_baku": {},          # varian kata → kata baku (isian bebas)
    "frasa_baku": {},         # frasa → frasa baku (isian bebas)
    "kata_pengisi": [],       # kata yang dibuang dari satu bagian isian bebas ('lantai keramik' = 'keramik')
    "jenjang": {"urutan": [], "kunci": {}},   # kelompok sinonim berurutan (jenjang pendidikan)
}

PRESET_UKT = {
    "penanda_kosong": ["", "-", "--", "?", "nan", "null", "none", "n/a", "na", "tidak tahu"],
    "singkatan": {r"\bkab\b": "kabupaten", r"\bkec\b": "kecamatan", r"\bkel\b": "kelurahan", r"\bprov\b": "provinsi"},
    "kata_tidak": ["tidak", "tidakada", "tidakdapat", "tidakpunya", "tidakpernah", "tidakmenerima", "belum",
                   "belumada", "belumpernah", "tak", "takada", "no", "none", "nihil"],
    "pemisah": [",", "&", "/", "+", ";", "dan", "atau", "(", ")"],
    "kata_umum": ["tanpa", "tanah", "tidak", "dak", "bagian", "sebagian", "dan", "atau", "dengan", "yang", "ada",
                  "belum", "sudah", "lain", "lainnya", "sengon", "seng", "dari", "di", "ke", "pakai"],
    "kata_baku": {"genting": "genteng", "gendeng": "genteng", "kramik": "keramik", "plaster": "plester",
                  "plesteran": "plester", "tegel": "ubin", "tekel": "ubin", "tehel": "ubin", "jubin": "ubin",
                  "herbel": "hebel", "batubata": "batu bata", "esbes": "asbes"},
    "frasa_baku": {"bata": "batu bata", "bata merah": "batu bata", "batu bata merah": "batu bata",
                   "genteng tanah liat": "genteng", "genteng tanah": "genteng"},
    "kata_pengisi": ["dari", "biasa", "bahan", "rumah"],
    "jenjang": {"urutan": ["tidak sekolah", "tidak lulus SD", "SD sederajat", "SMP sederajat", "SMA sederajat",
                           "D1", "D2", "D3", "D4/S1", "S2", "S3"],
                "kunci": {"tidaksekolah": 0, "belumsekolah": 0, "tidaklulussd": 1, "tidaktamatsd": 1,
                          "belumtamatsd": 1, "sd": 2, "mi": 2, "sdmi": 2, "sdsederajat": 2, "sdmisederajat": 2,
                          "paketa": 2, "smp": 3, "mts": 3, "sltp": 3, "smpmts": 3, "smpsederajat": 3,
                          "sltpsederajat": 3, "paketb": 3, "sma": 4, "ma": 4, "smk": 4, "slta": 4, "smama": 4,
                          "smasmk": 4, "smasederajat": 4, "sltasederajat": 4, "paketc": 4, "stm": 4, "smea": 4,
                          "d1": 5, "d2": 6, "d3": 7, "d4": 8, "s1": 8, "s2": 9, "s3": 10}},
}
PRESETS = {"umum (tanpa kata domain)": UMUM, "Indonesia / UKT": PRESET_UKT}
# regex pemisah lama (sebelum kamus proyek): dipakai langkah tersimpan yang belum punya parameter 'pemisah'
LEGACY_SEPARATORS = r"\s*(?:,|&|/|\+|;|\bdan\b|\batau\b|\(|\))\s*"

_active = copy.deepcopy(UMUM)


def use(kamus: dict | None) -> None:
    """Pakai kamus proyek (kunci yang tidak ada = nilai UMUM)."""
    global _active
    _active = {**copy.deepcopy(UMUM), **copy.deepcopy(kamus or {})}


def get(name: str):
    return _active[name]


def current() -> dict:
    return copy.deepcopy(_active)


def _escape(text: str) -> str:
    """Escape hanya karakter regex khusus (re.escape juga meng-escape '&', jadi teks regex lama berubah)."""
    return "".join("\\" + char if char in ".^$*+?{}[]\\|()" else char for char in text)


def separator_regex(items: list[str] | None = None, multi: bool = False) -> str:
    """Daftar pemisah → regex. Kata dapat batas \\b; multi=True tanpa '/' dan kurung (SD/MI tetap satu nilai)."""
    items = get("pemisah") if items is None else items
    if multi:
        items = [item for item in items if item not in ("/", "(", ")")]
    symbols = [_escape(item) for item in items if not item.isalpha() and item not in ("(", ")")]
    words = [rf"\b{_escape(item)}\b" for item in items if item.isalpha()]
    brackets = [_escape(item) for item in items if item in ("(", ")")]
    joined = "|".join(symbols + words + brackets)
    return rf"\s*(?:{joined})\s*" if not multi else joined


__all__ = ["UMUM", "PRESET_UKT", "PRESETS", "LEGACY_SEPARATORS", "use", "get", "current", "separator_regex"]
