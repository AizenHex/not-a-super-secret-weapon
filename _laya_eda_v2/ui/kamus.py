"""R18: setiap kolom tabel yang tampil punya ARTI (nilai apa, dihitung dari apa). Satu-satunya pintu tampil tabel
di ui/ adalah `show_table`; arti tampil sebagai tooltip judul kolom. Kolom tanpa arti = galat (bukan diam-diam).

Arti dicari berurutan: `meaning` (khusus tabel itu) → KAMUS (nama kolom umum di aplikasi) → `data_columns=True`
(tabel berisi kolom data pengguna: arti = "isi kolom data ini") → akhiran pratinjau `(sebelum)`/`(sesudah)`.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

KAMUS = {
    "kolom": "Nama kolom data yang dibahas di baris ini.",
    "langkah": "Nomor langkah di daftar langkah (sidebar).",
    "operasi": "Operasi langkah dan kolom yang diubahnya.",
    "galat": "Pesan galat asli bila langkah gagal (kosong = berhasil).",
    "keluaran": "Teks yang dicetak kode langkah (print).",
    "baris train": "Jumlah baris train sebelum → sesudah langkah.",
    "baris valid": "Jumlah baris valid sebelum → sesudah (tidak pernah berkurang).",
    "baris test": "Jumlah baris test sebelum → sesudah (tidak pernah berkurang).",
    "sel berubah": "Jumlah sel train yang isinya diubah langkah ini.",
    "kosong baru": "Sel kosong yang DIBUAT langkah ini, per kolom.",
    "tipe tebakan": "Tebakan tipe isi kolom: angka, angka-teks, kategori, isian bebas, id, kosong.",
    "dtype pandas": "Tipe kolom menurut pandas saat dibaca.",
    "kosong": "Jumlah sel kosong (NaN).",
    "kosong %": "Persen sel kosong dari semua baris.",
    "unik": "Jumlah nilai berbeda (tanpa kosong).",
    "contoh tersering": "Tiga nilai paling sering beserta jumlah barisnya.",
    "jumlah": "Jumlah baris yang berisi nilai/temuan ini.",
    "baris": "Jumlah baris (atau nomor baris, di tabel baris salah).",
    "contoh": "Contoh nilai asli dari data.",
    "nilai": "Nilai asli di data.",
    "alasan": "Kenapa usulan ini dibuat (aturan/kamus/skor kemiripan).",
    "setuju": "Dicentang = usulan dipakai saat langkah ditambah.",
    "usulan": "Nilai baku yang diusulkan.",
    "skor": "Skor kemiripan rapidfuzz 0–100 (100 = sama persis).",
    "model": "Nama model.",
    "kunci": "Kunci internal (model / kunci ejaan: huruf kecil, tanpa spasi dan tanda baca).",
    "status": "selesai / dilewati (alasan) / galat (pesan asli).",
    "skor CV": "Rata-rata skor metrik pilihan di fold validasi (pipeline di-fit ulang per fold).",
    "skor OOF": "Skor metrik di semua prediksi out-of-fold sekaligus, sesudah ambang/bobot kelas dari OOF.",
    "skor valid": "Skor model akhir (dilatih di seluruh train) di bagian valid split proyek; kosong = tanpa split.",
    "lama (s)": "Lama validasi silang model ini, detik.",
    "label asli": "Label yang tertulis di data.",
    "tebakan": "Kelas yang ditebak model (prediksi out-of-fold).",
    "porsi dari kelas asli": "baris ÷ jumlah baris kelas asli itu.",
    "salah": "Besar salah: 1 − peluang kelas asli (klasifikasi) atau |asli − tebakan| (regresi/berurutan).",
    "importance": "Turunnya skor bila isi kolom diacak (permutation importance, di fold validasi).",
    "stddev": "Simpangan baku importance antar pengacakan.",
    "kelas": "Nilai label (kelas).",
    "presisi": "Dari baris yang ditebak kelas ini, porsi yang benar (OOF).",
    "recall": "Dari baris kelas ini, porsi yang tertebak benar (OOF).",
    "f1": "Rata-rata harmonik presisi dan recall.",
    "pemeriksaan": "Aturan skema pandera yang dilanggar (tipe, rentang, nilai sah, kolom ada).",
    "contoh nilai": "Contoh nilai yang melanggar (maks 5).",
    "bagian": "Blok EDA (pandas, sebelum pipeline) atau Pipeline sklearn.",
    "transformer": "Kelas sklearn/feature_engine yang dipakai langkah ini di Pipeline.",
    "waktu": "Waktu latih dimulai.",
    "hash_langkah": "Sidik langkah EDA aktif + split + label: langkah beda = hash beda.",
    "metrik": "Metrik lomba yang dipakai menilai.",
    "parameter": "Setelan model dan validasi silang.",
    "skor_fold": "Skor per fold validasi.",
    "skor_rata": "Rata-rata skor fold.",
    "lama_detik": "Lama validasi silang, detik.",
    "setelan": "Semua setelan untuk membuka ulang percobaan (JSON).",
    "masalah": "Hal yang membuat submission ditolak validator.",
    "isi": "Isi format yang terbaca dari sample_submission.csv.",
    "arti": "Penjelasan.",
}


def meaning_of(column, meaning: dict | None = None, data_columns: bool = False) -> str | None:
    name = str(column)
    if meaning and name in meaning:
        return meaning[name]
    if name in KAMUS:
        return KAMUS[name]
    for suffix, text in ((" (sebelum)", "Isi sebelum langkah"), (" (sesudah)", "Isi sesudah langkah")):
        if name.endswith(suffix):
            return f"{text}, kolom data '{name[:-len(suffix)]}'."
    if data_columns:
        return f"Isi kolom data '{name}' (sesudah langkah yang dipilih)."
    return None


def show_table(data, meaning: dict | None = None, data_columns: bool = False, container=st, **kwargs) -> None:
    """Tampilkan tabel dengan arti di setiap judul kolom. Kolom tanpa arti → ValueError (R18)."""
    frame = data.to_frame() if isinstance(data, pd.Series) else data
    config = dict(kwargs.pop("column_config", None) or {})
    columns = list(frame.columns)
    if kwargs.get("hide_index") is not True and frame.index.name:
        columns.append(frame.index.name)
    missing = [c for c in columns if meaning_of(c, meaning, data_columns) is None]
    if missing:
        raise ValueError(f"R18: kolom tabel tanpa arti: {missing}. Tambahkan di ui/kamus.py atau argumen meaning.")
    for column in columns:
        text = meaning_of(column, meaning, data_columns)
        old = config.get(column)
        if isinstance(old, dict):
            config[column] = {**old, "help": old.get("help") or text}
        elif old is None:
            config[column] = st.column_config.Column(help=text)
    kwargs.setdefault("width", "stretch")
    container.dataframe(frame, column_config=config, **kwargs)


__all__ = ["KAMUS", "meaning_of", "show_table"]
