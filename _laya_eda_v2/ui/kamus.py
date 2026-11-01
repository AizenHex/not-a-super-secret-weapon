"""R18: setiap kolom tabel yang tampil punya ARTI (nilai apa, dihitung dari apa). Satu-satunya pintu tampil tabel
di ui/ adalah `show_table`; arti tampil sebagai tooltip judul kolom. Kolom tanpa arti = galat (bukan diam-diam).

Arti dicari berurutan: `meaning` (khusus tabel itu) → KAMUS (nama kolom umum di aplikasi) → `data_columns=True`
(tabel berisi kolom data pengguna: arti = "isi kolom data ini") → akhiran pratinjau `(sebelum)`/`(sesudah)`.
"""
from __future__ import annotations

import os

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
    # profil angka (describe)
    "count": "Jumlah nilai terisi.", "mean": "Rata-rata.", "std": "Simpangan baku.", "min": "Nilai terkecil.",
    "1%": "Persentil 1.", "25%": "Kuartil bawah (Q1).", "50%": "Median.", "75%": "Kuartil atas (Q3).",
    "99%": "Persentil 99.", "max": "Nilai terbesar.",
    # ejaan / isian bebas
    "asli": "Isi sel sebelum langkah.", "sesudah": "Isi/hasil sesudah langkah atau aturan.",
    "bahan": "Satu bahan hasil pecah isian bebas.", "% baris": "Persen baris train yang memuat bahan ini.",
    "ejaan": "Ejaan yang ada di data.", "jadi": "Ejaan baku hasil gabung.", "gabung ke": "Kunci tujuan gabung.",
    "frasa": "Bahan (satu atau beberapa kata) sesudah peta kata.", "kata asli": "Kata yang ditulis di data.",
    "catatan": "Peringatan/penjelasan untuk baris ini.", "kode": "Kode angka atau kunci.",
    "nilai unik": "Jumlah nilai berbeda di train.", "pakai": "Dicentang = kolom ikut langkah.",
    "sesudah aturan kunci": "Jumlah nilai berbeda sesudah aturan kunci (huruf kecil, singkatan, tanpa tanda baca).",
    "usulan gabung": "Jumlah usulan GABUNG (kunci berbeda, makna sama).",
    "samakan 2 → 1": "Dicentang = kata 2 diganti kata 1.", "kata 1": "Kata baku pertama.",
    "jumlah 1": "Jumlah kemunculan kata 1.", "kata 2": "Kata baku kedua (mirip kata 1).",
    "jumlah 2": "Jumlah kemunculan kata 2.", "sel berspasi lebih": "Jumlah sel dengan spasi di depan/belakang/ganda.",
    "penanda kosong": "Jumlah sel berisi penanda kosong palsu ('-', '?', 'null', ...).",
    "kata dibetulkan": "Jumlah kata yang diganti peta kata.", "usulan baku": "Ejaan baku yang diusulkan.",
    # outlier
    "jenis": "Jenis kolom angka (kontinu / diskret).", "median": "Median kolom (train).",
    "maks": "Nilai terbesar (train).", "sudah ada langkah": "Langkah outlier untuk kolom ini sudah ada.",
    "perlakuan": "Pilihanmu untuk kolom ini (biarkan / perbaiki dengan langkah).",
    "metode": "Metode batas: IQR, IQR-log, MAD, MAD-log, manual.", "k": "Pengali sebaran (IQR/MAD) untuk batas.",
    "batas bawah": "Batas bawah wajar dari train.", "batas atas": "Batas atas wajar dari train.",
    "aksi": "clip (potong ke batas), nan (jadikan kosong), drop (buang baris train).",
    "di luar batas": "Jumlah baris di luar batas.", "% data": "Persen baris di luar batas.",
    "maks wajar (di bawah batas)": "Nilai terbesar yang masih di bawah batas penjaga.",
    "kelompok": "Nilai kolom pengelompok.", "nol %": "Persen nilai nol.",
    # temuan
    "penanda": "Penanda kosong palsu yang ditemukan.", "% isi kolom": "Persen isi kolom yang berupa penanda ini.",
    "mungkin kategori": "Penanda sering (≥ 5%): mungkin jawaban ('tidak ada'), bukan kosong.",
    "satu-satunya nilai": "Satu-satunya nilai selain kosong (kolom checkbox).", "hanya format": "Beda hanya format.",
    "cramer_v": "Kekuatan hubungan dua kolom kategori (0–1, Cramér's V).", "left_column_name": "Kolom pertama.",
    "right_column_name": "Kolom kedua.", "kolom 1": "Kolom pertama dari pasangan.",
    "kolom 2": "Kolom kedua dari pasangan.", "Cramér's V": "Kekuatan hubungan dua kolom kategori (0–1).", "sama": "Persen baris yang isinya sama.",
    "urutan": "Urutan kategori dari terendah.",
    # split
    "train %": "Persen baris train per kelas label.", "valid %": "Persen baris valid per kelas label.",
    "selisih (poin %)": "valid % − train %.", "train": "Ringkasan di bagian train.", "valid": "Ringkasan di bagian valid.",
    # train vs test / AutoGluon / cleanlab
    "nilai baru": "Jumlah nilai di test yang tidak ada di train.", "baris test terdampak": "Baris test berisi nilai baru.",
    "KS": "Statistik Kolmogorov–Smirnov (0 = sebaran sama).", "p": "Nilai p uji KS.",
    "median train": "Median kolom di train.", "median test": "Median kolom di test.",
    "mentah": "Skor model di data mentah.", "sesudah langkah": "Skor model sesudah semua langkah.",
    "score_test": "Skor model di bagian uji (valid).", "score_val": "Skor validasi internal AutoGluon.",
    "fit_time": "Lama latih model, detik.", "p_value": "Nilai p importance (kecil = kolom benar-benar berpengaruh).",
    "skor label": "Skor kualitas label cleanlab (rendah = mungkin salah).",
    "tebakan model": "Kelas yang ditebak model OOF.", "yakin tebakan": "Peluang kelas tebakan menurut model.",
    "jawaban": "Jawaban Laya.", "konteks": "Konteks yang diberikan ke Laya.", "yakin": "Keyakinan jawaban Laya.",
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
    config = _config(frame, meaning, data_columns, kwargs.pop("column_config", None), kwargs.get("hide_index"))
    kwargs.setdefault("width", "stretch")
    container.dataframe(frame, column_config=config, **kwargs)


def _config(frame: pd.DataFrame, meaning, data_columns, config, hide_index) -> dict:
    config = dict(config or {})
    columns = list(frame.columns)
    if hide_index is not True and frame.index.name:
        columns.append(frame.index.name)
    missing = [c for c in columns if meaning_of(c, meaning, data_columns) is None]
    if missing and os.environ.get("LAYA_KAMUS_KUMPUL"):
        with open(os.environ["LAYA_KAMUS_KUMPUL"], "a", encoding="utf-8") as out:
            out.write(repr(missing) + "\n")
        missing = []
    if missing:
        raise ValueError(f"R18: kolom tabel tanpa arti: {missing}. Tambahkan di ui/kamus.py atau argumen meaning.")
    for column in columns:
        text = meaning_of(column, meaning, data_columns)
        old = config.get(column)
        if isinstance(old, dict):
            config[column] = {**old, "help": old.get("help") or text}
        elif old is None:
            config[column] = st.column_config.Column(help=text)
    return config


def edit_table(data: pd.DataFrame, meaning: dict | None = None, data_columns: bool = False, container=st, **kwargs):
    """st.data_editor dengan arti di setiap judul kolom (R18). Kembalikan tabel hasil edit."""
    kwargs["column_config"] = _config(data, meaning, data_columns, kwargs.pop("column_config", None),
                                      kwargs.get("hide_index"))
    kwargs.setdefault("width", "stretch")
    return container.data_editor(data, **kwargs)


__all__ = ["KAMUS", "meaning_of", "show_table", "edit_table"]
