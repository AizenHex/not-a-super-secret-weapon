"""Tab Cek angka: pandas mengecek kolom angka (Laya lemah soal angka)."""
import pandas as pd
import streamlit as st

import helpers
import outliers
from penjelasan import show_explanation


def column_counts(ctx, column):
    # (tidak terbaca, jumlah kosong, jumlah negatif/nol, outlier IQR) untuk satu kolom angka
    values = ctx["numbers"][column]
    unreadable = pd.Series(False, index=values.index)
    empty = values.isna().sum()
    if column in ctx["data"].columns:
        raw = ctx["data"][column]
        unreadable = raw.notna() & values.isna()
        empty = raw.isna().sum()
    outlier = outliers.iqr_outliers(values, ctx["use_log"])
    return unreadable, int(empty), int((values <= 0).sum()), outlier


def show_column(ctx, column):
    values = ctx["numbers"][column]
    unreadable, empty, non_positive, outlier = column_counts(ctx, column)
    st.markdown(f"#### {column}")
    first, second, third, fourth = st.columns(4)
    first.metric("tidak terbaca sebagai angka", int(unreadable.sum()))
    second.metric("kosong", empty)
    third.metric("negatif / nol", non_positive)
    fourth.metric("outlier IQR (semua baris)", int(outlier.sum()))
    with st.expander(f"lihat {column}"):
        if column in ctx["data"].columns:
            st.write("Nilai yang tidak bisa dibaca sebagai angka:")
            st.dataframe(ctx["data"].loc[unreadable, column].value_counts().head(30))
        st.dataframe(values.describe())
        order = values[outlier].sort_values(ascending=False).index
        st.write("Outlier (terbesar dulu), dengan alasannya:")
        shown = ctx["data"].loc[order].head(50).copy()
        shown["alasan"] = ctx["reasons"].reindex(shown.index)
        st.dataframe(shown)
    add_findings(ctx, column, int(unreadable.sum()), non_positive, int(outlier.sum()))


def add_findings(ctx, column, unreadable, non_positive, outlier):
    total = len(ctx["data"])
    if unreadable > 0:
        helpers.add_finding(
            ctx, "Cek angka", f"Kolom {column}: {unreadable} isian tidak terbaca sebagai angka "
            f"({100 * unreadable / total:.1f}%).", "Isiannya bukan angka (misal teks 'nego').",
            "Putuskan: jadikan kosong, isi dari sumber lain, atau buang barisnya.",
            10 * unreadable / total)
    if non_positive > 0:
        helpers.add_finding(
            ctx, "Cek angka", f"Kolom {column}: {non_positive} nilai negatif atau nol.",
            "Ukuran, harga, atau jumlah umumnya positif; skala log juga tidak bisa memakai nol.",
            "Cek apakah salah ketik tanda minus atau memang kosong.", 10 * non_positive / total)
    if outlier > 0:
        helpers.add_finding(
            ctx, "Cek angka", f"Kolom {column}: {outlier} outlier IQR di seluruh data.",
            "Nilainya di luar batas wajar Tukey (Q1 - 1,5 IQR sampai Q3 + 1,5 IQR).",
            "Bandingkan dengan kategorinya di tab Outlier.", 5 * outlier / total)


def show(ctx):
    st.write("Pandas mengecek angka (Laya lemah soal angka). Pilih kolom angka di bagian "
             "'Kolom angka' pada panel samping. Hasilnya dipakai lagi di tab Outlier.")
    numbers = ctx["numbers"]
    if len(numbers.columns) == 0:
        st.info("Belum ada kolom angka yang dipilih.")
    for column in numbers.columns:
        show_column(ctx, column)
    if ctx["ratio"] is not None:
        top, bottom = ctx["ratio"]
        st.caption(f"Kolom rasio `{top}_per_{bottom}` ikut dicek di atas dan bisa dipilih di tab "
                   "Outlier.")
        show_explanation("rasio")
    show_explanation("cek_angka")
