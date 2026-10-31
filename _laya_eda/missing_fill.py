"""Pilihan isi angka kosong di tab Kode EDA: satu cara untuk semua kolom, bisa diganti per kolom.
Nilai isinya dihitung kode (codegen.py) dari train, lalu dipakai juga untuk test."""
import pandas as pd
import streamlit as st

# cara -> keterangan; "biarkan" = tidak diisi (tidak ikut ke kode)
METHODS = {"biarkan": "biarkan kosong (NaN)", "median": "median train",
           "mean": "rata-rata (mean) train", "modus": "nilai terbanyak (modus) train",
           "nol": "isi 0"}
SAME = "(ikut pilihan umum)"


def candidates(ctx, table):
    # kolom angka (sesudah dibaca jadi angka) yang punya nilai kosong; label tidak diisi
    columns = []
    for column in table.columns:
        if column == ctx["label_column"] or not pd.api.types.is_numeric_dtype(table[column]):
            continue
        missing = table[column].isna().sum()
        if 0 < missing < len(table):
            columns.append(column)
    return columns


def method_name(method):
    if method == SAME:
        return SAME
    return METHODS[method]


def pick_overrides(ctx, columns, general):
    # satu pilihan per kolom; default ikut pilihan umum
    chosen = {}
    with st.expander(f"Atur per kolom ({len(columns)} kolom angka punya nilai kosong)"):
        for column in columns:
            method = st.selectbox(column, [SAME] + list(METHODS), format_func=method_name,
                                  key=f"isi_{column}__{ctx['folder']}")
            if method == SAME:
                method = general
            chosen[column] = method
    return chosen


def pick_fill(ctx, table):
    # hasil: {kolom: cara} hanya untuk kolom yang benar-benar diisi
    columns = candidates(ctx, table)
    if len(columns) == 0:
        st.caption("Tidak ada kolom angka yang kosong.")
        return {}
    general = st.selectbox("Isi angka kosong (semua kolom)", list(METHODS),
                           format_func=METHODS.get, key=f"isi_umum__{ctx['folder']}")
    chosen = pick_overrides(ctx, columns, general)
    fill = {}
    for column in chosen:
        if chosen[column] != "biarkan":
            fill[column] = chosen[column]
    return fill
