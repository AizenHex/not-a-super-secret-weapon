"""Membangun dict `ctx` yang dipakai semua tab. Dua tahap:
prepare(ctx)  : sebelum tab (teks, statistik teks, angka, lensa)
finish(ctx)   : setelah tab Kelompok & Lensa Laya (hasil Laya, tabel, vektor, alasan outlier)"""
import os

import numpy as np
import pandas as pd
import streamlit as st

import eda
import helpers
import outliers


def new_context(load_minilm, load_laya):
    # semua kunci selalu ada, walau tab terkait belum dibuka
    return {"data": None, "table": None, "texts": None, "columns": [], "label_column": None,
            "numbers": pd.DataFrame(), "stats": None, "vectors": None, "vector_rows": None,
            "folder": "", "use_log": True, "lenses": {}, "lens_done": [], "dimensions": [],
            "doubt_limit": 0.6, "csv_path": "", "load_minilm": load_minilm,
            "findings": [],
            # kunci tambahan (di luar kontrak dasar)
            "load_laya": load_laya, "mode": "teks", "date_column": None, "indonesian": True,
            "number_columns": [], "ratio": None, "reason_dim": None, "reasons": None,
            "reason_parts": {}, "distances": None, "group_rows": None, "doubt": None,
            "show_columns": []}


def build_numbers(ctx):
    data = ctx["data"]
    numbers = pd.DataFrame(index=data.index)
    for column in ctx["number_columns"]:
        numbers[column] = helpers.to_number(data[column], ctx["indonesian"])
    if ctx["ratio"] is not None:
        top, bottom = ctx["ratio"]
        numbers[f"{top}_per_{bottom}"] = numbers[top] / numbers[bottom].where(numbers[bottom] > 0)
    return numbers


def prepare(ctx):
    ctx["texts"] = helpers.row_texts(ctx["data"], ctx["columns"])
    ctx["stats"] = helpers.text_stats(ctx["texts"])
    ctx["numbers"] = build_numbers(ctx)
    ctx["lenses"] = helpers.load_lenses(ctx["folder"])


@st.cache_data
def cached_text_outliers(vectors, rows, categories):
    return outliers.text_outliers(vectors, rows, categories)


def add_group_column(ctx, table, dimensions):
    # kolom `kelompok` dari tab Temukan kategori (hanya baris yang ikut sampel)
    group_rows = pd.read_csv(f"{ctx['folder']}/kelompok.csv")
    table["kelompok"] = pd.Series(group_rows["kelompok"].to_numpy(), index=group_rows["row"])
    dimensions.append("kelompok")
    ctx["group_rows"] = group_rows


def build_table(ctx):
    # tabel gabungan: data asli + jawaban Laya + kelompok
    folder = ctx["folder"]
    results = helpers.lens_results(folder, ctx["lenses"], ctx["data"].index)
    table = ctx["data"].copy()
    for column in results.columns:
        table[column] = results[column]
    dimensions = []
    for name in ctx["lenses"]:
        if name in results.columns:
            dimensions.append(name)
    ctx["lens_done"] = list(dimensions)
    ctx["group_rows"] = None
    if os.path.exists(f"{folder}/kelompok.csv"):
        add_group_column(ctx, table, dimensions)
    if ctx["label_column"] is not None:
        label = ctx["data"][ctx["label_column"]]
        # label angka (regresi) atau teks yang hampir unik bukan kategori
        if not eda.is_regression_label(label) and label.nunique() <= eda.MAX_LABEL_CLASSES:
            dimensions.append(ctx["label_column"])
    ctx["table"] = table
    ctx["dimensions"] = dimensions


def load_vectors(ctx):
    ctx["vectors"] = None
    ctx["vector_rows"] = None
    if os.path.exists(f"{ctx['folder']}/vektor.npy"):
        ctx["vectors"] = np.load(f"{ctx['folder']}/vektor.npy")
        ctx["vector_rows"] = ctx["group_rows"]["row"].to_numpy()


def mark_doubt(ctx):
    table = ctx["table"]
    doubt = pd.Series(False, index=table.index)
    for name in ctx["lens_done"]:
        doubt = doubt | (table[f"{name}__yakin"] < ctx["doubt_limit"])
    table["laya_ragu"] = doubt
    ctx["doubt"] = doubt
    show_columns = list(ctx["columns"])
    for name in ctx["dimensions"]:
        if name not in show_columns:
            show_columns.append(name)
    for name in ctx["lens_done"]:
        show_columns.append(f"{name}__nilai")    # klik judul kolom untuk mengurutkan
    ctx["show_columns"] = show_columns


def pick_reason_dim(ctx):
    # alasan outlier angka/teks dihitung terhadap SATU kategori pembanding (pilihan di samping)
    ctx["reason_dim"] = None
    ctx["distances"] = None
    if len(ctx["dimensions"]) > 0:
        ctx["reason_dim"] = st.sidebar.selectbox("Pembanding kategori (untuk alasan outlier)",
                                                 ctx["dimensions"], key="reason_dim")
    if ctx["reason_dim"] is not None and ctx["vectors"] is not None:
        categories = ctx["table"][ctx["reason_dim"]]
        ctx["distances"] = cached_text_outliers(ctx["vectors"], ctx["vector_rows"], categories)


def finish(ctx):
    build_table(ctx)
    load_vectors(ctx)
    mark_doubt(ctx)
    pick_reason_dim(ctx)
    parts = outliers.collect_reasons(ctx)
    ctx["reason_parts"] = parts
    ctx["reasons"] = outliers.join_reasons(parts, ctx["data"].index)
