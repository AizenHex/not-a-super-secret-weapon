"""Tab Kode EDA: pilih apa yang dilakukan ke data (ganti label, buang baris, isi angka kosong,
fitur, buang kolom), lalu salin kodenya. Kode dibuat codegen.py dan DIJALANKAN juga di sini,
jadi train_siap di tab Split data dan Ekspor = hasil kode yang kamu tempel.
Dua bagian: pick_steps (dijalankan sebelum tab Split data) dan show_code (sesudahnya, supaya
kode split ikut ditulis)."""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

import category_encoding
import codegen
import custom_drop
import features
import helpers
import kamus
import missing_fill
from penjelasan import show_explanation

MINILM_PATH = "../../pretrained/minilm"    # sama dari _laya_eda dan dari folder notebook lomba


def notebook_path(csv_path):
    # notebook lomba ada di <repo>/<folder>/, sama dalamnya dengan _laya_eda.
    # "\" diganti "/" supaya path Windows tetap teks Python yang sah
    csv_path = csv_path.replace("\\", "/")
    if csv_path.startswith("../") or os.path.isabs(csv_path):
        return csv_path
    return f"../_laya_eda/{csv_path}"


def label_value(text, label):
    # label angka (0/1/2 atau 3.0) tetap angka di kode
    if pd.api.types.is_integer_dtype(label):
        return int(float(text))
    if pd.api.types.is_float_dtype(label):
        return float(text)
    return text


def notes_to_steps(ctx):
    # catatan tab Telusuri: 'buang' -> {baris: catatan}, 'ganti_label' -> {baris: label baru}
    notes = helpers.load_notes(ctx["folder"])
    label = None
    if ctx["label_column"] is not None:
        label = ctx["data"][ctx["label_column"]]
    manual_drop = {}
    new_labels = {}
    for row in notes.index:
        if notes.at[row, "keputusan"] == "buang":
            manual_drop[int(row)] = notes.at[row, "catatan"]
        if notes.at[row, "keputusan"] != "ganti_label" or notes.at[row, "label_baru"] == "":
            continue
        if label is not None:
            new_labels[int(row)] = label_value(notes.at[row, "label_baru"], label)
    return manual_drop, new_labels


def saved_features(ctx):
    # fitur yang lolos di tab Fitur → label (fitur_terpilih.json) jadi pilihan awal
    path = f"{ctx['folder']}/fitur_terpilih.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        content = json.load(file)
    chosen = []
    for item in content["fitur_terpilih"]:
        if item["grup"] in codegen.FEATURES and item["grup"] not in chosen:
            chosen.append(item["grup"])
    return chosen


def parsed_table(ctx):
    # data dengan kolom angka sudah jadi angka di kolom yang sama (seperti sesudah sel
    # "Ubah kolom angka"); dipakai untuk daftar pilihan dan cek aturan query
    table = ctx["data"].copy()
    for column in ctx["numbers"].columns:
        table[column] = ctx["numbers"][column]
    return table


@st.cache_data(max_entries=2, show_spinner=False)
def read_test_csv(path, modified):
    return pd.read_csv(path)


def read_test(ctx):
    path = st.text_input("CSV test (path relatif ke folder _laya_eda, seperti di samping; "
                         "kosongkan bila tidak ada)", "", key="kode_test").strip()
    if path == "":
        return None, None
    if not os.path.exists(path):
        st.error(f"File test `{path}` tidak ada, jadi test tidak ikut.")
        return None, None
    return path, read_test_csv(path, os.path.getmtime(path))


def pick_outlier(ctx, rules):
    if "outlier" not in rules or len(ctx["numbers"].columns) == 0:
        return []
    return st.multiselect("Cek outlier untuk kolom angka", list(ctx["numbers"].columns),
                          default=list(ctx["numbers"].columns),
                          key=f"kode_outlier_columns__{ctx['folder']}")


def pick_row_steps(ctx, table):
    st.markdown("**1. Buang baris (train saja)** · keputusan *buang* di tab Telusuri ikut "
                "otomatis")
    rules = st.multiselect("Aturan buang baris", list(codegen.RULES),
                           default=["duplikat", "teks_kosong"], format_func=codegen.RULES.get,
                           key=f"kode_rules__{ctx['folder']}")
    outlier_columns = pick_outlier(ctx, rules)
    query_rules = custom_drop.pick_rules(ctx, table)
    return {"drop_rules": rules, "outlier_columns": outlier_columns, "query_rules": query_rules}


def pick_category(ctx, rules, chosen):
    # kategori pembanding: untuk outlier per kategori dan fitur angka ÷ median kategori.
    # Kode tanpa Laya, jadi hanya kolom asli CSV (bukan lensa / kelompok)
    if "outlier" not in rules and "rasio_median" not in chosen:
        return None
    options = []
    for name in ctx["dimensions"] + features.category_candidates(ctx):
        if name in ctx["data"].columns and name not in options:
            options.append(name)
    if len(options) == 0:
        st.caption("Outlier per kategori dan angka ÷ median butuh kolom kategori di CSV.")
        return None
    return st.selectbox("Kategori pembanding (outlier per kategori, angka ÷ median)", options,
                        key=f"kode_category__{ctx['folder']}")


def pick_table_steps(ctx, table, rules):
    st.markdown("**2. Isi angka kosong (nilai dari train, dipakai sama untuk test)**")
    fill = missing_fill.pick_fill(ctx, table)
    st.markdown("**3. Tambah fitur (train dan test)**")
    chosen = st.multiselect("Fitur (pilihan awal = yang lolos di tab Fitur → label)",
                            list(codegen.FEATURES), default=saved_features(ctx),
                            format_func=codegen.FEATURES.get,
                            key=f"kode_features__{ctx['folder']}")
    category = pick_category(ctx, rules, chosen)
    st.markdown("**4. Encoding kolom kategori (train dan test)**")
    encoding, encode_drop = category_encoding.pick_encoding(ctx, table, date_columns(ctx))
    st.markdown("**5. Buang kolom (train dan test)**")
    drop_columns = custom_drop.pick_columns(ctx)
    return {"fill": fill, "features": chosen, "category": category, "drop_columns": drop_columns,
            "encoding": encoding, "encode_drop": encode_drop}


def without_label(ctx, columns):
    # kolom yang dipakai juga di test tidak boleh label (test tidak punya label)
    names = []
    for column in columns:
        if column != ctx["label_column"]:
            names.append(column)
    return names


def date_columns(ctx):
    columns = features.date_candidates(ctx)
    if ctx["date_column"] is not None and ctx["date_column"] not in columns:
        columns.append(ctx["date_column"])
    return without_label(ctx, columns)


def label_is_number(ctx):
    # label angka dengan banyak nilai = regresi (target encoding = rata-rata label)
    if ctx["label_column"] is None:
        return False
    label = ctx["data"][ctx["label_column"]]
    return bool(pd.api.types.is_numeric_dtype(label) and label.nunique() > 20)


def build_config(ctx, test_path, steps):
    manual_drop, new_labels = notes_to_steps(ctx)
    test_csv = None
    if test_path is not None:
        test_csv = notebook_path(test_path)
    config = {"csv_path": notebook_path(ctx["csv_path"]), "test_csv": test_csv,
              "columns": ctx["columns"], "label_column": ctx["label_column"],
              "number_columns": ctx["number_columns"], "indonesian_format": ctx["indonesian"],
              "use_log": ctx["use_log"], "ratio": ctx["ratio"], "manual_drop": manual_drop,
              "new_labels": new_labels,
              "number_features": without_label(ctx, ctx["numbers"].columns),
              "category_columns": without_label(ctx, features.category_candidates(ctx)),
              "date_columns": date_columns(ctx), "split_lines": None,
              "label_is_number": label_is_number(ctx), "minilm_path": MINILM_PATH}
    config.update(steps)
    return config


def make_printer(lines):
    # pengganti print() saat kode dijalankan di aplikasi: teksnya dikumpulkan untuk ditampilkan
    def printer(*values):
        texts = []
        for value in values:
            texts.append(str(value))
        lines.append(" ".join(texts))
    return printer


@st.cache_data(max_entries=4, show_spinner="Menjalankan kode EDA ...")
def run_script(script, data_key, test_key, _data, _test, _minilm):
    # sel kode yang sama dengan notebook, dijalankan pada salinan data. _data / _test tidak
    # di-hash (lambat); data_key dan test_key = (path, waktu ubah file) mewakilinya
    lines = []
    names = {"np": np, "pd": pd, "print": make_printer(lines), "train": _data.copy(),
             "MINILM": _minilm}
    if _test is not None:
        names["test"] = _test.copy()
    exec(script, names)
    return names["train"], names.get("test"), lines


def show_result(ctx, train, test, log):
    text = (f"**train_siap: {len(train)} baris × {len(train.columns)} kolom** "
            f"(CSV asli {len(ctx['data'])} baris)")
    if test is not None:
        text = text + f" · **test_siap: {len(test)} baris × {len(test.columns)} kolom**"
    st.write(text)
    with st.expander("Keluaran print() dari kode (dijalankan di sini)"):
        st.code("\n".join(log), language="text")
    kamus.show_table(train.head(20))
    st.caption("Jawaban lensa Laya tidak masuk train_siap karena kode ini tidak menjalankan "
               "Laya untuk test; jawaban itu ada di data_berlensa.csv (tab Ekspor).")


def pick_steps(ctx):
    st.write("Langkah yang kamu pilih di sini dilakukan ke data, lalu ditulis jadi kode di "
             "bawah. Kode yang sama dijalankan di sini, jadi hasilnya (train_siap) yang dibagi "
             "di tab **Split data** dan ditulis di tab **Ekspor**. Kolom angka dibaca jadi "
             "angka **di kolom yang sama** (nama kolom tidak diubah).")
    table = parsed_table(ctx)
    test_path, test = read_test(ctx)
    row_steps = pick_row_steps(ctx, table)
    steps = pick_table_steps(ctx, table, row_steps["drop_rules"])
    steps.update(row_steps)
    config = build_config(ctx, test_path, steps)
    script = codegen.app_script(codegen.build_cells(config))
    test_key = None
    if test_path is not None:
        test_key = (test_path, os.path.getmtime(test_path))
    minilm = None
    if "makna" in config["encoding"].values():
        minilm = ctx["load_minilm"]()
    train, ready_test, log = run_script(script, ctx["data_key"], test_key, ctx["data"], test,
                                        minilm)
    ctx["ready_train"] = train
    ctx["ready_test"] = ready_test
    show_result(ctx, train, ready_test, log)
    return config


def show_code(ctx, config):
    config["split_lines"] = ctx["split_lines"]    # dari tab Split data
    cells = codegen.build_cells(config)
    script = codegen.to_script(cells)
    st.subheader("Kode (tempel ke notebook)")
    st.write(f"**{len(cells)} sel** · {len(config['drop_rules'])} aturan buang · buang manual "
             f"{len(config['manual_drop'])} baris · ganti label {len(config['new_labels'])} "
             f"baris · isi kosong {len(config['fill'])} kolom · "
             f"{len(codegen.usable_features(config))} fitur · encoding "
             f"{len(config['encoding'])} kolom")
    first, second, third = st.columns(3)
    first.download_button("Unduh .py (sel # %%)", script.encode("utf-8"), "eda_otomatis.py",
                          key="kode_py")
    notebook = codegen.to_notebook(cells)
    second.download_button("Unduh .ipynb", notebook.encode("utf-8"), "eda_otomatis.ipynb",
                           key="kode_ipynb")
    if third.button("Tulis ke folder hasil", key="kode_write"):
        with open(f"{ctx['folder']}/eda_otomatis.py", "w", encoding="utf-8") as file:
            file.write(script)
        with open(f"{ctx['folder']}/eda_otomatis.ipynb", "w", encoding="utf-8") as file:
            file.write(notebook)
        st.success(f"Tersimpan di _laya_eda/{ctx['folder']}/eda_otomatis.py dan .ipynb")
    st.code(script, language="python")
    show_explanation("kode")
