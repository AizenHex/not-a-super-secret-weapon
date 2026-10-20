"""Tab Kode EDA: ubah pilihan + temuan + catatanmu jadi kode notebook (codegen.py)."""
import json
import os

import pandas as pd
import streamlit as st

import codegen
import features
import helpers
from penjelasan import show_explanation


def notebook_path(csv_path):
    # notebook lomba ada di _.NEW-GEN/<folder>/, sama dalamnya dengan _laya_eda
    if csv_path.startswith("../"):
        return csv_path
    return f"../_laya_eda/{csv_path}"


def notes_to_steps(ctx):
    # catatan tab Telusuri: 'buang' -> {baris: catatan}, 'ganti_label' -> {baris: label baru}
    notes = helpers.load_notes(ctx["folder"])
    number_label = False
    if ctx["label_column"] is not None:
        number_label = pd.api.types.is_integer_dtype(ctx["data"][ctx["label_column"]])
    manual_drop = {}
    new_labels = {}
    for row in notes.index:
        if notes.at[row, "keputusan"] == "buang":
            manual_drop[int(row)] = notes.at[row, "catatan"]
        if notes.at[row, "keputusan"] == "ganti_label" and notes.at[row, "label_baru"] != "":
            label = notes.at[row, "label_baru"]
            if number_label:
                label = int(label)    # label angka (0/1/2) tetap angka di kode
            new_labels[int(row)] = label
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
        if item["grup"] in codegen.FEATURES:
            chosen.append(item["grup"])
    return chosen


def pick_rules(ctx):
    default = ["duplikat", "teks_kosong"]
    if len(ctx["number_columns"]) > 0:
        default = default + ["tidak_terbaca", "negatif", "outlier"]
    return st.multiselect("Aturan buang baris (ditulis sebagai loop pandas)",
                          list(codegen.RULES), default=default, format_func=codegen.RULES.get,
                          key=f"kode_rules__{ctx['folder']}")


def pick_outlier(ctx, rules):
    if "outlier" not in rules:
        return [], None
    columns = st.multiselect("Cek outlier untuk kolom angka", list(ctx["numbers"].columns),
                             default=list(ctx["numbers"].columns),
                             key=f"kode_outlier_columns__{ctx['folder']}")
    # kode tanpa Laya: kategori pembanding hanya kolom asli CSV (bukan lensa / kelompok)
    category_options = []
    for name in ctx["dimensions"] + features.category_candidates(ctx):
        if name in ctx["data"].columns and name not in category_options:
            category_options.append(name)
    if len(columns) == 0 or len(category_options) == 0:
        st.caption("Outlier per kategori butuh kolom angka dan kolom kategori di CSV.")
        return [], None
    category = st.selectbox("... dibandingkan di dalam kategori", category_options,
                            key=f"kode_category__{ctx['folder']}")
    return columns, category


def date_columns(ctx):
    columns = features.date_candidates(ctx)
    if ctx["date_column"] is not None and ctx["date_column"] not in columns:
        columns.append(ctx["date_column"])
    return columns


def build_config(ctx):
    manual_drop, new_labels = notes_to_steps(ctx)
    test_csv = st.text_input("CSV test (kosongkan bila tidak ada; path dari folder notebook)",
                             "", key="kode_test")
    if test_csv.strip() == "":
        test_csv = None
    rules = pick_rules(ctx)
    outlier_columns, category = pick_outlier(ctx, rules)
    chosen = st.multiselect("Fitur yang ditambahkan (pilihan awal = yang lolos di tab Fitur)",
                            list(codegen.FEATURES), default=saved_features(ctx),
                            format_func=codegen.FEATURES.get,
                            key=f"kode_features__{ctx['folder']}")
    return {"csv_path": notebook_path(ctx["csv_path"]), "test_csv": test_csv,
            "columns": ctx["columns"], "label_column": ctx["label_column"],
            "number_columns": ctx["number_columns"], "indonesian_format": ctx["indonesian"],
            "use_log": ctx["use_log"], "ratio": ctx["ratio"],
            "manual_drop": manual_drop, "new_labels": new_labels, "drop_rules": rules,
            "outlier_columns": outlier_columns, "outlier_category": category,
            "features": chosen, "category_columns": features.category_candidates(ctx),
            "date_columns": date_columns(ctx)}


def show(ctx):
    st.write("Pilihan di sidebar, aturan buang, fitur, dan catatan *buang / ganti label* di "
             "tab Telusuri diubah jadi kode notebook. Kode ini **hanya pandas + numpy** (tanpa "
             "Laya, tanpa `_laya_eda`), jadi bisa ditempel ke notebook lomba.")
    config = build_config(ctx)
    cells = codegen.build_cells(config)
    script = codegen.to_script(cells)
    st.write(f"**{len(cells)} sel** · {len(config['drop_rules'])} aturan buang · buang manual "
             f"{len(config['manual_drop'])} baris · ganti label {len(config['new_labels'])} "
             f"baris · {len(codegen.usable_features(config))} fitur")
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
