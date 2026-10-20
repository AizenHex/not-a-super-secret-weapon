"""Tab Kode EDA: ubah pilihan + temuan + catatanmu jadi kode notebook (codegen.py)."""
import pandas as pd
import streamlit as st

import codegen
import helpers
from penjelasan import show_explanation


def notebook_path(csv_path):
    # notebook lomba ada di _.NEW-GEN/<folder>/, sama dalamnya dengan _laya_eda
    if csv_path.startswith("../"):
        return csv_path
    return f"../_laya_eda/{csv_path}"


def notes_to_steps(ctx):
    notes = helpers.load_notes(ctx["folder"])
    number_label = False
    if ctx["label_column"] is not None:
        number_label = pd.api.types.is_integer_dtype(ctx["data"][ctx["label_column"]])
    drop_rows = []
    new_labels = {}
    for row in notes.index:
        if notes.at[row, "keputusan"] == "buang":
            drop_rows.append(int(row))
        if notes.at[row, "keputusan"] == "ganti_label" and notes.at[row, "label_baru"] != "":
            label = notes.at[row, "label_baru"]
            if number_label:
                label = int(label)    # label angka (0/1/2) tetap angka di kode
            new_labels[int(row)] = label
    return drop_rows, new_labels


def used_lenses(ctx):
    lenses = {}
    for name in ctx["lens_done"]:
        lenses[name] = ctx["lenses"][name]
    return lenses


def pick_outlier(ctx):
    number_options = ["(tidak ada)"] + list(ctx["numbers"].columns)
    number = st.selectbox("Outlier per kategori untuk kolom angka", number_options,
                          key="kode_number")
    if number == "(tidak ada)":
        return None, None
    category_options = ctx["dimensions"]
    if len(category_options) == 0:
        st.caption("Belum ada kategori (lensa Laya atau label) untuk pembanding.")
        return None, None
    category = st.selectbox("... dibandingkan di dalam kategori", category_options,
                            key="kode_category")
    return number, category


def build_config(ctx):
    drop_rows, new_labels = notes_to_steps(ctx)
    test_csv = st.text_input("CSV test (kosongkan bila tidak ada; path dari folder notebook)",
                             "", key="kode_test")
    if test_csv.strip() == "":
        test_csv = None
    drop_duplicates = st.checkbox("Buang duplikat persis", value=True, key="kode_dup")
    number, category = pick_outlier(ctx)
    return {"csv_path": notebook_path(ctx["csv_path"]), "test_csv": test_csv,
            "columns": ctx["columns"], "label_column": ctx["label_column"],
            "number_columns": ctx["number_columns"], "indonesian_format": ctx["indonesian"],
            "use_log": ctx["use_log"], "ratio": ctx["ratio"], "outlier_number": number,
            "outlier_category": category, "lenses": used_lenses(ctx), "drop_rows": drop_rows,
            "new_labels": new_labels, "drop_duplicates": drop_duplicates}


def show(ctx):
    st.write("Pilihan di sidebar, lensa Laya yang sudah dijalankan, dan catatan "
             "*buang / ganti label* di tab Telusuri diubah jadi kode notebook. Kode ini tidak "
             "butuh `_laya_eda`, jadi bisa ditempel ke notebook lomba.")
    config = build_config(ctx)
    cells = codegen.build_cells(config)
    script = codegen.to_script(cells)
    st.write(f"**{len(cells)} sel** · buang {len(config['drop_rows'])} baris · ganti label "
             f"{len(config['new_labels'])} baris · {len(config['lenses'])} lensa Laya")
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
