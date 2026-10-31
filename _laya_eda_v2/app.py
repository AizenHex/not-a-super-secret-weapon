"""Laya EDA v2: alat bantu EDA. Menemukan masalah data, menjelaskan pilihan, menulis kode pipeline.

Jalankan: jalankan.bat (atau `python -m streamlit run app.py` dari folder ini).
Semua keputusan masuk ke resep (core/recipe.py) → kode (core/codegen.py) → dijalankan (core/runner.py).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import common, tab_clean, tab_code, tab_explore, tab_laya, tab_model  # noqa: E402

st.set_page_config(page_title="Laya EDA", layout="wide")

TABS = {
    "Data": tab_explore.show_data,
    "Ringkasan": tab_explore.show_summary,
    "Tipe kolom": tab_explore.show_types,
    "Kosong & penanda": tab_clean.show_missing,
    "Kategori": tab_clean.show_categories,
    "Isian bebas": tab_clean.show_free_text,
    "Outlier": tab_clean.show_outliers,
    "Baris": tab_clean.show_rows,
    "Isi kosong": tab_model.show_impute,
    "Label": tab_model.show_label,
    "Train vs test": tab_model.show_test,
    "Encoding": tab_model.show_encoding,
    "Split": tab_model.show_split,
    "Kode & ekspor": tab_code.show_code,
    "Laya": tab_laya.show_laya,
}

st.sidebar.title("Laya EDA")
st.sidebar.caption("Alat bantu EDA: tidak ada yang mengubah data sebelum kamu menyimpan ke resep.")
r = common.recipe()
st.sidebar.json({k: v for k, v in r.items() if v not in ([], {}, None, False)}, expanded=False)

for tab, show in zip(st.tabs(list(TABS)), TABS.values()):
    with tab:
        show()
