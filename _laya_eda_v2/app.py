"""Laya EDA v2: alat bantu EDA ala Data Wrangler / OpenRefine.

Jalankan: jalankan.bat (atau `python -m streamlit run app.py` dari folder ini).
Setiap perbaikan = satu langkah (core/ops.py) → satu blok kode pandas lurus → dijalankan berurutan
(core/steps.py). Tidak ada yang mengubah data sebelum kamu menambah langkah.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ui import common, tab_clean, tab_code, tab_explore, tab_laya, tab_model  # noqa: E402
from ui.flows import FLOWS  # noqa: E402

st.set_page_config(page_title="Laya EDA", layout="wide")

TABS = {
    "Data": tab_explore.show_data,
    "Temuan": tab_explore.show_findings,
    "Operasi": tab_clean.show_operations,
    "Ejaan": tab_clean.show_spelling,
    "Isian bebas": tab_clean.show_free_text,
    "Outlier": tab_clean.show_outliers,
    "Label": tab_model.show_label,
    "Train vs test": tab_model.show_test,
    "AutoGluon": tab_model.show_autogluon,
    "Kode & ekspor": tab_code.show_code,
    "Laya": tab_laya.show_laya,
}

st.sidebar.title("Laya EDA")
st.sidebar.caption("Alat bantu EDA: data mentah tidak berubah sampai kamu menambah langkah.")
with st.sidebar.expander("Cara kerja langkah (flowchart)"):
    st.graphviz_chart(FLOWS["langkah"], width="stretch")
common.sidebar_steps()

for tab, show in zip(st.tabs(list(TABS)), TABS.values()):
    with tab:
        show()
