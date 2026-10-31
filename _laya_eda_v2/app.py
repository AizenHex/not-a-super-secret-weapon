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

KEEP = ["insight_column", "op_group", "op_name", "spell_column", "spell_method", "free_column", "free_min",
        "free_cut", "outlier_column", "out_mode", "out_m", "out_a", "out_low", "out_high", "out_group", "label_model", "ag_time", "ag_presets",
        "spell_mode", "spell_all_method"]

st.sidebar.title("Laya EDA")
st.sidebar.caption("Alat bantu EDA: data mentah tidak berubah sampai kamu menambah langkah.")
with st.sidebar.expander("Cara kerja langkah (flowchart)"):
    st.graphviz_chart(FLOWS["langkah"], width="stretch")
common.restore_session()
tab_explore.sidebar_data()
common.sidebar_steps()

# pilihan di halaman yang sedang tidak dibuka tetap diingat (Streamlit menghapus state widget yang tidak tampil)
for name in KEEP:
    if name in st.session_state:
        st.session_state[name] = st.session_state[name]

# Hanya halaman yang dibuka yang dihitung (st.tabs menghitung SEMUA tab di setiap klik → terasa macet).
page = st.radio("Halaman", list(TABS), horizontal=True, key="page", label_visibility="collapsed")
st.divider()
try:
    TABS[page]()
finally:
    common.save_session()
