"""Bantuan bersama untuk tab: resep di session_state, flowchart, penjelasan, pipeline ter-cache."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core import recipe as rcp
from core.codegen import build_script
from core.runner import load_script
from ui.flows import FLOWS


def recipe() -> dict:
    if "recipe" not in st.session_state:
        st.session_state["recipe"] = rcp.new_recipe()
    return st.session_state["recipe"]


def recipe_key() -> str:
    return json.dumps(recipe(), sort_keys=True, ensure_ascii=False, default=str)


def train() -> pd.DataFrame | None:
    return st.session_state.get("train")


def test() -> pd.DataFrame | None:
    return st.session_state.get("test")


def features() -> pd.DataFrame:
    label = recipe()["label"]
    data = train()
    return data.drop(columns=[label]) if label in data else data


def header(flow: str, what: str, fits: str, danger: str) -> None:
    """Pembuka setiap tab: apa yang dilakukan, kapan cocok, kapan berbahaya, flowchart."""
    st.markdown(what)
    left, right = st.columns(2)
    left.success(f"**Kapan cocok.** {fits}")
    right.warning(f"**Kapan berbahaya.** {danger}")
    with st.expander("Cara kerja (flowchart)"):
        st.graphviz_chart(FLOWS[flow], width="stretch")


def saved(message: str) -> None:
    st.toast(message)


@st.cache_resource(max_entries=8)
def _module(key: str):
    return load_script(build_script(json.loads(key), {}))


def pipeline():
    """Modul pipeline dari resep saat ini. Semua pratinjau memakai fungsi di modul ini (P2)."""
    return _module(recipe_key())


@st.cache_data(max_entries=4, show_spinner="Menjalankan langkah tanpa statistik…")
def _prepared(key: str, data_id: str, _data: pd.DataFrame) -> pd.DataFrame:
    return _module(key).prepare(_data)


def prepared() -> pd.DataFrame:
    """Data setelah prepare() (tipe, penanda kosong, peta ejaan), dari kode pipeline sendiri."""
    return _prepared(recipe_key(), st.session_state.get("train_id", ""), features())


@st.cache_data(max_entries=2, show_spinner="Menjalankan pipeline penuh…")
def _run(key: str, data_id: str, _train: pd.DataFrame, _test: pd.DataFrame | None) -> dict:
    return _module(key).run(_train, _test)


def run_pipeline() -> dict:
    return _run(recipe_key(), st.session_state.get("train_id", ""), train(), test())


def object_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if not pd.api.types.is_numeric_dtype(data[column])]


def number_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]


def before_after(before: pd.Series, after: pd.Series, top: int = 15) -> None:
    """Dua tabel berdampingan: value_counts sebelum dan sesudah."""
    left, right = st.columns(2)
    left.caption(f"Sebelum: {before.nunique()} nilai unik, {before.isna().sum()} kosong")
    left.dataframe(before.value_counts(dropna=False).head(top), width="stretch")
    right.caption(f"Sesudah: {after.nunique()} nilai unik, {after.isna().sum()} kosong")
    right.dataframe(after.value_counts(dropna=False).head(top), width="stretch")


def need_data() -> bool:
    if train() is None:
        st.info("Muat data train dulu di tab **Data**.")
        return False
    return True
