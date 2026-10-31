"""Bantuan bersama: proyek (daftar langkah) di session_state, data per langkah (ter-cache), pratinjau, flowchart."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core import steps as engine
from core.ops import OPS, summary
from ui.flows import FLOWS


def project() -> dict:
    if "project" not in st.session_state:
        st.session_state["project"] = engine.new_project()
    return st.session_state["project"]


def label() -> str | None:
    return project()["label"]


def raw_train() -> pd.DataFrame | None:
    return st.session_state.get("train")


def raw_test() -> pd.DataFrame | None:
    return st.session_state.get("test")


def need_data() -> bool:
    if raw_train() is None:
        st.info("Muat data train dulu di tab **Data**.")
        return False
    return True


def header(flow: str, what: str, fits: str, danger: str) -> None:
    """Pembuka setiap tab: apa yang dilakukan, kapan cocok, kapan berbahaya, flowchart."""
    st.markdown(what)
    left, right = st.columns(2)
    left.success(f"**Kapan cocok.** {fits}")
    right.warning(f"**Kapan berbahaya.** {danger}")
    with st.expander("Cara kerja (flowchart)"):
        st.graphviz_chart(FLOWS[flow], width="stretch")


@st.cache_data(max_entries=12, show_spinner="Menjalankan langkah…")
def _run(key: str, data_id: str, _train: pd.DataFrame, _test: pd.DataFrame | None) -> dict:
    steps, lab = json.loads(key)
    return engine.run(steps, lab, _train, _test)


def state(upto: int | None = None) -> dict:
    """Hasil menjalankan langkah 1..upto (None = semua): {'train', 'test', 'log', 'error'}."""
    steps = project()["steps"][:upto]
    key = json.dumps([steps, label()], sort_keys=True, ensure_ascii=False, default=str)
    return _run(key, st.session_state.get("data_id", ""), raw_train(), raw_test())


def current() -> pd.DataFrame:
    """Train setelah semua langkah, tanpa label: dasar pilihan kolom dan pratinjau."""
    data = state()["train"]
    return data.drop(columns=[label()]) if label() in data else data


def object_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c])]


def number_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]


def op_help(op: str) -> None:
    spec = OPS[op]
    st.markdown(f"**{spec['title']}.** {spec['what']}")
    left, right = st.columns(2)
    left.success(f"**Kapan cocok.** {spec['fits']}")
    right.warning(f"**Kapan berbahaya.** {spec['danger']}")


def preview_step(op: str, params: dict, key: str, replace: int | None = None) -> None:
    """Pratinjau satu langkah (sebelum → sesudah) di atas data sesudah langkah sebelumnya.

    replace = None: tombol 'tambah langkah' (di akhir). replace = i: tombol 'simpan ubahan' untuk langkah i.
    """
    base = state(replace)
    candidate = [{"op": op, "params": params, "on": True}]
    result = engine.run(candidate, label(), base["train"], base["test"])
    with st.expander("Kode langkah ini"):
        st.code(engine.step_code(candidate[0]), language="python")
    if result["error"]:
        st.error(f"Langkah gagal: {result['error']['galat']}")
        return
    before, after = base["train"], result["train"]
    info = engine.diff(before, after)
    cols = st.columns(4)
    cols[0].metric("baris train", f"{len(after):,}", -len(info["baris dibuang"]) or None)
    cols[1].metric("kolom", after.shape[1], after.shape[1] - before.shape[1] or None)
    cols[2].metric("sel berubah", f"{int(info['sel berubah'].sum()):,}")
    cols[3].metric("baris test", f"{len(result['test']):,}")
    if result["log"][0]["keluaran"]:
        st.caption(result["log"][0]["keluaran"])
    if info["kolom baru"]:
        st.write(f"Kolom baru ({len(info['kolom baru'])}): " + ", ".join(info["kolom baru"][:30]))
    if info["kolom hilang"]:
        st.write("Kolom hilang: " + ", ".join(info["kolom hilang"]))
    if len(info["sel berubah"]):
        st.caption("Sel berubah per kolom: " + ", ".join(f"{c} ({n:,})" for c, n in info["sel berubah"].head(8).items()))
        st.dataframe(engine.changed_examples(before, after, info), width="stretch")
    if len(info["baris dibuang"]):
        st.caption("Contoh baris yang dibuang")
        st.dataframe(before.loc[info["baris dibuang"][:20]], width="stretch")
    if replace is None:
        if st.button(f"Tambah langkah: {OPS[op]['title']}", type="primary", key=f"add_{key}"):
            engine.add_step(project(), op, params)
            st.toast("Langkah ditambahkan")
            st.rerun()
    elif st.button(f"Simpan ubahan langkah {replace + 1}", type="primary", key=f"save_{key}"):
        project()["steps"][replace]["params"] = params
        st.toast("Langkah diubah")
        st.rerun()


def sidebar_steps() -> None:
    """Daftar langkah ala Data Wrangler: nyalakan/matikan, pindah, hapus; galat ditandai."""
    p = project()
    st.sidebar.subheader(f"Langkah ({len(p['steps'])})")
    if not p["steps"]:
        st.sidebar.caption("Belum ada langkah: data belum diubah sama sekali. Tambah langkah dari tab "
                           "Temuan, Operasi, Ejaan, Isian bebas, atau Outlier.")
        return
    log = {entry["langkah"]: entry for entry in state()["log"]} if raw_train() is not None else {}
    for i, step in enumerate(p["steps"]):
        entry = log.get(i + 1)
        on = step.get("on", True)
        mark = "⛔ " if entry and entry["galat"] else ("" if on else "⏸ ")
        box = st.sidebar.container(border=True)
        box.markdown(f"**{i + 1}.** {mark}{summary(step)}")
        if entry and entry["galat"]:
            box.error(entry["galat"])
        elif entry:
            box.caption(f"baris {entry['baris train']} · kolom {entry['kolom']}")
        cols = box.columns(4)
        if cols[0].button("⏸" if on else "▶", key=f"on_{i}", help="matikan / nyalakan"):
            step["on"] = not on
            st.rerun()
        if cols[1].button("↑", key=f"up_{i}", disabled=i == 0):
            p["steps"][i - 1], p["steps"][i] = p["steps"][i], p["steps"][i - 1]
            st.rerun()
        if cols[2].button("↓", key=f"down_{i}", disabled=i == len(p["steps"]) - 1):
            p["steps"][i + 1], p["steps"][i] = p["steps"][i], p["steps"][i + 1]
            st.rerun()
        if cols[3].button("✕", key=f"del_{i}", help="hapus langkah"):
            p["steps"].pop(i)
            st.rerun()
