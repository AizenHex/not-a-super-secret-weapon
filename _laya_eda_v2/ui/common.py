"""Bantuan bersama: proyek (daftar langkah) di session_state, data per langkah (ter-cache), pratinjau, flowchart."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from core import steps as engine
from core.ops import OPS, summary
from ui.flows import FLOWS


# Simpan otomatis: daftar langkah + label + path data ditulis ke file setiap berubah, dan dibaca lagi saat
# server/halaman dibuka ulang. Uji memakai file lain lewat LAYA_SESI agar tidak menimpa progresmu.
SESSION_FILE = Path(os.environ.get("LAYA_SESI", Path(__file__).resolve().parents[1] / "hasil" / "sesi_terakhir.json"))


def restore_session() -> None:
    """Dipanggil sekali di awal app.py, SEBELUM widget path data dibuat."""
    if "project" in st.session_state:
        return
    saved = {}
    if SESSION_FILE.exists():
        try:
            saved = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
    st.session_state["project"] = {"label": saved.get("label"), "steps": saved.get("steps", [])}
    for key in ("train_path", "test_path"):
        if saved.get(key):
            st.session_state[key] = saved[key]
    st.session_state["saved_session"] = json.dumps(saved, sort_keys=True, ensure_ascii=False, default=str)


def save_session() -> None:
    """Dipanggil di akhir app.py: tulis file hanya bila ada yang berubah."""
    p = project()
    data = {"label": p["label"], "steps": p["steps"], "train_path": st.session_state.get("train_path", ""),
            "test_path": st.session_state.get("test_path", "")}
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    if text != st.session_state.get("saved_session"):
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        st.session_state["saved_session"] = text


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
        st.info("Muat data train dulu di sidebar **Data**.")
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


def _key(steps: list[dict], lab: str | None) -> str:
    return json.dumps([steps, lab], sort_keys=True, ensure_ascii=False, default=str)


@st.cache_resource(max_entries=64, show_spinner="Menjalankan langkah…")
def _run(key: str, data_id: str, _train: pd.DataFrame, _test: pd.DataFrame | None) -> dict:
    """Bertahap: hasil langkah 1..n = langkah n dijalankan di atas hasil 1..n-1 (yang sudah ter-cache).

    Menambah/mengubah langkah terakhir hanya menjalankan langkah itu. cache_resource = tanpa salin/pickle,
    jadi hasilnya HANYA dibaca (engine.run selalu menyalin sebelum mengubah).
    """
    steps, lab = json.loads(key)
    if not steps:
        return engine.run([], lab, _train, _test)
    base = _run(_key(steps[:-1], lab), data_id, _train, _test)
    if base["error"] or not steps[-1].get("on", True):
        return base
    last = engine.run(steps[-1:], lab, base["train"], base["test"], first=len(steps))
    return {"train": last["train"], "test": last["test"], "log": base["log"] + last["log"], "error": last["error"]}


def state(upto: int | None = None) -> dict:
    """Hasil menjalankan langkah 1..upto (None = semua): {'train', 'test', 'log', 'error'}. Hanya dibaca."""
    return _run(_key(project()["steps"][:upto], label()), st.session_state.get("data_id", ""), raw_train(), raw_test())


def state_key(upto: int | None = None) -> str:
    """Kunci data di langkah ke-upto: dipakai juga untuk cache analisis (temuan, wawasan kolom)."""
    return _key(project()["steps"][:upto], label()) + st.session_state.get("data_id", "")


@st.cache_data(max_entries=32, show_spinner=False)
def analysis(name: str, key: str, _data: pd.DataFrame, *args):
    """audit.<name>(data, *args), di-cache per kunci data (state_key): tidak dihitung ulang tiap klik."""
    from core import audit

    return getattr(audit, name)(_data, *args)


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


def preview_steps(op: str, plan: list[dict], key: str) -> None:
    """Pratinjau beberapa langkah sejenis sekaligus (satu per kolom) + satu tombol untuk menambah semuanya.

    key = key data_editor asal usulan: dihapus sesudah menambah agar centang tidak terbawa (dan tidak dobel).
    """
    st.subheader(f"Pratinjau {len(plan)} langkah sekaligus")
    base = state()
    candidate = [{"op": op, "params": params, "on": True} for params in plan]
    result = engine.run(candidate, label(), base["train"], base["test"])
    if result["error"]:
        st.error(f"Langkah {result['error']['langkah']} gagal: {result['error']['galat']}")
        return
    info = engine.diff(base["train"], result["train"])
    cols = st.columns(3)
    cols[0].metric("baris train", f"{len(result['train']):,}", -len(info["baris dibuang"]) or None)
    cols[1].metric("sel berubah", f"{int(info['sel berubah'].sum()):,}")
    cols[2].metric("baris test", f"{len(result['test']):,}")
    if len(info["sel berubah"]):
        st.caption("Sel berubah per kolom: " + ", ".join(f"{c} ({n:,})" for c, n in info["sel berubah"].head(12).items()))
    with st.expander("Kode langkah-langkah ini"):
        st.code("\n\n".join(engine.step_code(step) for step in candidate), language="python")
    if st.button(f"Tambah {len(plan)} langkah: {OPS[op]['title']}", type="primary", key=f"add_many_{key}"):
        for params in plan:
            engine.add_step(project(), op, params)
        st.session_state.pop(key, None)
        st.toast(f"{len(plan)} langkah ditambahkan")
        st.rerun()


def sidebar_steps() -> None:
    """Daftar langkah ala Data Wrangler: nyalakan/matikan, pindah, hapus; galat ditandai."""
    p = project()
    st.sidebar.subheader(f"Langkah ({len(p['steps'])})")
    if not p["steps"]:
        st.sidebar.caption("Belum ada langkah: data belum diubah sama sekali. Tambah langkah dari tab "
                           "Temuan, Operasi, Ejaan, Isian bebas, atau Outlier.")
        return
    st.sidebar.caption(f"Tersimpan otomatis di `{SESSION_FILE.name}` (kembali lagi sesudah server dinyalakan ulang).")
    if st.sidebar.button("Kosongkan semua langkah", key="clear_steps"):
        st.session_state["confirm_clear"] = True
    if st.session_state.get("confirm_clear"):
        left, right = st.sidebar.columns(2)
        if left.button("Ya, kosongkan", type="primary", key="clear_yes"):
            p["steps"].clear()
            st.session_state["confirm_clear"] = False
            st.rerun()
        if right.button("Batal", key="clear_no"):
            st.session_state["confirm_clear"] = False
            st.rerun()
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
