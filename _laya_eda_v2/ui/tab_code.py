"""Tab Kode & ekspor: skrip lurus dari daftar langkah, log per langkah, unduh semua."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import split as splitter
from core import steps as engine
from ui.kamus import show_table
from ui import common

ROOT = Path(__file__).resolve().parents[1]


def show_code() -> None:
    common.header("kode",
                  "Split + daftar langkah jadi **satu file Python lurus**: baca data → split → satu blok `# %%` per "
                  "langkah → simpan. Tanpa `def`, pandas/numpy biasa. Aplikasi menjalankan blok yang sama persis, jadi "
                  "angka di sini = angka dari kode.",
                  "Untuk ditempel ke notebook lomba dan dijalankan dari atas ke bawah.",
                  "Mengubah data di luar kode ini (manual di Excel) membuat test diproses beda dari train.")
    if not common.need_data():
        return
    p, raw = common.project(), common.state(0)["train"]      # titik awal = train sesudah split, sebelum langkah
    train_path = st.text_input("Path train di skrip", st.session_state.get("train_path_input", "train.csv"),
                               key="train_path_input")
    test_path = st.text_input("Path test di skrip (kosong = tidak ada)",
                              "test.csv" if common.raw_test() is not None else "", key="test_path_input")
    script = engine.build_script(p, train_path, test_path)
    result = common.state()
    if result["error"]:
        st.error(f"Langkah {result['error']['langkah']} gagal: {result['error']['galat']}")
    clean, valid = result["train"], result["valid"]
    cols = st.columns(5)
    cols[0].metric("baris train", f"{len(clean):,}", len(clean) - len(raw) or None,
                   help="Selisih = baris yang dibuang langkahmu (dihitung dari train sesudah split).")
    cols[1].metric("baris valid", f"{len(valid):,}", help="Split: " + splitter.describe(common.split_config()))
    cols[2].metric("baris test", f"{len(result['test']):,}")
    cols[3].metric("kolom train", clean.shape[1], clean.shape[1] - raw.shape[1] or None)
    cols[4].metric("kosong tersisa (train)", f"{int(clean.isna().sum().sum()):,}")
    if not common.split_config():
        st.info("Belum ada split: skrip menulis `valid = train.iloc[:0]` (valid kosong). Pilih metode di halaman "
                "**Split** bila ingin bagian valid untuk menilai model.")
    st.subheader("Log per langkah")
    if result["log"]:
        log = pd.DataFrame(result["log"])
        if "kosong baru" in log:
            # langkah yang galat tidak punya 'kosong baru' (NaN di tabel): tampil kosong, bukan crash
            log["kosong baru"] = [", ".join(f"{c} +{n:,}" for c, n in gained.items()) if isinstance(gained, dict) else ""
                                  for gained in log["kosong baru"]]
        show_table(log.drop(columns=["kolom baru", "kolom hilang"], errors="ignore"), hide_index=True)
    for number, step in enumerate(p["steps"], 1):
        lines = len(engine.step_code(step).splitlines())
        if step.get("on", True) and lines > 40 and step["op"] not in ("free_text", "spelling"):   # dua itu = penggantinya
            st.warning(f"**Langkah {number} ({engine.summary(step)}) = {lines:,} baris kode.** Peta per nilai seperti ini "
                       "tidak berlaku untuk ejaan baru di test dan membuat skrip sulit dibaca. Ganti dengan **Satukan "
                       "ejaan** (tab Ejaan: satu aturan kunci + GABUNG pendek) atau **Bakukan isian bebas** (per kata), "
                       "lalu hapus langkah ini.")
    lost = [c for c in raw.columns if c not in clean.columns]
    if lost:
        st.caption("Kolom asli yang tidak ada lagi (dibuang/di-encode oleh langkahmu): " + ", ".join(lost))

    st.subheader("Kode")
    st.code(script, language="python", line_numbers=True)
    st.subheader("Unduh")
    cols = st.columns(3)
    cols[0].download_button("pipeline_eda.py", script, "pipeline_eda.py", "text/x-python")
    cols[1].download_button("pipeline_eda.ipynb", engine.to_notebook(script), "pipeline_eda.ipynb", "application/json")
    cols[2].download_button("langkah_eda.json", json.dumps(p, ensure_ascii=False, indent=2), "langkah_eda.json")
    cols = st.columns(3)
    cols[0].download_button("train_bersih.csv", clean.to_csv(index=False), "train_bersih.csv", "text/csv")
    if common.split_config():
        cols[1].download_button("valid_bersih.csv", valid.to_csv(index=False), "valid_bersih.csv", "text/csv")
    cols[2].download_button("test_bersih.csv", result["test"].to_csv(index=False), "test_bersih.csv", "text/csv")

    folder = ROOT / "hasil"
    if st.button(f"Simpan semuanya ke {folder}"):
        folder.mkdir(exist_ok=True)
        (folder / "pipeline_eda.py").write_text(script, encoding="utf-8")
        (folder / "pipeline_eda.ipynb").write_text(engine.to_notebook(script), encoding="utf-8")
        (folder / "langkah_eda.json").write_text(json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8")
        clean.to_csv(folder / "train_bersih.csv", index=False)
        if common.split_config():
            valid.to_csv(folder / "valid_bersih.csv", index=False)
        result["test"].to_csv(folder / "test_bersih.csv", index=False)
        st.success(f"Tersimpan di {folder}")
