"""Tab Submission (fitur extra E6): template kode `submission.csv` dari `sample_submission.csv` + validator.

Sample dibaca → kolom id & target, urutan id, tipe target. Template (blok lurus, tanpa def) = baca data → split +
blok EDA yang sama dengan Kode EDA → muat pipeline.joblib (tab Latih) → cek skema → prediksi → DataFrame baru,
kolom ditambah satu per satu mengikuti sample → validator → simpan. Validator juga bisa dipakai untuk file
submission mana pun yang diunggah.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import submission
from ui import common
from ui.kamus import show_table


def load_sample() -> pd.DataFrame | None:
    path = st.text_input("Path sample_submission.csv", st.session_state.get("sample_path", ""), key="sample_path")
    upload = st.file_uploader("… atau unggah sample_submission.csv", type="csv", key="sample_upload")
    if upload is not None:
        return pd.read_csv(upload)
    if path and Path(path).exists():
        return pd.read_csv(path)
    if path:
        st.error(f"File tidak ditemukan: {path}")
    return None


def show_submission() -> None:
    common.header("submission",
                  "Fitur **extra**: tulis **template kode** `submission.csv` yang mengikuti `sample_submission.csv` "
                  "persis (nama & urutan kolom, urutan id, tipe target `3` bukan `3.0`) dan **validator** yang menolak "
                  "submission rusak sebelum diunggah.",
                  "Akhir pengerjaan: sesudah pipeline disimpan di tab **Latih**.",
                  "Template memakai path di bawah; langkah EDA yang diubah sesudah pipeline disimpan tidak ikut "
                  "(template memakai langkah milik pipeline).")
    sample = load_sample()
    if sample is None:
        st.info("Isi path atau unggah `sample_submission.csv`.")
        return
    spec = submission.read_sample(sample)
    show_table(pd.DataFrame({"isi": ["kolom id", "kolom target", "jumlah baris", "target bilangan bulat",
                                     "kolom target lain (tidak diprediksi)"],
                             "nilai": [spec["id"], spec["target"], f"{spec['baris']:,}", "ya" if spec["bulat"] else "tidak",
                                       ", ".join(spec["target_lain"]) or "-"]}), hide_index=True)
    st.subheader("Template kode")
    pipeline_path = st.text_input("Path pipeline.joblib (dari tab Latih)", st.session_state.get("pipeline_path", ""),
                                  key="pipeline_path_input")
    meta_path = Path(f"{pipeline_path}.json")
    if not pipeline_path or not meta_path.exists():
        st.info("Simpan pipeline di tab **Latih** dulu (tombol 'Simpan pipeline.joblib'), atau isi path-nya.")
    else:
        info = json.loads(meta_path.read_text(encoding="utf-8"))
        left, right = st.columns(2)
        paths = {"train": left.text_input("Path train di template", st.session_state.get("train_path", "train.csv"),
                                          key="sub_train"),
                 "test": right.text_input("Path test di template", st.session_state.get("test_path", "") or "test.csv",
                                          key="sub_test"),
                 "sample": st.session_state.get("sample_path") or "sample_submission.csv",
                 "pipeline": pipeline_path}
        project = {"label": info["label"], "steps": info["langkah"], "split": info.get("split")}
        code = submission.template(project, info, spec, paths)
        st.caption(f"Model {info['model']} · tugas {info['tugas']} · metrik {info['metrik']} · skor CV "
                   f"{info.get('skor_cv', float('nan')):.4f} · disimpan dengan sklearn {info['sklearn']}.")
        st.code(code, language="python", line_numbers=True)
        st.download_button("buat_submission.py", code, "buat_submission.py", "text/x-python")
    st.subheader("Validator submission")
    upload = st.file_uploader("Unggah submission.csv untuk diperiksa", type="csv", key="check_upload")
    if upload is not None:
        text = upload.getvalue().decode("utf-8", errors="replace")
        made = pd.read_csv(upload)
        classes = None
        if common.raw_train() is not None and common.label() in common.raw_train():
            classes = common.raw_train()[common.label()].dropna().unique().tolist()
        problems = submission.validate(made, sample, classes, text)
        if problems:
            st.error(f"Submission DITOLAK ({len(problems)} masalah).")
            show_table(pd.DataFrame({"masalah": problems}), hide_index=True)
        else:
            st.success("Submission lolos validator: format sama dengan sample.")
