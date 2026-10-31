"""Tab Kode pipeline & ekspor: jalankan kode dari resep, tampilkan hasil, unduh semua."""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from core.codegen import build_script, to_notebook
from ui import common

ROOT = Path(__file__).resolve().parents[1]


def column_changes(before: pd.Index, after: pd.Index, recipe: dict) -> pd.DataFrame:
    rows = []
    for column in before:
        if column in recipe["drop_columns"]:
            rows.append({"kolom asli": column, "hasil": "dibuang (pilihanmu)"})
        elif column in recipe["encoding"]:
            new = [c for c in after if c.startswith(column) and c not in before]
            rows.append({"kolom asli": column, "hasil": f"{recipe['encoding'][column]['method']}: "
                         + (", ".join(new) if new else column)})
        elif column in after:
            rows.append({"kolom asli": column, "hasil": "tetap"})
        else:
            rows.append({"kolom asli": column, "hasil": "HILANG (tidak seharusnya)"})
    return pd.DataFrame(rows)


def show_code() -> None:
    common.header("kode",
                  "Resep diubah jadi **satu file Python** (pandas, numpy, scikit-learn, pandera, joblib). "
                  "Aplikasi menjalankan file yang sama persis, jadi angka di sini = angka dari kode.",
                  "Untuk dipakai ulang di notebook lomba: `run(train, test)` atau `transform(data_baru, state)`.",
                  "Mengubah data di luar kode ini (manual di Excel) membuat test diproses beda dari train.")
    if not common.need_data():
        return
    r, data = common.recipe(), common.train()
    paths = {"train": "data/train.csv", "test": "data/test.csv" if common.test() is not None else None,
             "output": "hasil"}
    script = build_script(r, paths)
    try:
        result = common.run_pipeline()
    except Exception as error:   # tampilkan galat kode yang dihasilkan apa adanya, jangan disembunyikan
        st.error(f"Pipeline gagal: {type(error).__name__}: {error}")
        st.code(script, language="python")
        return

    clean = result["train"]
    dropped = int(result["drop_reasons"].any(axis=1).sum())
    cols = st.columns(4)
    cols[0].metric("baris train", f"{len(clean):,}", f"-{dropped}" if dropped else None)
    cols[1].metric("kolom train", clean.shape[1], clean.shape[1] - data.shape[1] or None)
    cols[2].metric("baris test", "-" if result["test"] is None else f"{len(result['test']):,}")
    cols[3].metric("kosong tersisa", int(clean.isna().sum().sum()))
    for warning in result["warnings"]:
        st.warning(warning)
    if dropped:
        st.caption("Baris dibuang per alasan: " + ", ".join(
            f"{k}: {v}" for k, v in result["drop_reasons"].sum().items() if v))

    st.subheader("Kolom asli → hasil")
    changes = column_changes(data.columns.drop(r["label"], errors="ignore"), clean.columns, r)
    st.dataframe(changes, width="stretch", hide_index=True)
    st.subheader("Train bersih (20 baris pertama)")
    st.dataframe(clean.head(20), width="stretch")

    st.subheader("Kode")
    st.code(script, language="python", line_numbers=True)

    st.subheader("Unduh")
    cols = st.columns(3)
    cols[0].download_button("pipeline_eda.py", script, "pipeline_eda.py", "text/x-python")
    cols[1].download_button("pipeline_eda.ipynb", to_notebook(script), "pipeline_eda.ipynb", "application/json")
    cols[2].download_button("resep_eda.json", json.dumps(r, ensure_ascii=False, indent=2), "resep_eda.json")
    cols = st.columns(3)
    cols[0].download_button("train_bersih.csv", clean.to_csv(index=False), "train_bersih.csv", "text/csv")
    if result["test"] is not None:
        cols[1].download_button("test_bersih.csv", result["test"].to_csv(index=False), "test_bersih.csv")
    buffer = io.BytesIO()
    joblib.dump(result["state"], buffer)
    cols[2].download_button("pipeline_eda.joblib", buffer.getvalue(), "pipeline_eda.joblib")

    folder = ROOT / "hasil"
    if st.button(f"Simpan semuanya ke {folder}"):
        folder.mkdir(exist_ok=True)
        (folder / "pipeline_eda.py").write_text(script, encoding="utf-8")
        (folder / "pipeline_eda.ipynb").write_text(to_notebook(script), encoding="utf-8")
        (folder / "resep_eda.json").write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        clean.to_csv(folder / "train_bersih.csv", index=False)
        if result["test"] is not None:
            result["test"].to_csv(folder / "test_bersih.csv", index=False)
        joblib.dump(result["state"], folder / "pipeline_eda.joblib")
        st.success(f"Tersimpan di {folder}")
