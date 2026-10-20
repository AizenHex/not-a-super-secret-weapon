"""Tab Ekspor: satu file CSV berisi data + jawaban Laya + cek teks + angka + alasan + catatan."""
import streamlit as st

import helpers
from penjelasan import show_explanation

NOTE_COLUMNS = helpers.NOTE_COLUMNS


def build_export(ctx):
    export = ctx["table"].copy()
    for column in ["panjang", "jumlah_kata", "rasio_bukan_huruf", "duplikat", "duplikat_mirip"]:
        export[f"teks__{column}"] = ctx["stats"][column]
    for column in ctx["numbers"].columns:
        export[f"{column}__angka"] = ctx["numbers"][column]
    export["alasan"] = ctx["reasons"]
    notes = helpers.load_notes(ctx["folder"])
    for column in NOTE_COLUMNS:
        export[column] = notes[column].reindex(export.index).fillna("")
    return export


def show(ctx):
    st.write("Satu file berisi data asli + jawaban Laya (+ yakin) + kelompok + cek teks + angka "
             "+ alasan outlier + catatanmu. File ini dibaca notebook EDA program.")
    export = build_export(ctx)
    export_path = f"{ctx['folder']}/data_berlensa.csv"
    if st.button("Tulis file", type="primary"):
        export.to_csv(export_path, index_label="row")
        st.success(f"Tersimpan: _laya_eda/{export_path}")
    st.code(f'data = pd.read_csv("../_laya_eda/{export_path}")\n'
            'data = data[data["keputusan"] != "buang"]', language="python")
    st.dataframe(export.head(20))
    show_explanation("ekspor")
