"""Tab Telusuri + catatan: saring baris, lihat alasannya, isi keputusan dan catatan."""
import streamlit as st

import helpers
from penjelasan import show_explanation

DECISIONS = ["", "simpan", "buang", "ganti_label", "periksa_lagi"]
NOTE_COLUMNS = helpers.NOTE_COLUMNS


def filter_rows(ctx):
    table = ctx["table"]
    first, second, third = st.columns(3)
    filter_dim = first.selectbox("Saring menurut", ["(semua)"] + ctx["dimensions"])
    filter_values = []
    if filter_dim != "(semua)":
        filter_values = second.multiselect("Nilai", list(table[filter_dim].dropna().unique()))
    search = third.text_input("Cari kata")
    only_doubt = st.checkbox("Hanya baris yang membuat Laya ragu", key="only_doubt")
    only_reason = st.checkbox("Hanya baris yang punya alasan", key="only_reason")
    view = table
    if len(filter_values) > 0:
        view = view[view[filter_dim].isin(filter_values)]
    if search != "":
        view = view[ctx["texts"][view.index].str.contains(search, case=False, regex=False)]
    if only_doubt:
        view = view[view["laya_ragu"]]
    if only_reason:
        view = view[ctx["reasons"].reindex(view.index) != ""]
    return view


def store_notes(edited, notes, folder):
    for i in range(len(edited)):
        row = edited["row"].iloc[i]
        notes.loc[row, NOTE_COLUMNS] = edited[NOTE_COLUMNS].iloc[i].fillna("").to_list()
    helpers.save_notes(folder, notes)


def show(ctx):
    view = filter_rows(ctx)
    st.write(f"{len(view)} baris cocok · ditampilkan maks 500 · isi kolom keputusan / catatan "
             "lalu tekan **Simpan catatan**")
    notes = helpers.load_notes(ctx["folder"])
    shown = view.head(500)[ctx["show_columns"]].copy()
    shown.insert(0, "row", shown.index)
    shown["alasan"] = ctx["reasons"].reindex(shown.index)
    for column in NOTE_COLUMNS:
        shown[column] = notes[column].reindex(shown.index).fillna("")
    locked = list(shown.columns.difference(NOTE_COLUMNS))
    decision_box = st.column_config.SelectboxColumn("keputusan", options=DECISIONS)
    edited = st.data_editor(shown, column_config={"keputusan": decision_box}, disabled=locked,
                            hide_index=True, key="editor")
    if st.button("Simpan catatan"):
        store_notes(edited, notes, ctx["folder"])
        st.success(f"Tersimpan di _laya_eda/{ctx['folder']}/catatan.csv")
    show_explanation("telusuri", f"{int((ctx['reasons'] != '').sum())} baris punya alasan.")
