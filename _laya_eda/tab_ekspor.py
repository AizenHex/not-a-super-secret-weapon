"""Tab Ekspor: (1) satu file CSV berisi data + jawaban Laya + cek teks + angka + alasan + catatan,
(2) train siap pakai: catatan sudah diterapkan (buang baris, label baru) + fitur pilihan,
lalu buang kustom (kolom + aturan baris, custom_drop.py)."""
import json
import os

import pandas as pd
import streamlit as st

import custom_drop
import features
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


def saved_groups(ctx):
    # grup fitur yang lolos di tab Fitur → label jadi pilihan awal
    path = f"{ctx['folder']}/fitur_terpilih.json"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as file:
        content = json.load(file)
    chosen = []
    for item in content["fitur_terpilih"]:
        if item["grup"] in features.BUILDERS:
            chosen.append(item["grup"])
    return chosen


def feature_options(ctx):
    # pengaturan yang sama dengan tab Fitur → label, diisi pilihan otomatis yang wajar
    dimension = None
    candidates = features.dimension_candidates(ctx)
    if len(candidates) > 0:
        dimension = candidates[0]
    return {"dimension": dimension, "category_columns": features.category_candidates(ctx),
            "keywords": ["/bulan", "nego", "hubungi"],
            "date_columns": features.date_candidates(ctx)}


def apply_notes(ctx, train):
    # catatan tab Telusuri: ganti label dulu, baru buang baris
    notes = helpers.load_notes(ctx["folder"])
    label = ctx["label_column"]
    changed = 0
    if label is not None:
        number_label = pd.api.types.is_integer_dtype(train[label])
        for row in notes.index:
            if notes.at[row, "keputusan"] != "ganti_label" or notes.at[row, "label_baru"] == "":
                continue
            new_label = notes.at[row, "label_baru"]
            if number_label:
                new_label = int(new_label)
            train.loc[row, label] = new_label
            changed = changed + 1
    dropped = notes.index[notes["keputusan"] == "buang"]
    train = train.drop(index=dropped)
    return train, changed, len(dropped)


def build_train(ctx, groups, with_lenses, with_numbers):
    train = ctx["data"].copy()
    if with_numbers:
        for column in ctx["numbers"].columns:
            train[f"{column}_angka"] = ctx["numbers"][column]
    if with_lenses:
        for name in features.usable_lenses(ctx):
            train[name] = ctx["table"][name]
            train[f"{name}__nilai"] = ctx["table"][f"{name}__nilai"]
    options = feature_options(ctx)
    frames = [train]
    for group in groups:
        frame = features.BUILDERS[group](ctx, options)
        frames.append(frame.reindex(train.index))    # baris yang belum dibaca Laya = NaN
    train = pd.concat(frames, axis=1)    # sekali gabung, jauh lebih cepat dari satu per satu
    return apply_notes(ctx, train)


def group_title(name):
    return features.GROUPS[name][1]


def show_ready_train(ctx):
    st.subheader("Train siap pakai")
    st.write("Data asli dengan **catatanmu sudah diterapkan** (baris *buang* dibuang, label "
             "*ganti_label* diganti) + kolom fitur pilihanmu + buang kustom. File ini langsung "
             "dipakai model.")
    group_names = list(features.BUILDERS)
    groups = st.multiselect("Fitur yang ditambahkan (pilihan awal = yang lolos di tab Fitur)",
                            group_names, default=saved_groups(ctx),
                            format_func=group_title, key=f"ekspor_fitur__{ctx['folder']}")
    left, right = st.columns(2)
    with_lenses = left.checkbox("Sertakan jawaban lensa Laya", value=True, key="ekspor_lensa")
    with_numbers = right.checkbox("Sertakan kolom angka hasil baca (`<kolom>_angka`)",
                                  value=True, key="ekspor_angka")
    train, changed, dropped = build_train(ctx, groups, with_lenses, with_numbers)
    train, custom_dropped = custom_drop.show(ctx, train)
    ctx["ready_train"] = train    # dibagi train/validasi di tab Split data
    st.write(f"{len(train)} baris × {len(train.columns)} kolom · label diganti: {changed} · "
             f"dibuang (catatan): {dropped} · dibuang (aturan kustom): {custom_dropped}")
    path = f"{ctx['folder']}/train_siap.csv"
    first, second = st.columns(2)
    if first.button("Tulis train_siap.csv", key="ekspor_train_tulis"):
        train.to_csv(path, index_label="row")
        st.success(f"Tersimpan: _laya_eda/{path}")
    second.download_button("Unduh train_siap.csv", train.to_csv(index_label="row"),
                           "train_siap.csv", key="ekspor_train_unduh")
    st.dataframe(train.head(20))


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
    show_ready_train(ctx)
    show_explanation("ekspor")
