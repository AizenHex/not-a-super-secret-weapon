"""Buang kustom untuk train_siap (tab Ekspor): kolom yang tidak dipakai + aturan buang baris
(pandas query, satu per baris). Pilihan disimpan di hasil/<nama_csv>/buang_kustom.json supaya
tidak hilang saat aplikasi dibuka lagi."""
import json
import os

import pandas as pd
import streamlit as st

FILE_NAME = "buang_kustom.json"
RULE_HELP = ("Baris yang cocok dengan SALAH SATU aturan dibuang. Contoh:\n\n"
             "`panjang < 5`\n\n`label == \"netral\"`\n\n`harga_angka <= 0`\n\n"
             "`text.str.contains(\"http\")`\n\n`` `nama kolom` > 3 `` (nama berspasi pakai `)")


def load_choices(folder):
    path = f"{folder}/{FILE_NAME}"
    if not os.path.exists(path):
        return {"kolom": [], "aturan": ""}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_choices(folder, choices):
    with open(f"{folder}/{FILE_NAME}", "w", encoding="utf-8") as file:
        json.dump(choices, file, ensure_ascii=False, indent=2)


def rule_lines(rule_text):
    rules = []
    for line in rule_text.splitlines():
        if line.strip() != "":
            rules.append(line.strip())
    return rules


def mark_rows(train, rules):
    # tiap aturan = train.query(...); aturan salah tulis ditampilkan error-nya, lalu dilewati
    drop = pd.Series(False, index=train.index)
    counts = pd.DataFrame(columns=["aturan", "baris cocok"])
    for rule in rules:
        try:
            matched = train.query(rule, engine="python").index
        except Exception as error:
            st.error(f"Aturan `{rule}` tidak bisa dijalankan: {error}")
            continue
        drop[matched] = True
        counts.loc[len(counts)] = [rule, len(matched)]
    return drop, counts


def droppable_columns(ctx, train):
    # label dan kolom tanggal dipakai tab Split data, jadi tidak ikut pilihan
    columns = []
    for column in train.columns:
        if column not in [ctx["label_column"], ctx["date_column"]]:
            columns.append(column)
    return columns


def pick_choices(ctx, train):
    folder = ctx["folder"]
    saved = load_choices(folder)
    options = droppable_columns(ctx, train)
    default = []
    for column in saved["kolom"]:
        if column in options:
            default.append(column)
    left, right = st.columns(2)
    drop_columns = left.multiselect("Kolom yang dibuang", options, default=default,
                                    key=f"buang_kolom__{folder}")
    rule_text = right.text_area("Aturan buang baris (pandas query, satu per baris)",
                                saved["aturan"], help=RULE_HELP, key=f"buang_aturan__{folder}")
    choices = {"kolom": drop_columns, "aturan": rule_text}
    if choices != saved:
        save_choices(folder, choices)
    return drop_columns, rule_lines(rule_text)


def show(ctx, train):
    st.markdown("**Buang kustom** (diterapkan sesudah catatan tab Telusuri)")
    drop_columns, rules = pick_choices(ctx, train)
    drop, counts = mark_rows(train, rules)
    if len(counts) > 0:
        st.dataframe(counts, hide_index=True)
    if drop.sum() > 0:
        with st.expander(f"Contoh baris yang dibuang aturan ({int(drop.sum())} baris)"):
            st.dataframe(train[drop].head(20))
    train = train[~drop]
    train = train.drop(columns=drop_columns)
    return train, int(drop.sum())
