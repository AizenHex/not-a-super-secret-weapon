"""Buang kustom di tab Kode EDA: aturan buang baris (pandas query, satu per baris, train saja) dan
kolom yang dibuang (train dan test). Pilihan disimpan di hasil/<nama_csv>/buang_kustom.json
supaya tidak hilang saat aplikasi dibuka lagi."""
import json
import os

import pandas as pd
import streamlit as st

FILE_NAME = "buang_kustom.json"
RULE_HELP = ("Baris yang cocok dengan SALAH SATU aturan dibuang (train saja). Kolom angka sudah "
             "berupa angka, nama kolom tetap. Contoh:\n\n`label == \"netral\"`\n\n`harga <= 0`\n\n"
             "`text.str.len() < 5`\n\n`text.str.contains(\"http\")`\n\n"
             "`` `nama kolom` > 3 `` (nama berspasi pakai `)")


def load_choices(folder):
    path = f"{folder}/{FILE_NAME}"
    if not os.path.exists(path):
        return {"kolom": [], "aturan": ""}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_choice(folder, name, value):
    # simpan satu pilihan ("kolom" atau "aturan") tanpa mengubah yang lain
    choices = load_choices(folder)
    if choices[name] == value:
        return
    choices[name] = value
    with open(f"{folder}/{FILE_NAME}", "w", encoding="utf-8") as file:
        json.dump(choices, file, ensure_ascii=False, indent=2)


def rule_lines(rule_text):
    rules = []
    for line in rule_text.splitlines():
        if line.strip() != "":
            rules.append(line.strip())
    return rules


def check_rules(table, rules):
    # tiap aturan dicoba dengan table.query(...); aturan salah tulis ditampilkan error-nya lalu
    # tidak ikut ke kode (kode yang ditempel harus bisa jalan)
    good = []
    counts = pd.DataFrame(columns=["aturan", "baris cocok"])
    for rule in rules:
        try:
            matched = table.query(rule, engine="python").index
        except Exception as error:
            st.error(f"Aturan `{rule}` tidak bisa dijalankan, jadi tidak dipakai: {error}")
            continue
        good.append(rule)
        counts.loc[len(counts)] = [rule, len(matched)]
    return good, counts


def pick_rules(ctx, table):
    # table = data dengan kolom angka sudah jadi angka (sama dengan saat kode menjalankan aturan)
    folder = ctx["folder"]
    rule_text = st.text_area("Aturan buang baris (pandas query, satu per baris)",
                             load_choices(folder)["aturan"], help=RULE_HELP,
                             key=f"buang_aturan__{folder}")
    save_choice(folder, "aturan", rule_text)
    good, counts = check_rules(table, rule_lines(rule_text))
    if len(counts) > 0:
        st.dataframe(counts, hide_index=True)
    return good


def pick_columns(ctx):
    # label dan kolom tanggal dipakai tab Split data, jadi tidak ikut pilihan
    folder = ctx["folder"]
    options = []
    for column in ctx["data"].columns:
        if column not in [ctx["label_column"], ctx["date_column"]]:
            options.append(column)
    default = []
    for column in load_choices(folder)["kolom"]:
        if column in options:
            default.append(column)
    columns = st.multiselect("Kolom yang dibuang (train dan test)", options, default=default,
                             key=f"buang_kolom__{folder}")
    save_choice(folder, "kolom", columns)
    return columns
