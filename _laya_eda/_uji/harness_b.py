"""Harness kecil untuk menguji tab_fitur.py dan tab_test.py tanpa app.py.
Jalankan dari folder _laya_eda:  streamlit run _uji/harness_b.py
Semua hasil (vektor MiniLM, jawaban Laya CPU) disimpan di _uji/hasil_b/<skenario>/ dan
dipakai lagi pada jalan berikutnya."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.getcwd())
import helpers
import tab_fitur
import tab_test
from lenses import PRESETS

OUT_FOLDER = "_uji/hasil_b"
MAX_VECTOR_ROWS = 3000
MAX_LENS_ROWS = 300
EMOT = "../../data_latihan/text/classification/emot_train.csv"
SCENARIOS = {
    "emot (teks + label)": {
        "name": "emot_train", "csv": EMOT, "columns": ["text"], "label": "label",
        "numbers": [], "lenses": ["emosi", "kejelasan", "label_cocok"]},
    "properti (baris + label)": {
        "name": "properti", "csv": "contoh/properti.csv",
        "columns": ["judul", "deskripsi", "harga", "alamat", "luas_m2"], "label": "jenis_asli",
        "numbers": ["harga", "luas_m2"], "lenses": ["jenis_properti", "kelengkapan_alamat"]},
    "properti regresi (label luas_m2)": {
        "name": "properti_regresi", "csv": "contoh/properti.csv",
        "columns": ["judul", "deskripsi", "alamat", "harga"], "label": "luas_m2",
        "numbers": ["harga"], "lenses": ["jenis_properti"]},
}


def no_progress(value, text=""):
    # pengganti st.progress agar run_lens tidak menggambar apa pun
    return None


FAKE_BAR = SimpleNamespace(progress=no_progress)


@st.cache_resource
def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("../../pretrained/minilm", device="cpu")


@st.cache_resource
def load_laya_cpu():
    import laya
    return laya.load("../../pretrained/laya", subfolder="multilingual", device="cpu")


def make_vectors(data, texts, folder):
    # vektor MiniLM untuk <= 3000 baris acak, disimpan agar tidak diulang
    if not os.path.exists(f"{folder}/vektor.npy"):
        rows = data.sample(n=min(MAX_VECTOR_ROWS, len(data)), random_state=42).index
        rows = np.sort(rows.to_numpy())
        vectors = helpers.encode(load_minilm(), texts[rows])
        groups = helpers.find_groups(vectors, 8)
        np.save(f"{folder}/vektor.npy", vectors)
        np.save(f"{folder}/vektor_rows.npy", rows)
        table = pd.DataFrame({"row": rows, "kelompok": groups})
        table.to_csv(f"{folder}/kelompok.csv", index=False)
    return np.load(f"{folder}/vektor.npy"), np.load(f"{folder}/vektor_rows.npy")


def run_lenses(data, setting, lenses, folder):
    # jalankan tiap lensa sekali pada <= 300 baris dengan Laya di CPU
    order = data.sample(frac=1, random_state=helpers.SEED).index[:MAX_LENS_ROWS]
    for name in setting["lenses"]:
        if os.path.exists(helpers.lens_file(folder, name)):
            continue
        label_column = None
        if lenses[name].get("with_label"):
            label_column = setting["label"]
        helpers.run_lens(load_laya_cpu(), data, list(order), setting["columns"], label_column,
                         name, lenses[name], folder, FAKE_BAR)


def lens_table(data, folder, lenses):
    # data + kolom lensa + kelompok MiniLM, seperti tabel gabungan di app.py
    table = data.copy()
    results = helpers.lens_results(folder, lenses, data.index)
    for column in results.columns:
        table[column] = results[column]
    groups = pd.read_csv(f"{folder}/kelompok.csv")
    table["kelompok"] = pd.Series(groups["kelompok"].to_numpy(), index=groups["row"])
    return table


def read_numbers(data, setting):
    numbers = pd.DataFrame(index=data.index)
    for column in setting["numbers"]:
        numbers[column] = helpers.to_number(data[column], True)
    return numbers


def build_ctx(setting):
    folder = f"{OUT_FOLDER}/{setting['name']}"
    os.makedirs(folder, exist_ok=True)
    data = pd.read_csv(setting["csv"]).reset_index(drop=True)
    texts = helpers.row_texts(data, setting["columns"])
    lenses = {}
    for name in setting["lenses"]:
        lenses[name] = PRESETS[name]
    helpers.save_lenses(folder, lenses)
    run_lenses(data, setting, lenses, folder)
    vectors, vector_rows = make_vectors(data, texts, folder)
    return {"data": data, "table": lens_table(data, folder, lenses), "texts": texts,
            "columns": setting["columns"], "label_column": setting["label"],
            "numbers": read_numbers(data, setting), "stats": helpers.text_stats(texts),
            "vectors": vectors, "vector_rows": vector_rows, "folder": folder, "use_log": True,
            "lenses": lenses, "lens_done": list(lenses),
            "dimensions": list(lenses) + ["kelompok", setting["label"]], "doubt_limit": 0.6,
            "csv_path": setting["csv"], "load_minilm": load_minilm, "findings": []}


def add_doubt(ctx):
    doubt = pd.Series(False, index=ctx["table"].index)
    for name in ctx["lens_done"]:
        doubt = doubt | (ctx["table"][f"{name}__yakin"] < ctx["doubt_limit"])
    ctx["table"]["laya_ragu"] = doubt


st.set_page_config(page_title="Harness B", layout="wide")
choice = st.sidebar.radio("Skenario", list(SCENARIOS), key="harness_skenario")
ctx = build_ctx(SCENARIOS[choice])
add_doubt(ctx)
st.sidebar.caption(f"{len(ctx['data'])} baris · folder {ctx['folder']}")
tabs = st.tabs(["Fitur → label", "Train vs test"])
with tabs[0]:
    tab_fitur.show(ctx)
with tabs[1]:
    tab_test.show(ctx)
with st.expander("Temuan yang terkumpul (ctx['findings'])"):
    st.dataframe(pd.DataFrame(ctx["findings"]))
