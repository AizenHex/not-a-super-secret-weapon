"""Isian di samping (sidebar): data, kolom yang dibaca Laya, label, kolom angka, kolom tanggal.
Semua pilihan disimpan ke dict ctx supaya tab-tab tinggal memakainya."""
import os

import pandas as pd
import streamlit as st

import eda
import helpers

DEFAULT_CSV = "../../data_latihan/text/classification/emot_train.csv"
NO_CHOICE = "(tidak ada)"


@st.cache_data(max_entries=4, show_spinner=False)
def read_csv(csv_path, modified):
    # modified (waktu ubah file) ikut kunci cache: file berubah = dibaca ulang
    return pd.read_csv(csv_path).reset_index(drop=True)


def save_upload(uploaded):
    # tulis hanya kalau berbeda, supaya waktu ubah file (kunci cache) tidak berganti tiap klik
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/{uploaded.name}"
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) != uploaded.size:
        with open(csv_path, "wb") as file:
            file.write(uploaded.getbuffer())
    return csv_path


def pick_data(ctx):
    csv_path = st.sidebar.text_input("File CSV (path relatif ke folder _laya_eda)", DEFAULT_CSV)
    uploaded = st.sidebar.file_uploader("... atau unggah CSV", type="csv")
    if uploaded is not None:
        csv_path = save_upload(uploaded)
    ctx["data_key"] = (csv_path, os.path.getmtime(csv_path))
    ctx["data"] = read_csv(csv_path, ctx["data_key"][1])
    ctx["csv_path"] = csv_path
    ctx["folder"] = helpers.work_folder(csv_path)


def longest_text_column(data):
    # kolom teks = kolom bertipe object; default = yang isinya paling panjang
    text_like = []
    longest = 0
    default_text = None
    for column in data.columns:
        if data[column].dtype == object:
            text_like.append(column)
            lengths = data[column].astype(str).str.len()
            if lengths.mean() > longest:
                longest = lengths.mean()
                default_text = column
    return text_like, default_text


def pick_columns(ctx):
    data = ctx["data"]
    text_like, default_text = longest_text_column(data)
    if len(text_like) == 0:
        st.warning("CSV ini tidak punya kolom teks.")
        st.stop()
    mode = st.sidebar.radio("Mode", ["teks", "baris"],
                            help="teks = Laya membaca satu kolom. baris = Laya membaca beberapa "
                                 "kolom sekaligus (judul, alamat, harga, ...)")
    if mode == "teks":
        text_column = st.sidebar.selectbox("Kolom teks", text_like,
                                           index=text_like.index(default_text))
        ctx["columns"] = [text_column]
    else:
        ctx["columns"] = st.sidebar.multiselect("Kolom yang dibaca Laya", list(data.columns),
                                                default=text_like)
    if len(ctx["columns"]) == 0:
        st.warning("Pilih minimal satu kolom di samping.")
        st.stop()
    ctx["mode"] = mode


def warn_unique_label(label):
    # kolom teks yang hampir unik per baris (mis. kolom teks itu sendiri) bukan label kategori
    if pd.api.types.is_numeric_dtype(label) or label.nunique() <= eda.MAX_LABEL_CLASSES:
        return
    st.sidebar.warning(f"Kolom ini punya {label.nunique()} nilai berbeda, jadi bukan label "
                       "kategori. Pilih kolom label yang sebenarnya, atau (tidak ada).")


def pick_label_and_date(ctx):
    all_columns = list(ctx["data"].columns)
    label_choice = st.sidebar.selectbox("Kolom label (opsional)", [NO_CHOICE] + all_columns)
    ctx["label_column"] = None
    if label_choice != NO_CHOICE:
        ctx["label_column"] = label_choice
        warn_unique_label(ctx["data"][label_choice])
    date_choice = st.sidebar.selectbox("Kolom tanggal (opsional)", [NO_CHOICE] + all_columns)
    ctx["date_column"] = None
    if date_choice != NO_CHOICE:
        ctx["date_column"] = date_choice
    ctx["doubt_limit"] = st.sidebar.slider("Batas ragu: yakin Laya di bawah ini = ragu",
                                           0.0, 1.0, 0.6, 0.05)


def default_number_columns(data):
    # kolom bertipe angka, kecuali yang nilainya unik semua (biasanya nomor urut / ID)
    names = []
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]) and data[column].nunique() < len(data):
            names.append(column)
    return names


def pick_numbers(ctx):
    data = ctx["data"]
    with st.sidebar.expander("Kolom angka (dicek di tab Cek angka)"):
        ctx["number_columns"] = st.multiselect("Kolom angka yang dicek", list(data.columns),
                                               default=default_number_columns(data))
        ctx["indonesian"] = st.checkbox("Format Indonesia (titik = ribuan, koma = desimal, "
                                        "'Rp' dibuang)", value=True)
        ctx["use_log"] = st.checkbox("Outlier pada skala log (disarankan untuk harga, luas, "
                                     "jumlah)", value=True)
        ctx["ratio"] = None
        if len(ctx["number_columns"]) >= 2 and st.checkbox("Buat kolom rasio (misal harga per m2)"):
            top = st.selectbox("Rasio: pembilang", ctx["number_columns"])
            bottom = st.selectbox("Rasio: penyebut", ctx["number_columns"], index=1)
            ctx["ratio"] = (top, bottom)
