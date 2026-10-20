"""Laya EDA — melihat data teks / tabel dengan bantuan Laya. Jalankan lewat jalankan.bat.
app.py hanya mengatur urutan; isi tiap tab ada di tab_*.py, semua memakai dict `ctx`
(dibangun di context.py)."""
import streamlit as st
import torch

import context
import sidebar
import tab_angka
import tab_data
import tab_ekspor
import tab_fitur
import tab_kategori
import tab_kode
import tab_lensa
import tab_outlier
import tab_peta
import tab_profil
import tab_teks
import tab_telusuri
import tab_temuan
import tab_test

LAYA_FOLDER = "../../pretrained/laya"
MINILM_FOLDER = "../../pretrained/minilm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# daftar tab, urut dari kiri. Tab baru: tambah namanya di sini, lalu tambah satu blok
# `with tabs["nama"]: modul.show(ctx)` di bawah (sebelum Temuan utama dan Kode EDA).
TAB_NAMES = ["Data", "Profil & label", "Cek angka", "Teks & konsistensi", "Temukan kategori",
             "Lensa Laya", "Peta data", "Outlier", "Telusuri + catatan", "Ekspor",
             "Fitur → label", "Train vs test", "Temuan utama", "Kode EDA"]

st.set_page_config(page_title="Laya EDA", layout="wide")


@st.cache_resource    # model dimuat sekali saja, tidak tiap klik
def load_laya():
    import laya
    return laya.load(LAYA_FOLDER, subfolder="multilingual", device=DEVICE)


@st.cache_resource
def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MINILM_FOLDER, device=DEVICE)


# ================= sidebar: data dan kolom =================
ctx = context.new_context(load_minilm, load_laya)
st.sidebar.title("Laya EDA")
sidebar.pick_data(ctx)
sidebar.pick_columns(ctx)
sidebar.pick_label_and_date(ctx)
sidebar.pick_numbers(ctx)
st.sidebar.caption(f"Hasil disimpan di `_laya_eda/{ctx['folder']}` · perangkat: {DEVICE}")
context.prepare(ctx)

tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))

# ---- dua tab ini menulis file (vektor, kelompok, hasil Laya), jadi dijalankan sebelum finish ----
with tabs["Temukan kategori"]:
    tab_kategori.show(ctx)
with tabs["Lensa Laya"]:
    tab_lensa.show(ctx)

# ---- gabungkan hasil Laya + vektor + alasan outlier ke ctx (dipakai semua tab di bawah) ----
context.finish(ctx)

with tabs["Data"]:
    tab_data.show(ctx)
with tabs["Profil & label"]:
    tab_profil.show(ctx)
with tabs["Cek angka"]:
    tab_angka.show(ctx)
with tabs["Teks & konsistensi"]:
    tab_teks.show(ctx)
with tabs["Peta data"]:
    tab_peta.show(ctx)
with tabs["Outlier"]:
    tab_outlier.show(ctx)
with tabs["Telusuri + catatan"]:
    tab_telusuri.show(ctx)
with tabs["Ekspor"]:
    tab_ekspor.show(ctx)

with tabs["Fitur → label"]:
    tab_fitur.show(ctx)
with tabs["Train vs test"]:
    tab_test.show(ctx)

# ---- dua tab ini paling akhir: memakai temuan dan pilihan dari semua tab di atas ----
with tabs["Temuan utama"]:
    tab_temuan.show(ctx)
with tabs["Kode EDA"]:
    tab_kode.show(ctx)
