"""Tab Teks & konsistensi: nilai kategori tidak konsisten, konflik label, n-gram khas per kelas,
kosakata, kata alay, campuran bahasa."""
import pandas as pd
import streamlit as st

import eda_teks
import helpers
from penjelasan import show_explanation


@st.cache_data
def cached_inconsistent(series):
    return eda_teks.inconsistent_values(series)


@st.cache_data
def cached_conflicts(texts, labels):
    return eda_teks.label_conflicts(texts, labels)


@st.cache_data
def cached_ngrams(texts, labels):
    return eda_teks.class_ngrams(texts, labels)


@st.cache_data
def cached_vocabulary(texts):
    return eda_teks.vocabulary_stats(texts)


@st.cache_data
def cached_slang():
    return eda_teks.load_slang()


@st.cache_data
def cached_ratios(texts, slang):
    return eda_teks.word_ratios(texts, slang)


# ---------------- 1. nilai kategori tidak konsisten ----------------

def show_correction(ctx, column, table):
    st.dataframe(table, hide_index=True)
    mapping = eda_teks.correction_map(table)
    st.write("Peta koreksi (tempel ke notebook):")
    st.code(f'data["{column}"] = data["{column}"].replace({mapping})', language="python")
    affected = int(table["jumlah"].sum())
    helpers.add_finding(
        ctx, "Konsistensi", f"Kolom {column}: {len(table)} nilai tidak konsisten "
        f"({affected} baris).", "Nilai yang sama ditulis berbeda (huruf besar, spasi, tanda "
        "baca) atau mirip >= 90% (kemungkinan salah ketik).",
        "Gabungkan dengan peta koreksi di bagian ini.", 10 * affected / len(ctx["data"]))


def show_inconsistent(ctx):
    st.subheader("1. Nilai kategori yang tidak konsisten")
    names = eda_teks.short_text_columns(ctx["data"])
    if len(names) == 0:
        st.info("Tidak ada kolom teks pendek (rata-rata <= 40 huruf) yang bisa dicek.")
        return
    column = st.selectbox("Kolom yang dicek", names, key="consistency_column")
    table = cached_inconsistent(ctx["data"][column])
    st.write(f"Nilai yang bisa dirapikan: **{len(table)}** · baris yang terkena: "
             f"**{int(table['jumlah'].sum())}**")
    if len(table) == 0:
        st.success("Semua nilai konsisten (tidak ada yang beda hanya huruf besar/spasi/tanda "
                   "baca, tidak ada yang mirip >= 90%).")
    else:
        show_correction(ctx, column, table)
    show_explanation("konsistensi", f"{len(table)} nilai tidak konsisten di kolom {column}.")


# ---------------- 2. konflik label ----------------

def add_conflict_finding(ctx, groups, rows):
    helpers.add_finding(
        ctx, "Konflik label", f"{groups} kelompok teks sama tetapi label berbeda ({rows} baris).",
        "Teks yang sama (atau sama setelah huruf besar/tanda baca diabaikan) diberi label "
        "berbeda; paling banyak satu label bisa benar.",
        "Periksa tiap kelompok; pilih satu label atau buang barisnya.",
        10 * rows / len(ctx["data"]))


def show_conflicts(ctx):
    st.subheader("2. Konflik label (teks sama, label beda)")
    if ctx["label_column"] is None:
        st.info("Pilih kolom label di panel samping.")
        return
    labels = ctx["data"][ctx["label_column"]].fillna("(kosong)").astype(str)
    table = cached_conflicts(ctx["texts"], labels)
    groups = table["kelompok"].nunique()
    first, second = st.columns(2)
    first.metric("kelompok teks yang labelnya bentrok", groups)
    second.metric("baris yang terlibat", len(table))
    if len(table) == 0:
        st.success("Tidak ada teks sama dengan label berbeda.")
    else:
        st.dataframe(table, hide_index=True)
        add_conflict_finding(ctx, groups, len(table))
    show_explanation("konflik_label", f"{groups} kelompok bentrok, {len(table)} baris terlibat.")


# ---------------- 3. n-gram khas per kelas ----------------

def show_ngrams(ctx):
    st.subheader("3. Kata dan frasa paling khas per kelas")
    if ctx["label_column"] is None:
        st.info("Pilih kolom label di panel samping.")
        return
    labels = ctx["data"][ctx["label_column"]].fillna("(kosong)").astype(str)
    table = cached_ngrams(ctx["texts"], labels)
    chosen = st.selectbox("Kelas", list(table["kelas"].unique()), key="ngram_class")
    st.dataframe(table[table["kelas"] == chosen], hide_index=True)
    st.caption("persen_di_kelas = persen teks kelas ini yang memuat frasa itu; persen_di_lain = "
               "di semua teks kelas lain. chi2 besar = lebih khas.")
    top = table.groupby("kelas").head(1)
    helpers.add_finding(
        ctx, "Kata khas", "Frasa paling khas tiap kelas: " + describe_top(top) + ".",
        "Dipilih dengan chi-square satu-lawan-semua pada kemunculan (1-2 kata).",
        "Kata khas bisa jadi fitur kuat, tetapi waspada kata yang hanya penanda sumber data.",
        0.3)
    show_explanation("ngram", f"{len(table['kelas'].unique())} kelas dianalisis.")


def describe_top(top):
    parts = []
    for _, line in top.iterrows():
        parts.append(f"{line['kelas']}: '{line['ngram']}'")
    return "; ".join(parts)


# ---------------- 4. kosakata, kata alay, campuran bahasa ----------------

def show_vocabulary(ctx):
    st.subheader("4. Kosakata, kata alay, dan campuran bahasa")
    texts = ctx["texts"]
    stats = cached_vocabulary(texts)
    first, second, third = st.columns(3)
    first.metric("kosakata (kata berbeda)", stats["kosakata"])
    second.metric("kata muncul hanya 1x", f"{stats['persen_muncul_1x']}%")
    third.metric("kata per teks", stats["kata_per_teks"])
    slang = cached_slang()
    if slang is None:
        st.info("Kamus alay tidak ditemukan di data_latihan/preprocessing; rasio alay dilewati.")
        slang = set()
    ratios = cached_ratios(texts, slang)
    show_ratios(ctx, ratios, len(slang) > 0)
    add_vocabulary_findings(ctx, stats, ratios)
    show_explanation("kosakata_alay", f"{stats['persen_muncul_1x']}% kata muncul 1x; kosakata "
                                      f"{stats['kosakata']} kata.")


def show_ratios(ctx, ratios, has_slang):
    first, second = st.columns(2)
    if has_slang:
        first.metric("rata-rata rasio kata alay", f"{100 * ratios['rasio_alay'].mean():.1f}%")
    mixed = ratios["campuran_bahasa"].mean()
    second.metric("teks campur Indonesia + Inggris", f"{100 * mixed:.1f}%")
    if ctx["label_column"] is None:
        return
    table = pd.DataFrame({"label": ctx["data"][ctx["label_column"]].astype(str),
                          "rasio_alay_%": 100 * ratios["rasio_alay"],
                          "campuran_bahasa_%": 100 * ratios["campuran_bahasa"]})
    st.write("Rata-rata per label:")
    means = table.groupby("label").mean()
    st.dataframe(means.round(1))


def add_vocabulary_findings(ctx, stats, ratios):
    if stats["persen_muncul_1x"] >= 50:
        helpers.add_finding(
            ctx, "Kosakata", f"{stats['persen_muncul_1x']}% kata hanya muncul sekali "
            f"({stats['muncul_1x']} dari {stats['kosakata']}).",
            "Banyak kata langka (salah ketik, nama, slang) sehingga model berbasis kata sulit "
            "belajar dari kata itu.", "Pakai min_df, normalisasi slang, atau model makna "
            "(MiniLM).", 0.4)
    mean_alay = ratios["rasio_alay"].mean()
    if mean_alay >= 0.1:
        helpers.add_finding(
            ctx, "Kata alay", f"Rata-rata {100 * mean_alay:.1f}% kata per teks ada di kamus "
            "alay/slang.", "Kata dicocokkan dengan kamus alay Salsabila + kamus slang sendiri.",
            "Normalisasi slang (preprocessing teks Indonesia) sebelum TF-IDF.", mean_alay / 0.3)


def show(ctx):
    show_inconsistent(ctx)
    show_conflicts(ctx)
    show_ngrams(ctx)
    show_vocabulary(ctx)
