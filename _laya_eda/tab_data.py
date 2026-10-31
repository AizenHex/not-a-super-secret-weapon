"""Tab Data: bentuk tabel, ringkasan per kolom, statistik teks."""
import numpy as np
import pandas as pd
import streamlit as st

import helpers
import kamus
from penjelasan import show_explanation


def show_overview(ctx):
    data = ctx["data"]
    st.subheader(f"{len(data)} baris × {len(data.columns)} kolom")
    kamus.show_table(data.head(50))
    left, right = st.columns(2)
    per_column = pd.DataFrame()
    per_column["tipe"] = data.dtypes.astype(str)
    per_column["kosong"] = data.isna().sum()
    per_column["persen_kosong"] = (100 * data.isna().mean()).round(1)
    per_column["nilai_unik"] = data.nunique()
    left.write("**Per kolom**")
    left.dataframe(per_column)
    if ctx["label_column"] is not None:
        right.write(f"**Sebaran label `{ctx['label_column']}`**")
        right.bar_chart(data[ctx["label_column"]].value_counts(), horizontal=True)


def show_text_stats(ctx):
    stats = ctx["stats"]
    st.write("**Statistik teks** (kolom yang dibaca Laya)")
    first, second, third, fourth = st.columns(4)
    first.metric("duplikat persis", int(stats["duplikat"].sum()))
    second.metric("duplikat mirip (beda huruf/tanda baca)", int(stats["duplikat_mirip"].sum()))
    third.metric("≤ 3 kata", int((stats["jumlah_kata"] <= 3).sum()))
    fourth.metric("panjang median", int(stats["panjang"].median()))
    counts, edges = np.histogram(stats["panjang"], bins=40)
    st.bar_chart(pd.Series(counts, index=edges[:-1].round()), x_label="panjang (huruf)")


def show_extremes(ctx):
    data = ctx["data"]
    stats = ctx["stats"]
    columns = ctx["columns"]
    with st.expander("Duplikat terbanyak"):
        kamus.show_table(ctx["texts"][stats["duplikat"]].value_counts().head(30))
    with st.expander("Teks paling pendek"):
        kamus.show_table(data.loc[stats["panjang"].nsmallest(30).index, columns])
    with st.expander("Teks paling panjang"):
        kamus.show_table(data.loc[stats["panjang"].nlargest(30).index, columns])
    with st.expander("Paling banyak simbol / angka (bukan huruf)"):
        kamus.show_table(data.loc[stats["rasio_bukan_huruf"].nlargest(30).index, columns])


def add_findings(ctx):
    stats = ctx["stats"]
    total = len(stats)
    exact = int(stats["duplikat"].sum())
    near = int(stats["duplikat_mirip"].sum())
    short = int((stats["jumlah_kata"] <= 3).sum())
    if exact + near > 0:
        helpers.add_finding(
            ctx, "Data", f"{exact} baris duplikat persis dan {near} baris duplikat mirip "
            f"({100 * (exact + near) / total:.1f}% dari {total} baris).",
            "Teks yang sama muncul lebih dari sekali (setelah huruf besar/tanda baca diabaikan).",
            "Cek apakah duplikat berlabel sama; buang salah satunya sebelum membagi data latih "
            "dan uji.", 10 * (exact + near) / total)
    if short > 0:
        helpers.add_finding(
            ctx, "Data", f"{short} baris hanya 3 kata atau kurang ({100 * short / total:.1f}%).",
            "Teks sependek ini sering kekurangan informasi untuk ditebak labelnya.",
            "Lihat contohnya di bagian 'Teks paling pendek'; putuskan dibuang atau dipertahankan.",
            5 * short / total)


def show(ctx):
    show_overview(ctx)
    show_text_stats(ctx)
    show_extremes(ctx)
    add_findings(ctx)
    show_explanation("data_stats", f"{int(ctx['stats']['duplikat'].sum())} duplikat persis, "
                                   f"panjang median {int(ctx['stats']['panjang'].median())} huruf.")
