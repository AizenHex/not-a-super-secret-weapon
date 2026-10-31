"""Tab Peta data: hasil Laya per lensa, ringkasan per kategori, tabel silang, peta 2D makna."""
import altair as alt
import pandas as pd
import streamlit as st

import eda_teks
import helpers
import kamus
from penjelasan import show_explanation

SILHOUETTE_TEXT = [(0.5, "terpisah jelas"), (0.25, "terpisah cukup"), (0.1, "terpisah lemah"),
                   (0.0, "hampir seluruhnya bertumpuk")]


@st.cache_data
def cached_summary(categories, stats, labels, texts):
    return helpers.category_summary(categories, stats, labels, texts)


@st.cache_data
def cached_tsne(vectors, seed):
    return eda_teks.tsne_2d(vectors, seed)


@st.cache_data
def cached_separation(vectors, labels):
    return eda_teks.separation_table(vectors, labels)


def show_counts(ctx):
    table = ctx["table"]
    read_rows = pd.Series(False, index=table.index)
    for name in ctx["lens_done"]:
        read_rows = read_rows | table[name].notna()
    st.write(f"Baris yang sudah dibaca Laya: **{int(read_rows.sum())}** dari {len(ctx['data'])} · "
             f"Laya ragu (yakin < {ctx['doubt_limit']}) di minimal satu lensa: "
             f"**{int(ctx['doubt'].sum())}**")
    grid = st.columns(2)
    for number, name in enumerate(ctx["lens_done"]):
        unsure = int((table[f"{name}__yakin"] < ctx["doubt_limit"]).sum())
        grid[number % 2].write(f"**{name}** · ragu: {unsure} baris")
        grid[number % 2].bar_chart(table[name].value_counts(), horizontal=True)
    add_doubt_findings(ctx)


def add_doubt_findings(ctx):
    table = ctx["table"]
    for name in ctx["lens_done"]:
        doubtful = table[table[f"{name}__yakin"] < ctx["doubt_limit"]]
        if len(doubtful) == 0:
            continue
        first = doubtful[f"{name}__yakin"].idxmin()
        reasons = ctx["reason_parts"].get(first, [""])
        helpers.add_finding(
            ctx, "Laya ragu", f"Lensa {name}: Laya ragu di {len(doubtful)} baris "
            f"({100 * len(doubtful) / len(table):.1f}% dari baris yang sudah dibaca). "
            f"Paling ragu: baris {first}.", " ".join(reasons[-1:]),
            "Periksa baris ini dengan mata sendiri; kategori lensa mungkin kurang jelas.",
            min(0.7, 2 * len(doubtful) / len(table)))


def show_summary(ctx):
    st.subheader("Ringkasan per kategori")
    dim = st.selectbox("Kategori", ctx["dimensions"], key="summary_dim")
    labels = None
    if ctx["label_column"] is not None and dim != ctx["label_column"]:
        labels = ctx["data"][ctx["label_column"]]
    kamus.show_table(cached_summary(ctx["table"][dim], ctx["stats"], labels, ctx["texts"]))
    st.caption("kata_khas = kata yang jauh lebih sering muncul di kategori ini daripada di "
               "semua teks. Bahan insight untuk slide.")
    show_explanation("ringkasan_kategori")


def top_values(values, limit=30):
    # kolom dengan ribuan nilai berbeda: 30 tersering tetap, sisanya jadi "(lainnya)"
    values = values.astype(str)
    common = values.value_counts().index[:limit]
    return values.where(values.isin(common), "(lainnya)")


def crosstab_as(rows, columns, show_as):
    rows = top_values(rows)
    columns = top_values(columns)
    cross = pd.crosstab(rows, columns)
    if show_as == "% per baris":
        cross = (100 * pd.crosstab(rows, columns, normalize="index")).round(1)
    if show_as == "% per kolom":
        cross = (100 * pd.crosstab(rows, columns, normalize="columns")).round(1)
    return cross


def show_cross(ctx):
    st.subheader("Tabel silang")
    st.caption("Contoh: label asli × lensa emosi → sel yang tidak sejalan = kemungkinan salah "
               "label. Pilih nilai di bawah tabel untuk melihat barisnya.")
    table = ctx["table"]
    dimensions = ctx["dimensions"]
    left, middle, right = st.columns(3)
    dim_a = left.selectbox("Baris tabel", dimensions, key="dim_a")
    dim_b = middle.selectbox("Kolom tabel", dimensions, index=len(dimensions) - 1, key="dim_b")
    show_as = right.radio("Isi sel", ["jumlah", "% per baris", "% per kolom"], horizontal=True)
    cross = crosstab_as(table[dim_a], table[dim_b], show_as)
    kamus.show_table(cross.style.background_gradient(cmap="Blues", axis=None))
    value_a = left.selectbox(f"{dim_a} =", list(cross.index), key="value_a")
    value_b = middle.selectbox(f"{dim_b} =", list(cross.columns), key="value_b")
    same_a = top_values(table[dim_a]) == value_a
    same_b = top_values(table[dim_b]) == value_b
    picked = table[same_a & same_b]
    st.write(f"{len(picked)} baris")
    view = picked[ctx["show_columns"]].head(300).copy()
    view["alasan"] = ctx["reasons"].reindex(view.index)
    kamus.show_table(view)
    show_explanation("tabel_silang")


# ---------------- peta 2D makna ----------------

def silhouette_meaning(value):
    for limit, text in SILHOUETTE_TEXT:
        if value >= limit:
            return text
    return "bertumpuk dan sering salah sisi"


def show_separation(ctx, vectors, labels, dim):
    keep = labels.notna().to_numpy()
    kept_labels = labels[keep].astype(str).to_numpy()
    if len(set(kept_labels)) < 2:
        st.info("Butuh minimal dua kelas untuk menghitung pemisahan.")
        return
    table, overall = cached_separation(vectors[keep], kept_labels)
    kamus.show_table(table.sort_values("silhouette_rata2"), hide_index=True)
    meaning = silhouette_meaning(overall)
    st.write(f"Silhouette rata-rata (kosinus, vektor asli 384 dimensi): **{overall:.3f}** → kelas "
             f"pada `{dim}` {meaning}. Kelas dengan silhouette kecil / persen lebih mirip kelas "
             "lain tinggi = sulit dibedakan hanya dari makna teksnya.")
    worst = table.sort_values("silhouette_rata2").iloc[0]
    helpers.add_finding(
        ctx, "Peta makna", f"Pemisahan kelas {dim}: silhouette rata-rata {overall:.3f} "
        f"({meaning}); paling bertumpuk: '{worst['kelas']}' ({worst['silhouette_rata2']}).",
        f"{worst['persen_lebih_mirip_kelas_lain']}% teks kelas '{worst['kelas']}' lebih mirip "
        f"kelas lain; kelas terdekat: '{worst['kelas_terdekat']}'.",
        "Kelas yang bertumpuk sulit dibedakan; pertimbangkan menggabungkan atau menambah fitur.",
        0.5)


def draw_map(ctx, positions, dim):
    rows = ctx["vector_rows"][positions]
    vectors = ctx["vectors"][positions]
    xy, explained = cached_tsne(vectors, helpers.SEED)
    labels = ctx["table"][dim].reindex(rows)
    frame = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "row": rows})
    frame["warna"] = labels.astype(str).to_numpy()
    frame["teks"] = ctx["texts"][rows].str.slice(0, 150).to_numpy()
    chart = alt.Chart(frame).mark_circle(size=30, opacity=0.7)
    chart = chart.encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), color="warna:N",
                         tooltip=["row", "warna", "teks"]).interactive()
    st.altair_chart(chart, width="stretch")
    st.caption(f"PCA 50 dimensi (menyimpan {100 * explained:.0f}% ragam) lalu t-SNE, "
               f"{len(frame)} titik. Kelas yang bertumpuk = sulit dibedakan.")
    show_separation(ctx, vectors, labels, dim)


def show_map_2d(ctx):
    st.subheader("Peta 2D makna (PCA + t-SNE)")
    if ctx["vectors"] is None:
        st.info("Tekan **Kelompokkan** di tab Temukan kategori dulu (membuat vektor MiniLM).")
        return
    total = len(ctx["vectors"])
    n_points = st.slider("Jumlah titik (sampel acak)", min(100, total), min(5000, total),
                         min(2000, total), key="tsne_points")
    dim = st.selectbox("Warnai menurut", ctx["dimensions"], key="tsne_color")
    if st.button("Buat peta 2D", key="tsne_button"):
        st.session_state["tsne_n"] = n_points
    if "tsne_n" in st.session_state:
        positions = eda_teks.sample_positions(total, st.session_state["tsne_n"], helpers.SEED)
        draw_map(ctx, positions, dim)
        show_explanation("peta_2d", f"{len(positions)} titik pada peta.")


def show(ctx):
    if len(ctx["dimensions"]) == 0:
        st.info("Belum ada hasil. Jalankan Laya di tab Lensa Laya (atau Kelompokkan di tab "
                "Temukan kategori).")
        return
    show_counts(ctx)
    show_explanation("ragu_laya")
    show_summary(ctx)
    show_cross(ctx)
    show_map_2d(ctx)
