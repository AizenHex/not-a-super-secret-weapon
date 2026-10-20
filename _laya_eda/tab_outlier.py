"""Tab Outlier: angka dan teks yang janggal DI DALAM kategorinya, lengkap dengan alasannya."""
import altair as alt
import pandas as pd
import streamlit as st

import helpers
import outliers
from penjelasan import show_explanation


@st.cache_data
def cached_pairs(vectors, rows, threshold):
    return helpers.similar_pairs(vectors, rows, threshold)


def add_number_findings(ctx, dim):
    # satu temuan per kolom angka: berapa baris janggal hanya di kategorinya + contoh alasannya
    total = len(ctx["data"])
    for column in ctx["numbers"].columns:
        checked = outliers.compare_outliers(ctx["numbers"][column], ctx["table"][dim],
                                            ctx["use_log"])
        local = checked[checked["status"] == "janggal hanya di kategorinya"]
        if len(local) == 0:
            continue
        first = local.sort_values("kali_median").index[0]
        example = outliers.number_reason(column, local.loc[first], dim)
        helpers.add_finding(
            ctx, "Outlier angka", f"Kolom {column}: {len(local)} baris janggal hanya di dalam "
            f"kategori {dim} (wajar kalau dilihat seluruh data). Contoh baris {first}.",
            example, "Periksa baris-baris ini; kemungkinan salah ketik, salah satuan, atau salah "
            "kategori.", 10 * len(local) / total)


def show_number_table(ctx, checked, dim):
    per_group = checked.groupby("kelompok")
    summary = per_group["nilai"].describe()
    summary = summary[["count", "25%", "50%", "75%"]]
    summary["outlier_di_kategori"] = per_group["outlier"].sum()
    summary["outlier_di_keseluruhan"] = per_group["outlier_keseluruhan"].sum()
    st.dataframe(summary)
    return per_group


def show_boxplot(ctx, checked, column, dim):
    box_data = checked[["nilai", "kelompok"]].copy()
    box_data["kelompok"] = box_data["kelompok"].astype(str)
    scale = alt.Scale(type="linear")
    if ctx["use_log"]:
        box_data = box_data[box_data["nilai"] > 0]
        scale = alt.Scale(type="log")
    x_axis = alt.X("nilai:Q", scale=scale, title=column)
    box = alt.Chart(box_data).mark_boxplot()
    box = box.encode(x=x_axis, y=alt.Y("kelompok:N", title=dim))
    st.altair_chart(box, width="stretch")


def show_flagged(ctx, checked, column, dim):
    st.write("Yang paling menarik biasanya **janggal hanya di kategorinya**: wajar kalau "
             "dilihat global, padahal aneh untuk jenisnya.")
    st.dataframe(checked["status"].value_counts())
    status = st.selectbox("Tampilkan status", ["janggal hanya di kategorinya",
                                               "janggal di keduanya",
                                               "janggal hanya di keseluruhan", "normal"])
    flagged = checked[checked["status"] == status].sort_values("kali_median")
    st.write(f"{len(flagged)} baris · kali_median kecil = terlalu murah/kecil, "
             "besar = terlalu mahal/besar")
    view = flagged.join(ctx["data"][ctx["columns"]]).head(300)
    view["alasan_kolom_ini"] = ""
    for row, line in view.iterrows():
        if line["status"] != "normal":
            view.at[row, "alasan_kolom_ini"] = outliers.number_reason(column, line, dim)
    view["alasan"] = ctx["reasons"].reindex(view.index)
    st.dataframe(view)


def show_histogram(ctx, per_group, column, dim):
    numbers = ctx["numbers"]
    chosen = st.selectbox(f"Histogram: semua data vs {dim} =", list(per_group.groups))
    same = numbers[column][ctx["table"][dim] == chosen]
    histogram = helpers.compare_histogram(numbers[column], same, ctx["use_log"])
    x_name = column
    if ctx["use_log"]:
        x_name = f"log10 {column}"
    st.line_chart(histogram, x_label=x_name, y_label="% baris")


def show_numbers(ctx, dim):
    st.subheader("Angka: dibanding kategorinya vs dibanding keseluruhan")
    if len(ctx["numbers"].columns) == 0:
        st.info("Pilih kolom angka di panel samping dulu.")
        return
    column = st.selectbox("Kolom angka", list(ctx["numbers"].columns))
    checked = outliers.compare_outliers(ctx["numbers"][column], ctx["table"][dim],
                                        ctx["use_log"])
    per_group = show_number_table(ctx, checked, dim)
    show_boxplot(ctx, checked, column, dim)
    show_histogram(ctx, per_group, column, dim)
    show_flagged(ctx, checked, column, dim)
    add_number_findings(ctx, dim)
    local = int((checked["status"] == "janggal hanya di kategorinya").sum())
    show_explanation("outlier_angka", f"{local} baris janggal hanya di kategorinya pada kolom "
                                      f"{column}.")


def add_text_findings(ctx, dim, distances):
    far = distances[distances["jauh"]]
    stray = distances[distances["nyasar"]]
    if len(stray) > 0:
        first = stray["jarak_ke_kategorinya"].idxmax()
        helpers.add_finding(
            ctx, "Outlier teks", f"{len(stray)} teks lebih dekat maknanya ke kategori {dim} lain "
            f"({100 * len(stray) / len(distances):.1f}% dari {len(distances)} teks yang punya "
            f"vektor). Contoh baris {first}.", outliers.text_reason(dim, stray.loc[first]),
            f"Kalau {dim} adalah label asli: kemungkinan salah label. Kalau lensa Laya: "
            "kemungkinan salah kategori.", min(0.7, 1.5 * len(stray) / len(distances)))
    if len(far) > 0:
        first = far["jarak_ke_kategorinya"].idxmax()
        helpers.add_finding(
            ctx, "Outlier teks", f"{len(far)} teks jauh dari pusat makna kategorinya "
            f"(contoh baris {first}).", outliers.text_reason(dim, far.loc[first]),
            "Teks seperti ini bisa tidak nyambung dengan kategorinya atau berisi hal lain.",
            min(0.7, 3 * len(far) / len(distances)))


def show_text_table(ctx, distances, text_view):
    odd = distances[distances[text_view]]
    odd = odd.sort_values("jarak_ke_kategorinya", ascending=False)
    view = odd.join(ctx["data"][ctx["columns"]]).head(300)
    view["alasan"] = ctx["reasons"].reindex(view.index)
    st.dataframe(view)


def show_text(ctx, dim):
    st.subheader("Teks: jauh dari kategorinya atau nyasar")
    distances = ctx["distances"]
    if distances is None:
        st.info("Tekan **Kelompokkan** di tab Temukan kategori dulu (sekalian membuat vektor "
                "MiniLM).")
        return
    first, second = st.columns(2)
    first.metric("jauh dari kategorinya", int(distances["jauh"].sum()))
    second.metric("lebih dekat ke kategori lain (nyasar)", int(distances["nyasar"].sum()))
    st.caption("Jarak dihitung dari makna (MiniLM), bukan kata. Nyasar pada lensa Laya = "
               "kemungkinan salah kategori; nyasar pada label asli = kemungkinan salah label.")
    text_view = st.radio("Tampilkan", ["jauh", "nyasar"], horizontal=True)
    show_text_table(ctx, distances, text_view)
    add_text_findings(ctx, dim, distances)
    show_explanation("outlier_teks", f"{int(distances['nyasar'].sum())} teks nyasar, "
                                     f"{int(distances['jauh'].sum())} teks jauh.")


def show_similar(ctx):
    st.subheader("Teks yang maknanya hampir sama (kata-katanya beda)")
    threshold = st.slider("Batas kemiripan", 0.80, 0.99, 0.93, 0.01)
    texts = ctx["texts"]
    unique_rows = []
    seen = set()
    for position, row in enumerate(ctx["vector_rows"]):
        if texts[row] not in seen:
            seen.add(texts[row])
            unique_rows.append(position)
    rows = ctx["vector_rows"][unique_rows]
    pairs = cached_pairs(ctx["vectors"][unique_rows], rows, threshold)
    pairs["teks_a"] = texts[pairs["row_a"]].to_numpy()
    pairs["teks_b"] = texts[pairs["row_b"]].to_numpy()
    st.write(f"{len(pairs)} pasangan (duplikat persis sudah dibuang; maks 3000)")
    st.dataframe(pairs.head(300), hide_index=True)
    show_explanation("makna_sama", f"{len(pairs)} pasangan pada batas kemiripan {threshold}.")


def position_line(ctx, column, row, dim):
    # posisi satu nilai: persentil, median, outlier (seluruh data dan dalam kategorinya)
    values = ctx["numbers"][column]
    category = ctx["table"].at[row, dim]
    line = {"kolom": column, "nilai": values[row]}
    if pd.isna(values[row]):
        return line
    overall = outliers.describe_value(values, row, ctx["use_log"])
    line["persentil_keseluruhan"] = overall["persentil"]
    line["median_keseluruhan"] = overall["median"]
    line["outlier_keseluruhan"] = overall["outlier"]
    if not pd.isna(category):
        same = values[ctx["table"][dim] == category]
        inside = outliers.describe_value(same, row, ctx["use_log"])
        line[f"persentil_di_{dim}={category}"] = inside["persentil"]
        line["median_kategori"] = inside["median"]
        line["outlier_kategori"] = inside["outlier"]
    return line


def position_table(ctx, row, dim):
    lines = []
    for column in ctx["numbers"].columns:
        lines.append(position_line(ctx, column, row, dim))
    return lines


def show_card(ctx, dim):
    st.subheader("Periksa satu baris: outlier atau bukan?")
    data = ctx["data"]
    row = st.number_input("Nomor baris (kolom `row` di tab Telusuri)", 0, len(data) - 1, 0)
    st.dataframe(ctx["table"].loc[[row], ctx["show_columns"]])
    parts = ctx["reason_parts"].get(row, [])
    st.write(f"**Alasan baris {row}: {len(parts)} ditandai** (dibanding kategori `{dim}`)")
    for text in parts:
        st.markdown("- " + text)
    if len(parts) == 0:
        st.success("Tidak ada alasan yang ditandai untuk baris ini.")
    lines = position_table(ctx, row, dim)
    if len(lines) > 0:
        st.dataframe(pd.DataFrame(lines), hide_index=True)
    if pd.isna(ctx["table"].at[row, dim]):
        st.caption(f"Baris ini belum punya nilai `{dim}`; hanya dibanding keseluruhan.")
    show_explanation("kartu_baris", f"baris {row} punya {len(parts)} alasan.")


def show(ctx):
    dim = ctx["reason_dim"]
    if dim is None:
        st.info("Belum ada kategori. Jalankan Laya di tab Lensa Laya (atau Kelompokkan di tab "
                "Temukan kategori).")
        return
    st.write(f"Pembanding kategori: **{dim}** (ganti di panel samping: 'Pembanding kategori').")
    show_numbers(ctx, dim)
    if ctx["vectors"] is not None:
        show_text(ctx, dim)
        show_similar(ctx)
    else:
        st.subheader("Teks: jauh dari kategorinya atau nyasar")
        st.info("Tekan **Kelompokkan** di tab Temukan kategori dulu (sekalian membuat vektor "
                "MiniLM).")
    show_card(ctx, dim)
