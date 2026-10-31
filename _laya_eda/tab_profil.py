"""Tab Profil & label: profil tiap kolom, hubungan dengan label, korelasi, data kosong, waktu."""
import altair as alt
import pandas as pd
import streamlit as st

import eda
import helpers
import kamus
from outliers import decimal_text
from penjelasan import show_explanation

THRESHOLD_NOTE = ("Ambang ukuran efek (Cohen): Cramer V dan rho: 0,1 kecil / 0,3 sedang / "
                  "0,5 besar. "
                  "epsilon2 (Kruskal-Wallis): 0,01 kecil / 0,06 sedang / 0,14 besar. Signifikan = "
                  "p < 0,05. Pada data besar, hubungan yang sangat lemah pun bisa signifikan: "
                  "lihat ukuran efeknya.")


# ---------------- 1. profil per kolom ----------------

def column_values(ctx, column):
    # kolom angka terpilih memakai hasil parsing (numbers); kolom lain apa adanya
    if column in ctx["numbers"].columns:
        return ctx["numbers"][column]
    return ctx["data"][column]


def show_distribution(ctx, profile):
    options = list(profile["kolom"])
    for column in ctx["numbers"].columns:
        if column not in options:
            options.append(column)
    column = st.selectbox("Lihat sebaran kolom", options, key="profile_column")
    values = column_values(ctx, column)
    if pd.api.types.is_numeric_dtype(values):
        scale = alt.Scale(type="linear")
        if st.checkbox("Sumbu log (untuk sebaran miring)", key="profile_log"):
            values = values[values > 0]
            scale = alt.Scale(type="log")
        chart = alt.Chart(pd.DataFrame({"nilai": values}))
        chart = chart.mark_bar().encode(alt.X("nilai:Q", bin=alt.Bin(maxbins=40), scale=scale),
                                        alt.Y("count()", title="jumlah baris"))
        st.altair_chart(chart, width="stretch")
    else:
        st.bar_chart(values.value_counts().head(25), horizontal=True)


def add_profile_findings(ctx, profile):
    for _, line in profile.iterrows():
        if line["saran"] == "":
            continue
        helpers.add_finding(
            ctx, "Profil kolom", f"Kolom {line['kolom']}: {line['saran']}.",
            f"Tipe {line['tipe']}, {line['unik']} nilai unik, skew {line['skew']}, "
            f"kosong {line['kosong_persen']}%.",
            "Rapikan kolom ini sebelum membuat model (ubah ke angka, skala log, atau gabung "
            "kategori langka).", 0.4)


@st.cache_data(max_entries=4, show_spinner=False)
def cached_profile(data_key, indonesian, _data):
    # data_key = (path, waktu ubah file) mewakili _data, jadi data tidak di-hash tiap klik
    return eda.profile_columns(_data, indonesian)


def show_profile(ctx):
    st.subheader("1. Profil per kolom")
    profile = cached_profile(ctx["data_key"], ctx["indonesian"], ctx["data"])
    kamus.show_table(profile, hide_index=True)
    show_distribution(ctx, profile)
    add_profile_findings(ctx, profile)
    problems = profile[profile["saran"] != ""]
    show_explanation("profil_kolom", f"{len(problems)} dari {len(profile)} kolom punya catatan.")


# ---------------- 2. hubungan dengan label ----------------

def chart_number_by_class(column, feature, label):
    frame = pd.DataFrame({"nilai": feature, "label": label.astype(str)}).dropna()
    box = alt.Chart(frame).mark_boxplot()
    box = box.encode(x=alt.X("nilai:Q", title=column), y=alt.Y("label:N"))
    st.altair_chart(box, width="stretch")


def chart_category_by_class(column, feature, label):
    counts = pd.crosstab(feature, label.astype(str))
    counts = counts.loc[counts.sum(axis=1).nlargest(30).index]
    lines = []
    for value in counts.index:
        for class_name in counts.columns:
            lines.append({"nilai": str(value), "label": class_name,
                          "jumlah": counts.at[value, class_name]})
    chart = alt.Chart(pd.DataFrame(lines)).mark_bar()
    chart = chart.encode(x=alt.X("sum(jumlah):Q", stack="normalize", title="porsi"),
                         y=alt.Y("nilai:N", title=column), color="label:N")
    st.altair_chart(chart, width="stretch")


def show_relation_chart(ctx, features, column):
    label = ctx["data"][ctx["label_column"]]
    feature, is_number = features[column]
    if is_number and eda.is_regression_label(label):
        frame = pd.DataFrame({"nilai": feature, "label": label}).dropna()
        st.scatter_chart(frame, x="nilai", y="label")
    elif is_number:
        chart_number_by_class(column, feature, label)
    else:
        chart_category_by_class(column, feature, label)


def add_relation_findings(ctx, table):
    strong = table[table["efek"] != eda.EFFECT_NAMES[0]]
    if len(strong) == 0:
        helpers.add_finding(
            ctx, "Hubungan dengan label", "Tidak ada kolom yang berhubungan berarti dengan label.",
            "Semua uji menunjukkan efek diabaikan (di bawah ambang kecil).",
            "Label mungkin hanya bisa dibaca dari isi teks, atau fitur lain perlu dibuat.", 0.4)
    for _, line in strong.head(3).iterrows():
        advice = "Kolom ini calon fitur penting."
        if line["ukuran_efek"] >= 0.5 and line["nama_efek"] != "epsilon2":
            advice = ("Hubungan sangat kuat: cek apakah kolom ini bocoran (turunan dari label) "
                      "sebelum dipakai.")
        helpers.add_finding(
            ctx, "Hubungan dengan label", f"Kolom {line['kolom']} berhubungan dengan label "
            f"({line['uji']}, {line['nama_efek']} = {decimal_text(line['ukuran_efek'])}, efek "
            f"{line['efek']}).", line["tafsir"], advice, line["ukuran_efek"] / 0.5)


def show_relations(ctx):
    st.subheader("2. Hubungan tiap kolom dengan label")
    if ctx["label_column"] is None:
        st.info("Pilih kolom label di panel samping untuk melihat hubungan kolom dengan label.")
        return
    table = eda.label_relations(ctx["data"], ctx["label_column"], ctx["numbers"], ctx["stats"])
    if len(table) == 0:
        st.info("Tidak ada kolom yang bisa diuji terhadap label ini.")
        return
    kamus.show_table(table, hide_index=True)
    st.caption(THRESHOLD_NOTE)
    features = eda.feature_columns(ctx["data"], ctx["label_column"], ctx["numbers"], ctx["stats"])
    column = st.selectbox("Lihat kolom terhadap label", list(table["kolom"]), key="relation_col")
    show_relation_chart(ctx, features, column)
    add_relation_findings(ctx, table)
    best = table.iloc[0]
    strongest = f"{best['nama_efek']} = {decimal_text(best['ukuran_efek'])}, efek {best['efek']}"
    show_explanation("hubungan_label", f"terkuat: {best['kolom']} ({strongest}).")


# ---------------- 3. korelasi antar kolom angka ----------------

def draw_heatmap(long):
    color = alt.Color("rho:Q", scale=alt.Scale(domain=[-1, 1], scheme="blueorange"))
    chart = alt.Chart(long).mark_rect()
    chart = chart.encode(x="kolom_a:N", y="kolom_b:N", color=color,
                         tooltip=["kolom_a", "kolom_b", "rho"])
    st.altair_chart(chart, width="stretch")


def add_correlation_findings(ctx, pairs):
    for _, line in pairs.iterrows():
        helpers.add_finding(
            ctx, "Korelasi", f"Kolom {line['kolom_a']} dan {line['kolom_b']} hampir sama "
            f"(rho = {line['rho']}).", "Urutan nilai kedua kolom hampir identik.",
            "Pertahankan salah satu supaya model tidak menghitung informasi yang sama dua kali.",
            0.5)


def show_correlation(ctx):
    st.subheader("3. Korelasi antar kolom angka (Spearman)")
    numbers = ctx["numbers"]
    if len(numbers.columns) < 2:
        st.info("Pilih minimal dua kolom angka di panel samping.")
        return
    matrix = eda.spearman_matrix(numbers)
    draw_heatmap(eda.correlation_long(matrix))
    pairs = eda.redundant_pairs(matrix)
    st.write(f"Pasangan redundan (|rho| > {eda.REDUNDANT_LIMIT}): **{len(pairs)}**")
    kamus.show_table(pairs, hide_index=True)
    add_correlation_findings(ctx, pairs)
    show_explanation("korelasi", f"{len(pairs)} pasangan redundan dari {len(matrix.columns)} "
                                 "kolom.")


# ---------------- 4. pola data kosong ----------------

def add_missing_findings(ctx, table):
    for _, line in table.iterrows():
        helpers.add_finding(
            ctx, "Data kosong", f"Kolom {line['kolom']} kosong {line['persen_kosong']}% "
            f"({line['kosong']} baris).", "Nilai tidak terisi.",
            "Putuskan: isi (imputasi), jadikan kategori 'kosong', atau buang kolomnya.",
            line["persen_kosong"] / 50)
        tafsir = line.get("tafsir")
        if isinstance(tafsir, str) and eda.related(tafsir):
            helpers.add_finding(
                ctx, "Data kosong", f"Kosongnya kolom {line['kolom']} berkaitan dengan label "
                f"(Cramer V = {line['cramers_v']}).", line["tafsir"],
                "Kosong itu sendiri informasi: jangan diisi mentah, tambahkan penanda 'kosong'.",
                line["cramers_v"] / 0.5)


def show_missing(ctx):
    st.subheader("4. Pola data kosong")
    table = eda.missing_table(ctx["data"], ctx["label_column"])
    if len(table) == 0:
        st.success("Tidak ada sel kosong di seluruh data.")
    else:
        kamus.show_table(table, hide_index=True)
        st.bar_chart(table, x="kolom", y="persen_kosong")
        together = eda.missing_together(ctx["data"])
        if len(together) > 0:
            st.write("Kolom yang kosong di baris yang sama:")
            kamus.show_table(together, hide_index=True)
        add_missing_findings(ctx, table)
    show_explanation("kosong", f"{len(table)} kolom punya sel kosong.")


# ---------------- 5. waktu ----------------

def add_time_findings(ctx, drift):
    if drift is None or drift["p_value"] >= eda.ALPHA or drift["cramers_v"] < 0.1:
        return
    helpers.add_finding(
        ctx, "Waktu", f"Sebaran label bergeser: label '{drift['label']}' dari "
        f"{drift['awal']:.1f}% (separuh awal) menjadi {drift['akhir']:.1f}% (separuh akhir).",
        f"Chi-square awal vs akhir: {eda.p_text(drift['p_value'])}, Cramer V = "
        f"{decimal_text(drift['cramers_v'])}.",
        "Bagi data latih/uji menurut waktu, bukan acak, supaya evaluasi jujur.",
        drift["cramers_v"] / 0.5)


def show_label_over_time(ctx, dates):
    label = ctx["data"][ctx["label_column"]]
    st.write("Sebaran label per bulan (%)")
    st.line_chart(eda.label_by_month(dates, label))
    drift = eda.drift_test(dates, label)
    if drift is not None:
        st.write(f"Separuh awal vs akhir: {eda.p_text(drift['p_value'])}, Cramer V = "
                 f"{decimal_text(drift['cramers_v'])}; label paling berubah: "
                 f"**{drift['label']}** ({drift['awal']:.1f}% menjadi {drift['akhir']:.1f}%)")
    return drift


def show_time(ctx):
    st.subheader("5. Waktu")
    if ctx["date_column"] is None:
        st.info("Pilih 'Kolom tanggal' di panel samping untuk melihat jumlah baris per bulan "
                "dan pergeseran label.")
        return
    raw = ctx["data"][ctx["date_column"]]
    dates = eda.parse_dates(raw)
    st.write(f"Tanggal tidak terbaca: **{int((dates.isna() & raw.notna()).sum())}** dari "
             f"{len(dates)} baris")
    st.bar_chart(eda.monthly_counts(dates))
    drift = None
    if ctx["label_column"] is not None:
        drift = show_label_over_time(ctx, dates)
    add_time_findings(ctx, drift)
    show_explanation("waktu", f"{len(dates.dropna())} tanggal terbaca dari {len(dates)} baris.")


def show(ctx):
    show_profile(ctx)
    show_relations(ctx)
    show_correlation(ctx)
    show_missing(ctx)
    show_time(ctx)
