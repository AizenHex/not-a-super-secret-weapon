"""Laya EDA — melihat data teks / tabel dengan bantuan Laya. Jalankan lewat jalankan.bat."""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch

import helpers
from lenses import PRESETS

DEFAULT_CSV = "../../data_latihan/text/classification/emot_train.csv"
LAYA_FOLDER = "../../pretrained/laya"
MINILM_FOLDER = "../../pretrained/minilm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TYPE_NAMES = {"pilih satu": "choice", "ya / tidak": "noul", "skala": "score"}
SPEED = 300    # perkiraan jawaban Laya per detik di GPU (hasil uji, teks ±180 huruf)
DECISIONS = ["", "simpan", "buang", "ganti_label", "periksa_lagi"]
NOTE_COLUMNS = ["keputusan", "label_baru", "catatan"]

st.set_page_config(page_title="Laya EDA", layout="wide")


@st.cache_resource    # model dimuat sekali saja, tidak tiap klik
def load_laya():
    import laya
    return laya.load(LAYA_FOLDER, subfolder="multilingual", device=DEVICE)


@st.cache_resource
def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MINILM_FOLDER, device=DEVICE)


# hitungan berat di tab 5-6: diulang hanya bila pilihannya berubah
@st.cache_data
def cached_summary(categories, stats, labels, texts):
    return helpers.category_summary(categories, stats, labels, texts)


@st.cache_data
def cached_text_outliers(vectors, rows, categories):
    return helpers.text_outliers(vectors, rows, categories)


@st.cache_data
def cached_pairs(vectors, rows, threshold):
    return helpers.similar_pairs(vectors, rows, threshold)


# ================= sidebar: data dan kolom =================
st.sidebar.title("Laya EDA")
csv_path = st.sidebar.text_input("File CSV (path relatif ke folder _laya_eda)", DEFAULT_CSV)
uploaded = st.sidebar.file_uploader("... atau unggah CSV", type="csv")
if uploaded is not None:
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/{uploaded.name}"
    with open(csv_path, "wb") as file:
        file.write(uploaded.getbuffer())

data = pd.read_csv(csv_path)
data = data.reset_index(drop=True)
folder = helpers.work_folder(csv_path)
all_columns = list(data.columns)

# kolom teks = kolom bertipe object; default = yang isinya paling panjang
text_like = []
longest = 0
for column in all_columns:
    if data[column].dtype == object:
        text_like.append(column)
        mean_length = data[column].astype(str).str.len().mean()
        if mean_length > longest:
            longest = mean_length
            default_text = column

mode = st.sidebar.radio("Mode", ["teks", "baris"],
                        help="teks = Laya membaca satu kolom. baris = Laya membaca beberapa kolom "
                             "sekaligus (judul, alamat, harga, ...)")
if mode == "teks":
    text_column = st.sidebar.selectbox("Kolom teks", text_like, index=text_like.index(default_text))
    columns = [text_column]
else:
    columns = st.sidebar.multiselect("Kolom yang dibaca Laya", all_columns, default=text_like)
if len(columns) == 0:
    st.warning("Pilih minimal satu kolom di samping.")
    st.stop()

label_choice = st.sidebar.selectbox("Kolom label (opsional)", ["(tidak ada)"] + all_columns)
label_column = None
if label_choice != "(tidak ada)":
    label_column = label_choice
doubt_limit = st.sidebar.slider("Batas ragu: yakin Laya di bawah ini = ragu", 0.0, 1.0, 0.6, 0.05)
st.sidebar.caption(f"Hasil disimpan di `_laya_eda/{folder}` · perangkat: {DEVICE}")

texts = helpers.row_texts(data, columns)
lenses = helpers.load_lenses(folder)

tabs = st.tabs(["1 Data", "2 Cek angka", "3 Temukan kategori", "4 Lensa Laya", "5 Peta data",
                "6 Outlier", "7 Telusuri + catatan", "8 Ekspor"])
tab_data = tabs[0]
tab_numbers = tabs[1]
tab_groups = tabs[2]
tab_lenses = tabs[3]
tab_map = tabs[4]
tab_outlier = tabs[5]
tab_browse = tabs[6]
tab_export = tabs[7]

# ================= 1 Data =================
with tab_data:
    st.subheader(f"{len(data)} baris × {len(all_columns)} kolom")
    st.dataframe(data.head(50))
    left, right = st.columns(2)
    per_column = pd.DataFrame()
    per_column["tipe"] = data.dtypes.astype(str)
    per_column["kosong"] = data.isna().sum()
    per_column["persen_kosong"] = (100 * data.isna().mean()).round(1)
    per_column["nilai_unik"] = data.nunique()
    left.write("**Per kolom**")
    left.dataframe(per_column)
    if label_column is not None:
        right.write(f"**Sebaran label `{label_column}`**")
        right.bar_chart(data[label_column].value_counts(), horizontal=True)

    stats = helpers.text_stats(texts)
    st.write("**Statistik teks** (kolom yang dibaca Laya)")
    first, second, third, fourth = st.columns(4)
    first.metric("duplikat persis", int(stats["duplikat"].sum()))
    second.metric("duplikat mirip (beda huruf/tanda baca)", int(stats["duplikat_mirip"].sum()))
    third.metric("≤ 3 kata", int((stats["jumlah_kata"] <= 3).sum()))
    fourth.metric("panjang median", int(stats["panjang"].median()))
    counts, edges = np.histogram(stats["panjang"], bins=40)
    st.bar_chart(pd.Series(counts, index=edges[:-1].round()), x_label="panjang (huruf)")
    with st.expander("Duplikat terbanyak"):
        st.dataframe(texts[stats["duplikat"]].value_counts().head(30))
    with st.expander("Teks paling pendek"):
        st.dataframe(data.loc[stats["panjang"].nsmallest(30).index, columns])
    with st.expander("Teks paling panjang"):
        st.dataframe(data.loc[stats["panjang"].nlargest(30).index, columns])
    with st.expander("Paling banyak simbol / angka (bukan huruf)"):
        st.dataframe(data.loc[stats["rasio_bukan_huruf"].nlargest(30).index, columns])

# ================= 2 Cek angka =================
with tab_numbers:
    st.write("Pandas mengecek angka (Laya lemah soal angka). Hasilnya dipakai lagi di tab 6.")
    numeric_default = []
    for column in all_columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric_default.append(column)
    number_columns = st.multiselect("Kolom angka yang dicek", all_columns, default=numeric_default)
    indonesian = st.checkbox("Format Indonesia (titik = ribuan, koma = desimal, 'Rp' dibuang)",
                             value=True)
    use_log = st.checkbox("Outlier pada skala log (disarankan untuk harga, luas, jumlah)",
                          value=True)
    numbers = pd.DataFrame(index=data.index)
    for column in number_columns:
        values = helpers.to_number(data[column], indonesian)
        numbers[column] = values
        unreadable = data[column].notna() & values.isna()
        outlier = helpers.iqr_outliers(values, use_log)
        st.markdown(f"#### {column}")
        first, second, third, fourth = st.columns(4)
        first.metric("tidak terbaca sebagai angka", int(unreadable.sum()))
        second.metric("kosong", int(data[column].isna().sum()))
        third.metric("negatif / nol", int((values <= 0).sum()))
        fourth.metric("outlier IQR (semua baris)", int(outlier.sum()))
        with st.expander(f"lihat {column}"):
            st.write("Nilai yang tidak bisa dibaca sebagai angka:")
            st.dataframe(data.loc[unreadable, column].value_counts().head(30))
            st.dataframe(values.describe())
            order = values[outlier].sort_values(ascending=False).index
            st.write("Outlier (terbesar dulu):")
            st.dataframe(data.loc[order].head(50))
    if len(number_columns) >= 2:
        st.markdown("#### Kolom rasio (misal harga per m²)")
        first, second = st.columns(2)
        top = first.selectbox("Pembilang", number_columns)
        bottom = second.selectbox("Penyebut", number_columns, index=1)
        numbers[f"{top}_per_{bottom}"] = numbers[top] / numbers[bottom].where(numbers[bottom] > 0)
        st.caption(f"Kolom `{top}_per_{bottom}` bisa dipilih di tab 6 → outlier per kategori.")

# ================= 3 Temukan kategori =================
with tab_groups:
    st.write("Laya hanya bisa **memilih** dari daftar kategori yang kita tulis. Di sini MiniLM "
             "mengelompokkan teks yang mirip, supaya kamu tahu kategori apa yang sebenarnya ada.")
    n_groups = st.slider("Jumlah kelompok", 3, 40, 12)
    max_rows = st.number_input("Maksimal baris (sampel acak)", 50, len(data), min(20000, len(data)))
    st.caption("Vektor MiniLM ikut disimpan dan dipakai lagi di tab 6 (outlier teks, teks "
               "yang maknanya sama).")
    if st.button("Kelompokkan"):
        sample = data.sample(n=int(max_rows), random_state=helpers.SEED).index
        with st.spinner("MiniLM membaca teks ..."):
            vectors = helpers.encode(load_minilm(), texts[sample])
        groups = helpers.find_groups(vectors, n_groups)
        keywords = helpers.group_keywords(texts[sample], groups)
        np.save(f"{folder}/vektor.npy", vectors)
        pd.DataFrame({"row": sample, "kelompok": groups}).to_csv(f"{folder}/kelompok.csv",
                                                                 index=False)
        info = pd.DataFrame({"kelompok": range(n_groups)})
        info["kata_kunci"] = info["kelompok"].map(keywords)
        info.to_csv(f"{folder}/kelompok_info.csv", index=False)

    if os.path.exists(f"{folder}/kelompok_info.csv"):
        group_rows = pd.read_csv(f"{folder}/kelompok.csv")
        info = pd.read_csv(f"{folder}/kelompok_info.csv")
        sizes = group_rows["kelompok"].value_counts()
        for group, keywords in zip(info["kelompok"], info["kata_kunci"]):
            members = group_rows.loc[group_rows["kelompok"] == group, "row"]
            with st.expander(f"Kelompok {group} · {sizes[group]} baris · {keywords}"):
                st.dataframe(data.loc[members.head(10), columns])

        st.write("**Jadikan lensa.** Beri nama tiap kelompok. Nama sama = kelompok digabung. "
                 "Nama dikosongkan = kelompok tidak dipakai. Penjelasan boleh diperbaiki.")
        naming = pd.DataFrame({"kelompok": info["kelompok"]})
        naming["nama"] = "kelompok_" + info["kelompok"].astype(str)
        naming["penjelasan"] = info["kata_kunci"]
        named = st.data_editor(naming, disabled=["kelompok"], hide_index=True, key="naming")
        lens_name = st.text_input("Nama lensa baru", "topik")
        if st.button("Jadikan lensa"):
            criteria = {}
            for group_name, description in zip(named["nama"], named["penjelasan"]):
                group_name = str(group_name).strip()
                if group_name == "" or group_name == "None":
                    continue
                if group_name in criteria:
                    criteria[group_name] = criteria[group_name] + ", " + description
                else:
                    criteria[group_name] = description
            lenses[lens_name] = {"type": "choice", "criteria": criteria,
                                 "instructions": "Teks ini termasuk kategori apa?"}
            helpers.save_lenses(folder, lenses)
            if os.path.exists(helpers.lens_file(folder, lens_name)):
                os.remove(helpers.lens_file(folder, lens_name))
            st.success(f"Lensa `{lens_name}` ({len(criteria)} pilihan) dibuat. Jalankan di tab 4.")

# ================= 4 Lensa Laya =================
with tab_lenses:
    overview = []
    for name in lenses:
        lens = lenses[name]
        options = ""
        if "criteria" in lens:
            options = ", ".join(lens["criteria"])
        overview.append({"lensa": name, "jenis": lens["type"], "pertanyaan": lens["instructions"],
                         "pilihan": options,
                         "sudah_dihitung": len(helpers.done_rows(folder, name))})
    st.write("**Lensa aktif** (satu lensa = satu pertanyaan yang dijawab Laya untuk tiap baris)")
    if len(overview) == 0:
        st.info("Belum ada lensa. Tambah lensa siap pakai atau buat sendiri di bawah.")
    else:
        st.dataframe(pd.DataFrame(overview), hide_index=True)

    left, right = st.columns(2)
    with left:
        preset = st.selectbox("Lensa siap pakai (isi lengkap ada di lenses.py)", list(PRESETS))
        st.json(PRESETS[preset], expanded=False)
        if st.button("Tambah lensa ini"):
            lenses[preset] = PRESETS[preset]
            helpers.save_lenses(folder, lenses)
            st.rerun()
        removed = st.selectbox("Hapus lensa", ["-"] + list(lenses))
        if st.button("Hapus") and removed != "-":
            del lenses[removed]
            helpers.save_lenses(folder, lenses)
            if os.path.exists(helpers.lens_file(folder, removed)):
                os.remove(helpers.lens_file(folder, removed))
            st.rerun()
    with right:
        with st.form("lensa_baru"):
            st.write("**Buat / ubah lensa** (nama sama = ganti, hasil lamanya dihapus)")
            name = st.text_input("Nama lensa (tanpa spasi), misal: topik")
            type_name = st.radio("Jenis", list(TYPE_NAMES), horizontal=True)
            question = st.text_input("Pertanyaan untuk Laya", "Apa topik utama teks ini?")
            options_text = st.text_area("Pilihan, satu per baris. Pilih satu: `nama: penjelasan`. "
                                        "Skala: urut dari terendah. Ya/tidak: kosongkan.")
            with_label = st.checkbox("Laya juga melihat kolom label (untuk mencari label salah)")
            if st.form_submit_button("Simpan lensa") and name.strip() != "":
                lens = {"type": TYPE_NAMES[type_name], "instructions": question}
                if lens["type"] != "noul":
                    lens["criteria"] = helpers.parse_options(options_text, lens["type"])
                if with_label:
                    lens["with_label"] = True
                lenses[name.strip()] = lens
                helpers.save_lenses(folder, lenses)
                if os.path.exists(helpers.lens_file(folder, name.strip())):
                    os.remove(helpers.lens_file(folder, name.strip()))
                st.rerun()

    st.divider()
    st.write("**Jalankan Laya.** Urutan acak selalu sama, jadi menambah jumlah baris nanti hanya "
             "menghitung baris baru. Boleh dihentikan (tombol Stop kanan atas); hasil tersimpan.")
    use_all = st.checkbox(f"Semua baris ({len(data)})")
    n_rows = st.number_input("Jumlah baris", 10, len(data), min(1000, len(data)))
    if use_all:
        n_rows = len(data)
    order = data.sample(frac=1, random_state=helpers.SEED).index
    target = order[:int(n_rows)]
    todo = {}
    for name in lenses:
        done = helpers.done_rows(folder, name)
        rows = []
        for row in target:
            if row not in done:
                rows.append(row)
        todo[name] = rows
    total = 0
    for name in todo:
        total = total + len(todo[name])
    st.write(f"Perlu dihitung: **{total}** jawaban · perkiraan **{total / SPEED / 60:.1f} menit**")
    if st.button("Jalankan Laya", type="primary", disabled=total == 0):
        judge = load_laya()
        for name in lenses:
            if len(todo[name]) == 0:
                continue
            lens_label = None
            if lenses[name].get("with_label"):
                if label_column is None:
                    st.warning(f"Lensa {name} butuh kolom label; pilih di samping. Dilewati.")
                    continue
                lens_label = label_column
            bar = st.progress(0.0, text=f"lensa {name} ...")
            helpers.run_lens(judge, data, todo[name], columns, lens_label, name, lenses[name],
                             folder, bar)
        st.success("Selesai. Lihat tab 5 Peta data.")

# ================= tabel gabungan untuk tab 5-8 =================
results = helpers.lens_results(folder, lenses, data.index)
table = data.copy()
for column in results.columns:
    table[column] = results[column]
dimensions = []
for name in lenses:
    if name in results.columns:
        dimensions.append(name)
lens_done = list(dimensions)
if os.path.exists(f"{folder}/kelompok.csv"):
    group_rows = pd.read_csv(f"{folder}/kelompok.csv")
    table["kelompok"] = pd.Series(group_rows["kelompok"].to_numpy(), index=group_rows["row"])
    dimensions.append("kelompok")
vectors = None
if os.path.exists(f"{folder}/vektor.npy"):
    vectors = np.load(f"{folder}/vektor.npy")
    vector_rows = group_rows["row"].to_numpy()
if label_column is not None:
    dimensions.append(label_column)

doubt = pd.Series(False, index=table.index)
for name in lens_done:
    doubt = doubt | (table[f"{name}__yakin"] < doubt_limit)
table["laya_ragu"] = doubt

show_columns = list(columns)
for name in dimensions:
    if name not in show_columns:
        show_columns.append(name)
for name in lens_done:
    show_columns.append(f"{name}__nilai")    # klik judul kolom untuk mengurutkan

# ================= 5 Peta data =================
with tab_map:
    if len(dimensions) == 0:
        st.info("Belum ada hasil. Jalankan Laya di tab 4 (atau Kelompokkan di tab 3).")
    else:
        read_rows = results.notna().any(axis=1).sum()
        st.write(f"Baris yang sudah dibaca Laya: **{read_rows}** dari {len(data)} · "
                 f"Laya ragu (yakin < {doubt_limit}) di minimal satu lensa: **{doubt.sum()}**")
        grid = st.columns(2)
        for number, name in enumerate(lens_done):
            unsure = int((table[f"{name}__yakin"] < doubt_limit).sum())
            grid[number % 2].write(f"**{name}** · ragu: {unsure} baris")
            grid[number % 2].bar_chart(table[name].value_counts(), horizontal=True)

        st.subheader("Ringkasan per kategori")
        summary_dim = st.selectbox("Kategori", dimensions, key="summary_dim")
        labels = None
        if label_column is not None and summary_dim != label_column:
            labels = data[label_column]
        st.dataframe(cached_summary(table[summary_dim], stats, labels, texts))
        st.caption("kata_khas = kata yang jauh lebih sering muncul di kategori ini daripada di "
                   "semua teks. Bahan insight untuk slide.")

        st.subheader("Tabel silang")
        st.caption("Contoh: label asli × lensa emosi → sel yang tidak sejalan = kemungkinan salah "
                   "label. Pilih nilai di bawah tabel untuk melihat barisnya.")
        left, middle, right = st.columns(3)
        dim_a = left.selectbox("Baris tabel", dimensions, key="dim_a")
        dim_b = middle.selectbox("Kolom tabel", dimensions, index=len(dimensions) - 1, key="dim_b")
        show_as = right.radio("Isi sel", ["jumlah", "% per baris", "% per kolom"], horizontal=True)
        cross = pd.crosstab(table[dim_a], table[dim_b])
        if show_as == "% per baris":
            cross = (100 * pd.crosstab(table[dim_a], table[dim_b], normalize="index")).round(1)
        if show_as == "% per kolom":
            cross = (100 * pd.crosstab(table[dim_a], table[dim_b], normalize="columns")).round(1)
        st.dataframe(cross.style.background_gradient(cmap="Blues", axis=None))
        value_a = left.selectbox(f"{dim_a} =", list(cross.index), key="value_a")
        value_b = middle.selectbox(f"{dim_b} =", list(cross.columns), key="value_b")
        picked = table[(table[dim_a] == value_a) & (table[dim_b] == value_b)]
        st.write(f"{len(picked)} baris")
        st.dataframe(picked[show_columns].head(300))

# ================= 6 Outlier =================
with tab_outlier:
    if len(dimensions) == 0:
        st.info("Belum ada kategori. Jalankan Laya di tab 4 (atau Kelompokkan di tab 3).")
    else:
        dim = st.selectbox("Bandingkan di dalam kategori", dimensions, key="dim_outlier")

        st.subheader("Angka: dibanding kategorinya vs dibanding keseluruhan")
        if len(numbers.columns) == 0:
            st.info("Pilih kolom angka di tab 2 dulu.")
        else:
            number_column = st.selectbox("Kolom angka", list(numbers.columns))
            checked = helpers.compare_outliers(numbers[number_column], table[dim], use_log)
            per_group = checked.groupby("kelompok")
            summary = per_group["nilai"].describe()
            summary = summary[["count", "25%", "50%", "75%"]]
            summary["outlier_di_kategori"] = per_group["outlier"].sum()
            summary["outlier_di_keseluruhan"] = per_group["outlier_keseluruhan"].sum()
            st.dataframe(summary)

            box_data = checked[["nilai", "kelompok"]].copy()
            box_data["kelompok"] = box_data["kelompok"].astype(str)
            scale = alt.Scale(type="linear")
            if use_log:
                box_data = box_data[box_data["nilai"] > 0]
                scale = alt.Scale(type="log")
            x_axis = alt.X("nilai:Q", scale=scale, title=number_column)
            box = alt.Chart(box_data).mark_boxplot()
            box = box.encode(x=x_axis, y=alt.Y("kelompok:N", title=dim))
            st.altair_chart(box, width="stretch")

            chosen = st.selectbox(f"Histogram: semua data vs {dim} =", list(per_group.groups))
            same = numbers[number_column][table[dim] == chosen]
            histogram = helpers.compare_histogram(numbers[number_column], same, use_log)
            x_name = number_column
            if use_log:
                x_name = f"log10 {number_column}"
            st.line_chart(histogram, x_label=x_name, y_label="% baris")

            st.write("Yang paling menarik biasanya **janggal hanya di kategorinya**: wajar kalau "
                     "dilihat global, padahal aneh untuk jenisnya.")
            st.dataframe(checked["status"].value_counts())
            status = st.selectbox("Tampilkan status", ["janggal hanya di kategorinya",
                                                       "janggal di keduanya",
                                                       "janggal hanya di keseluruhan", "normal"])
            flagged = checked[checked["status"] == status].sort_values("kali_median")
            st.write(f"{len(flagged)} baris · kali_median kecil = terlalu murah/kecil, "
                     "besar = terlalu mahal/besar")
            st.dataframe(flagged.join(data[columns]).head(300))

        st.subheader("Teks: jauh dari kategorinya atau nyasar")
        if vectors is None:
            st.info("Tekan **Kelompokkan** di tab 3 dulu (sekalian membuat vektor MiniLM).")
        else:
            distances = cached_text_outliers(vectors, vector_rows, table[dim])
            first, second = st.columns(2)
            first.metric("jauh dari kategorinya", int(distances["jauh"].sum()))
            second.metric("lebih dekat ke kategori lain (nyasar)", int(distances["nyasar"].sum()))
            st.caption("Jarak dihitung dari makna (MiniLM), bukan kata. Nyasar pada lensa Laya = "
                       "kemungkinan salah kategori; nyasar pada label asli = kemungkinan salah "
                       "label.")
            text_view = st.radio("Tampilkan", ["jauh", "nyasar"], horizontal=True)
            odd = distances[distances[text_view]]
            odd = odd.sort_values("jarak_ke_kategorinya", ascending=False)
            st.dataframe(odd.join(data[columns]).head(300))

            st.subheader("Teks yang maknanya hampir sama (kata-katanya beda)")
            threshold = st.slider("Batas kemiripan", 0.80, 0.99, 0.93, 0.01)
            unique_rows = []
            seen = set()
            for position, row in enumerate(vector_rows):
                if texts[row] not in seen:
                    seen.add(texts[row])
                    unique_rows.append(position)
            pairs = cached_pairs(vectors[unique_rows], vector_rows[unique_rows], threshold)
            pairs["teks_a"] = texts[pairs["row_a"]].to_numpy()
            pairs["teks_b"] = texts[pairs["row_b"]].to_numpy()
            st.write(f"{len(pairs)} pasangan (duplikat persis sudah dibuang; maks 3000)")
            st.dataframe(pairs.head(300), hide_index=True)

        st.subheader("Periksa satu baris: outlier atau bukan?")
        row = st.number_input("Nomor baris (kolom `row` di tab 7)", 0, len(data) - 1, 0)
        st.dataframe(table.loc[[row], show_columns])
        category = table.at[row, dim]
        lines = []
        for column in numbers.columns:
            values = numbers[column]
            line = {"kolom": column, "nilai": values[row]}
            if pd.isna(values[row]):
                lines.append(line)
                continue
            overall = helpers.describe_value(values, row, use_log)
            line["persentil_keseluruhan"] = overall["persentil"]
            line["median_keseluruhan"] = overall["median"]
            line["outlier_keseluruhan"] = overall["outlier"]
            if not pd.isna(category):
                inside = helpers.describe_value(values[table[dim] == category], row, use_log)
                line[f"persentil_di_{dim}={category}"] = inside["persentil"]
                line["median_kategori"] = inside["median"]
                line["outlier_kategori"] = inside["outlier"]
            lines.append(line)
        if len(lines) > 0:
            st.dataframe(pd.DataFrame(lines), hide_index=True)
        if vectors is not None and row in distances.index:
            info = distances.loc[row]
            st.write(f"Teks: jarak ke kategorinya **{info['jarak_ke_kategorinya']:.3f}** · "
                     f"kategori terdekat: **{info['kategori_terdekat']}** · jauh: {info['jauh']}")
        if pd.isna(category):
            st.caption(f"Baris ini belum punya nilai `{dim}`; hanya dibanding keseluruhan.")

# ================= 7 Telusuri + catatan =================
with tab_browse:
    first, second, third = st.columns(3)
    filter_dim = first.selectbox("Saring menurut", ["(semua)"] + dimensions)
    filter_values = []
    if filter_dim != "(semua)":
        filter_values = second.multiselect("Nilai", list(table[filter_dim].dropna().unique()))
    search = third.text_input("Cari kata")
    only_doubt = st.checkbox("Hanya baris yang membuat Laya ragu")
    view = table
    if len(filter_values) > 0:
        view = view[view[filter_dim].isin(filter_values)]
    if search != "":
        view = view[texts[view.index].str.contains(search, case=False, regex=False)]
    if only_doubt:
        view = view[view["laya_ragu"]]
    st.write(f"{len(view)} baris cocok · ditampilkan maks 500 · isi kolom keputusan / catatan "
             "lalu tekan **Simpan catatan**")

    notes = helpers.load_notes(folder)
    shown = view.head(500)[show_columns].copy()
    shown.insert(0, "row", shown.index)
    for column in NOTE_COLUMNS:
        shown[column] = notes[column].reindex(shown.index).fillna("")
    locked = []
    for column in shown.columns:
        if column not in NOTE_COLUMNS:
            locked.append(column)
    decision_box = st.column_config.SelectboxColumn("keputusan", options=DECISIONS)
    edited = st.data_editor(shown, column_config={"keputusan": decision_box}, disabled=locked,
                            hide_index=True, key="editor")
    if st.button("Simpan catatan"):
        for i in range(len(edited)):
            row = edited["row"].iloc[i]
            notes.loc[row, NOTE_COLUMNS] = edited[NOTE_COLUMNS].iloc[i].fillna("").to_list()
        helpers.save_notes(folder, notes)
        st.success(f"Tersimpan di _laya_eda/{folder}/catatan.csv")

# ================= 8 Ekspor =================
with tab_export:
    st.write("Satu file berisi data asli + jawaban Laya (+ yakin) + kelompok + cek teks + angka "
             "+ catatanmu. File ini dibaca notebook EDA program.")
    export = table.copy()
    for column in ["panjang", "jumlah_kata", "rasio_bukan_huruf", "duplikat", "duplikat_mirip"]:
        export[f"teks__{column}"] = stats[column]
    for column in numbers.columns:
        export[f"{column}__angka"] = numbers[column]
    notes = helpers.load_notes(folder)
    for column in NOTE_COLUMNS:
        export[column] = notes[column].reindex(export.index).fillna("")
    export_path = f"{folder}/data_berlensa.csv"
    if st.button("Tulis file", type="primary"):
        export.to_csv(export_path, index_label="row")
        st.success(f"Tersimpan: _laya_eda/{export_path}")
    st.code(f'data = pd.read_csv("../_laya_eda/{export_path}")\n'
            'data = data[data["keputusan"] != "buang"]', language="python")
    st.dataframe(export.head(20))
