"""Fungsi bantu untuk app.py: statistik teks, cek angka, kelompok MiniLM, dan menjalankan Laya.
Semua hasil disimpan ke folder hasil/<nama_csv>/ supaya tidak dihitung ulang."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

RESULT_FOLDER = "hasil"
BATCH_SIZE = 32        # teks per putaran GPU (32 paling cepat di laptop ini)
CHUNK_ROWS = 256       # hasil Laya disimpan ke file tiap sekian baris
SEED = 42


def work_folder(csv_path):
    name = os.path.splitext(os.path.basename(csv_path))[0]
    folder = f"{RESULT_FOLDER}/{name}"
    os.makedirs(folder, exist_ok=True)
    return folder


def row_texts(data, columns):
    # mode teks: isi satu kolom; mode baris: "kolom: nilai | kolom: nilai"
    if len(columns) == 1:
        return data[columns[0]].fillna("").astype(str)
    texts = pd.Series("", index=data.index)
    for column in columns:
        texts = texts + column + ": " + data[column].fillna("").astype(str) + " | "
    return texts


def text_stats(texts):
    stats = pd.DataFrame(index=texts.index)
    stats["panjang"] = texts.str.len()
    stats["jumlah_kata"] = texts.str.split().str.len()
    letters = texts.str.count(r"[A-Za-z]")
    stats["rasio_bukan_huruf"] = 1 - letters / stats["panjang"].clip(lower=1)
    stats["duplikat"] = texts.duplicated(keep=False)
    simple = texts.str.lower()
    simple = simple.str.replace(r"[^a-z0-9]", "", regex=True)
    stats["duplikat_mirip"] = simple.duplicated(keep=False) & ~stats["duplikat"]
    return stats


# ---------------- cek angka ----------------

def to_number(series, indonesian_format):
    if pd.api.types.is_numeric_dtype(series):
        return series
    text = series.astype(str).str.lower()
    text = text.str.replace("rp", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    if indonesian_format:
        text = text.str.replace(".", "", regex=False)    # titik = pemisah ribuan
        text = text.str.replace(",", ".", regex=False)   # koma = desimal
    else:
        text = text.str.replace(",", "", regex=False)
    number_text = text.str.extract(r"^(-?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(number_text, errors="coerce")


def iqr_outliers(values, use_log):
    # skala log cocok untuk harga/luas: 15 juta vs 1,5 miliar terlihat jauh
    if use_log:
        values = np.log10(values.where(values > 0))
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    spread = 1.5 * (q3 - q1)
    return (values < q1 - spread) | (values > q3 + spread)


def group_outliers(values, groups, use_log):
    # outlier IQR di dalam tiap kelompok: dibandingkan dengan baris sejenis saja
    table = pd.DataFrame({"nilai": values, "kelompok": groups})
    table = table.dropna()
    scaled = table["nilai"]
    if use_log:
        scaled = np.log10(scaled.where(scaled > 0))
    grouped = scaled.groupby(table["kelompok"])
    q1 = grouped.transform("quantile", 0.25)
    q3 = grouped.transform("quantile", 0.75)
    spread = (q3 - q1) * 1.5
    grouped = table.groupby("kelompok")["nilai"]
    table["median_kelompok"] = grouped.transform("median")
    table["kali_median"] = table["nilai"] / table["median_kelompok"]
    table["outlier"] = (scaled < q1 - spread) | (scaled > q3 + spread)
    return table


def compare_outliers(values, groups, use_log):
    # dua pembanding: semua baris vs baris sekategori saja
    table = group_outliers(values, groups, use_log)
    table["outlier_keseluruhan"] = iqr_outliers(values, use_log)[table.index]
    table["persentil_keseluruhan"] = (100 * values.rank(pct=True)[table.index]).round()
    table["persentil_kategori"] = (100 * table.groupby("kelompok")["nilai"].rank(pct=True)).round()
    in_group = table["outlier"]
    overall = table["outlier_keseluruhan"]
    table["status"] = "normal"
    table.loc[in_group & overall, "status"] = "janggal di keduanya"
    table.loc[in_group & ~overall, "status"] = "janggal hanya di kategorinya"
    table.loc[~in_group & overall, "status"] = "janggal hanya di keseluruhan"
    return table


def describe_value(values, row, use_log):
    # posisi satu nilai di antara values: persentil, median, outlier atau bukan
    values = values.dropna()
    value = values[row]
    position = {"persentil": round(100 * (values < value).mean()), "median": values.median()}
    position["outlier"] = bool(iqr_outliers(values, use_log)[row])
    return position


# ---------------- kelompok MiniLM ----------------

def encode(encoder, texts):
    # vektor makna tiap teks (panjang 1, jadi kemiripan = perkalian titik)
    return encoder.encode(list(texts), batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False)


def find_groups(vectors, n_groups):
    model = KMeans(n_clusters=n_groups, n_init=4, random_state=SEED)
    return model.fit_predict(vectors)


def group_keywords(texts, groups, n_words=8):
    # kata khas tiap kelompok: TF-IDF rata-rata di kelompok dikurangi rata-rata semua teks
    vectorizer = TfidfVectorizer(max_features=20000, min_df=2)
    matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    overall = np.asarray(matrix.mean(axis=0)).ravel()
    groups = np.asarray(groups)
    keywords = {}
    for group in pd.unique(groups):
        inside = np.asarray(matrix[groups == group].mean(axis=0)).ravel()
        top = (inside - overall).argsort()[::-1][:n_words]
        keywords[group] = ", ".join(words[top])
    return keywords


def category_summary(categories, stats, labels, texts):
    # satu baris per kategori: jumlah, panjang teks, label dominan, kata khas
    frame = pd.DataFrame({"kategori": categories, "panjang": stats["panjang"], "label": labels})
    frame = frame.dropna(subset=["kategori"])
    grouped = frame.groupby("kategori")
    summary = pd.DataFrame()
    summary["jumlah"] = grouped.size()
    summary["persen"] = (100 * summary["jumlah"] / len(frame)).round(1)
    summary["panjang_median"] = grouped["panjang"].median()
    if frame["label"].notna().any():
        shares = pd.crosstab(frame["kategori"], frame["label"], normalize="index")
        summary["label_terbanyak"] = shares.idxmax(axis=1)
        summary["persen_label_itu"] = (100 * shares.max(axis=1)).round(1)
    summary["kata_khas"] = pd.Series(group_keywords(texts[frame.index], frame["kategori"]))
    return summary.sort_values("jumlah", ascending=False)


def text_outliers(vectors, rows, categories):
    # jarak makna tiap teks ke pusat kategorinya; "nyasar" = lebih dekat ke kategori lain
    labels = categories[rows].to_numpy()
    keep = pd.notna(labels)
    vectors = vectors[keep]
    labels = labels[keep]
    names = list(pd.unique(labels))
    centers = []
    for name in names:
        center = vectors[labels == name].mean(axis=0)
        centers.append(center / np.linalg.norm(center))
    similarity = vectors @ np.array(centers).T
    own = []
    for label in labels:
        own.append(names.index(label))
    table = pd.DataFrame({"kategori": labels}, index=np.asarray(rows)[keep])
    table["jarak_ke_kategorinya"] = 1 - similarity[np.arange(len(labels)), own]
    table["kategori_terdekat"] = np.array(names, dtype=object)[similarity.argmax(axis=1)]
    table["nyasar"] = table["kategori_terdekat"] != table["kategori"]
    checked = group_outliers(table["jarak_ke_kategorinya"], table["kategori"], False)
    table["jauh"] = checked["outlier"] & (checked["kali_median"] > 1)   # hanya yang terlalu jauh
    return table


def similar_pairs(vectors, rows, threshold, max_pairs=3000):
    # pasangan teks yang maknanya hampir sama walau kata-katanya beda
    rows = np.asarray(rows)
    pairs = []
    for start in range(0, len(vectors), 1000):
        similarity = vectors[start:start + 1000] @ vectors.T
        first, second = np.nonzero(similarity >= threshold)
        for i, j in zip(first, second):
            if start + i < j:
                pairs.append({"row_a": rows[start + i], "row_b": rows[j],
                              "kemiripan": round(float(similarity[i, j]), 3)})
        if len(pairs) >= max_pairs:
            break
    return pd.DataFrame(pairs[:max_pairs], columns=["row_a", "row_b", "kemiripan"])


def compare_histogram(values, inside, use_log, bins=30):
    # sebaran (persen) semua data vs satu kategori, di sumbu yang sama
    values = values.dropna()
    inside = inside.dropna()
    if use_log:
        values = np.log10(values[values > 0])
        inside = np.log10(inside[inside > 0])
    edges = np.histogram_bin_edges(values, bins=bins)
    all_counts, _ = np.histogram(values, edges)
    inside_counts, _ = np.histogram(inside, edges)
    chart = pd.DataFrame(index=edges[:-1].round(2))
    chart["semua data (%)"] = 100 * all_counts / max(all_counts.sum(), 1)
    chart["kategori ini (%)"] = 100 * inside_counts / max(inside_counts.sum(), 1)
    return chart


# ---------------- lensa Laya ----------------

def load_lenses(folder):
    path = f"{folder}/lenses.json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def save_lenses(folder, lenses):
    with open(f"{folder}/lenses.json", "w", encoding="utf-8") as file:
        json.dump(lenses, file, ensure_ascii=False, indent=2)


def lens_file(folder, name):
    return f"{folder}/laya_{name}.csv"


def parse_options(text, lens_type):
    # satu pilihan per baris; untuk "pilih satu" boleh "nama: penjelasan"
    lines = []
    for line in text.splitlines():
        if line.strip() != "":
            lines.append(line.strip())
    if lens_type == "score":
        return lines
    criteria = {}
    for line in lines:
        name, _, description = line.partition(":")
        description = description.strip()
        if description == "":
            description = name.strip()
        criteria[name.strip()] = description
    return criteria


def make_states(data, rows, columns, label_column):
    # mode teks tanpa label: kirim teksnya saja; selain itu kirim dict kolom -> nilai
    states = []
    for row in rows:
        if len(columns) == 1 and label_column is None:
            states.append(str(data.at[row, columns[0]]))
            continue
        state = {}
        for column in columns:
            state[column] = str(data.at[row, column])
        if label_column is not None:
            state["label"] = str(data.at[row, label_column])
        states.append(state)
    return states


def read_answer(answer, lens):
    # hasil: (jawaban, nilai, yakin). nilai = peluang "ya" atau skor skala
    if answer["type"] == "choice":
        return answer["choice"], answer["answer_confidence"], answer["answer_confidence"]
    if answer["type"] == "score":
        level = int(round(answer["score"]))
        text = f"{level} {lens['criteria'][level]}"
        return text, answer["score"], answer["answer_confidence"]
    choice = "tidak"
    if answer["noul"] >= 0.5:
        choice = "ya"
    return choice, answer["noul"], answer["answer_confidence"]


def run_lens(judge, data, rows, columns, label_column, name, lens, folder, bar):
    question = {"type": lens["type"], "instructions": lens["instructions"]}
    if lens["type"] != "noul":
        question["criteria"] = lens["criteria"]
    path = lens_file(folder, name)
    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows[start:start + CHUNK_ROWS]
        states = make_states(data, chunk, columns, label_column)
        results = judge.predict_batch(states, {name: question}, batch_size=BATCH_SIZE)
        lines = []
        for row, result in zip(chunk, results):
            answer, value, confidence = read_answer(result["answers"][name], lens)
            lines.append({"row": row, "answer": answer, "value": value, "confidence": confidence})
        first_write = not os.path.exists(path)
        pd.DataFrame(lines).to_csv(path, mode="a", header=first_write, index=False)
        done = start + len(chunk)
        bar.progress(done / len(rows), text=f"lensa {name}: {done} / {len(rows)} baris")


def done_rows(folder, name):
    path = lens_file(folder, name)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["row"])


def lens_results(folder, lenses, index):
    table = pd.DataFrame(index=index)
    for name in lenses:
        path = lens_file(folder, name)
        if not os.path.exists(path):
            continue
        result = pd.read_csv(path)
        result = result.drop_duplicates("row", keep="last")
        result = result.set_index("row")
        table[name] = result["answer"]
        table[f"{name}__yakin"] = result["confidence"]
        table[f"{name}__nilai"] = result["value"]
    return table


# ---------------- catatan manual ----------------

def load_notes(folder):
    path = f"{folder}/catatan.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["keputusan", "label_baru", "catatan"])
    text_types = {"keputusan": str, "label_baru": str, "catatan": str}
    return pd.read_csv(path, index_col="row", dtype=text_types, keep_default_na=False)


def save_notes(folder, notes):
    keep = (notes["keputusan"] != "") | (notes["label_baru"] != "") | (notes["catatan"] != "")
    notes = notes[keep]
    notes.to_csv(f"{folder}/catatan.csv", index_label="row")
