"""Fungsi bantu: statistik teks, baca angka, kelompok MiniLM, menjalankan Laya, catatan.
Outlier + alasannya ada di outliers.py.
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
NOTE_COLUMNS = ["keputusan", "label_baru", "catatan"]


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


def simple_keys(texts):
    # kunci sederhana: huruf kecil, hanya a-z dan 0-9 ("Halo, Dunia!" sama dengan "halo dunia")
    keys = texts.str.lower()
    return keys.str.replace(r"[^a-z0-9]+", "", regex=True)


def text_stats(texts, keys=None):
    # keys = simple_keys(texts) yang sudah dihitung (context.py menyimpannya di cache)
    if keys is None:
        keys = simple_keys(texts)
    stats = pd.DataFrame(index=texts.index)
    stats["panjang"] = texts.str.len()
    stats["jumlah_kata"] = texts.str.split().str.len()
    letters = texts.str.count(r"[A-Za-z]")
    stats["rasio_bukan_huruf"] = 1 - letters / stats["panjang"].clip(lower=1)
    stats["duplikat"] = texts.duplicated(keep=False)
    stats["duplikat_mirip"] = keys.duplicated(keep=False) & ~stats["duplikat"]
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


def similar_pairs(vectors, rows, threshold, max_pairs=3000):
    # pasangan teks yang maknanya hampir sama walau kata-katanya beda
    rows = np.asarray(rows)
    pairs = []
    for start in range(0, len(vectors), 1000):
        similarity = vectors[start:start + 1000] @ vectors.T
        first, second = np.nonzero(similarity >= threshold)
        for i, other in zip(first, second):
            if start + i < other:
                pairs.append({"row_a": rows[start + i], "row_b": rows[other],
                              "kemiripan": round(float(similarity[i, other]), 3)})
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


def best_other(probabilities, chosen):
    # pilihan dengan peluang terbesar selain yang dipilih Laya
    second = ""
    second_value = -1.0
    for name in probabilities:
        if name != chosen and probabilities[name] > second_value:
            second = name
            second_value = probabilities[name]
    return second, second_value


def read_choice(answer):
    probabilities = answer["probabilities"]
    chosen = answer["choice"]
    second, second_value = best_other(probabilities, chosen)
    return {"answer": chosen, "value": answer["answer_confidence"],
            "first_value": probabilities[chosen], "second": second, "second_value": second_value}


def read_score(answer, lens):
    # skor skala dibulatkan ke tingkat terdekat; pilihan kedua = tingkat lain yang paling mungkin
    level = int(round(answer["score"]))
    probabilities = answer["probabilities"]
    second, second_value = best_other(probabilities, str(level))
    return {"answer": f"{level} {lens['criteria'][level]}", "value": answer["score"],
            "first_value": probabilities[str(level)],
            "second": f"{second} {lens['criteria'][int(second)]}", "second_value": second_value}


def read_noul(answer):
    # ya/tidak: value = peluang "ya"; pilihan kedua = jawaban yang berlawanan
    yes = answer["noul"]
    if yes >= 0.5:
        return {"answer": "ya", "value": yes, "first_value": yes, "second": "tidak",
                "second_value": 1 - yes}
    return {"answer": "tidak", "value": yes, "first_value": 1 - yes, "second": "ya",
            "second_value": yes}


def read_answer(answer, lens):
    # satu jawaban Laya -> satu baris file: jawaban, nilai, yakin, pilihan pertama & kedua
    if answer["type"] == "choice":
        line = read_choice(answer)
    elif answer["type"] == "score":
        line = read_score(answer, lens)
    else:
        line = read_noul(answer)
    line["confidence"] = answer["answer_confidence"]
    return line


NEW_LENS_COLUMNS = ["first_value", "second", "second_value"]


def upgrade_old_file(path):
    # file lama (sebelum ada pilihan kedua) diberi kolom kosong supaya baris baru bisa ditambahkan
    old_columns = list(pd.read_csv(path, nrows=0).columns)
    if "second" in old_columns:
        return
    old = pd.read_csv(path)
    for column in NEW_LENS_COLUMNS:
        old[column] = np.nan
    old.to_csv(path, index=False)


def answer_chunk(judge, data, chunk, columns, label_column, name, lens):
    # Laya menjawab satu potongan baris -> tabel siap ditulis ke file
    question = {"type": lens["type"], "instructions": lens["instructions"]}
    if lens["type"] != "noul":
        question["criteria"] = lens["criteria"]
    states = make_states(data, chunk, columns, label_column)
    results = judge.predict_batch(states, {name: question}, batch_size=BATCH_SIZE)
    lines = []
    for row, result in zip(chunk, results):
        line = read_answer(result["answers"][name], lens)
        line["row"] = row
        lines.append(line)
    return pd.DataFrame(lines)[["row", "answer", "value", "confidence"] + NEW_LENS_COLUMNS]


def run_lens(judge, data, rows, columns, label_column, name, lens, folder, bar):
    path = lens_file(folder, name)
    if os.path.exists(path):
        upgrade_old_file(path)
    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows[start:start + CHUNK_ROWS]
        answers = answer_chunk(judge, data, chunk, columns, label_column, name, lens)
        first_write = not os.path.exists(path)
        answers.to_csv(path, mode="a", header=first_write, index=False)
        done = start + len(chunk)
        bar.progress(done / len(rows), text=f"lensa {name}: {done} / {len(rows)} baris")


def done_rows(folder, name):
    path = lens_file(folder, name)
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["row"])


def lens_results(folder, lenses, index):
    # kolom per lensa: <nama>, __yakin, __nilai, __pertama_nilai, __kedua, __kedua_nilai
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
        for column, suffix in [("first_value", "__pertama_nilai"), ("second", "__kedua"),
                               ("second_value", "__kedua_nilai")]:
            table[name + suffix] = np.nan    # file lama tidak punya kolom ini
            if column in result.columns:
                table[name + suffix] = result[column]
    return table


def add_finding(ctx, section, finding, reason, advice, importance):
    # temuan penting satu bagian; importance 0..1 dipakai untuk mengurutkan panel Temuan utama
    ctx["findings"].append({"bagian": section, "temuan": finding, "alasan": reason,
                            "saran": advice, "penting": round(min(max(importance, 0.0), 1.0), 3)})


# ---------------- catatan manual ----------------

def load_notes(folder):
    path = f"{folder}/catatan.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=NOTE_COLUMNS)
    text_types = {"keputusan": str, "label_baru": str, "catatan": str}
    return pd.read_csv(path, index_col="row", dtype=text_types, keep_default_na=False)


def save_notes(folder, notes):
    keep = (notes["keputusan"] != "") | (notes["label_baru"] != "") | (notes["catatan"] != "")
    notes = notes[keep]
    notes.to_csv(f"{folder}/catatan.csv", index_label="row")
