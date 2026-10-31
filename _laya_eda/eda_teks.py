"""EDA teks: kategori tidak konsisten, konflik label, n-gram per kelas, kosakata, kata alay,
campuran bahasa, dan peta 2D makna. Semua fungsi murni (tanpa streamlit)."""
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples

import outliers

SLANG_FOLDER = "../../data_latihan/preprocessing"    # kamus alay yang sudah ada di toolkit
SLANG_FILES = ["kamus_alay_salsabila.csv", "kamus_slang.csv"]
SIMILAR_RATIO = 90        # rapidfuzz ratio >= ini = nilai dianggap sama dengan salah ketik
MAX_DISTINCT = 1500       # kolom dengan nilai unik lebih banyak dari ini tidak dicek kemiripannya
SHORT_TEXT = 40           # kolom "teks pendek" = rata-rata panjang isi <= ini
MAX_CLASSES = 30          # n-gram dihitung untuk paling banyak sekian kelas terbesar
ENGLISH_WORDS = {"the", "and", "is", "are", "you", "your", "my", "this", "that", "with", "for",
                 "not", "have", "has", "was", "were", "be", "but", "just", "like", "love", "so",
                 "it", "of", "to", "in", "on", "at", "what", "when", "how", "who", "will", "can",
                 "we", "he", "she", "they", "from", "all", "out", "up", "get", "good", "very"}
INDONESIAN_WORDS = {"yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "tidak",
                    "ada", "akan", "juga", "pada", "saya", "aku", "kamu", "kita", "sudah",
                    "belum", "bisa", "atau", "karena", "jadi", "kalau", "tapi", "banget", "aja",
                    "gak", "nggak", "udah", "dong", "sih", "lagi", "apa", "bukan", "mau", "dia",
                    "mereka", "kami", "yg", "dgn", "banyak", "lebih", "sangat"}


def words_of(text):
    return re.findall(r"\w+", text.lower())


# ---------------- kategori tidak konsisten ----------------

def short_text_columns(data):
    # kolom teks pendek: bertipe object, isi rata-rata pendek, nilai uniknya tidak terlalu banyak
    names = []
    for column in data.columns:
        if data[column].dtype != object:
            continue
        present = data[column].dropna().astype(str)
        lengths = present.str.len()
        if len(lengths) > 0 and lengths.mean() <= SHORT_TEXT:
            if data[column].nunique() <= MAX_DISTINCT:
                names.append(column)
    return names


def key_table(series):
    # per kunci sederhana (huruf kecil tanpa tanda baca): jumlah total dan varian terbanyak dulu
    present = series.dropna().astype(str)
    counts = present.value_counts()
    keys = {}
    for value in counts.index:
        key = outliers.simple_text(value)
        if key == "":
            continue
        if key not in keys:
            keys[key] = {"total": 0, "variants": []}
        keys[key]["total"] = keys[key]["total"] + int(counts[value])
        keys[key]["variants"].append((value, int(counts[value])))    # sudah urut terbanyak
    return keys


def same_key_lines(keys):
    # varian yang berbeda hanya karena huruf besar, spasi, atau tanda baca
    lines = []
    for key in keys:
        variants = keys[key]["variants"]
        for value, count in variants[1:]:
            lines.append({"nilai": value, "jumlah": count, "ganti_ke": variants[0][0],
                          "alasan": "beda hanya huruf besar/spasi/tanda baca"})
    return lines


def differ_only_in_digits(first, second):
    return re.sub(r"\d", "", first) == re.sub(r"\d", "", second)


def similar_key_lines(keys):
    # kunci yang mirip (rapidfuzz ratio >= SIMILAR_RATIO) digabung ke kunci yang lebih sering muncul
    totals = {}
    for key in keys:
        totals[key] = keys[key]["total"]
    names = list(pd.Series(totals).sort_values(ascending=False).index[:MAX_DISTINCT])
    scores = process.cdist(names, names, scorer=fuzz.ratio, score_cutoff=SIMILAR_RATIO)
    taken = set()
    lines = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if names[first] in taken or names[second] in taken or scores[first, second] == 0:
                continue
            if differ_only_in_digits(names[first], names[second]):
                continue
            taken.add(names[second])
            lines.append(merge_line(keys[names[first]], keys[names[second]], scores[first, second]))
    return lines


def merge_line(keep, merged, score):
    return {"nilai": merged["variants"][0][0], "jumlah": merged["total"],
            "ganti_ke": keep["variants"][0][0],
            "alasan": f"mirip {score:.0f}% dengan nilai yang lebih sering muncul "
                      "(kemungkinan salah ketik)"}


def inconsistent_values(series):
    # tabel saran koreksi: nilai, jumlah, ganti_ke, alasan
    keys = key_table(series)
    lines = same_key_lines(keys)
    lines = lines + similar_key_lines(keys)
    return pd.DataFrame(lines, columns=["nilai", "jumlah", "ganti_ke", "alasan"])


def correction_map(table):
    mapping = {}
    for value, target in zip(table["nilai"], table["ganti_ke"]):
        mapping[value] = target
    return mapping


# ---------------- konflik label ----------------

def label_conflicts(texts, labels, keys):
    # teks sama (persis atau setelah huruf besar/tanda baca diabaikan) tetapi label berbeda
    # keys = kunci sederhana tiap teks (helpers.simple_keys)
    lines = []
    groups = outliers.groups_by_key(keys)
    number = 0
    for members in groups.values():
        label_set = set(labels[members])
        if len(label_set) < 2:
            continue
        number = number + 1
        exact = texts[members].nunique() == 1
        kind = "sama persis"
        if not exact:
            kind = "sama setelah huruf besar/tanda baca diabaikan"
        for row in members:
            lines.append({"kelompok": number, "jenis": kind, "row": row, "label": labels[row],
                          "teks": texts[row]})
    return pd.DataFrame(lines, columns=["kelompok", "jenis", "row", "label", "teks"])


# ---------------- n-gram, kosakata, kata alay, bahasa ----------------

def class_ngrams(texts, labels, top=10):
    # n-gram (1-2 kata) yang paling khas per kelas: chi-square satu-lawan-semua
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, binary=True, max_features=30000)
    matrix = vectorizer.fit_transform(texts)
    words = vectorizer.get_feature_names_out()
    lines = []
    for label in labels.value_counts().index[:MAX_CLASSES]:
        target = (labels == label).to_numpy()
        scores, _ = chi2(matrix, target)
        inside = np.asarray(matrix[target].mean(axis=0)).ravel()
        outside = np.asarray(matrix[~target].mean(axis=0)).ravel()
        scores = np.nan_to_num(scores)
        scores[inside <= outside] = 0
        for index in scores.argsort()[::-1][:top]:
            lines.append({"kelas": label, "ngram": words[index],
                          "persen_di_kelas": round(100 * inside[index], 1),
                          "persen_di_lain": round(100 * outside[index], 1),
                          "chi2": round(scores[index], 1)})
    return pd.DataFrame(lines)


def vocabulary_stats(texts):
    counter = Counter()
    total_words = 0
    for text in texts:
        words = words_of(text)
        counter.update(words)
        total_words = total_words + len(words)
    hapax = 0
    for word in counter:
        if counter[word] == 1:
            hapax = hapax + 1
    return {"kata_total": total_words, "kosakata": len(counter), "muncul_1x": hapax,
            "persen_muncul_1x": round(100 * hapax / max(len(counter), 1), 1),
            "kata_per_teks": round(total_words / max(len(texts), 1), 1)}


def load_slang():
    # kata alay/slang dari kamus toolkit; None kalau file kamus tidak ada
    words = set()
    for name in SLANG_FILES:
        path = f"{SLANG_FOLDER}/{name}"
        if not os.path.exists(path):
            return None
        table = pd.read_csv(path)
        for slang, standard in zip(table["slang"], table["baku"]):
            if str(slang) != str(standard):
                words.add(str(slang).lower())
    return words


def word_ratios(texts, slang):
    # per teks: rasio kata alay, jumlah kata Inggris, jumlah kata Indonesia
    lines = []
    for text in texts:
        words = words_of(text)
        counts = {"alay": 0, "inggris": 0, "indonesia": 0}
        for word in words:
            if word in slang:
                counts["alay"] = counts["alay"] + 1
            if word in ENGLISH_WORDS:
                counts["inggris"] = counts["inggris"] + 1
            if word in INDONESIAN_WORDS:
                counts["indonesia"] = counts["indonesia"] + 1
        counts["rasio_alay"] = counts["alay"] / max(len(words), 1)
        lines.append(counts)
    table = pd.DataFrame(lines, index=texts.index)
    table["campuran_bahasa"] = (table["inggris"] > 0) & (table["indonesia"] > 0)
    return table


# ---------------- peta 2D makna ----------------

def sample_positions(count, max_points, seed):
    if count <= max_points:
        return np.arange(count)
    picked = np.random.RandomState(seed).choice(count, max_points, replace=False)
    return np.sort(picked)


def tsne_2d(vectors, seed):
    # PCA (<= 50 dimensi) dulu supaya t-SNE cepat, lalu t-SNE ke 2 dimensi di CPU
    components = min(50, vectors.shape[1], len(vectors) - 1)
    pca = PCA(n_components=components, random_state=seed)
    reduced = pca.fit_transform(vectors)
    perplexity = min(30, max(5, len(vectors) // 4))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed)
    return tsne.fit_transform(reduced), float(pca.explained_variance_ratio_.sum())


def separation_table(vectors, labels):
    # silhouette (kosinus) per kelas + kelas terdekat menurut pusat makna
    silhouette = silhouette_samples(vectors, labels, metric="cosine")
    names, centers = outliers.category_centers(vectors, labels)
    similarity = centers @ centers.T
    np.fill_diagonal(similarity, -np.inf)
    lines = []
    for number, name in enumerate(names):
        inside = silhouette[labels == name]
        nearest = similarity[number].argmax()
        lines.append({"kelas": name, "jumlah": len(inside),
                      "silhouette_rata2": round(float(inside.mean()), 3),
                      "persen_lebih_mirip_kelas_lain": round(100 * float((inside < 0).mean()), 1),
                      "kelas_terdekat": names[nearest],
                      "kemiripan_pusat": round(float(similarity[number, nearest]), 3)})
    return pd.DataFrame(lines), float(silhouette.mean())
