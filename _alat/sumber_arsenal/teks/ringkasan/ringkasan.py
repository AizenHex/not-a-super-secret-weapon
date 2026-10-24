# %% [markdown]
# # Ringkasan teks ekstraktif: pilih kalimat terpenting dari artikel + nilai ROUGE
#
# **Butuh:** `articles` (kolom teks panjang). Opsional: ringkasan contoh untuk menilai.
# **Menghasilkan:** kolom `summary`, skor ROUGE-1/2/L, `submission.csv`.
#
# Ekstraktif = ringkasan disusun dari kalimat asli (tanpa model generatif, cepat, offline).
# Acuan: IndoLEM memakai LEAD-3 (3 kalimat pertama) sebagai baseline IndoSum, dan baseline ini
# sulit dikalahkan untuk berita (inti berita ada di awal). TextRank di IndoSum: ROUGE-1 ±0,48.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | `split_sentences` | **ya** |
# | 2 | `rouge_n`, `rouge_l` (ROUGE ditulis sendiri) | untuk menilai |
# | 3 | LEAD-k: k kalimat pertama | **baseline wajib** |
# | 4 | TextRank (graf kemiripan kalimat, PageRank) | coba |
# | 5 | embedding: kalimat paling dekat ke makna artikel | coba |
# | 6 | bandingkan + `submission.csv` | **ya** |
#
# Ringkasan abstraktif (menulis kalimat baru) butuh model generatif; `pretrained/qwen_05b`
# terlalu kecil untuk ringkasan bahasa Indonesia yang baik → ekstraktif lebih aman.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Artikel berita + ringkasan buatan manusia (kunci).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

articles = pd.read_csv(f"{DEMO_FOLDER}/text/generasi/test.csv")
answers = pd.read_csv(f"{DEMO_FOLDER}/text/generasi/test_labels.csv")
print(articles.shape, answers.shape)
print(answers["ringkasan"].iloc[0][:200])

# %% [markdown]
# ## 1. `split_sentences(text)` → list kalimat
# Pisah di `.`/`!`/`?` yang diikuti spasi + huruf besar/angka/kutip. Singkatan umum
# ("dr.", "No.") dilindungi dulu supaya tidak terpotong. Dateline berita di awal artikel
# ("Jakarta, CNN Indonesia - -") dibuang karena tidak pernah ada di ringkasan.

# %%
import re

TEXT = "artikel"   # UBAH
ABBREVIATIONS = ["dr", "drs", "ir", "prof", "no", "jl", "kab", "kec", "tbk", "st", "sp"]  # UBAH
DATELINE = r"^[A-Z][\w .]{1,30}, [\w .]{1,30}? -+ (- )?"   # "Jakarta, CNN Indonesia - - "


def split_sentences(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = re.sub(DATELINE, "", text)
    for short in ABBREVIATIONS:
        text = re.sub(rf"\b({short})\. ", r"\1<titik> ", text, flags=re.IGNORECASE)
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(])", text)
    sentences = []
    for piece in pieces:
        piece = piece.replace("<titik>", ".").strip()
        if len(piece.split()) >= 3:
            sentences.append(piece)
    return sentences


articles["sentences"] = articles[TEXT].apply(split_sentences)
sentence_counts = articles["sentences"].str.len()
print(sentence_counts.describe().round(1))
print(articles["sentences"].iloc[0][:3])

# %% [markdown]
# ## 2. ROUGE ditulis sendiri: `rouge_n` (kata/2 kata sama) dan `rouge_l` (urutan terpanjang)
# Skor F1 antara ringkasan tebakan dan ringkasan kunci, 0..1.

# %%
import re


def words_of(text):
    return re.findall(r"\w+", str(text).lower())


def overlap_f1(guess_units, truth_units):
    shared = 0
    for unit in set(guess_units):
        shared += min(guess_units.count(unit), truth_units.count(unit))
    if shared == 0:
        return 0.0
    precision, recall = shared / len(guess_units), shared / len(truth_units)
    return 2 * precision * recall / (precision + recall)


def rouge_n(guess, truth, n):
    guess_words, truth_words = words_of(guess), words_of(truth)
    guess_units, truth_units = [], []
    for i in range(len(guess_words) - n + 1):
        guess_units.append(tuple(guess_words[i:i + n]))
    for i in range(len(truth_words) - n + 1):
        truth_units.append(tuple(truth_words[i:i + n]))
    return overlap_f1(guess_units, truth_units)


print(round(rouge_n("adele belum boleh bicara", "adele belum boleh berbicara", 1), 3))

# %% [markdown]
# `rouge_l`: panjang urutan kata sama terpanjang (LCS, pemrograman dinamis) → F1.

# %%
import numpy as np


def rouge_l(guess, truth):
    guess_words, truth_words = words_of(guess), words_of(truth)
    if not guess_words or not truth_words:
        return 0.0
    table = np.zeros((len(guess_words) + 1, len(truth_words) + 1), dtype=int)
    for row in range(1, len(guess_words) + 1):
        for column in range(1, len(truth_words) + 1):
            if guess_words[row - 1] == truth_words[column - 1]:
                table[row, column] = table[row - 1, column - 1] + 1
            else:
                table[row, column] = max(table[row - 1, column], table[row, column - 1])
    common = table[-1, -1]
    if common == 0:
        return 0.0
    precision, recall = common / len(guess_words), common / len(truth_words)
    return 2 * precision * recall / (precision + recall)


print(round(rouge_l("adele belum boleh bicara lagi", "adele belum boleh berbicara"), 3))

# %% [markdown]
# `rouge_report(summaries, references)` → rata-rata ROUGE-1/2/L.

# %%
import numpy as np


def rouge_report(summaries, references):
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for guess, truth in zip(summaries, references):
        scores["rouge1"].append(rouge_n(guess, truth, 1))
        scores["rouge2"].append(rouge_n(guess, truth, 2))
        scores["rougeL"].append(rouge_l(guess, truth))
    result = {}
    for name, values in scores.items():
        result[name] = round(float(np.mean(values)), 4)
    return result


print(rouge_report(["adele belum boleh bicara"], ["adele belum boleh berbicara"]))

# %% [markdown]
# ## 3. LEAD-k: k kalimat pertama
# `SUMMARY_SENTENCES` ≈ rata-rata jumlah kalimat ringkasan contoh (lihat di bawah).

# %%
SUMMARY_SENTENCES = 3   # UBAH

reference_lengths = answers["ringkasan"].apply(split_sentences).str.len()
print("kalimat per ringkasan kunci (median):", reference_lengths.median())
lead_summaries = []
for sentences in articles["sentences"]:
    lead_summaries.append(" ".join(sentences[:SUMMARY_SENTENCES]))
print("LEAD:", rouge_report(lead_summaries, answers["ringkasan"]))

# %% [markdown]
# ## 4. TextRank: kalimat yang paling "mirip banyak kalimat lain" = terpenting
# Graf: simpul = kalimat, bobot sisi = cosine TF-IDF. PageRank dihitung manual (iterasi).
# Kalimat terpilih dikembalikan ke urutan aslinya.

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def textrank_summary(sentences, count, damping=0.85, steps=50):
    if len(sentences) <= count:
        return " ".join(sentences)
    vectors = TfidfVectorizer().fit_transform(sentences)
    similarity = (vectors @ vectors.T).toarray()
    np.fill_diagonal(similarity, 0)
    row_sums = similarity.sum(axis=1, keepdims=True)
    transition = similarity / np.where(row_sums == 0, 1, row_sums)
    score = np.ones(len(sentences)) / len(sentences)
    for _ in range(steps):
        score = (1 - damping) / len(sentences) + damping * transition.T @ score
    chosen = sorted(np.argsort(-score)[:count])
    picked = []
    for i in chosen:
        picked.append(sentences[i])
    return " ".join(picked)


textrank_summaries = []
for sentences in articles["sentences"]:
    textrank_summaries.append(textrank_summary(sentences, SUMMARY_SENTENCES))
print("TextRank:", rouge_report(textrank_summaries, answers["ringkasan"]))

# %% [markdown]
# ## 5. Embedding: kalimat paling dekat ke vektor rata-rata artikel (+ bonus posisi awal)
# `POSITION_BONUS` menambah skor kalimat awal (berita: inti di depan). 0 = tanpa bonus.

# %%
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

PRETRAINED_FOLDER = "pretrained"   # UBAH
POSITION_BONUS = 0.1               # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)


def embedding_summary(sentences, count):
    if len(sentences) <= count:
        return " ".join(sentences)
    vectors = encoder.encode(sentences, normalize_embeddings=True)
    center = vectors.mean(axis=0)
    score = vectors @ (center / np.linalg.norm(center))
    score = score + POSITION_BONUS / (1 + np.arange(len(sentences)))
    picked = []
    for i in sorted(np.argsort(-score)[:count]):
        picked.append(sentences[i])
    return " ".join(picked)


embedding_summaries = []
for sentences in articles["sentences"]:
    embedding_summaries.append(embedding_summary(sentences, SUMMARY_SENTENCES))
print("embedding:", rouge_report(embedding_summaries, answers["ringkasan"]))

# %% [markdown]
# ## 6. Pilih metode terbaik → `submission.csv`

# %%
import pandas as pd

articles["summary"] = lead_summaries   # UBAH: textrank_summaries / embedding_summaries
submission = pd.DataFrame()
submission["id"] = articles["id"]              # UBAH
submission["ringkasan"] = articles["summary"]  # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head(3).to_string(max_colwidth=80))
