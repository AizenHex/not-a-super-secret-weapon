# %% [markdown]
# # Teks — Kata Kunci dan Ringkasan Ekstraktif (TF-IDF)
# **Kapan dipakai:** soal meminta kata kunci per dokumen, atau ringkasan singkat, tanpa label.
#
# **Langkah:**
# 1. TF-IDF memberi bobot tiap kata per dokumen (kata khas dokumen itu = bobot tinggi).
# 2. Kata kunci = kata dengan bobot tertinggi.
# 3. Ringkasan = kalimat dengan total bobot kata tertinggi (urutan asli dipertahankan).
#
# **Data demo:** news4 (berita berbahasa Inggris).

# %%
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/classification_en"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/news4_train.csv"

TEXT_COL = "text"
ID_COL = "id"
STOP_WORDS = "english"     # teks Indonesia: isi dengan daftar stopword sendiri, atau None
N_KEYWORDS = 5
N_SENTENCES = 2

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
data[TEXT_COL] = data[TEXT_COL].fillna("")
print(data.shape)
data.head()

# %% [markdown]
# ## EDA: panjang dokumen (jumlah kata)

# %%
word_counts = data[TEXT_COL].str.split().str.len()
word_counts.describe()

# %% [markdown]
# ## Bobot TF-IDF tiap kata per dokumen
# `min_df=2` = kata harus muncul di ≥ 2 dokumen, `max_df=0.5` = kata yang ada di > 50% dokumen
# dibuang (terlalu umum).

# %%
vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, min_df=2, max_df=0.5, sublinear_tf=True)
X = vectorizer.fit_transform(data[TEXT_COL])
vocabulary = vectorizer.get_feature_names_out()
print("dokumen x kata:", X.shape)

# %% [markdown]
# ## Fungsi kata kunci
# - `word_weights`: kamus {kata: bobot} untuk satu dokumen.
# - `keywords`: kata dengan bobot terbesar.

# %%
def word_weights(row_number):
    row = X[row_number]
    weights = {}
    for column, weight in zip(row.indices, row.data):
        weights[vocabulary[column]] = weight
    return weights


def keywords(weights):
    ranked = pd.Series(weights).sort_values(ascending=False)
    top_words = list(ranked.index[:N_KEYWORDS])
    return ", ".join(top_words)


# coba pada dokumen pertama
first_weights = word_weights(0)
print("kata kunci:", keywords(first_weights))

# %% [markdown]
# ## Fungsi ringkasan
# Pecah teks per kalimat, jumlahkan bobot kata di tiap kalimat, ambil kalimat terbaik lalu
# susun sesuai urutan aslinya.

# %%
def summarize(text, weights):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scores = []
    for sentence in sentences:
        total = 0
        for word in re.findall(r"\w+", sentence.lower()):
            total += weights.get(word, 0)
        scores.append(total)
    best = np.argsort(scores)[::-1][:N_SENTENCES]
    chosen = []
    for position in sorted(best):
        chosen.append(sentences[position])
    return " ".join(chosen)


# coba pada dokumen pertama
print("ringkasan:", summarize(data.loc[0, TEXT_COL], first_weights))

# %% [markdown]
# ## Jalankan untuk semua dokumen
# Tidak ada kunci jawaban: cek manual beberapa contoh.

# %%
all_keywords = []
all_summaries = []
for i in range(len(data)):
    weights = word_weights(i)
    all_keywords.append(keywords(weights))
    all_summaries.append(summarize(data.loc[i, TEXT_COL], weights))

result = pd.DataFrame()
result["id"] = data[ID_COL]
result["kata_kunci"] = all_keywords
result["ringkasan"] = all_summaries
result.to_csv("hasil.csv", index=False)
result.head()
