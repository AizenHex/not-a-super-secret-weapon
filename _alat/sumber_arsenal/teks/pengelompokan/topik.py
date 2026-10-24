# %% [markdown]
# # Topik teks tanpa label: kelompokkan teks, beri kata kunci per topik, lihat tren waktu
#
# **Butuh:** `texts` (kolom teks, opsional kolom tanggal). **Menghasilkan:** `texts["topic"]`,
# tabel kata kunci per topik, contoh teks per topik, tren topik per bulan, `outputs/topik.csv`.
#
# Contoh soal: "temukan topik keluhan pelanggan", "isu apa yang naik bulan ini", EDA teks
# sebelum melabeli sendiri (beri nama per topik → label awal).
#
# | cara | sel | catatan |
# |---|---|---|
# | NMF di atas TF-IDF | 2 | cepat, jumlah topik ditentukan sendiri, baseline |
# | gaya BERTopic: embedding → UMAP → HDBSCAN → c-TF-IDF | 3–5 | jumlah topik dicari sendiri |
#
# Topik `-1` pada cara BERTopic = teks yang tidak masuk kelompok mana pun.
#
# Studi pembanding (berita, tweet, teks pendek Hindi) umumnya menemukan BERTopic memberi topik
# lebih koheren daripada NMF/LDA; NMF tetap baseline yang layak dan jauh lebih cepat.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `clean_text` + stopword | **ya** |
# | 2 | NMF → `texts["topic_nmf"]` + kata teratas | baseline |
# | 3 | embedding + UMAP + HDBSCAN → `texts["topic"]` | **ya** |
# | 4 | `topic_keywords` (c-TF-IDF) → kata kunci per topik | **ya** |
# | 5 | contoh teks paling khas per topik | **ya** |
# | 6 | tren topik per bulan | kalau ada tanggal |
# | 7 | simpan + beri nama topik | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Tweet bertanggal (Inggris) dengan label topik; label hanya untuk menilai di akhir.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

texts = pd.read_csv(f"{DEMO_FOLDER}/text/topik_waktu/tweet_topic.csv")
texts["label_first"] = texts["label_name"].fillna("kosong")
texts["label_first"] = texts["label_first"].str.split("|").str[0]
print(texts.shape)
print(texts["label_first"].value_counts().head(8))

# %% [markdown]
# ## 1. Konfigurasi + `clean_text` + `STOPWORDS`
# Kata umum ("yang", "dan", "the") harus dibuang supaya kata kunci topik bermakna.
# Teks Indonesia: pakai daftar Sastrawi (baris `#`) + tambah kata sendiri.

# %%
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TEXT = "text"   # UBAH
DATE = "date"   # UBAH: None kalau tidak ada tanggal
STOPWORDS = set(ENGLISH_STOP_WORDS) | {"amp", "rt", "username", "just", "like"}   # UBAH
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | {"yg", "aja", "nya"}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|\{\{?\w+\}?\}|\{@|@\}|@\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = []
    for word in text.split():
        if word not in STOPWORDS and len(word) > 2:
            words.append(word)
    return " ".join(words)


texts["clean"] = texts[TEXT].apply(clean_text)
print(texts[[TEXT, "clean"]].head(3).to_string(max_colwidth=70))

# %% [markdown]
# ## 2. NMF di atas TF-IDF → `texts["topic_nmf"]` + 10 kata teratas per topik
# `N_TOPICS` coba beberapa (8, 12, 20), pilih yang kata kuncinya paling "masuk akal".

# %%
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

N_TOPICS = 10   # UBAH
tfidf = TfidfVectorizer(min_df=5, max_df=0.5, ngram_range=(1, 2))
matrix = tfidf.fit_transform(texts["clean"])
nmf = NMF(n_components=N_TOPICS, init="nndsvda", random_state=42, max_iter=400)
weights = nmf.fit_transform(matrix)            # (teks, topik)
texts["topic_nmf"] = weights.argmax(axis=1)
words = np.array(tfidf.get_feature_names_out())
for topic in range(N_TOPICS):
    top = np.argsort(-nmf.components_[topic])[:10]
    print(f"topik {topic} ({(texts['topic_nmf'] == topic).sum()} teks):", ", ".join(words[top]))

# %% [markdown]
# ## 3. Gaya BERTopic: embedding → UMAP 5 dimensi → HDBSCAN → `texts["topic"]`
# Embedding pakai teks ASLI (bukan `clean`). UMAP meringkas 384 angka jadi 5 supaya HDBSCAN
# bisa menemukan kelompok padat. `MIN_TOPIC_SIZE` besar → topik lebih sedikit & lebar.
# Topik `-1` = teks yang tidak masuk kelompok mana pun (bukan error).

# %%
import hdbscan
import torch
import umap
from sentence_transformers import SentenceTransformer

PRETRAINED_FOLDER = "pretrained"   # UBAH
MIN_TOPIC_SIZE = 40                # UBAH: 15–100 tergantung jumlah teks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)
vectors = encoder.encode(texts[TEXT].tolist(), batch_size=64, normalize_embeddings=True,
                         show_progress_bar=True)
reducer = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                    random_state=42)
reduced = reducer.fit_transform(vectors)
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_TOPIC_SIZE, metric="euclidean",
                            cluster_selection_method="eom")
texts["topic"] = clusterer.fit_predict(reduced)
print(texts["topic"].nunique() - 1, "topik |", (texts["topic"] == -1).mean().round(3),
      "porsi teks tanpa topik")

# %% [markdown]
# ## 4. `topic_keywords(texts, column)` → kata kunci per topik (c-TF-IDF)
# Semua teks satu topik digabung jadi satu "dokumen besar", lalu kata yang sering di topik
# itu tapi jarang di topik lain mendapat skor tinggi (cara BERTopic, Grootendorst 2022).

# %%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def topic_keywords(table, column, top_words=10):
    joined = table.groupby(column)["clean"].apply(" ".join)
    counter = CountVectorizer(ngram_range=(1, 2), min_df=2)
    counts = counter.fit_transform(joined).toarray()
    counts = counts.astype(float)
    term_share = counts / counts.sum(axis=1, keepdims=True)              # tf per topik
    average_words = counts.sum() / len(counts)
    importance = term_share * np.log(1 + average_words / counts.sum(axis=0))
    words = np.array(counter.get_feature_names_out())
    rows = []
    for row, topic in enumerate(joined.index):
        top = np.argsort(-importance[row])[:top_words]
        rows.append({"topic": topic, "size": int((table[column] == topic).sum()),
                     "keywords": ", ".join(words[top])})
    return pd.DataFrame(rows)


keywords = topic_keywords(texts, "topic")
print(keywords.to_string(max_colwidth=90))

# %% [markdown]
# ## 5. Contoh teks paling khas per topik (paling dekat ke pusat topik)

# %%
import numpy as np

EXAMPLES = 3   # UBAH
for topic in sorted(texts["topic"].unique()):
    if topic == -1:
        continue
    members = np.where(texts["topic"] == topic)[0]
    center = vectors[members].mean(axis=0)
    closest = members[np.argsort(-(vectors[members] @ center))[:EXAMPLES]]
    print(f"\n== topik {topic}:", keywords.loc[keywords["topic"] == topic, "keywords"].iloc[0][:60])
    for i in closest:
        print("  -", texts[TEXT].iloc[i][:110])

# %% [markdown]
# ## 6. Tren topik per bulan (porsi teks tiap topik per bulan)
# Topik yang porsinya melonjak = isu yang sedang naik.

# %%
import matplotlib.pyplot as plt
import pandas as pd

texts["month"] = pd.to_datetime(texts[DATE]).dt.to_period("M")
monthly = pd.crosstab(texts["month"], texts["topic"], normalize="index")
monthly = monthly.drop(columns=-1, errors="ignore")
biggest = texts.loc[texts["topic"] != -1, "topic"].value_counts().head(6).index   # UBAH
fig, ax = plt.subplots(figsize=(10, 4))
for topic in biggest:
    ax.plot(monthly.index.astype(str), monthly[topic], marker="o", label=f"topik {topic}")
ax.set_ylabel("porsi teks")
ax.legend(fontsize=8)
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ## 7. Beri nama topik + simpan `outputs/topik.csv`
# Isi `TOPIC_NAMES` setelah membaca sel 4–5. Banyak teks `-1`? Turunkan `MIN_TOPIC_SIZE`,
# atau masukkan ke topik terdekat: `hdbscan.approximate_predict` / topik NMF.

# %%
import os

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

TOPIC_NAMES = {}   # UBAH: {0: "olahraga", 1: "musik", ...}
os.makedirs("outputs", exist_ok=True)
texts["topic_name"] = texts["topic"].map(TOPIC_NAMES)
texts["topic_name"] = texts["topic_name"].fillna("lainnya")
texts.to_csv("outputs/topik.csv", index=False)
keywords.to_csv("outputs/topik_kata_kunci.csv", index=False)
for column in ["topic_nmf", "topic"]:      # hanya demo: dibandingkan dengan label asli
    print(column, "| NMI:", round(normalized_mutual_info_score(texts["label_first"],
                                                                texts[column]), 4),
          "| ARI:", round(adjusted_rand_score(texts["label_first"], texts[column]), 4))
