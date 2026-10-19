# %% [markdown]
# # Teks — Retrieval (cari dokumen paling relevan untuk tiap query)
# **Kapan dipakai:** soal memberi kumpulan dokumen (korpus) dan pertanyaan/query, lalu
# meminta K dokumen paling relevan per query.
#
# **Langkah:** beri skor tiap pasangan (query, dokumen) → urutkan → ambil K teratas →
# evaluasi P@10, MRR, nDCG@10.
#
# **Data demo:** news4. Korpus = train, query = sampel test berlabel. Dokumen dianggap relevan
# bila labelnya sama dengan label query.
#
# **Saat lomba:** file kunci (`news4_test_labels.csv`) tidak ada. Hapus baris yang membacanya
# dan sel Evaluasi.

# %%
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/classification_en"   # GANTI saat lomba
CORPUS_CSV = f"{DATA_FOLDER}/news4_train.csv"
QUERY_CSV = f"{DATA_FOLDER}/news4_test.csv"
QUERY_KEY_CSV = f"{DATA_FOLDER}/news4_test_labels.csv"   # hanya ada di data latihan
EMBEDDING_MODEL = "../../pretrained/minilm"

TEXT_COL = "text"
LABEL_COL = "label"
ID_COL = "id"
TOP_K = 10
N_QUERIES = 300          # demo: pakai 300 query saja supaya cepat
SEED = 42

# %% [markdown]
# ## Baca data

# %%
corpus = pd.read_csv(CORPUS_CSV)
queries = pd.read_csv(QUERY_CSV)
queries[LABEL_COL] = pd.read_csv(QUERY_KEY_CSV)[LABEL_COL]   # urutan baris sama
queries = queries.sample(N_QUERIES, random_state=SEED)
corpus[TEXT_COL] = corpus[TEXT_COL].fillna("")
queries[TEXT_COL] = queries[TEXT_COL].fillna("")
print("korpus:", corpus.shape, "| query:", queries.shape)
print(corpus[LABEL_COL].value_counts())
corpus.head()

# %% [markdown]
# ## Model: skor relevansi
# Hasilnya `scores`: baris = query, kolom = dokumen korpus.
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.
#
# - BM25: pencocokan kata, klasik dan kuat.
# - TF-IDF cosine (kata / huruf): sederhana; versi huruf tahan salah ketik.
# - Embedding: memahami makna (sinonim), butuh model pretrained.

# %%
corpus_words = []
for text in corpus[TEXT_COL]:
    corpus_words.append(re.findall(r"\w+", text.lower()))
bm25 = BM25Okapi(corpus_words)
score_rows = []
for text in queries[TEXT_COL]:
    score_rows.append(bm25.get_scores(re.findall(r"\w+", text.lower())))
scores = np.vstack(score_rows)

# vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
# vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
# vectorizer.fit(corpus[TEXT_COL])
# scores = cosine_similarity(vectorizer.transform(queries[TEXT_COL]),
#                            vectorizer.transform(corpus[TEXT_COL]))

# encoder = SentenceTransformer(EMBEDDING_MODEL)
# query_vectors = encoder.encode(list(queries[TEXT_COL]), normalize_embeddings=True)
# corpus_vectors = encoder.encode(list(corpus[TEXT_COL]), normalize_embeddings=True)
# scores = query_vectors @ corpus_vectors.T

top_documents = np.argsort(-scores, axis=1)[:, :TOP_K]   # posisi K dokumen teratas per query
print(scores.shape)

# %% [markdown]
# ## Evaluasi (memakai label, hanya untuk latihan)
# - P@10: persen dokumen relevan di 10 teratas.
# - MRR: rata-rata 1 / posisi dokumen relevan pertama.
# - nDCG@10: dokumen relevan di posisi atas diberi nilai lebih besar.

# %%
corpus_labels = corpus[LABEL_COL].to_numpy()
discount = 1 / np.log2(np.arange(2, TOP_K + 2))   # bobot posisi 1, 2, 3, ...

precisions = []
reciprocal_ranks = []
ndcgs = []
for row, query_label in enumerate(queries[LABEL_COL]):
    relevant = (corpus_labels[top_documents[row]] == query_label).astype(float)
    precisions.append(relevant.mean())
    reciprocal_ranks.append(0)
    if relevant.any():
        reciprocal_ranks[-1] = 1 / (np.argmax(relevant) + 1)
    n_relevant = min(TOP_K, (corpus_labels == query_label).sum())
    ndcgs.append((relevant * discount).sum() / discount[:n_relevant].sum())

print("P@10:", round(np.mean(precisions), 4))
print("MRR:", round(np.mean(reciprocal_ranks), 4))
print("nDCG@10:", round(np.mean(ndcgs), 4))

# %% [markdown]
# ## Hasil: K dokumen teratas per query
# Id dokumen dipisah spasi, urut dari paling relevan.

# %%
corpus_ids = corpus[ID_COL].astype(str).to_numpy()
top_id_text = []
for row in top_documents:
    top_id_text.append(" ".join(corpus_ids[row]))

result = pd.DataFrame()
result["query_id"] = queries[ID_COL].to_numpy()
result["top10_id"] = top_id_text
result.to_csv("hasil.csv", index=False)
result.head()
