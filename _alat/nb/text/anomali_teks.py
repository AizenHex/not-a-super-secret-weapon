# %% [markdown]
# # Teks — Anomali (teks aneh / spam / di luar topik, tanpa label)
# **Kapan dipakai:** sebagian teks "tidak pada tempatnya" (spam, bahasa lain, acak, topik lain)
# dan soal meminta menandai atau mengurutkan teks paling janggal, tanpa label.
#
# **Cara kerja:** embedding MiniLM multibahasa → skor kejanggalan = rata-rata jarak cosine ke
# k tetangga terdekat (kNN). Teks biasa punya banyak tetangga dekat; teks aneh sendirian.
# Alternatif (baris `#`): IsolationForest, LocalOutlierFactor.
#
# **Data demo:** 1.000 ulasan/tweet SmSA + 90 teks asing yang disisipkan: 40 abstrak arXiv
# (bahasa Inggris), 10 spam buatan (SINTETIS), 40 paragraf berita/ensiklopedia (topik lain).
# Kunci jenis teks hanya dipakai untuk mengukur AUC.

# %%
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text"   # GANTI saat lomba
NORMAL_CSV = f"{DATA_FOLDER}/sentiment/smsa_train.csv"
ARXIV_CSV = f"{DATA_FOLDER}/arxiv/arxiv_train.csv"
NEWS_CSV = f"{DATA_FOLDER}/qa/facqa_train.csv"
EMBEDDING_MODEL = "Z:/GEMASTIK-DATABASE/pretrained/minilm"

TEXT_COL = "text"
N_NORMAL = 1000
N_FOREIGN = 40        # per jenis (bahasa lain, topik lain), hanya untuk demo
N_SPAM = 10           # spam saling mirip: kalau banyak, kNN menganggapnya "ramai" = normal
N_NEIGHBORS = 50      # harus lebih besar dari kelompok anomali yang saling mirip (spam)
ANOMALY_SHARE = 0.08  # perkiraan bagian teks anomali, menentukan ambang label
SEED = 42

# %% [markdown]
# ## Siapkan data demo (HAPUS sel ini saat lomba, ganti dengan `pd.read_csv` data soal)
# Hasil: tabel `data` dengan kolom `text`, dan `jenis` (kunci jawaban, bukan input).

# %%
normal = pd.read_csv(NORMAL_CSV)
normal = normal.sample(N_NORMAL, random_state=SEED)
arxiv = pd.read_csv(ARXIV_CSV)
arxiv = arxiv.sample(N_FOREIGN, random_state=SEED)
news = pd.read_csv(NEWS_CSV)
news = news.drop_duplicates("konteks")
news = news.sample(N_FOREIGN, random_state=SEED)

random_generator = np.random.default_rng(SEED)
spam = []
for i in range(N_SPAM):
    code = "".join(random_generator.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 6))
    prize = str(random_generator.integers(10, 99))
    phone = str(random_generator.integers(10**9, 10**10))
    spam.append("SELAMAT anda menang hadiah " + prize + " JUTA!!! klik http://bit.ly/" + code
                + " sekarang juga, hubungi 08" + phone)

data = pd.DataFrame()
data["text"] = list(normal["text"]) + list(arxiv["text"].str[:400]) + spam + list(news["konteks"])
data["jenis"] = (["normal"] * N_NORMAL + ["bahasa_lain"] * N_FOREIGN + ["spam"] * N_SPAM
                 + ["topik_lain"] * N_FOREIGN)
data = data.sample(frac=1, random_state=SEED)
data = data.reset_index(drop=True)
data["id"] = range(1, len(data) + 1)
print(data.shape)
print(data["jenis"].value_counts())

# %% [markdown]
# ## Embedding MiniLM
# `normalize_embeddings=True` = panjang tiap vektor 1, sehingga cosine mudah dihitung.

# %%
encoder = SentenceTransformer(EMBEDDING_MODEL)
embeddings = encoder.encode(list(data[TEXT_COL]), batch_size=64, normalize_embeddings=True)
print(embeddings.shape)

# %% [markdown]
# ## Model: skor kejanggalan
# kNN: rata-rata jarak ke k tetangga terdekat. Tetangga pertama adalah teks itu sendiri
# (jarak 0), jadi minta k+1 lalu buang kolom pertama.
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.

# %%
neighbor_finder = NearestNeighbors(n_neighbors=N_NEIGHBORS + 1, metric="cosine")
neighbor_finder.fit(embeddings)
distances, _ = neighbor_finder.kneighbors(embeddings)
scores = distances[:, 1:].mean(axis=1)

# IsolationForest (score_samples kecil = janggal, jadi dibalik dengan tanda minus)
# forest = IsolationForest(n_estimators=300, random_state=SEED)
# forest.fit(embeddings)
# scores = -forest.score_samples(embeddings)

# LocalOutlierFactor (membandingkan kepadatan dengan tetangga sendiri)
# lof = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, metric="cosine")
# lof.fit(embeddings)
# scores = -lof.negative_outlier_factor_

data["skor"] = scores
threshold = np.quantile(scores, 1 - ANOMALY_SHARE)
data["anomali"] = (scores > threshold).astype(int)
print("ambang skor:", round(threshold, 3), "| ditandai anomali:", data["anomali"].sum())

# %% [markdown]
# ## Teks paling janggal
# Cek manual: apakah 15 teratas memang aneh?

# %%
pd.set_option("display.max_colwidth", 90)
most_strange = data.sort_values("skor", ascending=False)
most_strange[["skor", "jenis", "text"]].head(15)

# %% [markdown]
# ## Ukur dengan kunci jawaban (HAPUS saat lomba)
# AUC 1 = semua teks asing di atas semua teks normal, 0,5 = tebakan acak.

# %%
is_foreign = (data["jenis"] != "normal").astype(int)
print("AUC semua jenis asing:", round(roc_auc_score(is_foreign, scores), 3))
for kind in ["bahasa_lain", "spam", "topik_lain"]:
    rows = data["jenis"].isin(["normal", kind])
    auc = roc_auc_score(is_foreign[rows], scores[rows])
    print("AUC normal vs", kind, ":", round(auc, 3))
flagged_share = data.groupby("jenis")["anomali"].mean()
print(flagged_share.round(2))

# %% [markdown]
# ## Simpan hasil

# %%
submission = pd.DataFrame()
submission["id"] = data["id"]
submission["skor_anomali"] = data["skor"]
submission["anomali"] = data["anomali"]
submission.to_csv("submission.csv", index=False)
submission.head()
