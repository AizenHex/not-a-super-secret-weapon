# %% [markdown]
# # Recsys — Content-based + Hybrid (termasuk pengguna baru / cold start)
# **Kapan dipakai:** item punya atribut (genre, kategori, merek, deskripsi) dan banyak
# pengguna/item yang riwayatnya sedikit, sehingga model dari interaksi saja lemah.
#
# **Idenya:**
# - **Content-based:** profil pengguna = atribut item yang ia sukai. Item yang atributnya mirip
#   profil direkomendasikan. Bisa untuk item baru yang belum pernah dirating.
# - **Hybrid:** skor konten digabung skor popularitas (atau SVD). Pengguna baru yang profilnya
#   kosong otomatis jatuh ke bagian popularitas.
#
# **Langkah:**
# 1. Split menurut waktu, susun matriks pengguna x item.
# 2. Skor konten, popularitas, SVD → gabungkan.
# 3. Evaluasi precision@K/recall@K untuk pengguna lama dan baru, simpan rekomendasi.
#
# **Data demo:** MovieLens 100k (`movies.csv` berisi 19 kolom genre 0/1).

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/recsys/ml-100k"   # GANTI saat lomba
RATINGS_CSV = f"{DATA_FOLDER}/ratings.csv"
ITEMS_CSV = f"{DATA_FOLDER}/movies.csv"

USER_COL = "user"
ITEM_COL = "item"
RATING_COL = "rating"
TIME_COL = "timestamp"
NOT_ATTRIBUTE_COLS = ["item", "title", "release_date"]   # kolom item yang bukan atribut
TOP_K = 10
RELEVANT_RATING = 4       # rating >= 4 dianggap disukai
TRAIN_FRACTION = 0.8      # porsi interaksi tertua untuk latih
CONTENT_WEIGHT = 0.5      # bobot skor konten di hybrid (sisanya popularitas/SVD)
NEW_USER_LIMIT = 10       # pengguna dengan <= 10 interaksi latih dianggap "baru"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
ratings = pd.read_csv(RATINGS_CSV)
ratings = ratings.sort_values(TIME_COL)
items = pd.read_csv(ITEMS_CSV)
attribute_cols = items.columns.drop(NOT_ATTRIBUTE_COLS)
print(ratings.shape, items.shape, "| atribut:", len(attribute_cols))
items.head()

# %% [markdown]
# ## Langkah 1 — split menurut waktu + matriks pengguna x item

# %%
cut = int(len(ratings) * TRAIN_FRACTION)
train = ratings.iloc[:cut]
test = ratings.iloc[cut:]
test = test[test[RATING_COL] >= RELEVANT_RATING]
N_USERS = ratings[USER_COL].max() + 1
N_ITEMS = ratings[ITEM_COL].max() + 1

train_matrix = sp.csr_matrix((np.ones(len(train)), (train[USER_COL], train[ITEM_COL])),
                             shape=(N_USERS, N_ITEMS))
test_matrix = sp.csr_matrix((np.ones(len(test)), (test[USER_COL], test[ITEM_COL])),
                            shape=(N_USERS, N_ITEMS))
train_counts = np.asarray(train_matrix.sum(axis=1)).ravel()     # interaksi latih per pengguna
print("interaksi latih:", train_matrix.nnz, "| uji (disukai):", test_matrix.nnz)

# %% [markdown]
# ## Langkah 2a — skor konten
# Bobot tiap rating = rating dikurangi rata-rata rating pengguna itu (suka = positif, tidak
# suka = negatif). Profil pengguna = jumlah atribut item yang ia rating, dikali bobotnya.
# Skor konten = kemiripan profil dengan atribut tiap item.

# %%
item_attributes = np.zeros((N_ITEMS, len(attribute_cols)))
item_attributes[items[ITEM_COL]] = items[attribute_cols].to_numpy()
lengths = np.linalg.norm(item_attributes, axis=1, keepdims=True)
item_attributes = item_attributes / np.maximum(lengths, 1e-9)    # panjang vektor jadi 1

user_mean = train.groupby(USER_COL)[RATING_COL].transform("mean")
weights = train[RATING_COL] - user_mean
weight_matrix = sp.csr_matrix((weights, (train[USER_COL], train[ITEM_COL])),
                              shape=(N_USERS, N_ITEMS))
profiles = weight_matrix @ item_attributes                      # pengguna x atribut
content_scores = profiles @ item_attributes.T                   # pengguna x item
print(content_scores.shape)

# %% [markdown]
# ## Langkah 2b — skor popularitas dan SVD

# %%
popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
popular_scores = np.tile(popularity, (N_USERS, 1)).astype(float)

svd = TruncatedSVD(n_components=32, random_state=SEED)
user_factors = svd.fit_transform(train_matrix)
svd_scores = user_factors @ svd.components_
print(popular_scores.shape, svd_scores.shape)

# %% [markdown]
# ## Langkah 2c — gabungkan skor
# Skor tiap model disamakan skalanya ke 0–1 per pengguna dulu, baru dijumlah dengan bobot.
# Pilih SATU baris `scores = ...`: hapus `#` di depan baris yang ingin dipakai.

# %%
def scale_rows(scores):
    low = scores.min(axis=1, keepdims=True)
    spread = scores.max(axis=1, keepdims=True) - low
    spread[spread == 0] = 1                     # baris yang semuanya sama: hindari bagi nol
    return (scores - low) / spread


# coba pada skor konten 3 pengguna pertama
print(scale_rows(content_scores[:3]).max(axis=1))

scores = (CONTENT_WEIGHT * scale_rows(content_scores)
          + (1 - CONTENT_WEIGHT) * scale_rows(popular_scores))     # konten + popularitas
# scores = (CONTENT_WEIGHT * scale_rows(content_scores)
#           + (1 - CONTENT_WEIGHT) * scale_rows(svd_scores))       # konten + SVD
# scores = content_scores.copy()                                   # konten saja
# scores = popular_scores.copy()                                   # popularitas saja

scores[train_matrix.nonzero()] = -np.inf          # item yang sudah dilihat tidak direkomendasikan
top_items = np.argsort(-scores, axis=1)[:, :TOP_K]

# %% [markdown]
# ## Langkah 3a — evaluasi pengguna lama vs pengguna baru

# %%
test_dense = test_matrix.toarray()
hits = np.take_along_axis(test_dense, top_items, axis=1).sum(axis=1)
relevant = test_dense.sum(axis=1)
has_test = relevant > 0
is_new = train_counts <= NEW_USER_LIMIT

for group_name in ["semua", "lama", "baru"]:
    group = has_test
    if group_name == "lama":
        group = has_test & ~is_new
    if group_name == "baru":
        group = has_test & is_new
    precision = (hits[group] / TOP_K).mean()
    recall = (hits[group] / relevant[group]).mean()
    print(f"{group_name:6s} n={group.sum():4d} | precision@{TOP_K} {precision:.4f} | "
          f"recall@{TOP_K} {recall:.4f}")

# %% [markdown]
# ## Langkah 3b — simpan rekomendasi (satu baris per pengguna-item)
# Demo: pengguna yang punya data uji. Saat lomba: daftar pengguna dari file test.

# %%
titles = pd.Series(items["title"].to_numpy(), index=items[ITEM_COL])
target_users = np.unique(test[USER_COL])
recommended = top_items[target_users].ravel()

submission = pd.DataFrame()
submission["user"] = np.repeat(target_users, TOP_K)
submission["rank"] = np.tile(np.arange(1, TOP_K + 1), len(target_users))
submission["item"] = recommended
submission["title"] = submission["item"].map(titles)
submission.to_csv("submission.csv", index=False)
submission.head(TOP_K)
