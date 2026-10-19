# %% [markdown]
# # Recsys — Rekomendasi Top-N (daftar K item untuk tiap pengguna)
# **Kapan dipakai:** ada riwayat interaksi pengguna–item (beli, tonton, klik, rating) dan
# diminta merekomendasikan K item yang belum pernah dilihat tiap pengguna.
#
# **Idenya:** interaksi disusun jadi matriks pengguna x item (1 = pernah). Model memberi `scores`
# (pengguna x item). Item yang sudah dilihat dibuang, K skor tertinggi = rekomendasi.
#
# **Langkah:**
# 1. Split menurut waktu: interaksi terbaru jadi uji.
# 2. Hitung `scores` dengan satu model.
# 3. Evaluasi precision@K dan recall@K, simpan rekomendasi.
#
# **Data demo:** MovieLens 100k, rating dianggap "pernah menonton". Rating >= 4 di masa uji =
# film yang benar-benar disukai.

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/recsys/ml-100k"   # GANTI saat lomba
RATINGS_CSV = f"{DATA_FOLDER}/ratings.csv"

USER_COL = "user"         # nomor pengguna 0, 1, 2, ...
ITEM_COL = "item"         # nomor item 0, 1, 2, ...
RATING_COL = "rating"
TIME_COL = "timestamp"
TOP_K = 10                    # jumlah item yang direkomendasikan
RELEVANT_RATING = 4       # rating >= 4 dianggap disukai
TRAIN_FRACTION = 0.8      # porsi interaksi tertua untuk latih
SEED = 42

# %% [markdown]
# ## Baca data

# %%
ratings = pd.read_csv(RATINGS_CSV)
ratings = ratings.sort_values(TIME_COL)
print(ratings.shape)
print("user:", ratings[USER_COL].nunique(), "| item:", ratings[ITEM_COL].nunique())
ratings.head()

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
print("interaksi latih:", train_matrix.nnz, "| uji (disukai):", test_matrix.nnz)

# %% [markdown]
# ## Langkah 2 — model
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.

# %%
popularity = np.asarray(train_matrix.sum(axis=0)).ravel()
scores = np.tile(popularity, (N_USERS, 1)).astype(float)       # item terpopuler untuk semua

# item_similarity = cosine_similarity(train_matrix.T)          # item mirip yang pernah dilihat
# scores = train_matrix @ item_similarity

# als = AlternatingLeastSquares(factors=64, regularization=0.05, iterations=15,
#                               use_gpu=False, random_state=SEED)
# als.fit((train_matrix * 40).astype(np.float32), show_progress=False)
# scores = np.asarray(als.user_factors) @ np.asarray(als.item_factors).T

# bpr = BayesianPersonalizedRanking(factors=64, iterations=100, use_gpu=False,
#                                   random_state=SEED)
# bpr.fit(train_matrix.astype(np.float32), show_progress=False)
# scores = np.asarray(bpr.user_factors) @ np.asarray(bpr.item_factors).T

scores[train_matrix.nonzero()] = -np.inf          # item yang sudah dilihat tidak direkomendasikan
top_items = np.argsort(-scores, axis=1)[:, :TOP_K]    # TOP_K item skor tertinggi per pengguna
print(top_items[:3])

# %% [markdown]
# ## Langkah 3a — evaluasi precision@K dan recall@K
# - precision@K = dari K rekomendasi, berapa porsi yang disukai di masa uji.
# - recall@K = dari film yang disukai di masa uji, berapa porsi yang masuk K rekomendasi.

# %%
test_dense = test_matrix.toarray()
hits = np.take_along_axis(test_dense, top_items, axis=1).sum(axis=1)
relevant = test_dense.sum(axis=1)
has_test = relevant > 0
print(f"precision@{TOP_K}:", round((hits[has_test] / TOP_K).mean(), 4))
print(f"recall@{TOP_K}   :", round((hits[has_test] / relevant[has_test]).mean(), 4))
print("pengguna dievaluasi:", has_test.sum())

# %% [markdown]
# ## Langkah 3b — simpan rekomendasi (satu baris per pengguna-item)
# Demo: pengguna yang punya data uji. Saat lomba: daftar pengguna dari file test.

# %%
target_users = np.unique(test[USER_COL])
users = np.repeat(target_users, TOP_K)
ranks = np.tile(np.arange(1, TOP_K + 1), len(target_users))
items = top_items[target_users].ravel()

submission = pd.DataFrame()
submission["user"] = users
submission["rank"] = ranks
submission["item"] = items
submission.to_csv("submission.csv", index=False)
submission.head(TOP_K)
