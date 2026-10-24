# %% [markdown]
# # Recsys — Rekomendasi Sekuensial (tebak item **berikutnya**)
# **Kapan dipakai:** "setelah membeli/menonton ini, apa berikutnya?" Urutan waktu penting:
# keranjang belanja per sesi, playlist, halaman yang dibuka berurutan.
#
# **Idenya:** **transisi** (Markov) = hitung berapa kali item B muncul tepat sesudah item A di
# urutan semua pengguna. Rekomendasi = item B dengan hitungan tertinggi dari item terakhir
# pengguna. Sederhana, cepat, dan sering sulit dikalahkan.
#
# **Langkah:**
# 1. Sembunyikan item terakhir tiap pengguna (target), sisanya = riwayat.
# 2. Hitung matriks transisi dari riwayat, beri skor tiap item.
# 3. Evaluasi hit@K dan MRR@K, simpan rekomendasi.
#
# **Data demo:** MovieLens 100k, diurutkan `timestamp` per pengguna.

# %%
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/recsys/ml-100k"   # GANTI saat lomba
EVENTS_CSV = f"{DATA_FOLDER}/ratings.csv"

USER_COL = "user"         # nomor pengguna 0, 1, 2, ...
ITEM_COL = "item"         # nomor item 0, 1, 2, ...
TIME_COL = "timestamp"
TOP_K = 10
WINDOW = 5                # alternatif co-occurrence: B dihitung bila <= 5 langkah sesudah A
N_LAST = 5                # alternatif: jumlah item terakhir pengguna yang dipakai

# %% [markdown]
# ## Baca data

# %%
events = pd.read_csv(EVENTS_CSV)
events = events.sort_values([USER_COL, TIME_COL, ITEM_COL], kind="stable")
N_USERS = events[USER_COL].max() + 1
N_ITEMS = events[ITEM_COL].max() + 1
print(events.shape)
print("user:", events[USER_COL].nunique(), "| item:", events[ITEM_COL].nunique())
events.head()

# %% [markdown]
# ## Langkah 1 — item terakhir tiap pengguna = target

# %%
target = events.groupby(USER_COL).tail(1)
history = events.drop(target.index)
history_matrix = sp.csr_matrix((np.ones(len(history)), (history[USER_COL], history[ITEM_COL])),
                               shape=(N_USERS, N_ITEMS))
print("riwayat:", len(history), "| target:", len(target))

# %% [markdown]
# ## Langkah 2a — matriks pasangan A → B
# `pair_matrix(table, distance)`: baris = item A, kolom = item B, isi = berapa kali B muncul
# `distance` langkah sesudah A. Jarak 1 = transisi langsung.

# %%
def pair_matrix(table, distance):
    next_items = table.groupby(USER_COL)[ITEM_COL].shift(-distance)
    has_next = next_items.notna()
    first = table[ITEM_COL][has_next]
    second = next_items[has_next].astype(int)
    return sp.csr_matrix((np.ones(len(first)), (first, second)), shape=(N_ITEMS, N_ITEMS))


transitions = pair_matrix(history, 1)
print("pasangan transisi unik:", transitions.nnz)

co_occurrence = transitions.copy()
for distance in range(2, WINDOW + 1):
    co_occurrence = co_occurrence + pair_matrix(history, distance) / distance   # jauh = kecil

last_one = history.groupby(USER_COL).tail(1)
last_one_matrix = sp.csr_matrix((np.ones(len(last_one)), (last_one[USER_COL], last_one[ITEM_COL])),
                                shape=(N_USERS, N_ITEMS))
last_n = history.groupby(USER_COL).tail(N_LAST)
last_n_matrix = sp.csr_matrix((np.ones(len(last_n)), (last_n[USER_COL], last_n[ITEM_COL])),
                              shape=(N_USERS, N_ITEMS))

# %% [markdown]
# ## Langkah 2b — skor
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.

# %%
scores = last_one_matrix @ transitions                     # Markov: dari item terakhir
scores = scores.toarray()

# scores = last_n_matrix @ co_occurrence                   # dari N item terakhir, jendela
# scores = scores.toarray()

# item_similarity = cosine_similarity(history_matrix.T)    # item yang mirip N item terakhir
# scores = last_n_matrix @ item_similarity

popularity = np.asarray(history_matrix.sum(axis=0)).ravel()
scores = scores + 1e-6 * popularity          # pemecah seri: item populer didahulukan
scores[history_matrix.nonzero()] = -np.inf   # item yang sudah dilihat tidak direkomendasikan
top_items = np.argsort(-scores, axis=1)[:, :TOP_K]

# %% [markdown]
# ## Langkah 3a — evaluasi hit@K dan MRR@K
# - hit@K = porsi pengguna yang item terakhirnya ada di K tebakan.
# - MRR@K = rata-rata 1/peringkat (peringkat 1 = 1, peringkat 2 = 0.5, tidak ada = 0).

# %%
guesses = top_items[target[USER_COL].to_numpy()]
is_correct = guesses == target[ITEM_COL].to_numpy()[:, None]
found = is_correct.any(axis=1)
ranks = is_correct.argmax(axis=1) + 1
reciprocal = np.zeros(len(found))
reciprocal[found] = 1 / ranks[found]
print(f"hit@{TOP_K}:", round(found.mean(), 4))
print(f"MRR@{TOP_K}:", round(reciprocal.mean(), 4))

# %% [markdown]
# ## Langkah 3b — simpan rekomendasi item berikutnya
# Saat lomba: `history = events` (semua data) supaya item terakhir ikut dipakai, jalankan ulang.

# %%
target_users = target[USER_COL].to_numpy()

submission = pd.DataFrame()
submission["user"] = np.repeat(target_users, TOP_K)
submission["rank"] = np.tile(np.arange(1, TOP_K + 1), len(target_users))
submission["item"] = guesses.ravel()
submission.to_csv("submission.csv", index=False)
submission.head(TOP_K)
