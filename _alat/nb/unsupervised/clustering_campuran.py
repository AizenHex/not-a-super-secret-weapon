# %% [markdown]
# # Unsupervised — Clustering Data Campuran (kolom angka + kategori)
# **Kapan dipakai:** data untuk dikelompokkan berisi kolom angka DAN kategori (kota, jenis
# kelamin, tipe produk). KMeans biasa hanya paham angka.
#
# **Idenya:** dua cara:
# - **K-Prototypes** (paket `kmodes`): jarak biasa untuk angka + jumlah kategori yang beda.
# - **Jarak Gower** + Agglomerative: tiap kolom diberi jarak 0–1, lalu dirata-rata.
#
# **Langkah:**
# 1. Siapkan kolom angka (skala disamakan) dan kolom kategori.
# 2. Hitung matriks jarak Gower (untuk model Agglomerative dan nilai silhouette).
# 3. Latih model, lihat profil cluster, simpan label cluster.
#
# **Data demo:** Penguins (angka: paruh, sirip, berat, kategori: pulau, jenis kelamin).
# `species` hanya untuk cek.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/penguins.csv"

LABEL_COL = "species"    # hanya untuk cek
NUMBER_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORY_COLS = ["island", "sex"]
N_CLUSTERS = 3
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
print(data.shape)
missing_per_column = data.isna().sum()
print("missing total:", missing_per_column.sum())
print(data[NUMBER_COLS].describe().round(1))
data.head()

# %% [markdown]
# ## Langkah 1 — kolom angka + kategori
# K-Prototypes perlu tahu nomor kolom yang kategori (`CATEGORY_POSITIONS`).

# %%
scaler = StandardScaler()
numbers = scaler.fit_transform(data[NUMBER_COLS])
X_mixed = np.column_stack([numbers, data[CATEGORY_COLS].to_numpy()])
X_mixed = X_mixed.astype(object)
CATEGORY_POSITIONS = []
for i in range(len(CATEGORY_COLS)):
    CATEGORY_POSITIONS.append(len(NUMBER_COLS) + i)
print(X_mixed[:3])

# %% [markdown]
# ## Langkah 2 — jarak Gower
# Kolom angka: selisih dibagi rentang kolom (hasil 0–1). Kolom kategori: 0 kalau sama, 1 kalau
# beda. Jarak dua baris = rata-rata semua kolom.

# %%
def gower_distance(table):
    values = table[NUMBER_COLS].to_numpy(dtype=float)
    value_range = values.max(axis=0) - values.min(axis=0)
    total = np.zeros((len(table), len(table)))
    for i in range(len(NUMBER_COLS)):
        column = values[:, i]
        total += np.abs(column[:, None] - column[None, :]) / value_range[i]
    for name in CATEGORY_COLS:
        column = table[name].to_numpy()
        total += column[:, None] != column[None, :]
    return total / (len(NUMBER_COLS) + len(CATEGORY_COLS))


# coba pada 5 baris pertama
print(gower_distance(data.head()).round(2))

gower = gower_distance(data)
print("matriks Gower:", gower.shape)

# %% [markdown]
# ## Langkah 3 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = KPrototypes(n_clusters=N_CLUSTERS, init="Cao", random_state=SEED)
labels = model.fit_predict(X_mixed, categorical=CATEGORY_POSITIONS)

# model = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="precomputed",
#                                 linkage="average")
# labels = model.fit_predict(gower)                   # pakai matriks Gower

print("silhouette (jarak Gower):", round(silhouette_score(gower, labels, metric="precomputed"), 3))
print(pd.Series(labels).value_counts())

pca = PCA(n_components=2)
X_2d = pca.fit_transform(numbers)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=12)
plt.title("Cluster, PCA 2D dari kolom angka")
plt.show()

# %% [markdown]
# ## Profil tiap cluster
# Angka: rata-rata per cluster. Kategori: jumlah tiap nilai per cluster.

# %%
profile = data[NUMBER_COLS].groupby(labels).mean()
print(profile.round(1))
for name in CATEGORY_COLS:
    print(pd.crosstab(labels, data[name], rownames=["cluster"]))

# %% [markdown]
# ## Simpan label cluster

# %%
submission = pd.DataFrame()
submission["row"] = data.index
submission["cluster"] = labels
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# ARI = kemiripan cluster dengan label asli (1 = sama persis, 0 = acak).

# %%
print("ARI:", round(adjusted_rand_score(data[LABEL_COL], labels), 3))
pd.crosstab(labels, data[LABEL_COL], rownames=["cluster"])
