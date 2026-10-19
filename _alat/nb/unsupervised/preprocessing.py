# %% [markdown]
# # Unsupervised — Preprocessing (scaling dan transformasi kolom angka)
# **Kapan dipakai:** sebelum clustering, PCA, KNN, SVM, atau model linear, yaitu model yang
# peka terhadap besar kecilnya angka. Juga saat kolom sangat miring (banyak nilai kecil,
# sedikit nilai sangat besar).
#
# **Idenya:** tiap kolom diubah ke skala yang setara (misal rata-rata 0, simpangan 1) atau
# bentuknya dibuat lebih simetris. Efeknya dicek lewat skewness, histogram, dan clustering.
#
# **Langkah:**
# 1. Baca data, lihat rentang dan kemiringan tiap kolom.
# 2. Transformasi kolom.
# 3. Bandingkan hasil KMeans sebelum dan sesudah, simpan data hasil transformasi.
#
# **Data demo:** Wine (13 kolom kimia, `proline` bernilai ratusan, kolom lain kecil).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/wine.csv"

DROP_COLS = ["cultivar"]    # kolom yang tidak ditransformasi (label, id)
PLOT_COL = "proline"        # kolom yang digambar histogramnya
N_CLUSTERS = 3
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
X = data.drop(columns=DROP_COLS)
missing_per_column = X.isna().sum()
print(X.shape)
print("missing total:", missing_per_column.sum())
X.describe().round(2)

# %% [markdown]
# ## Langkah 2 — transformasi
# Pilih SATU: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
scaler = StandardScaler()                       # rata-rata 0, simpangan 1
# scaler = MinMaxScaler()                       # tiap kolom jadi 0..1
# scaler = RobustScaler()                       # pakai median, tahan nilai ekstrem
# scaler = PowerTransformer(method="yeo-johnson")   # kolom miring dibuat lebih simetris
# scaler = QuantileTransformer(output_distribution="normal", n_quantiles=100,
#                              random_state=SEED)   # paksa jadi bentuk lonceng

X_new = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

effect = pd.DataFrame()
effect["skew_before"] = X.skew()
effect["skew_after"] = X_new.skew()
effect["min_after"] = X_new.min()
effect["max_after"] = X_new.max()
print(effect.round(2))

fig, axes = plt.subplots(1, 2, figsize=(9, 3))
axes[0].hist(X[PLOT_COL], bins=30)
axes[0].set_title(f"{PLOT_COL} sebelum")
axes[1].hist(X_new[PLOT_COL], bins=30)
axes[1].set_title(f"{PLOT_COL} sesudah")
plt.show()

# %% [markdown]
# ## Langkah 3 — efek ke clustering
# Silhouette tidak bisa dibandingkan langsung antar skala, jadi lihat juga ARI di sel terakhir.

# %%
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
labels_before = kmeans.fit_predict(X)
labels_after = kmeans.fit_predict(X_new)
print("silhouette sebelum:", round(silhouette_score(X, labels_before), 3))
print("silhouette sesudah:", round(silhouette_score(X_new, labels_after), 3))

# %% [markdown]
# ## Simpan data hasil transformasi

# %%
submission = pd.DataFrame()
for name in X.columns:
    submission[name] = X_new[name]
for name in DROP_COLS:
    submission[name] = data[name]
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# ARI = kemiripan cluster dengan label asli (1 = sama persis, 0 = acak).

# %%
print("ARI sebelum:", round(adjusted_rand_score(data["cultivar"], labels_before), 3))
print("ARI sesudah:", round(adjusted_rand_score(data["cultivar"], labels_after), 3))
