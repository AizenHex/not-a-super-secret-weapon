# %% [markdown]
# # Unsupervised — Clustering (kolom angka)
# **Kapan dipakai:** diminta mengelompokkan data (pelanggan, wilayah, produk) tanpa label,
# lalu menjelaskan tiap kelompok.
#
# **Idenya:** baris yang jaraknya dekat dimasukkan ke kelompok (cluster) yang sama. Skala
# kolom disamakan dulu supaya kolom bernilai besar tidak mendominasi jarak.
#
# **Langkah:**
# 1. Baca data, samakan skala.
# 2. Pilih jumlah cluster dengan silhouette (makin tinggi makin baik).
# 3. Latih model, lihat profil tiap cluster, simpan label cluster.
#
# **Data demo:** Wine (13 kolom kimia, `cultivar` = jenis anggur asli, hanya untuk cek).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, Birch, KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/wine.csv"
# DATA_CSV = f"{DATA_FOLDER}/moons_SINTETIS.csv"    # bentuk bulan sabit: LABEL_COL = "true"

LABEL_COL = "cultivar"   # hanya untuk cek, kolom lain dipakai sebagai feature
N_CLUSTERS = 3
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
print(data.shape)
missing_per_column = data.isna().sum()
print("missing total:", missing_per_column.sum())
data.head()

# %% [markdown]
# ## Langkah 1 — samakan skala
# Gambar PCA 2D = semua kolom diringkas jadi 2 sumbu, hanya untuk dilihat.

# %%
X = data.drop(columns=[LABEL_COL])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
plt.scatter(X_2d[:, 0], X_2d[:, 1], s=12)
plt.title("Data dalam PCA 2D")
plt.show()

# %% [markdown]
# ## Langkah 2 — pilih jumlah cluster
# Silhouette tinggi dan Davies-Bouldin rendah = cluster rapat dan saling terpisah.
# Pilih angka terbaik, tulis di `N_CLUSTERS` pada Konfigurasi.

# %%
for count in range(2, 9):
    kmeans = KMeans(n_clusters=count, n_init=10, random_state=SEED)
    trial_labels = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, trial_labels)
    davies = davies_bouldin_score(X_scaled, trial_labels)
    print(count, "cluster | silhouette:", round(silhouette, 3), "| DB:", round(davies, 3))

# %% [markdown]
# ## Langkah 3 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.
# DBSCAN/HDBSCAN menentukan jumlah cluster sendiri dan memberi label -1 untuk "noise".

# %%
model = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED)
# model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
# model = GaussianMixture(n_components=N_CLUSTERS, random_state=SEED)   # cluster lonjong
# model = SpectralClustering(n_clusters=N_CLUSTERS, affinity="nearest_neighbors",
#                            random_state=SEED)                       # bentuk bulan sabit
# model = Birch(n_clusters=N_CLUSTERS)                               # data sangat besar
# model = DBSCAN(eps=2.0, min_samples=8)                             # eps perlu dicoba-coba
# model = HDBSCAN(min_cluster_size=10)

labels = model.fit_predict(X_scaled)
print("silhouette:", round(silhouette_score(X_scaled, labels), 3))
print(pd.Series(labels).value_counts())

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=12)
plt.title("Hasil clustering, PCA 2D")
plt.show()

# %% [markdown]
# ## Profil tiap cluster
# Rata-rata tiap kolom per cluster = bahan untuk menjelaskan "cluster 0 itu yang seperti apa".

# %%
profile = X.groupby(labels).mean()
profile.round(2)

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
