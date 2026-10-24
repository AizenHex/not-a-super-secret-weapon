# %% [markdown]
# # Unsupervised — Reduksi Dimensi (banyak kolom → 2 kolom)
# **Kapan dipakai:** kolom sangat banyak (piksel, embedding, sensor) dan perlu digambar,
# dipercepat, atau diringkas sebelum clustering.
#
# **Idenya:** tiap baris dipetakan ke sedikit sumbu baru. PCA mempertahankan sebaran terbesar
# (cepat, bisa dipakai ulang di data baru). t-SNE/UMAP mempertahankan "siapa dekat siapa"
# (bagus untuk gambar, lambat).
#
# **Langkah:**
# 1. Baca data, samakan skala.
# 2. Ringkas jadi 2 dimensi, gambar.
# 3. Simpan koordinat 2D tiap baris.
#
# **Data demo:** Digits (gambar angka 8x8 = 64 kolom piksel, `digit` hanya untuk warna/cek).

# %%
import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, Isomap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/digits.csv"

LABEL_COL = "digit"     # hanya untuk warna gambar dan cek, tidak dipakai melatih
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
X = data.drop(columns=[LABEL_COL])
print(data.shape)

fig, axes = plt.subplots(1, 10, figsize=(10, 1.3))
for i in range(10):
    pixels = X.iloc[i].to_numpy()
    axes[i].imshow(pixels.reshape(8, 8), cmap="gray")
    axes[i].axis("off")
plt.show()

# %% [markdown]
# ## Langkah 1 — samakan skala

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# ## Langkah 2 — ringkas jadi 2 dimensi
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = PCA(n_components=2, random_state=SEED)
# model = TSNE(n_components=2, perplexity=30, random_state=SEED)    # bagus untuk gambar
# model = umap.UMAP(n_components=2, random_state=SEED)             # mirip t-SNE, lebih cepat
# model = TruncatedSVD(n_components=2, random_state=SEED)          # data jarang (TF-IDF)
# model = KernelPCA(n_components=2, kernel="rbf", random_state=SEED)
# model = Isomap(n_components=2, n_neighbors=10)

embedding = model.fit_transform(X_scaled)
plt.scatter(embedding[:, 0], embedding[:, 1], s=8)
plt.title("Data dalam 2 dimensi")
plt.show()

# %% [markdown]
# ## Langkah 3 — simpan koordinat 2D

# %%
submission = pd.DataFrame()
submission["row"] = data.index
submission["dim1"] = embedding[:, 0]
submission["dim2"] = embedding[:, 1]
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# Warna = label asli. Silhouette tinggi = tiap digit membentuk kelompok sendiri.

# %%
points = plt.scatter(embedding[:, 0], embedding[:, 1], c=data[LABEL_COL], cmap="tab10", s=8)
plt.colorbar(points, label="digit")
plt.show()
print("silhouette vs digit:", round(silhouette_score(embedding, data[LABEL_COL]), 3))
