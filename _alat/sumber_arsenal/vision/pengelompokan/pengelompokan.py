# %% [markdown]
# # Pengelompokan gambar (clustering): kelompokkan gambar mirip TANPA label
#
# **Butuh:** `images` (kolom `path`). **Menghasilkan:** `images["cluster"]`, grid contoh per
# kelompok, `outputs/kelompok.csv`, dan (opsional) label per kelompok.
#
# Dipakai untuk:
# - soal clustering langsung (kelompokkan foto, jumlah kelompok ditentukan/dicari);
# - **mempercepat melabeli sendiri**: daripada melabeli 2000 gambar satu per satu, lihat 20
#   kelompok, beri nama tiap kelompok, lalu perbaiki yang nyasar saja;
# - cek data: kelompok aneh = gambar rusak/kembar/salah folder.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | `embed_images` (vektor norma 1) | **ya** |
# | 2 | PCA + coba beberapa jumlah kelompok (skor silhouette) | disarankan |
# | 3 | KMeans dengan `N_CLUSTERS` → `cluster` + jarak ke pusat | **ya** |
# | 4 | grid contoh per kelompok (lihat mata) | **ya** |
# | 5 | beri nama tiap kelompok → label awal + daftar yang perlu dicek | untuk melabeli |
# | 6 | peta 2D semua gambar | opsional |
#
# Gambar berupa karakter/aksara/kode (bukan foto): vektor piksel 32×32 sering lebih baik
# daripada backbone foto; lihat `vision/ocr/karakter_label_sendiri.ipynb` bagian 3.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Foto ras kucing/anjing; labelnya pura-pura tidak ada (hanya dipakai untuk menilai di akhir).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

images = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
images["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + images["id"]
images = images.head(400).reset_index(drop=True)

# %% [markdown]
# ## 1. `embed_images(paths)` → vektor norma 1

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

PRETRAINED_FOLDER = "pretrained"   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
backbone = models.convnext_tiny(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"))
backbone.classifier[2] = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
embed_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# %% [markdown]
# Fungsi `embed_images` + terapkan ke SEMUA gambar → `vectors`.

# %%
def embed_images(paths, batch_size=64):
    result = []
    for start in tqdm(range(0, len(paths), batch_size)):
        batch = []
        for path in paths[start:start + batch_size]:
            image = ImageOps.exif_transpose(Image.open(path))
            batch.append(embed_transform(image.convert("RGB")))
        with torch.no_grad():
            result.append(backbone(torch.stack(batch).to(DEVICE)).cpu().numpy())
    result = np.concatenate(result)
    return result / np.linalg.norm(result, axis=1, keepdims=True)


vectors = embed_images(images["path"].tolist())
print(vectors.shape)

# %% [markdown]
# ## 2. PCA + cari jumlah kelompok → `reduced`
# PCA meringkas 768 angka jadi 50 (lebih cepat, lebih sedikit noise). Silhouette: -1..1,
# makin tinggi makin terpisah. Ambil puncaknya, atau jumlah yang diminta soal.

# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

reduced = PCA(n_components=50, random_state=42).fit_transform(vectors)
for n_clusters in [4, 6, 8, 10, 12, 15, 20]:          # UBAH
    found = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(reduced)
    print(n_clusters, "kelompok | silhouette:", round(silhouette_score(reduced, found), 4))

# %% [markdown]
# ## 3. KMeans → `images["cluster"]`, `images["distance"]` (jarak ke pusat kelompok)
# Jarak besar = gambar kurang cocok dengan kelompoknya (cek dulu yang ini).
# Alternatif tanpa menentukan jumlah: `AgglomerativeClustering(n_clusters=None,
# distance_threshold=...)` atau `HDBSCAN(min_cluster_size=10)` (label -1 = noise).

# %%
import numpy as np
from sklearn.cluster import KMeans

N_CLUSTERS = 10   # UBAH

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
images["cluster"] = kmeans.fit_predict(reduced)
centers = kmeans.cluster_centers_[images["cluster"]]
images["distance"] = np.linalg.norm(reduced - centers, axis=1)
print(images["cluster"].value_counts().sort_index())

# %% [markdown]
# ## 4. Grid contoh per kelompok (satu baris = satu kelompok)
# `PER_CLUSTER` gambar terdekat ke pusat = wajah khas kelompok itu.

# %%
import matplotlib.pyplot as plt
from PIL import Image

PER_CLUSTER = 6   # UBAH

fig, axes = plt.subplots(N_CLUSTERS, PER_CLUSTER, figsize=(1.6 * PER_CLUSTER, 1.6 * N_CLUSTERS),
                         squeeze=False)
for cluster in range(N_CLUSTERS):
    members = images[images["cluster"] == cluster].sort_values("distance")
    paths = members["path"].head(PER_CLUSTER).tolist()
    for col in range(PER_CLUSTER):
        axes[cluster, col].axis("off")
        if col < len(paths):
            axes[cluster, col].imshow(Image.open(paths[col]).convert("RGB"))
    axes[cluster, 0].set_title(f"kelompok {cluster}", fontsize=8, loc="left")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Nama per kelompok → label awal + `outputs/kelompok.csv` untuk dicek
# Isi `CLUSTER_NAMES` setelah melihat grid. Kelompok campur? Beri nama `"cek"`, atau naikkan
# `N_CLUSTERS`. Buka CSV di Excel: urut per kelompok, jarak terbesar di atas → perbaiki kolom
# `label`. Hasilnya jadi train untuk `vision/klasifikasi/embedding.ipynb`.

# %%
import os

CLUSTER_NAMES = {}   # UBAH: {0: "persian", 1: "pug", 2: "cek", ...}

os.makedirs("outputs", exist_ok=True)
images["label_awal"] = images["cluster"].map(CLUSTER_NAMES)
images["label_awal"] = images["label_awal"].fillna("cek")
to_check = images.sort_values(["cluster", "distance"], ascending=[True, False])
to_check.to_csv("outputs/kelompok.csv", index=False)
print(pd.crosstab(images["cluster"], images["label"]))   # hanya kalau label asli ada (demo)

# %% [markdown]
# ## 6. Peta 2D semua gambar (PCA 2 dimensi), warna = kelompok

# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

points = PCA(n_components=2, random_state=42).fit_transform(reduced)
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(points[:, 0], points[:, 1], c=images["cluster"], cmap="tab20", s=10)
ax.legend(*scatter.legend_elements(), fontsize=7, title="kelompok", loc="best")
plt.show()
