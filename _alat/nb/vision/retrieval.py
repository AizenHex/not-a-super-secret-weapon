# %% [markdown]
# # Vision — Retrieval (embedding pretrained + tetangga terdekat)
# **Kapan dipakai:** soal "cari gambar paling mirip" (produk serupa, foto duplikat, pencarian
# katalog) untuk tiap gambar kueri.
#
# **Langkah:** gambar → embedding CNN pretrained (beku, tanpa latih) → NearestNeighbors cosine
# → k gambar galeri terdekat. **Recall@k** = persen kueri yang punya minimal satu gambar relevan
# (label sama) di k hasil teratas.
#
# **Data demo:** CIFAR-10 kecil. Galeri = data `train`, kueri = data `test`.

# %%
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/classification/cifar10_small"  # GANTI
GALLERY_CSV = f"{DATA_FOLDER}/train.csv"
QUERY_CSV = f"{DATA_FOLDER}/test.csv"
QUERY_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
IMAGE_SIZE = 224
BATCH_SIZE = 128
TOP_K = 10
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
# normalisasi warna ImageNet (wajib untuk bobot torchvision)
MEAN = torch.tensor([0.485, 0.456, 0.406])
MEAN = MEAN.view(1, 3, 1, 1).to(DEVICE)    # bentuk (1, 3 warna, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225])
STD = STD.view(1, 3, 1, 1).to(DEVICE)

# %% [markdown]
# ## Baca data

# %%
gallery = pd.read_csv(GALLERY_CSV)
queries = pd.read_csv(QUERY_CSV)
print("galeri:", gallery.shape, "| kueri:", queries.shape)
print(gallery[TARGET].value_counts())
gallery.head()

# %% [markdown]
# ## Backbone pretrained beku
# Lapisan klasifikasi terakhir diganti `nn.Identity()` supaya keluarannya embedding.
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok resnet50.

# %%
backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()

# backbone = models.vit_b_16(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/vit_b_16/model.pth", map_location="cpu"))
# backbone.heads = nn.Identity()

# backbone = models.convnext_tiny(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth",
#                                     map_location="cpu"))
# backbone.classifier[2] = nn.Identity()

# backbone = models.efficientnet_b0(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/efficientnet_b0/model.pth",
#                                     map_location="cpu"))
# backbone.classifier = nn.Identity()

backbone = backbone.to(DEVICE)
backbone.eval()

# %% [markdown]
# ## Embedding semua gambar
# Gambar → tensor 0–1 → diperbesar ke `IMAGE_SIZE` → dinormalisasi → backbone.

# %%
def embed_images(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        arrays = []
        for path in paths[start:start + BATCH_SIZE]:
            arrays.append(np.asarray(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB")))
        batch = torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2)
        batch = batch.float().to(DEVICE) / 255
        batch = functional.interpolate(batch, size=IMAGE_SIZE, mode="bilinear")
        with torch.no_grad():
            batches.append(backbone((batch - MEAN) / STD).cpu())
    vectors = torch.cat(batches)
    return functional.normalize(vectors, dim=-1).numpy()


gallery_vectors = embed_images(list(gallery[PATH_COL]))
query_vectors = embed_images(list(queries[PATH_COL]))
print("galeri:", gallery_vectors.shape, "| kueri:", query_vectors.shape)

# %% [markdown]
# ## Model: tetangga terdekat (cosine)
# `neighbor_rows[i]` = nomor baris galeri untuk TOP_K gambar termirip dengan kueri ke-i.

# %%
index = NearestNeighbors(n_neighbors=TOP_K, metric="cosine")
index.fit(gallery_vectors)
distances, neighbor_rows = index.kneighbors(query_vectors)
gallery_paths = gallery[PATH_COL].to_numpy()
gallery_labels = gallery[TARGET].to_numpy()

# %% [markdown]
# ## Hasil per kueri (gambar galeri teratas)

# %%
result = pd.DataFrame()
result["id"] = queries["id"]
result["top1_path"] = gallery_paths[neighbor_rows[:, 0]]
result["top1_label"] = gallery_labels[neighbor_rows[:, 0]]
result["top1_jarak"] = distances[:, 0]
result.to_csv("hasil.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Relevan = label gambar galeri sama dengan label kueri.

# %%
query_labels = pd.read_csv(QUERY_KEY_CSV)[TARGET].to_numpy()
is_relevant = gallery_labels[neighbor_rows] == query_labels[:, None]
for top in [1, 5, 10]:
    found = is_relevant[:, :top].any(axis=1)
    print(f"Recall@{top}:", round(found.mean(), 4))
print("Precision@10:", round(is_relevant.mean(), 4))
