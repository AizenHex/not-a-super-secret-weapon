# %% [markdown]
# # Vision — Retrieval (embedding pretrained + tetangga terdekat)
# **Kapan dipakai:** soal "cari gambar paling mirip" (produk serupa, foto duplikat, pencarian
# katalog) untuk tiap gambar kueri.
#
# **Bentuk data yang didukung:** galeri dan kueri masing-masing CSV `id` (+ `label` bila ada) +
# folder gambar (aktif), atau semua gambar di folder (sel alternatif, tanpa CSV).
#
# **Langkah:** gambar → embedding CNN pretrained (beku, tanpa latih) → NearestNeighbors cosine
# → k gambar galeri terdekat. **Recall@k** = persen kueri yang punya minimal satu gambar relevan
# (label sama) di k hasil teratas.
#
# **Data demo:** `pets` (10 ras kucing/anjing, ukuran beragam, ada GIF/grayscale/RGBA).
# Galeri = gambar `train`, kueri = gambar `test`. Relevan = ras sama.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
GALLERY_CSV = f"{DATA_FOLDER}/train.csv"
QUERY_CSV = f"{DATA_FOLDER}/test.csv"
GALLERY_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
QUERY_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
QUERY_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../pretrained"
OUTPUT_FOLDER = "../../outputs/vision/retrieval"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar
TARGET = "label"        # kolom label galeri (hanya untuk mengukur / menampilkan)
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
IMAGE_SIZE = 224
BATCH_SIZE = 64
TOP_K = 10
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# %% [markdown]
# ## Baca data — CSV + folder gambar (aktif)

# %%
gallery = pd.read_csv(GALLERY_CSV)
queries = pd.read_csv(QUERY_CSV)
gallery["path"] = GALLERY_IMAGE_FOLDER + "/" + gallery[ID_COL].astype(str) + IMAGE_EXTENSION
queries["path"] = QUERY_IMAGE_FOLDER + "/" + queries[ID_COL].astype(str) + IMAGE_EXTENSION
print("galeri:", gallery.shape, "| kueri:", queries.shape)
print("contoh path:", gallery["path"].iloc[0])

# %% [markdown]
# ## (Alternatif) tanpa CSV: semua gambar di folder

# %%
# gallery = pd.DataFrame()
# gallery["path"] = sorted(glob.glob(f"{GALLERY_IMAGE_FOLDER}/*"))
# gallery[ID_COL] = gallery["path"].apply(os.path.basename)
# gallery[TARGET] = ""
# queries = pd.DataFrame()
# queries["path"] = sorted(glob.glob(f"{QUERY_IMAGE_FOLDER}/*"))
# queries[ID_COL] = queries["path"].apply(os.path.basename)

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
# Tiap gambar: `convert("RGB")` (GIF/grayscale/RGBA) → `ImageOps.fit` (potong tengah ke persegi)
# → dinormalisasi → backbone. Objek sering terpotong? Ganti `fit` dengan `ImageOps.pad`.

# %%
def embed_images(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        arrays = []
        for path in paths[start:start + BATCH_SIZE]:
            image = Image.open(path).convert("RGB")
            arrays.append(np.asarray(ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))))
        batch = torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2)
        batch = (batch.float() / 255 - MEAN) / STD
        with torch.no_grad():
            batches.append(backbone(batch.to(DEVICE)).cpu())
    return functional.normalize(torch.cat(batches), dim=-1).numpy()


gallery_vectors = embed_images(list(gallery["path"]))
query_vectors = embed_images(list(queries["path"]))
print("galeri:", gallery_vectors.shape, "| kueri:", query_vectors.shape)

# %% [markdown]
# ## Model: tetangga terdekat (cosine)
# `neighbor_rows[i]` = nomor baris galeri untuk TOP_K gambar termirip dengan kueri ke-i.

# %%
index = NearestNeighbors(n_neighbors=TOP_K, metric="cosine")
index.fit(gallery_vectors)
distances, neighbor_rows = index.kneighbors(query_vectors)
gallery_ids = gallery[ID_COL].to_numpy()
gallery_labels = gallery[TARGET].to_numpy()

# %% [markdown]
# ## Contoh: kueri (kiri) dan 5 gambar galeri termirip

# %%
fig, axes = plt.subplots(4, 6, figsize=(13, 9))
for ax in axes.ravel():
    ax.axis("off")
for row in range(4):
    axes[row, 0].imshow(Image.open(queries["path"][row]).convert("RGB"))
    axes[row, 0].set_title("KUERI " + queries[ID_COL][row], fontsize=8)
    for column in range(1, 6):
        gallery_row = neighbor_rows[row, column - 1]
        axes[row, column].imshow(Image.open(gallery["path"][gallery_row]).convert("RGB"))
        title = f"{gallery_labels[gallery_row]}\njarak {distances[row, column - 1]:.2f}"
        axes[row, column].set_title(title, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/contoh_retrieval.png", dpi=90)
plt.show()

# %% [markdown]
# ## Hasil per kueri
# Format umum: satu baris per kueri, kolom berisi id galeri teratas dipisah spasi. Sesuaikan
# dengan format soal (mis. hanya top-1, atau satu baris per pasangan kueri-galeri).

# %%
top_ids = []
for rows in neighbor_rows:
    top_ids.append(" ".join(gallery_ids[rows]))
result = pd.DataFrame()
result[ID_COL] = queries[ID_COL]
result["top1"] = gallery_ids[neighbor_rows[:, 0]]
result["top1_jarak"] = distances[:, 0]
result[f"top{TOP_K}"] = top_ids
result.to_csv(f"{OUTPUT_FOLDER}/hasil.csv", index=False)
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

# %% [markdown]
# ## Masalah umum
# - **Hasil teratas mirip latar/warna, bukan objek:** objek kecil di gambar besar. Potong objek
#   dulu (`deteksi`), atau coba backbone lain (ConvNeXt/ViT, atau CLIP dari `klasifikasi_clip`).
# - **Galeri sangat besar (≥ 100 ribu):** `NearestNeighbors` lambat; hitung `query @ gallery.T`
#   per potongan kueri, atau pakai `faiss` bila tersedia.
# - **Kueri juga ada di galeri:** hasil teratas = dirinya sendiri (jarak 0); buang kolom pertama.
