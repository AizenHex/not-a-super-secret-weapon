# %% [markdown]
# # Vision — Deteksi Anomali: Autoencoder vs Jarak Embedding
# **Kapan dipakai:** data latih hanya berisi gambar **normal**, dan di test harus diberi skor
# "seberapa tidak biasa" tiap gambar. Foto produk seragam dengan cacat kecil (goresan, noda)?
# Pakai `defect` (PatchCore).
#
# **Dua skor dibandingkan:**
# 1. **Autoencoder konvolusi**: belajar menyalin ulang gambar normal → gambar tidak biasa disalin
#    lebih buruk → skor = error rekonstruksi (MSE). Cocok bila anomali = perubahan piksel (noda,
#    warna aneh) pada gambar yang seragam.
# 2. **Jarak embedding (kNN)**: embedding ResNet50 pretrained → skor = rata-rata jarak ke
#    `N_NEIGHBORS` gambar normal terdekat. Tanpa latih. Cocok bila anomali = ISI gambar berbeda.
#
# **Bentuk data:** CSV `id` + folder gambar (aktif), atau semua gambar di folder (alternatif).
#
# **Data demo:** `pets`. Normal = foto kucing (train disaring hanya kucing), anomali = foto anjing
# di test. Hasil demo: ROC-AUC autoencoder 0.64, embedding 1.00, karena beda kucing vs anjing
# adalah beda isi, bukan beda piksel.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../pretrained"
OUTPUT_FOLDER = "../../outputs/vision/autoencoder_anomali"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
NORMAL_COL = "spesies"  # HANYA DEMO: kolom penentu normal
NORMAL_VALUE = "kucing"
IMAGE_SIZE = 128        # untuk autoencoder, embedding selalu 224
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_NEIGHBORS = 5         # tetangga terdekat untuk skor embedding
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# %% [markdown]
# ## Baca data — CSV + folder gambar (aktif)

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["path"] = TRAIN_IMAGE_FOLDER + "/" + train[ID_COL].astype(str) + IMAGE_EXTENSION
test["path"] = TEST_IMAGE_FOLDER + "/" + test[ID_COL].astype(str) + IMAGE_EXTENSION
print("contoh path:", train["path"].iloc[0])

# %% [markdown]
# ## (Alternatif) tanpa CSV: semua gambar di folder train = normal

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# test = pd.DataFrame()
# test["path"] = sorted(glob.glob(f"{TEST_IMAGE_FOLDER}/*"))
# test[ID_COL] = test["path"].apply(os.path.basename)

# %% [markdown]
# ## Simulasi: train hanya berisi gambar normal (HAPUS saat lomba)

# %%
train = train[train[NORMAL_COL] == NORMAL_VALUE].reset_index(drop=True)
print("latih (normal):", len(train), "| test:", len(test))

# %% [markdown]
# ## Gambar → tensor
# `convert("RGB")` menyamakan GIF/grayscale/RGBA, `ImageOps.fit` memotong tengah ke persegi.
# Disimpan uint8 (hemat memori), diubah ke 0–1 saat dipakai.

# %%
def load_images(paths, size):
    images = np.zeros((len(paths), size, size, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        images[i] = ImageOps.fit(Image.open(path).convert("RGB"), (size, size))
    return torch.tensor(images).permute(0, 3, 1, 2)


X_train = load_images(train["path"], IMAGE_SIZE)
X_test = load_images(test["path"], IMAGE_SIZE)
print("latih:", tuple(X_train.shape), "| test:", tuple(X_test.shape))

# %% [markdown]
# ## Model 1: autoencoder konvolusi
# Encoder mengecilkan gambar 128 → 8 piksel (16 kanal), decoder membesarkannya lagi 8 → 128.
# Bagian tengah sempit memaksa model hanya mengingat pola umum gambar normal.

# %%
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),       # 64
    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),      # 32
    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),     # 16
    nn.Conv2d(128, 16, 3, stride=2, padding=1), nn.ReLU(),     # 8
    nn.ConvTranspose2d(16, 128, 2, stride=2), nn.ReLU(),       # 16
    nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),       # 32
    nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),        # 64
    nn.ConvTranspose2d(32, 3, 2, stride=2), nn.Sigmoid(),      # 128
)
net = net.to(DEVICE)

# %% [markdown]
# ## Latih (hanya gambar normal)

# %%
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    net.train()
    order = torch.randperm(len(X_train))
    total_loss = 0
    for start in range(0, len(order), BATCH_SIZE):
        batch = X_train[order[start:start + BATCH_SIZE]].float().to(DEVICE) / 255
        loss = functional.mse_loss(net(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
    if (epoch + 1) % 5 == 0:
        print("epoch", epoch + 1, "loss", round(total_loss / len(X_train), 5))

# %% [markdown]
# ## Skor autoencoder = error rekonstruksi tiap gambar test (per batch, hemat memori GPU)

# %%
def reconstruction_errors(images):
    net.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = images[start:start + BATCH_SIZE].float().to(DEVICE) / 255
            errors.append(((net(batch) - batch) ** 2).mean(dim=(1, 2, 3)).cpu())
    return torch.cat(errors).numpy()


ae_scores = reconstruction_errors(X_test)
print("skor min/rata-rata/maks:", ae_scores.min().round(5), ae_scores.mean().round(5),
      ae_scores.max().round(5))

# %% [markdown]
# ## Lihat rekonstruksi: 4 skor terendah (atas) dan 4 tertinggi (bawah)
# Baris ganjil = asli, baris genap = salinan autoencoder.

# %%
chosen = np.concatenate([np.argsort(ae_scores)[:4], np.argsort(ae_scores)[-4:]])
net.eval()
with torch.no_grad():
    copies = net(X_test[chosen].float().to(DEVICE) / 255).cpu()
fig, axes = plt.subplots(4, 4, figsize=(9, 9))
for number, row in enumerate(chosen):
    grid_row = (number // 4) * 2
    axes[grid_row, number % 4].imshow(X_test[row].permute(1, 2, 0))
    axes[grid_row, number % 4].set_title(f"skor {ae_scores[row]:.4f}", fontsize=8)
    axes[grid_row + 1, number % 4].imshow(copies[number].permute(1, 2, 0))
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/rekonstruksi.png", dpi=90)
plt.show()

# %% [markdown]
# ## Model 2: jarak embedding ResNet50 pretrained (kNN, tanpa latih)
# Skor = rata-rata jarak cosine ke `N_NEIGHBORS` gambar normal terdekat. Jauh dari semua
# gambar normal = tidak biasa.

# %%
backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()


def embed(images):
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            batch = (images[start:start + 64].float() / 255 - MEAN) / STD
            batches.append(backbone(batch.to(DEVICE)).cpu())
    return functional.normalize(torch.cat(batches), dim=-1).numpy()


train_vectors = embed(load_images(train["path"], 224))
test_vectors = embed(load_images(test["path"], 224))
index = NearestNeighbors(n_neighbors=N_NEIGHBORS, metric="cosine")
index.fit(train_vectors)
knn_scores = index.kneighbors(test_vectors)[0].mean(axis=1)

# %% [markdown]
# ## Hasil per gambar (skor besar = lebih mungkin anomali)
# Pilih SATU skor untuk dikumpulkan; keduanya ikut disimpan untuk dibandingkan.

# %%
scores = knn_scores
# scores = ae_scores

result = pd.DataFrame()
result[ID_COL] = test[ID_COL]
result["skor_anomali"] = scores
result["skor_autoencoder"] = ae_scores
result["skor_knn"] = knn_scores
result.to_csv(f"{OUTPUT_FOLDER}/hasil.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# ROC-AUC 0.5 = acak, 1 = sempurna. PR-AUC dibandingkan dengan baseline = proporsi anomali.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
is_anomaly = (test_key[NORMAL_COL] != NORMAL_VALUE).astype(int)
print("proporsi anomali di test:", round(is_anomaly.mean(), 3))
for name, values in [("autoencoder", ae_scores), ("embedding kNN", knn_scores)]:
    print(f"{name:14s} ROC-AUC {roc_auc_score(is_anomaly, values):.4f}",
          f"| PR-AUC {average_precision_score(is_anomaly, values):.4f}")

# %% [markdown]
# ## Masalah umum
# - **Autoencoder ROC-AUC ≈ 0.5 atau < 0.5:** anomali berupa beda isi (seperti demo), atau
#   gambar anomali justru lebih polos sehingga lebih mudah disalin. Pakai skor embedding.
# - **Rekonstruksi buram semua:** wajar untuk autoencoder kecil; yang penting gambar anomali
#   LEBIH buruk. Coba `EPOCHS` lebih banyak atau `IMAGE_SIZE` lebih kecil (64).
# - **Perlu label normal/anomali:** pilih ambang dari persentil skor (mis. 95% gambar train
#   normal di bawah ambang), seperti di `kelas_baru`.
