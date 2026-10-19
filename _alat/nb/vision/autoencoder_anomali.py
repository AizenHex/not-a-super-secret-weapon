# %% [markdown]
# # Vision — Deteksi Anomali dengan Autoencoder Konvolusi
# **Kapan dipakai:** data latih hanya berisi gambar **normal** (produk bagus, kondisi biasa),
# dan di test harus ditandai gambar yang tidak biasa.
#
# **Langkah:** autoencoder belajar menyalin ulang gambar normal → gambar tidak biasa disalin
# lebih buruk → skor anomali = error rekonstruksi (MSE) per gambar.
#
# **PERINGATAN:** ROC-AUC demo 0.38 (lebih buruk dari acak, karena kelas CIFAR berbeda isi
# tetapi mirip tekstur). Belum layak dipakai; lihat handoff §6 dan `vision/defect`.
#
# **Data demo:** CIFAR-10 kecil. Kelas `NORMAL_CLASS` jadi satu-satunya data latih, semua kelas
# lain di test dianggap anomali.

# %%
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
NORMAL_CLASS = "automobile"
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data + ubah gambar jadi tensor
# Tensor gambar: (jumlah, 3 warna, tinggi, lebar), nilai 0–1.

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())


def load_images(paths):
    arrays = []
    for path in paths:
        arrays.append(np.asarray(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB")))
    images = torch.tensor(np.stack(arrays))
    return images.permute(0, 3, 1, 2).float() / 255


normal_rows = train[TARGET] == NORMAL_CLASS          # latih hanya kelas normal
X_train = load_images(train.loc[normal_rows, PATH_COL])
X_test = load_images(test[PATH_COL])
print("latih (normal):", X_train.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Model: autoencoder konvolusi kecil
# Encoder mengecilkan gambar 32→8 piksel, decoder membesarkannya lagi 8→32.

# %%
net = nn.Sequential(
    nn.Conv2d(3, 32, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 16, 3, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(16, 64, 2, stride=2),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 2, stride=2),
    nn.ReLU(),
    nn.Conv2d(32, 3, 3, padding=1),
    nn.Sigmoid(),
)
net = net.to(DEVICE)

# %% [markdown]
# ## Latih (hanya gambar normal)

# %%
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
for epoch in range(EPOCHS):
    order = torch.randperm(len(X_train))
    for start in range(0, len(order), BATCH_SIZE):
        batch = X_train[order[start:start + BATCH_SIZE]].to(DEVICE)
        loss = functional.mse_loss(net(batch), batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch", epoch + 1, "loss", round(loss.item(), 5))

# %% [markdown]
# ## Skor anomali = error rekonstruksi tiap gambar test

# %%
net.eval()
with torch.no_grad():
    test_on_device = X_test.to(DEVICE)
    squared_error = (net(test_on_device) - test_on_device) ** 2
    scores = squared_error.mean(dim=(1, 2, 3))     # rata-rata per gambar
scores = scores.cpu().numpy()
print("skor min/rata-rata/maks:", scores.min().round(5), scores.mean().round(5),
      scores.max().round(5))

# %% [markdown]
# ## Hasil per gambar (skor besar = lebih mungkin anomali)

# %%
result = pd.DataFrame()
result["id"] = test["id"]
result["skor_anomali"] = scores
result.to_csv("hasil.csv", index=False)
result.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# PR-AUC dibandingkan dengan baseline = proporsi anomali (skor acak).

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
is_anomaly = (test_key[TARGET] != NORMAL_CLASS).astype(int)
print("ROC-AUC:", round(roc_auc_score(is_anomaly, scores), 4))
print("PR-AUC:", round(average_precision_score(is_anomaly, scores), 4),
      "| baseline PR-AUC:", round(is_anomaly.mean(), 4))
