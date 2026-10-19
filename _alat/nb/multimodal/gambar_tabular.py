# %% [markdown]
# # Multimodal — Gambar + Tabel (embedding gambar digabung kolom tabel)
# **Kapan dipakai:** tiap baris punya foto DAN kolom tabel: foto produk + harga/berat, foto
# rumah + luas/lokasi, rontgen + data pasien.
#
# **Idenya:** gambar diubah jadi embedding (deretan angka) oleh model pretrained yang tidak
# dilatih ulang. Kolom tabel di-log, disamakan skalanya, kategori di-one-hot. Keduanya
# **digabung jadi satu tabel** lalu dilatih satu model ("early fusion").
#
# **Langkah:**
# 1. Embedding gambar train dan test.
# 2. Olah kolom tabel, gabungkan dengan embedding.
# 3. Latih model, evaluasi F1, buat submission.
#
# **Data demo:** CIFAR-10 kecil + kolom acak `berat_kg`, `panjang_m`, `kecepatan_kmj`,
# `warna_dominan` (sengaja tidak berguna, jadi "tabel saja" seharusnya lemah).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/multimodal"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/gambar_tabular/train.csv"
TEST_CSV = f"{DATA_FOLDER}/gambar_tabular/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/gambar_tabular/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "Z:/GEMASTIK-DATABASE/pretrained"

ID_COL = "id"
PATH_COL = "path"       # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
NUMBER_COLS = ["berat_kg", "panjang_m", "kecepatan_kmj"]
CATEGORY_COLS = ["warna_dominan"]
IMAGE_SIZE = 224
BATCH_SIZE = 128
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406])
MEAN = MEAN.view(1, 3, 1, 1).to(DEVICE)    # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225])
STD = STD.view(1, 3, 1, 1).to(DEVICE)

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Langkah 1a — backbone pretrained beku
# Lapisan klasifikasi terakhir diganti `nn.Identity()` supaya keluarannya embedding.
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok resnet50.

# %%
backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()

# backbone = models.convnext_tiny(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth",
#                                     map_location="cpu"))
# backbone.classifier[2] = nn.Identity()

# backbone = models.resnet18(weights=None)     # paling cepat
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
# backbone.fc = nn.Identity()

backbone = backbone.to(DEVICE)
backbone.eval()

# %% [markdown]
# ## Langkah 1b — embedding semua gambar
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
    return torch.cat(batches).numpy()


# coba pada 2 gambar pertama
print(embed_images(list(train[PATH_COL][:2])).shape)

image_train = embed_images(list(train[PATH_COL]))
image_test = embed_images(list(test[PATH_COL]))
print("embedding train:", image_train.shape, "| test:", image_test.shape)

# %% [markdown]
# ## Langkah 2 — kolom tabel + gabungan
# Angka: log(1 + x) lalu skala disamakan. Kategori: one-hot (satu kolom 0/1 per nilai).

# %%
number_steps = make_pipeline(FunctionTransformer(np.log1p), StandardScaler())
table_steps = ColumnTransformer([
    ("number", number_steps, NUMBER_COLS),
    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORY_COLS),
])
table_train = table_steps.fit_transform(train)
table_test = table_steps.transform(test)

fused_train = np.hstack([image_train, table_train])
fused_test = np.hstack([image_test, table_test])
print("tabel:", table_train.shape, "| gabungan:", fused_train.shape)

# %% [markdown]
# ## Pilih feature (satu baris aktif)
# Coba ketiganya untuk melihat apakah penggabungan menaikkan skor (bahan presentasi).

# %%
X = fused_train
X_test = fused_test
# X = image_train                 # gambar saja (X_test = image_test)
# X = table_train                 # tabel saja (X_test = table_test)
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                  random_state=SEED)

# %% [markdown]
# ## Langkah 3 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
# model = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=SEED)

model.fit(X_train, y_train)
pred_val = model.predict(X_val)
print("F1 macro validasi:", round(f1_score(y_val, pred_val, average="macro"), 4))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("F1 macro test:", round(f1_score(test_key[TARGET], predictions, average="macro"), 4))
