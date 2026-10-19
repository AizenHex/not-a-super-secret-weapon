# %% [markdown]
# # Vision — Klasifikasi Embedding Pretrained (beku + LogisticRegression)
# **Kapan dipakai:** klasifikasi gambar dengan data latih ratusan–ribuan gambar dan waktu
# terbatas. Biasanya jauh lebih baik dari fitur klasik dan jauh lebih cepat dari fine-tune.
#
# **Langkah:** backbone pretrained dibekukan → tiap gambar jadi embedding (vektor angka) →
# model sklearn biasa dilatih di embedding.
#
# **Data demo:** CIFAR-10 kecil (gambar 32x32, 10 kelas).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
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

# backbone = models.resnet18(weights=None)     # paling cepat
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
# backbone.fc = nn.Identity()

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
    return torch.cat(batches).numpy()


X_all = embed_images(list(train[PATH_COL]))
X_test = embed_images(list(test[PATH_COL]))
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_all, train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = make_pipeline(StandardScaler(), LinearSVC(C=0.01))
# model = KNeighborsClassifier(n_neighbors=10, metric="cosine")
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = model.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X_all, train[TARGET])
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
