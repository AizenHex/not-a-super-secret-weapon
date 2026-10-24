# %% [markdown]
# # Vision — Klasifikasi Gambar Multilabel (satu gambar boleh punya banyak label)
# **Kapan dipakai:** tag foto (pantai + orang + perahu), jenis kerusakan jalan, penyakit daun
# ganda. Target = satu kolom 0/1 per label.
#
# **Langkah:** backbone pretrained beku → embedding → satu model per label
# (`OneVsRestClassifier`) → ambang per label dicari dari prediksi out-of-fold (bukan 0.5 semua).
# Metrik: F1 macro (rata-rata per label), F1 micro, dan subset accuracy (semua label tepat).
#
# **Data demo:** mosaik 2×2 gambar CIFAR (64×64), tiap mosaik berisi 1–4 kelas berbeda.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/multilabel"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../../pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
LABELS = ["airplane", "automobile", "bird", "cat", "deer",
          "dog", "frog", "horse", "ship", "truck"]          # kolom target 0/1
IMAGE_SIZE = 224
BATCH_SIZE = 64
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
missing = train.isna().sum()
print("missing:", missing.sum())
train.head()

# %% [markdown]
# ## EDA: frekuensi tiap label dan jumlah label per gambar

# %%
print(train[LABELS].sum().sort_values())
labels_per_image = train[LABELS].sum(axis=1)
labels_per_image = labels_per_image.value_counts()
print("label per gambar:", labels_per_image.sort_index().to_dict())

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(Image.open(f"{DATA_FOLDER}/{train[PATH_COL][i]}"))
    present = []
    for label in LABELS:
        if train[label][i] == 1:
            present.append(label)
    ax.set_title(", ".join(present), fontsize=8)
    ax.axis("off")
plt.show()

# %% [markdown]
# ## Backbone pretrained beku
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok resnet50.

# %%
backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()

# backbone = models.convnext_tiny(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth",
#                                     map_location="cpu"))
# backbone.classifier[2] = nn.Identity()

# backbone = models.vit_b_16(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/vit_b_16/model.pth", map_location="cpu"))
# backbone.heads = nn.Identity()

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
y_all = train[LABELS].to_numpy()      # bentuk (n_gambar, n_label)
print("train:", X_all.shape, "| test:", X_test.shape, "| target:", y_all.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2,
                                                  random_state=SEED)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai. Semua pilihan mengeluarkan
# `predict_proba` berbentuk (n_gambar, n_label). `ClassifierChain` memakai tebakan label
# sebelumnya sebagai fitur (berguna bila label sering muncul bersama).

# %%
model = OneVsRestClassifier(make_pipeline(StandardScaler(),
                                          LogisticRegression(C=0.05, max_iter=3000)))
# model = OneVsRestClassifier(LGBMClassifier(n_estimators=200, verbose=-1, random_state=SEED))
# model = ClassifierChain(make_pipeline(StandardScaler(),
#                                       LogisticRegression(C=0.05, max_iter=3000)),
#                         random_state=SEED)

# %% [markdown]
# ## Ambang per label dari prediksi out-of-fold (cross-validation di data latih)
# Untuk tiap label dicoba ambang 0.10, 0.15, …, 0.90; dipilih yang F1-nya tertinggi.

# %%
def find_thresholds(proba, y_true):
    thresholds = []
    for column in range(y_true.shape[1]):
        best_score = -1
        best_threshold = 0.5
        for candidate in np.arange(0.1, 0.91, 0.05):
            score = f1_score(y_true[:, column], proba[:, column] >= candidate)
            if score > best_score:
                best_score = score
                best_threshold = candidate
        thresholds.append(best_threshold)
    return np.array(thresholds)


folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_proba = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")
thresholds = find_thresholds(oof_proba, y_train)
print(pd.Series(thresholds, index=LABELS).round(2))

# %% [markdown]
# ## Evaluasi di validasi: ambang 0.5 vs ambang per label

# %%
model.fit(X_train, y_train)
val_proba = model.predict_proba(X_val)
for name, cutoff in [("ambang 0.5", 0.5), ("ambang per label", thresholds)]:
    val_predictions = (val_proba >= cutoff).astype(int)
    print(name,
          "| F1 macro", round(f1_score(y_val, val_predictions, average="macro"), 4),
          "| F1 micro", round(f1_score(y_val, val_predictions, average="micro"), 4),
          "| subset acc", round(accuracy_score(y_val, val_predictions), 4))

val_predictions = (val_proba >= thresholds).astype(int)
print(classification_report(y_val, val_predictions, target_names=LABELS, zero_division=0))

# %% [markdown]
# ## Latih ulang di semua data + submission (satu kolom 0/1 per label)

# %%
oof_proba = cross_val_predict(model, X_all, y_all, cv=folds, method="predict_proba")
thresholds = find_thresholds(oof_proba, y_all)
model.fit(X_all, y_all)
predictions = (model.predict_proba(X_test) >= thresholds).astype(int)

submission = pd.DataFrame()
submission["id"] = test["id"]
for column, label in enumerate(LABELS):
    submission[label] = predictions[:, column]
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[LABELS].to_numpy()
print("F1 macro test:", round(f1_score(test_key, predictions, average="macro"), 4))
print("subset accuracy test:", round(accuracy_score(test_key, predictions), 4))
