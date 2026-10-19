# %% [markdown]
# # Video — Klasifikasi Video (embedding per frame + arah perubahan)
# **Kapan dipakai:** satu label per video: jenis aktivitas dari CCTV, gerakan olahraga,
# bahasa isyarat, kejadian lalu lintas.
#
# **Idenya:** `N_FRAMES` frame diambil merata, tiap frame diubah jadi embedding oleh CNN
# pretrained yang tidak dilatih ulang. Satu video diringkas jadi:
# - **rata-rata** embedding = isi adegan.
# - **akhir − awal** = arah perubahan. Rata-rata saja tidak tahu urutan: video "ke kanan" dan
#   "ke kiri" rata-ratanya sama.
#
# **Langkah:**
# 1. Baca `N_FRAMES` frame tiap video.
# 2. Embedding tiap frame, ringkas jadi satu baris per video.
# 3. Latih model, evaluasi F1, buat submission.
#
# **Data demo:** video sintetis 96×96, bola bergerak (kanan/kiri/atas/bawah/membesar/diam).

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/video"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "Z:/GEMASTIK-DATABASE/pretrained"

ID_COL = "id"
PATH_COL = "path"       # path video, relatif terhadap DATA_FOLDER
TARGET = "label"
N_FRAMES = 16           # frame diambil merata dari awal sampai akhir video
N_EDGE = 4              # jumlah frame untuk "awal" dan "akhir"
GRID = 2                # peta fitur dirata-rata per kotak 2x2 (posisi objek tetap tersimpan)
IMAGE_SIZE = 224
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
# ## Langkah 1 — baca `N_FRAMES` frame tiap video
# Hasil satu video = array (N_FRAMES, tinggi, lebar, 3).

# %%
def read_video(path):
    video = cv2.VideoCapture(f"{DATA_FOLDER}/{path}")
    frames = []
    ok, frame = video.read()
    while ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ok, frame = video.read()
    positions = np.linspace(0, len(frames) - 1, N_FRAMES)
    positions = positions.round().astype(int)
    return np.stack(frames)[positions]


# coba pada video pertama: frame awal, tengah, akhir
first_video = read_video(train[PATH_COL][0])
print(first_video.shape)
fig, axes = plt.subplots(1, 3, figsize=(7, 3))
axes[0].imshow(first_video[0])
axes[1].imshow(first_video[N_FRAMES // 2])
axes[2].imshow(first_video[-1])
axes[0].set_title(train[TARGET][0])
plt.show()

# %% [markdown]
# ## Langkah 2a — backbone pretrained beku
# ResNet tanpa 2 lapisan terakhir = peta fitur. Pilih SATU blok.

# %%
network = models.resnet18(weights=None)
network.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
# network = models.resnet50(weights=None)
# network.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))

layers = list(network.children())
backbone = nn.Sequential(*layers[:-2])       # buang avgpool + fc
backbone = backbone.to(DEVICE)
backbone.eval()

# %% [markdown]
# ## Langkah 2b — satu baris feature per video

# %%
def video_features(frames):
    batch = torch.tensor(frames).permute(0, 3, 1, 2)
    batch = batch.float().to(DEVICE) / 255
    batch = functional.interpolate(batch, size=IMAGE_SIZE, mode="bilinear")
    with torch.no_grad():
        feature_map = backbone((batch - MEAN) / STD)
    pooled = functional.adaptive_avg_pool2d(feature_map, GRID)
    per_frame = pooled.flatten(1)
    per_frame = per_frame.cpu().numpy()
    average = per_frame.mean(axis=0)
    change = per_frame[-N_EDGE:].mean(axis=0) - per_frame[:N_EDGE].mean(axis=0)
    return np.concatenate([average, change])


# coba pada video pertama
print(video_features(first_video).shape)

train_rows = []
for path in train[PATH_COL]:
    train_rows.append(video_features(read_video(path)))
test_rows = []
for path in test[PATH_COL]:
    test_rows.append(video_features(read_video(path)))
features_train = np.stack(train_rows)
features_test = np.stack(test_rows)
print("train:", features_train.shape, "| test:", features_test.shape)

# %% [markdown]
# ## Pilih feature (satu baris aktif)
# Separuh pertama kolom = rata-rata, separuh kedua = akhir − awal.

# %%
half = features_train.shape[1] // 2
X = features_train
X_test = features_test
# X = features_train[:, :half]          # rata-rata saja (X_test = features_test[:, :half])
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, stratify=y,
                                                  random_state=SEED)

# %% [markdown]
# ## Langkah 3 — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)

model.fit(X_train, y_train)
pred_val = model.predict(X_val)
print("F1 macro validasi:", round(f1_score(y_val, pred_val, average="macro"), 4))
print(classification_report(y_val, pred_val))
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
