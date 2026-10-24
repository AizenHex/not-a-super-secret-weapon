# %% [markdown]
# # Vision — Klasifikasi Klasik (HOG + histogram warna + model sklearn, tanpa deep learning)
# **Kapan dipakai:** tidak ada GPU dan tidak boleh/tidak bisa memakai model pretrained, gambar
# sederhana (ikon, tekstur, warna dominan, latar seragam), atau sebagai pembanding cepat.
# Untuk foto sungguhan (hewan, orang, benda) hampir selalu kalah jauh dari
# `klasifikasi_embedding`: lihat skor demo di bawah.
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# **Langkah:** baca tabel → cek label & file gambar → semua gambar jadi RGB 128x128 (WAJIB sama
# ukurannya, kalau tidak panjang fitur HOG berbeda) → fitur HOG (bentuk/tepi) + histogram warna
# HSV → model sklearn → cross-validation → submission.
#
# **Data demo:** `pets` = foto 10 ras kucing/anjing (Oxford-IIIT Pet). Sengaja kotor: jumlah per
# kelas 200 → 30, ukuran 100–3000 px, ada GIF/PNG berekstensi `.jpg`, grayscale, RGBA.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai sample_submission.csv
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
OUTPUT_FOLDER = "../../outputs/vision/klasifikasi_klasik"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
IMAGE_SIZE = 128        # semua gambar diseragamkan ke ukuran ini
HOG_CELL = 16           # ukuran kotak HOG (piksel), kecil = fitur lebih rinci dan lebih panjang
COLOR_BINS = 16         # jumlah kotak histogram per kanal warna
VAL_FRACTION = 0.2
N_FOLDS = 5
SEED = 42

# %% [markdown]
# ## Baca data — bentuk 1: CSV `id,label` + folder gambar (aktif)
# Kolom `path` = lokasi file gambar. `astype(str)` supaya id berupa angka (`101`) juga bisa.

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["path"] = TRAIN_IMAGE_FOLDER + "/" + train[ID_COL].astype(str) + IMAGE_EXTENSION
test["path"] = TEST_IMAGE_FOLDER + "/" + test[ID_COL].astype(str) + IMAGE_EXTENSION
print("train:", train.shape, "| test:", test.shape)
print("contoh path:", train["path"].iloc[0])
train.head()

# %% [markdown]
# ## (Alternatif) bentuk 2: folder per kelas `train/<kelas>/*.jpg`
# Nama subfolder = label. Test biasanya satu folder tanpa label. Hapus `#` untuk memakai.

# %%
# rows = []
# for class_name in sorted(os.listdir(TRAIN_IMAGE_FOLDER)):
#     for path in sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/{class_name}/*")):
#         rows.append([os.path.basename(path), class_name, path])
# train = pd.DataFrame(rows, columns=[ID_COL, TARGET, "path"])
# test = pd.DataFrame()
# test["path"] = sorted(glob.glob(f"{TEST_IMAGE_FOLDER}/*"))
# test[ID_COL] = test["path"].apply(os.path.basename)   # cocokkan dengan format sample_submission
# print("train:", train.shape, "| test:", test.shape)

# %% [markdown]
# ## (Alternatif) bentuk 3: label di nama file `Abyssinian_12.jpg`
# `rsplit("_", n=1)` memotong di garis bawah TERAKHIR: `shiba_inu_7` → `shiba_inu`.

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# train[ID_COL] = train["path"].apply(os.path.basename)
# train[TARGET] = train[ID_COL].str.rsplit("_", n=1).str[0]
# print(train[TARGET].value_counts())

# %% [markdown]
# ## EDA: label
# Rasio kelas terbanyak/tersedikit > 3 = tidak seimbang: pakai F1 macro, split stratified, dan
# `class_weight="balanced"` di model. Akurasi menipu bila kelas tidak seimbang.

# %%
counts = train[TARGET].value_counts()
print(counts)
print("jumlah kelas:", len(counts), "| rasio terbanyak/tersedikit:",
      round(counts.max() / counts.min(), 1))
print("id ganda:", train[ID_COL].duplicated().sum(), "| label kosong:", train[TARGET].isna().sum())
counts.plot.barh(figsize=(6, 4), title="jumlah gambar per kelas")
plt.show()

# %% [markdown]
# ## EDA: file gambar (ada? format asli? ukuran?)
# Ekstensi belum tentu benar: `format` dibaca dari isi file. `mode` RGB = berwarna, L = abu-abu,
# RGBA = ada transparansi, P = palet (GIF). Semuanya nanti diubah ke RGB.

# %%
def image_info(paths):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            rows.append([path, "TIDAK ADA", "", 0, 0])
            continue
        image = Image.open(path)
        rows.append([path, image.format, image.mode, image.width, image.height])
    return pd.DataFrame(rows, columns=["path", "format", "mode", "width", "height"])


train_info = image_info(train["path"])
test_info = image_info(test["path"])
print("format train:", train_info["format"].value_counts().to_dict())
print("format test :", test_info["format"].value_counts().to_dict())
print("mode train:", train_info["mode"].value_counts().to_dict())
print("mode test :", test_info["mode"].value_counts().to_dict())
sizes = pd.DataFrame()
sizes["train_width"] = train_info["width"].describe()
sizes["test_width"] = test_info["width"].describe()
sizes["train_height"] = train_info["height"].describe()
sizes["test_height"] = test_info["height"].describe()
print(sizes.round())
train_info[(train_info["format"] != "JPEG") | (train_info["mode"] != "RGB")]

# %% [markdown]
# ## EDA: lihat contoh gambar per kelas
# Cek dengan mata: apakah label masuk akal, objeknya di tengah, ada gambar aneh/salah label.

# %%
classes = sorted(counts.index)
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax in axes.ravel():
    ax.axis("off")
for ax, class_name in zip(axes.ravel(), classes):
    row = train[train[TARGET] == class_name].iloc[0]
    image = Image.open(row["path"]).convert("RGB")
    ax.imshow(image)
    ax.set_title(f"{class_name}\n{image.width}x{image.height}", fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Gambar → array numpy berukuran sama
# `convert("RGB")` menyamakan semua mode jadi 3 kanal. `ImageOps.fit` memotong bagian tengah
# lalu mengecilkan ke persegi `IMAGE_SIZE`. Ganti dengan `ImageOps.pad` bila objek di tepi.

# %%
def load_images(paths):
    images = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        images[i] = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))
        # images[i] = ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE))
    return images      # (jumlah, tinggi, lebar, 3)


images_all = load_images(train["path"])
images_test = load_images(test["path"])
print("train:", images_all.shape, "| test:", images_test.shape)

# %% [markdown]
# ## Fitur: HOG + histogram warna HSV
# - HOG = arah tepi di tiap kotak `HOG_CELL` piksel → menangkap BENTUK.
# - Histogram HSV = sebaran warna (H = jenis warna, S = kepekatan, V = terang) → menangkap WARNA,
#   lebih tahan perubahan cahaya daripada RGB.

# %%
def image_features(image):
    shape_part = hog(image, orientations=9, pixels_per_cell=(HOG_CELL, HOG_CELL),
                     cells_per_block=(2, 2), channel_axis=-1)
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    parts = [shape_part]
    for channel in range(3):
        counts_part, edges = np.histogram(hsv[:, :, channel], bins=COLOR_BINS, range=(0, 256))
        parts.append(counts_part / counts_part.sum())     # proporsi, bukan jumlah piksel
    return np.concatenate(parts)


train_features = []
for image in images_all:
    train_features.append(image_features(image))
test_features = []
for image in images_test:
    test_features.append(image_features(image))
X_all = np.array(train_features)
X_test = np.array(test_features)
print("fitur train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi (stratified)

# %%
y_all = train[TARGET].to_numpy()
train_rows, val_rows = train_test_split(np.arange(len(y_all)), test_size=VAL_FRACTION,
                                        stratify=y_all, random_state=SEED)
X_train = X_all[train_rows]
X_val = X_all[val_rows]
y_train = y_all[train_rows]
y_val = y_all[val_rows]
print("train:", X_train.shape, "| validasi:", X_val.shape)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced", random_state=SEED))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000,
#                                                            class_weight="balanced"))
# model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=SEED)
# model = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=SEED)
# model = LGBMClassifier(n_estimators=300, class_weight="balanced", verbose=-1, random_state=SEED)
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# Baca confusion matrix: baris = label asli, kolom = tebakan. Angka di luar diagonal = salah.

# %%
val_predictions = model.predict(X_val)
print("akurasi:", round((val_predictions == y_val).mean(), 4))
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=90)
plt.show()

# %% [markdown]
# ## Cross-validation 5 fold (skor lebih stabil daripada satu split)

# %%
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
scores = cross_val_score(model, X_all, y_all, cv=folds, scoring="f1_macro")
print("F1 macro per fold:", scores.round(4))
print("rata-rata:", round(scores.mean(), 4), "| simpangan:", round(scores.std(), 4))

# %% [markdown]
# ## Latih ulang di SEMUA data train, lalu submission
# Format kolom harus sama dengan `sample_submission.csv` (di sini `id,label`).

# %%
model.fit(X_all, y_all)
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv(f"{OUTPUT_FOLDER}/submission.csv", index=False)
print(submission.shape)
print(submission[TARGET].value_counts())
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("urutan id sama:", (test_key["id"] == submission[ID_COL]).all())
print("F1 macro test:", round(f1_score(test_key["label"], predictions, average="macro"), 4))

# %% [markdown]
# ## Masalah umum
# - **`ValueError: setting an array element with a sequence`:** gambar berbeda ukuran → panjang
#   fitur HOG berbeda. Semua gambar harus lewat `load_images` (ukuran sama).
# - **`FileNotFoundError`:** lihat `contoh path` di sel Baca data (`IMAGE_EXTENSION`, folder).
# - **`UnidentifiedImageError`:** file rusak; buang barisnya: `train = train[train["path"] != ...]`.
# - **Lambat:** `HOG_CELL` 16 → 32 atau `IMAGE_SIZE` 128 → 64.
# - **Skor rendah pada foto sungguhan:** wajar untuk fitur klasik; pakai `klasifikasi_embedding`.
