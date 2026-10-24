# %% [markdown]
# # Vision — Klasifikasi CLIP Zero-shot (tanpa latih) + linear probe
# **Kapan dipakai:** label latih sangat sedikit atau TIDAK ADA, dan kelasnya bisa ditulis dengan
# kata bahasa Inggris (cat, dog, car, Persian cat, beagle). Punya cukup label (puluhan per
# kelas)? Bagian "linear probe" di bawah atau `klasifikasi_embedding` biasanya lebih bagus.
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
# Tidak ada train sama sekali? Cukup isi `CLASS_TEXT` manual dan baca test saja.
#
# **Langkah:** nama kelas → kalimat bahasa Inggris (beberapa template, dirata-rata) → CLIP
# mengubah gambar dan kalimat jadi vektor → tiap gambar diberi kelas yang kalimatnya paling mirip.
#
# **Data demo:** `pets` = foto 10 ras kucing/anjing (Oxford-IIIT Pet). Nama kelas berupa kode
# (`staffordshire_bull_terrier`) sehingga perlu diterjemahkan ke kalimat.
# Bobot: `pretrained/clip_b32`.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi
# `CLASS_TEXT`: nama kelas soal → teks bahasa Inggris untuk CLIP. Kelas yang tidak diisi memakai
# namanya sendiri dengan `_` diganti spasi. Label bahasa Indonesia WAJIB diterjemahkan
# (CLIP dilatih dengan teks Inggris): `{"kucing": "cat", "mobil": "car"}`.

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai sample_submission.csv
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
CLIP_FOLDER = "../../pretrained/clip_b32"
OUTPUT_FOLDER = "../../outputs/vision/klasifikasi_clip"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
CLASS_TEXT = {"Bengal": "Bengal cat", "Persian": "Persian cat", "Siamese": "Siamese cat",
              "Egyptian_Mau": "Egyptian Mau cat", "pug": "pug dog", "beagle": "beagle dog"}
TEMPLATES = ["a photo of a {}, a type of pet.", "a photo of a {}.", "a close-up photo of a {}.",
             "a blurry photo of a {}."]   # beberapa kalimat dirata-rata = lebih stabil
BATCH_SIZE = 64
VAL_FRACTION = 0.2
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

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
# Zero-shot tidak memakai label untuk belajar; label hanya untuk MENGUKUR. Kelas tidak seimbang
# tetap berarti metrik yang dipakai harus F1 macro.

# %%
counts = train[TARGET].value_counts()
print(counts)
print("jumlah kelas:", len(counts), "| rasio terbanyak/tersedikit:",
      round(counts.max() / counts.min(), 1))
print("id ganda:", train[ID_COL].duplicated().sum(), "| label kosong:", train[TARGET].isna().sum())

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
train_info[(train_info["format"] != "JPEG") | (train_info["mode"] != "RGB")]

# %% [markdown]
# ## Kelas → kalimat untuk CLIP
# Periksa daftar ini dengan teliti: teks yang jelek = skor jelek. Teks yang lebih spesifik
# ("Persian cat" bukan "Persian") membantu, karena "Persian" juga bisa berarti karpet/bahasa.

# %%
classes = sorted(train[TARGET].unique())
# classes = ["kucing", "anjing"]        # tanpa train: tulis daftar kelas manual
class_texts = []
for class_name in classes:
    text = CLASS_TEXT.get(class_name, str(class_name).replace("_", " "))
    class_texts.append(text)
    print(f"{class_name:30} → {text}")

# %% [markdown]
# ## Model CLIP + vektor teks tiap kelas (rata-rata semua template)

# %%
clip = CLIPModel.from_pretrained(CLIP_FOLDER, local_files_only=True)
processor = CLIPProcessor.from_pretrained(CLIP_FOLDER, local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()

text_vectors = []
for text in class_texts:
    prompts = []
    for template in TEMPLATES:
        prompts.append(template.format(text))
    encoded = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        vectors = clip.get_text_features(**encoded).pooler_output
    vectors = functional.normalize(vectors, dim=-1).mean(0)
    text_vectors.append(functional.normalize(vectors, dim=0).cpu())
text_vectors = torch.stack(text_vectors).numpy()
print("vektor teks:", text_vectors.shape, "| contoh prompt:", TEMPLATES[0].format(class_texts[0]))

# %% [markdown]
# ## Vektor gambar
# `processor` sudah mengecilkan & memotong tengah gambar ke 224x224; cukup `convert("RGB")`.

# %%
def image_vectors(paths):
    batches = []
    paths = list(paths)
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(path).convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            batches.append(clip.get_image_features(**encoded).pooler_output.cpu())
    return functional.normalize(torch.cat(batches), dim=-1).numpy()


X_all = image_vectors(train["path"])
X_test = image_vectors(test["path"])
print("vektor gambar train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Zero-shot: evaluasi di SEMUA train (tidak ada yang dilatih, jadi semua bisa dipakai menguji)
# Kemiripan = perkalian vektor gambar dan vektor teks; kelas dengan kemiripan tertinggi menang.

# %%
y_all = train[TARGET].to_numpy()
similarity = X_all @ text_vectors.T
zero_shot = np.array(classes)[similarity.argmax(axis=1)]
print("F1 macro zero-shot:", round(f1_score(y_all, zero_shot, average="macro"), 4))
print(classification_report(y_all, zero_shot, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_all, zero_shot, xticks_rotation=90)
plt.show()

# %% [markdown]
# ## Linear probe: punya label? latih LogisticRegression di vektor gambar CLIP
# Biasanya jauh lebih bagus dari zero-shot bila tiap kelas punya puluhan gambar.

# %%
train_rows, val_rows = train_test_split(np.arange(len(y_all)), test_size=VAL_FRACTION,
                                        stratify=y_all, random_state=SEED)
probe = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
probe.fit(X_all[train_rows], y_all[train_rows])
probe_val = probe.predict(X_all[val_rows])
zero_shot_val = zero_shot[val_rows]
print("validasi yang sama | zero-shot F1:",
      round(f1_score(y_all[val_rows], zero_shot_val, average="macro"), 4),
      "| linear probe F1:", round(f1_score(y_all[val_rows], probe_val, average="macro"), 4))

# %% [markdown]
# ## Submission
# Pilih SATU: zero-shot (aktif, tanpa label) atau linear probe (dilatih ulang di semua train).

# %%
predictions = np.array(classes)[(X_test @ text_vectors.T).argmax(axis=1)]
# probe.fit(X_all, y_all)
# predictions = probe.predict(X_test)

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
# - **Satu kelas hampir selalu menang:** teksnya terlalu umum ("animal") atau ambigu. Buat teks
#   tiap kelas lebih spesifik di `CLASS_TEXT`.
# - **Label bahasa Indonesia / kode (`K01`):** wajib diterjemahkan di `CLASS_TEXT`.
# - **`OSError ... clip_b32`:** cek `CLIP_FOLDER`; harus berisi `config.json` dan bobot.
# - **`FileNotFoundError` gambar:** lihat `contoh path` di sel Baca data
#   (`IMAGE_EXTENSION`, folder).
# - **Skor zero-shot rendah untuk kelas teknis** (jenis cacat produk, penyakit daun): CLIP tidak
#   mengenal istilahnya; pakai linear probe atau `klasifikasi_embedding`.
