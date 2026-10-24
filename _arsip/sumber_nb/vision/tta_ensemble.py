# %% [markdown]
# # Vision — TTA + Ensemble (gabung beberapa backbone dan versi gambar)
# **Kapan dipakai:** klasifikasi gambar sudah jalan (`klasifikasi_embedding`), ingin skor naik
# sedikit lagi tanpa melatih jaringan. Biasanya dipakai di akhir lomba.
# - **TTA** (test-time augmentation): tebak gambar asli DAN gambar dibalik kiri-kanan, lalu
#   rata-ratakan peluangnya.
# - **Ensemble**: rata-rata peluang dari beberapa backbone (ResNet50 + ConvNeXt + CLIP).
# Naiknya biasanya kecil (0–2 poin). Tabel di bawah menunjukkan apakah benar-benar membantu.
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# **Langkah:** baca → RGB 224x224 → embedding tiap backbone untuk gambar asli & dibalik →
# LogisticRegression per backbone → bandingkan tanpa TTA / dengan TTA / ensemble → submission.
#
# **Data demo:** `pets` = foto 10 ras kucing/anjing (Oxford-IIIT Pet), jumlah per kelas 200 → 30,
# ada GIF/PNG berekstensi `.jpg`, grayscale, RGBA.

# %%
import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torchvision import models
from transformers import CLIPModel

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai sample_submission.csv
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../pretrained"
OUTPUT_FOLDER = "../../outputs/vision/tta_ensemble"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
IMAGE_SIZE = 224        # wajib 224 untuk CLIP
BATCH_SIZE = 64         # CUDA out of memory? turunkan ke 32
VAL_FRACTION = 0.2
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CLIP_MEAN = torch.tensor([0.4815, 0.4578, 0.4082]).view(1, 3, 1, 1)   # normalisasi khas CLIP
CLIP_STD = torch.tensor([0.2686, 0.2613, 0.2758]).view(1, 3, 1, 1)

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
print(train[TARGET].value_counts())

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
# ## Gambar → tensor (EDA lengkap ada di `klasifikasi_embedding`)
# `convert("RGB")` menyamakan GIF/grayscale/RGBA jadi 3 kanal; `ImageOps.fit` memotong tengah
# lalu mengecilkan ke persegi `IMAGE_SIZE`.

# %%
def load_images(paths):
    images = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        images[i] = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))
    return torch.tensor(images).permute(0, 3, 1, 2)     # (jumlah, 3, tinggi, lebar)


images_all = load_images(train["path"])
images_test = load_images(test["path"])
print("train:", tuple(images_all.shape), "| test:", tuple(images_test.shape))

# %% [markdown]
# ## Tiga backbone beku
# Ingin backbone lain? Tambah satu baris di `backbones`: nama → (model, mean, std).
# Hapus baris untuk mempercepat.

# %%
resnet = models.resnet50(weights=None)
resnet.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
resnet.fc = nn.Identity()
convnext = models.convnext_tiny(weights=None)
convnext.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth", map_location="cpu"))
convnext.classifier[2] = nn.Identity()
clip = CLIPModel.from_pretrained(f"{PRETRAINED}/clip_b32", local_files_only=True)


def clip_image_vectors(batch):
    return clip.get_image_features(pixel_values=batch).pooler_output


backbones = {"resnet50": (resnet, IMAGENET_MEAN, IMAGENET_STD),
             "convnext": (convnext, IMAGENET_MEAN, IMAGENET_STD),
             "clip": (clip_image_vectors, CLIP_MEAN, CLIP_STD)}
for net in [resnet, convnext, clip]:
    net.to(DEVICE)
    net.eval()

# %% [markdown]
# ## Embedding: tiap backbone × (gambar asli, gambar dibalik)
# `embeddings[nama]["train"]` = embedding gambar asli, `["train_flip"]` = gambar dibalik.

# %%
def embed(forward, mean, std, images, flip):
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = (images[start:start + BATCH_SIZE].float() / 255 - mean) / std
            if flip:
                batch = batch.flip(3)       # balik kiri-kanan
            batches.append(forward(batch.to(DEVICE)).cpu())
    return torch.cat(batches).numpy()


embeddings = {}
for name, (forward, mean, std) in backbones.items():
    embeddings[name] = {}
    embeddings[name]["train"] = embed(forward, mean, std, images_all, flip=False)
    embeddings[name]["train_flip"] = embed(forward, mean, std, images_all, flip=True)
    embeddings[name]["test"] = embed(forward, mean, std, images_test, flip=False)
    embeddings[name]["test_flip"] = embed(forward, mean, std, images_test, flip=True)
    print(name, embeddings[name]["train"].shape)

# %% [markdown]
# ## Split train / validasi (stratified)

# %%
y_all = train[TARGET].to_numpy()
train_rows, val_rows = train_test_split(np.arange(len(y_all)), test_size=VAL_FRACTION,
                                        stratify=y_all, random_state=SEED)
y_val = y_all[val_rows]
print("train:", len(train_rows), "| validasi:", len(val_rows))

# %% [markdown]
# ## Model per backbone: tanpa TTA vs dengan TTA
# Model dilatih dengan gambar asli + gambar dibalik (augmentasi gratis, data latih jadi 2x).
# `predict_proba` = peluang tiap kelas; TTA = rata-rata peluang gambar asli dan dibalik.

# %%
def fit_model(features, features_flip, y_part):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000,
                                                               class_weight="balanced"))
    model.fit(np.vstack([features, features_flip]), np.concatenate([y_part, y_part]))
    return model


val_probabilities = {}
rows = []
for name in backbones:
    features = embeddings[name]
    model = fit_model(features["train"][train_rows], features["train_flip"][train_rows],
                      y_all[train_rows])
    plain = model.predict_proba(features["train"][val_rows])
    tta = (plain + model.predict_proba(features["train_flip"][val_rows])) / 2
    val_probabilities[name] = tta
    classes = model.classes_
    plain_f1 = f1_score(y_val, classes[plain.argmax(1)], average="macro")
    tta_f1 = f1_score(y_val, classes[tta.argmax(1)], average="macro")
    rows.append([name, round(plain_f1, 4), round(tta_f1, 4)])
pd.DataFrame(rows, columns=["backbone", "F1 tanpa TTA", "F1 dengan TTA"])

# %% [markdown]
# ## Ensemble: rata-rata peluang TTA semua backbone
# Pilih SATU baris `WEIGHTS`. Bobot lebih besar untuk backbone yang skornya lebih tinggi di tabel.
# Jangan menyetel bobot terlalu teliti di validasi kecil: hasilnya bisa kebetulan.

# %%
WEIGHTS = {"resnet50": 1, "convnext": 1, "clip": 1}
# WEIGHTS = {"resnet50": 1, "convnext": 2, "clip": 2}
ensemble_val = 0
for name in backbones:
    ensemble_val = ensemble_val + WEIGHTS[name] * val_probabilities[name]
val_predictions = classes[ensemble_val.argmax(1)]
print("ENSEMBLE + TTA F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))

# %% [markdown]
# ## Latih ulang di SEMUA data train, lalu submission

# %%
ensemble_test = 0
for name in backbones:
    features = embeddings[name]
    model = fit_model(features["train"], features["train_flip"], y_all)
    tta = (model.predict_proba(features["test"]) + model.predict_proba(features["test_flip"])) / 2
    ensemble_test = ensemble_test + WEIGHTS[name] * tta
predictions = model.classes_[ensemble_test.argmax(1)]

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
# - **Ensemble lebih jelek dari backbone terbaik:** backbone lemah ikut menarik turun. Hapus dari
#   `backbones`, atau beri bobot kecil.
# - **TTA flip menurunkan skor:** gambar yang arahnya bermakna (teks, angka, tanda panah, kiri vs
#   kanan). Jangan pakai flip; pakai versi lain (sedikit diperbesar/diperkecil).
# - **`CUDA out of memory`:** `BATCH_SIZE` 64 → 32, atau hapus satu backbone.
# - **Urutan kelas antar model berbeda:** di sini aman, karena semua model sklearn dilatih dengan
#   label yang sama (`classes_` terurut). Menggabungkan model torch? Samakan urutan kelasnya dulu.
