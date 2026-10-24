# %% [markdown]
# # Vision — Label Berisik (cari gambar yang salah label dengan cleanlab)
# **Kapan dipakai:** label gambar dibuat manusia dan kamu curiga sebagian salah: kelas mirip
# tertukar, anotator terburu-buru, atau `analisis_kesalahan` menemukan banyak "yakin tapi salah".
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# **Langkah:** embedding ResNet50 → peluang out-of-fold → cleanlab menandai gambar yang labelnya
# bertentangan dengan tebakan model yang yakin → lihat grid → buang ATAU perbaiki → latih ulang.
#
# **Data demo:** `pets` (10 ras kucing/anjing, Oxford-IIIT Pet). Sebagian label **sengaja
# ditukar** antar ras yang mirip supaya deteksinya bisa diukur. Saat lomba, HAPUS sel
# "Rusak sebagian label".

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from cleanlab.filter import find_label_issues
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai sample_submission.csv
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "../../../pretrained"
OUTPUT_FOLDER = "../../../outputs/vision/label_berisik"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
SWAP_FRACTION = 0.3     # HANYA DEMO: proporsi label yang ditukar
SWAPS = {"Bengal": "Abyssinian", "Persian": "Siamese", "beagle": "staffordshire_bull_terrier"}
IMAGE_SIZE = 224
BATCH_SIZE = 64
N_FOLDS = 5
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)   # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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

# %% [markdown]
# ## (Alternatif) bentuk 3: label di nama file `Abyssinian_12.jpg`

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# train[ID_COL] = train["path"].apply(os.path.basename)
# train[TARGET] = train[ID_COL].str.rsplit("_", n=1).str[0]

# %% [markdown]
# ## Rusak sebagian label (HANYA DEMO, HAPUS sel ini saat lomba)
# Menukar label antar ras yang mirip, persis kesalahan anotator sungguhan.

# %%
rng = np.random.default_rng(SEED)
true_labels = train[TARGET].to_numpy().copy()
noisy_labels = true_labels.copy()
for right_class in SWAPS:
    hit = (true_labels == right_class) & (rng.random(len(train)) < SWAP_FRACTION)
    noisy_labels[hit] = SWAPS[right_class]
train[TARGET] = noisy_labels
is_corrupted = noisy_labels != true_labels
print("label dirusak:", is_corrupted.sum(), "dari", len(train))

# %% [markdown]
# ## Gambar → tensor → embedding ResNet50 beku
# `convert("RGB")` menyamakan GIF/grayscale/RGBA; `ImageOps.fit` memotong tengah lalu mengecilkan.

# %%
def load_images(paths):
    images = np.zeros((len(paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, path in enumerate(paths):
        image = Image.open(path).convert("RGB")
        images[i] = ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE))
    return torch.tensor(images).permute(0, 3, 1, 2)     # (jumlah, 3, tinggi, lebar)


def embed(images):
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            batch = (images[start:start + BATCH_SIZE].float() / 255 - MEAN) / STD
            batches.append(backbone(batch.to(DEVICE)).cpu())
    return torch.cat(batches).numpy()


backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
X_all = embed(load_images(train["path"]))
X_test = embed(load_images(test["path"]))
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Peluang out-of-fold lalu cari label bermasalah
# Out-of-fold = tiap gambar ditebak model yang TIDAK melihat gambar itu saat latih. Kalau model
# tidak melihatnya tapi tetap yakin itu kelas lain → labelnya patut dicurigai.
# Pilih SATU model (harus punya `predict_proba`).

# %%
encoder = LabelEncoder()
y_numbers = encoder.fit_transform(train[TARGET])     # nomor kelas 0, 1, 2, …

model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
# model = KNeighborsClassifier(n_neighbors=10, metric="cosine")

folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probabilities = cross_val_predict(model, X_all, y_numbers, cv=folds, method="predict_proba")

issue_rows = find_label_issues(labels=y_numbers, pred_probs=oof_probabilities,
                               return_indices_ranked_by="self_confidence", n_jobs=1)
print("dicurigai salah label:", len(issue_rows), "dari", len(train))

suspects = pd.DataFrame()
suspects[ID_COL] = train[ID_COL].to_numpy()[issue_rows]
suspects["label_tertulis"] = train[TARGET].to_numpy()[issue_rows]
suspects["tebakan_model"] = encoder.classes_[oof_probabilities[issue_rows].argmax(1)]
suspects.to_csv(f"{OUTPUT_FOLDER}/dicurigai.csv", index=False)
suspects.head(10)

# %% [markdown]
# ## Lihat 10 gambar paling mencurigakan
# Judul: label tertulis → tebakan model. Gambar jelas cocok dengan tebakan = labelnya salah.

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for ax in axes.ravel():
    ax.axis("off")
for ax, row in zip(axes.ravel(), issue_rows[:10]):
    ax.imshow(Image.open(train["path"].iloc[row]).convert("RGB"))
    guess = encoder.classes_[oof_probabilities[row].argmax()]
    ax.set_title(f"{train[TARGET].iloc[row]}\n→ {guess}", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/dicurigai.png", dpi=100)
plt.show()

# %% [markdown]
# ## Latih ulang dengan label bersih
# Pilih SATU cara: BUANG baris bermasalah (aktif, aman) atau PERBAIKI labelnya dengan tebakan
# model (hapus `#` di 4 baris alternatif). Memperbaiki lebih berisiko bila model sering salah.

# %%
keep = np.ones(len(train), dtype=bool)
keep[issue_rows] = False
clean_labels = train[TARGET].to_numpy()
# clean_labels = train[TARGET].to_numpy().copy()
# for row in issue_rows:
#     clean_labels[row] = encoder.classes_[oof_probabilities[row].argmax()]
# keep = np.ones(len(train), dtype=bool)

clean_model = clone(model)
clean_model.fit(X_all[keep], clean_labels[keep])
print("baris dipakai:", keep.sum(), "dari", len(train))

# %% [markdown]
# ## Submission

# %%
predictions = clean_model.predict(X_test)

submission = pd.DataFrame()
submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv(f"{OUTPUT_FOLDER}/submission.csv", index=False)
print(submission.shape)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Seberapa tepat deteksi label rusak, dan apakah membersihkan label menaikkan skor test?
# Precision = dari yang ditandai, berapa yang memang rusak. Recall = dari yang rusak, berapa
# yang tertangkap.

# %%
flagged = np.zeros(len(train), dtype=bool)
flagged[issue_rows] = True
print("precision deteksi:", round(precision_score(is_corrupted, flagged), 3))
print("recall deteksi   :", round(recall_score(is_corrupted, flagged), 3))

noisy_model = clone(model)
noisy_model.fit(X_all, train[TARGET])
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
noisy_f1 = f1_score(test_key, noisy_model.predict(X_test), average="macro")
clean_f1 = f1_score(test_key, predictions, average="macro")
print("F1 macro test, label berisik      :", round(noisy_f1, 4))
print("F1 macro test, setelah dibersihkan:", round(clean_f1, 4))

# %% [markdown]
# ## Masalah umum
# - **Hampir semua gambar kelas kecil ditandai:** model lemah di kelas itu, bukan labelnya salah.
#   Cek grid dulu; jangan buang buta.
# - **`find_label_issues` kosong:** model kurang yakin; pakai backbone lebih kuat.
# - **Yang ditandai sedikit tapi label jelas banyak salah:** label salah yang konsisten (seluruh
#   kelas A ditulis B) tidak bisa dideteksi; harus dicek manual.
# - **Skor test turun setelah membuang:** terlalu banyak data bagus terbuang; pakai hanya 50%
#   teratas `issue_rows` (sudah urut dari yang paling mencurigakan).
