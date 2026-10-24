# %% [markdown]
# # Vision — Kelas Baru di Test (open-set: tolak yang "tidak dikenal")
# **Kapan dipakai:** test bisa berisi gambar dari kelas yang **tidak ada** di train, dan soal
# meminta label khusus (mis. "lainnya"/"unknown") untuk gambar seperti itu.
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# **Langkah:** embedding CLIP → klasifier biasa → skor "keanehan" tiap gambar → gambar dengan
# skor di atas ambang diberi label `OTHER_LABEL`. Skor utama: 1 − probabilitas tertinggi.
# Alternatif: jarak ke pusat kelas, jarak kNN (baris `#`).
#
# **Data demo:** `pets` (10 ras kucing/anjing). Train hanya 7 ras; `Bengal`, `shiba_inu`,
# `yorkshire_terrier` sengaja disembunyikan dan muncul di test. Bengal sulit: mirip Abyssinian
# dan Egyptian_Mau yang ada di train.

# %%
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai sample_submission.csv
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
CLIP_FOLDER = "../../pretrained/clip_b32"
OUTPUT_FOLDER = "../../outputs/vision/kelas_baru"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar (juga kolom id di submission)
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
HIDDEN_CLASSES = ["Bengal", "shiba_inu", "yorkshire_terrier"]   # HANYA DEMO, [] saat lomba
OTHER_LABEL = "lainnya"   # GANTI sesuai format soal
ACCEPT_SHARE = 0.95       # 95% gambar dikenal diterima. Turunkan (0.85) = lebih banyak tertangkap
BATCH_SIZE = 64
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data — bentuk 1: CSV `id,label` + folder gambar (aktif)

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["path"] = TRAIN_IMAGE_FOLDER + "/" + train[ID_COL].astype(str) + IMAGE_EXTENSION
test["path"] = TEST_IMAGE_FOLDER + "/" + test[ID_COL].astype(str) + IMAGE_EXTENSION
print("contoh path:", train["path"].iloc[0])

# %% [markdown]
# ## (Alternatif) bentuk 2: folder per kelas `train/<kelas>/*.jpg`

# %%
# rows = []
# for class_name in sorted(os.listdir(TRAIN_IMAGE_FOLDER)):
#     for path in sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/{class_name}/*")):
#         rows.append([os.path.basename(path), class_name, path])
# train = pd.DataFrame(rows, columns=[ID_COL, TARGET, "path"])

# %% [markdown]
# ## (Alternatif) bentuk 3: label di nama file `Abyssinian_12.jpg`

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# train[ID_COL] = train["path"].apply(os.path.basename)
# train[TARGET] = train[ID_COL].str.rsplit("_", n=1).str[0]

# %% [markdown]
# ## Simulasi kelas baru (HANYA DEMO)
# Kelas di `HIDDEN_CLASSES` dibuang dari train. Saat lomba isi `HIDDEN_CLASSES = []`.

# %%
train = train[~train[TARGET].isin(HIDDEN_CLASSES)].reset_index(drop=True)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Embedding CLIP pretrained (beku)
# CLIP dilatih di banyak sekali jenis gambar, jadi gambar asing lebih mudah dibedakan.
# `convert("RGB")` menyamakan GIF/grayscale/RGBA.

# %%
clip = CLIPModel.from_pretrained(CLIP_FOLDER, local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()
processor = CLIPProcessor.from_pretrained(CLIP_FOLDER, local_files_only=True)


def embed_images(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(path).convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(**encoded)
        batches.append(output.pooler_output.cpu().numpy())
    return np.concatenate(batches)


X_train = embed_images(list(train["path"]))
X_test = embed_images(list(test["path"]))
print("train:", X_train.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split + klasifier untuk kelas yang dikenal
# Validasi (semuanya kelas dikenal) dipakai untuk memilih ambang.

# %%
X_part, X_val, y_part, y_val = train_test_split(
    X_train, train[TARGET], test_size=0.2, random_state=SEED, stratify=train[TARGET])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
model.fit(X_part, y_part)
print("akurasi validasi (kelas dikenal):", round(accuracy_score(y_val, model.predict(X_val)), 4))

# %% [markdown]
# ## Skor "tidak dikenal" dan ambang
# Skor tinggi = gambar terlihat asing. Ambang = persentil skor validasi (semua validasi dikenal).
# Pilih SATU blok skor: hapus `#` di blok yang ingin dipakai, beri `#` pada blok lain.

# %%
val_scores = 1 - model.predict_proba(X_val).max(axis=1)
test_scores = 1 - model.predict_proba(X_test).max(axis=1)

# Alternatif 1: jarak ke pusat kelas terdekat
# centers = []
# for name in model.classes_:
#     centers.append(X_part[y_part.to_numpy() == name].mean(axis=0))
# centers = np.array(centers)
# val_scores = np.linalg.norm(X_val[:, None, :] - centers[None], axis=2).min(axis=1)
# test_scores = np.linalg.norm(X_test[:, None, :] - centers[None], axis=2).min(axis=1)

# Alternatif 2: rata-rata jarak ke 5 tetangga terdekat di train
# knn = NearestNeighbors(n_neighbors=5)
# knn.fit(X_part)
# val_scores = knn.kneighbors(X_val)[0].mean(axis=1)
# test_scores = knn.kneighbors(X_test)[0].mean(axis=1)

threshold = np.quantile(val_scores, ACCEPT_SHARE)
print("ambang:", round(threshold, 4))

# %% [markdown]
# ## Sebaran skor: validasi (dikenal) vs test
# Test yang berisi kelas baru biasanya punya "ekor" skor tinggi di kanan garis ambang.

# %%
plt.hist(val_scores, bins=30, alpha=0.6, density=True, label="validasi (dikenal)")
plt.hist(test_scores, bins=30, alpha=0.6, density=True, label="test")
plt.axvline(threshold, color="black", linestyle="--", label="ambang")
plt.xlabel("skor tidak dikenal")
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/sebaran_skor.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Prediksi test + submission

# %%
predictions = model.predict(X_test).astype(object)
predictions[test_scores > threshold] = OTHER_LABEL
print(pd.Series(predictions).value_counts())

submission = pd.DataFrame()
submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv(f"{OUTPUT_FOLDER}/submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# AUROC: seberapa baik skor memisahkan gambar kelas baru dari kelas dikenal (1 = sempurna).

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("urutan id sama:", (test_key[ID_COL].to_numpy() == submission[ID_COL].to_numpy()).all())
truth = test_key[TARGET].to_numpy().astype(object)
is_new = np.isin(truth, HIDDEN_CLASSES)
truth[is_new] = OTHER_LABEL
print("AUROC dikenal vs baru:", round(roc_auc_score(is_new, test_scores), 3))
print("akurasi total (dengan label lainnya):", round(accuracy_score(truth, predictions), 3))
print("F1 macro (dengan label lainnya):", round(f1_score(truth, predictions, average="macro"), 3))
print("gambar baru yang tertangkap:", round((predictions[is_new] == OTHER_LABEL).mean(), 3))
print("gambar dikenal yang salah ditolak:", round((predictions[~is_new] == OTHER_LABEL).mean(), 3))
pd.crosstab(test_key[TARGET].to_numpy()[is_new], predictions[is_new], rownames=["asli"],
            colnames=["tebakan"])

# %% [markdown]
# ## Masalah umum
# - **Kelas baru mirip kelas lama tidak tertangkap** (di demo: Bengal ditebak Egyptian_Mau
#   atau Abyssinian): batas alami metode ini. Turunkan `ACCEPT_SHARE`, atau coba skor alternatif.
# - **Terlalu banyak gambar dikenal ditolak:** naikkan `ACCEPT_SHARE` (0.98).
# - **Tahu berapa banyak gambar baru di test** (mis. soal menyebut 20%)? Ganti ambang dengan
#   `np.quantile(test_scores, 1 - 0.2)`.
# - **Tahu nama kelas barunya** tetapi tidak ada contoh gambarnya? Pakai zero-shot di
#   `klasifikasi_clip`.
