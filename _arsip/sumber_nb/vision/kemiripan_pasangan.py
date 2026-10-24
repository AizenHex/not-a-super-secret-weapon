# %% [markdown]
# # Vision — Kemiripan Pasangan Gambar (sama / mirip atau bukan?)
# **Kapan dipakai:** soal memberi dua gambar per baris dan bertanya "apakah objek/produk/kategori
# yang sama?" (verifikasi, deteksi duplikat produk, pencocokan foto).
#
# **Bentuk data:** CSV pasangan `id,gambar_a,gambar_b,mirip`; kolom gambar berisi path relatif
# terhadap `DATA_FOLDER` (atau hanya nama file; isi `IMAGE_PREFIX`).
#
# **Langkah:** embedding CLIP tiap gambar → (1) cosine similarity pasangan sebagai pembanding →
# (2) fitur pasangan |a−b| dan a*b → LogisticRegression (aktif) → ambang dipilih di validasi.
# Tidak ada pasangan berlabel sama sekali? Lewati sel (2); cosine + ambang tetap jalan.
#
# **Data demo:** pasangan foto `pets`, "mirip" = ras sama, 50:50. Sepertiga pasangan "tidak mirip"
# berasal dari spesies yang sama (kucing vs kucing lain ras), jadi tidak bisa ditebak dari
# kucing/anjing saja.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/pasangan_train.csv"
TEST_CSV = f"{DATA_FOLDER}/pasangan_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/pasangan_test_labels.csv"   # hanya ada di data latihan
IMAGE_PREFIX = f"{DATA_FOLDER}/"   # ditempel di depan isi kolom gambar, mis. f"{DATA_FOLDER}/img/"
CLIP_FOLDER = "../../pretrained/clip_b32"
OUTPUT_FOLDER = "../../outputs/vision/kemiripan_pasangan"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"
IMAGE_A_COL = "gambar_a"
IMAGE_B_COL = "gambar_b"
TARGET = "mirip"
BATCH_SIZE = 64
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
for table in [train, test]:
    table["path_a"] = IMAGE_PREFIX + table[IMAGE_A_COL].astype(str)
    table["path_b"] = IMAGE_PREFIX + table[IMAGE_B_COL].astype(str)
print("train:", train.shape, "| test:", test.shape)
print("contoh path:", train["path_a"].iloc[0])
print(train[TARGET].value_counts())

# %% [markdown]
# ## Embedding CLIP (beku), dinormalisasi supaya perkalian titik = cosine
# Satu gambar bisa muncul di banyak pasangan, jadi tiap gambar unik di-embed SEKALI saja lalu
# dicari lewat kamus `vector_of[path]`.

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
    vectors = np.concatenate(batches)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # panjang tiap baris = 1


all_paths = pd.concat([train["path_a"], train["path_b"], test["path_a"], test["path_b"]])
unique_paths = list(all_paths.unique())
vector_of = dict(zip(unique_paths, embed_images(unique_paths)))
print("gambar unik:", len(unique_paths))

# %% [markdown]
# ## Susun vektor A dan B per pasangan

# %%
def pair_vectors(paths):
    vectors = []
    for path in paths:
        vectors.append(vector_of[path])
    return np.array(vectors)


A_train = pair_vectors(train["path_a"])
B_train = pair_vectors(train["path_b"])
A_test = pair_vectors(test["path_a"])
B_test = pair_vectors(test["path_b"])
print(A_train.shape, A_test.shape)

# %% [markdown]
# ## Split train / validasi + cosine similarity
# Cosine 1 = arah sama persis.

# %%
y_train = train[TARGET].to_numpy()
train_rows, val_rows = train_test_split(np.arange(len(train)), test_size=0.3, random_state=SEED,
                                        stratify=y_train)
similarity_train = (A_train * B_train).sum(axis=1)
similarity_test = (A_test * B_test).sum(axis=1)
val_auc = roc_auc_score(y_train[val_rows], similarity_train[val_rows])
print("AUC cosine di validasi:", round(val_auc, 3))

# %% [markdown]
# ## Fitur pasangan → LogisticRegression (aktif, menimpa skor cosine)
# Model belajar sendiri dimensi embedding mana yang penting untuk "mirip" (di demo: AUC test
# cosine 0.82 → 0.93). |a−b| dan a*b tidak bergantung urutan A/B. Tanpa label pasangan? Beri `#`
# di semua baris sel ini.

# %%
pair_train = np.hstack([np.abs(A_train - B_train), A_train * B_train])
pair_test = np.hstack([np.abs(A_test - B_test), A_test * B_test])
pair_model = LogisticRegression(max_iter=3000, C=1.0)
pair_model.fit(pair_train[train_rows], y_train[train_rows])
similarity_train = pair_model.predict_proba(pair_train)[:, 1]
similarity_test = pair_model.predict_proba(pair_test)[:, 1]
val_auc = roc_auc_score(y_train[val_rows], similarity_train[val_rows])
print("AUC fitur pasangan di validasi:", round(val_auc, 3))

# %% [markdown]
# ## Pilih ambang di validasi (F1 tertinggi)

# %%
best_f1 = 0
threshold = 0.5
for candidate in np.arange(0.0, 1.0, 0.005):
    guess = similarity_train[val_rows] > candidate
    candidate_f1 = f1_score(y_train[val_rows], guess)
    if candidate_f1 > best_f1:
        best_f1 = candidate_f1
        threshold = candidate
print("ambang:", round(threshold, 3), "| F1 validasi:", round(best_f1, 3))

# %% [markdown]
# ## Sebaran skor di validasi: mirip vs tidak mirip
# Makin sedikit tumpang tindih, makin mudah dipisah. Tumpang tindih = pasangan sulit.

# %%
val_y = y_train[val_rows]
val_similarity = similarity_train[val_rows]
plt.hist(val_similarity[val_y == 1], bins=30, alpha=0.6, label="mirip")
plt.hist(val_similarity[val_y == 0], bins=30, alpha=0.6, label="tidak mirip")
plt.axvline(threshold, color="black", linestyle="--", label="ambang")
plt.legend()
plt.savefig(f"{OUTPUT_FOLDER}/sebaran_skor.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Prediksi test + submission

# %%
predictions = (similarity_test > threshold).astype(int)
print("proporsi diprediksi mirip:", round(predictions.mean(), 3))

submission = pd.DataFrame()
submission[ID_COL] = test[ID_COL]
submission[TARGET] = predictions
submission.to_csv(f"{OUTPUT_FOLDER}/submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
print("urutan id sama:", (test_key[ID_COL].to_numpy() == submission[ID_COL].to_numpy()).all())
print("AUC test    :", round(roc_auc_score(test_key[TARGET], similarity_test), 3))
print("akurasi test:", round(accuracy_score(test_key[TARGET], predictions), 3))
print("F1 test     :", round(f1_score(test_key[TARGET], predictions), 3))

# %% [markdown]
# ## Masalah umum
# - **AUC rendah:** "mirip" di soal bukan kemiripan visual umum (mis. produk sama beda sudut).
#   Coba embedding lain (ResNet/ConvNeXt dari `klasifikasi_embedding`) atau `C` lain.
# - **Gambar test juga muncul di pasangan train:** wajar; yang dilatih adalah pasangan, bukan
#   gambar. Tetapi validasi bisa terlalu optimis bila pasangan sama muncul dua kali.
# - **Proporsi mirip di test jauh dari train:** ambang F1 bergantung proporsi; pilih ambang lain
#   (mis. `np.quantile(similarity_test, 1 - proporsi)` bila proporsi test diketahui).
# - **`KeyError` di `vector_of`:** path di CSV beda bentuk (backslash, spasi); cek `IMAGE_PREFIX`.
