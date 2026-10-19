# %% [markdown]
# # Vision — Kemiripan Pasangan Gambar (sama / mirip atau bukan?)
# **Kapan dipakai:** soal memberi dua gambar per baris dan bertanya "apakah objek/produk/kategori
# yang sama?" (verifikasi, deteksi duplikat produk, pencocokan foto).
#
# **Langkah:** embedding CLIP tiap gambar → cosine similarity pasangan → ambang dipilih di data
# validasi → "mirip" bila cosine di atas ambang. Tanpa melatih apa pun (hanya ambang).
# Alternatif (baris `#`): fitur pasangan |a−b| dan a*b → LogisticRegression.
#
# **Data demo:** pasangan gambar CIFAR-10, "mirip" = kelas sama. Pasangan dibuat acak 50:50.

# %%
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
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/pasangan_cifar"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/pasangan_train.csv"
TEST_CSV = f"{DATA_FOLDER}/pasangan_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/pasangan_test_labels.csv"   # hanya ada di data latihan
IMAGE_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"
CLIP_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/clip_b32"

IMAGE_A_COL = "gambar_a"   # path gambar, relatif terhadap IMAGE_FOLDER
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
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Embedding CLIP (beku), dinormalisasi supaya perkalian titik = cosine
# Gambar A dan gambar B diubah jadi vektor dengan cara yang sama.

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
            images.append(Image.open(f"{IMAGE_FOLDER}/{path}").convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(**encoded)
        batches.append(output.pooler_output.cpu().numpy())
    vectors = np.concatenate(batches)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # panjang tiap baris = 1


A_train = embed_images(list(train[IMAGE_A_COL]))
B_train = embed_images(list(train[IMAGE_B_COL]))
A_test = embed_images(list(test[IMAGE_A_COL]))
B_test = embed_images(list(test[IMAGE_B_COL]))
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
# ## Alternatif: fitur pasangan → LogisticRegression
# Model belajar sendiri dimensi embedding mana yang penting untuk "mirip".
# Hapus `#` di semua baris untuk memakai (uji demo: AUC test cosine 0.86 → 0.92).

# %%
# pair_train = np.hstack([np.abs(A_train - B_train), A_train * B_train])
# pair_test = np.hstack([np.abs(A_test - B_test), A_test * B_test])
# pair_model = LogisticRegression(max_iter=3000, C=1.0)
# pair_model.fit(pair_train[train_rows], y_train[train_rows])
# similarity_train = pair_model.predict_proba(pair_train)[:, 1]
# similarity_test = pair_model.predict_proba(pair_test)[:, 1]
# val_auc = roc_auc_score(y_train[val_rows], similarity_train[val_rows])
# print("AUC fitur pasangan di validasi:", round(val_auc, 3))

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
# ## Prediksi test + submission

# %%
predictions = (similarity_test > threshold).astype(int)
print("proporsi diprediksi mirip:", round(predictions.mean(), 3))

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["mirip"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
print("AUC test    :", round(roc_auc_score(test_key, similarity_test), 3))
print("akurasi test:", round(accuracy_score(test_key, predictions), 3))
print("F1 test     :", round(f1_score(test_key, predictions), 3))
