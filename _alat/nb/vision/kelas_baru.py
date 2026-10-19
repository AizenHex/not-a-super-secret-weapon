# %% [markdown]
# # Vision — Kelas Baru di Test (open-set: tolak yang "tidak dikenal")
# **Kapan dipakai:** test bisa berisi gambar dari kelas yang **tidak ada** di train, dan soal
# meminta label khusus (mis. "lainnya"/"unknown") untuk gambar seperti itu.
#
# **Langkah:** embedding CLIP → klasifier biasa → skor "keanehan" tiap gambar → gambar dengan
# skor di atas ambang diberi label `OTHER_LABEL`. Skor utama: 1 − probabilitas tertinggi.
# Alternatif: jarak ke pusat kelas, jarak kNN (baris `#`).
#
# **Data demo:** CIFAR-10 kecil. Train hanya 7 kelas, 3 kelas sengaja disembunyikan dan muncul
# di test.

# %%
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
CLIP_FOLDER = "../../pretrained/clip_b32"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
HIDDEN_CLASSES = ["frog", "horse", "truck"]   # demo saja: kelas yang tidak ada di train
OTHER_LABEL = "lainnya"                        # GANTI sesuai format soal
ACCEPT_SHARE = 0.95   # 95% gambar dikenal diterima. Turunkan (0.85) = lebih banyak tertangkap
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train = train[~train[TARGET].isin(HIDDEN_CLASSES)]   # HAPUS saat lomba (simulasi kelas baru)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Embedding CLIP pretrained (beku)
# CLIP dilatih di banyak sekali jenis gambar, jadi gambar asing lebih mudah dibedakan
# (uji demo AUROC: CLIP 0.86, ResNet18 0.67).

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
            images.append(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(**encoded)
        batches.append(output.pooler_output.cpu().numpy())
    return np.concatenate(batches)


X_train = embed_images(list(train[PATH_COL]))
X_test = embed_images(list(test[PATH_COL]))
print("train:", X_train.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split + klasifier untuk kelas yang dikenal
# Validasi dipakai untuk memilih ambang.

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
# ## Prediksi test + submission

# %%
predictions = model.predict(X_test).astype(object)
predictions[test_scores > threshold] = OTHER_LABEL
print(pd.Series(predictions).value_counts())

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# AUROC: seberapa baik skor memisahkan gambar kelas baru dari kelas dikenal (1 = sempurna).

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
truth = test_key[TARGET].to_numpy().astype(object)
is_new = np.isin(truth, HIDDEN_CLASSES)
truth[is_new] = OTHER_LABEL
print("AUROC dikenal vs baru:", round(roc_auc_score(is_new, test_scores), 3))
print("akurasi total (dengan label lainnya):", round(accuracy_score(truth, predictions), 3))
print("gambar baru yang tertangkap:", round((predictions[is_new] == OTHER_LABEL).mean(), 3))
print("gambar dikenal yang salah ditolak:", round((predictions[~is_new] == OTHER_LABEL).mean(), 3))
