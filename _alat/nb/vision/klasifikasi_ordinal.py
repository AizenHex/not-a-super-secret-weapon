# %% [markdown]
# # Vision — Klasifikasi Ordinal (kelas berurutan: tingkat keparahan, kematangan, usia)
# **Kapan dipakai:** kelas punya urutan (0 < 1 < 2 < 3 < 4) dan salah tebak jauh lebih buruk
# daripada salah tebak dekat. Metrik biasanya QWK (quadratic weighted kappa) atau MAE.
#
# **Langkah:** embedding CLIP → regresi (Ridge, tahu bahwa 3 lebih dekat ke 4 daripada ke 0) →
# angka diubah jadi kelas dengan **ambang yang dicari** (bukan pembulatan 0.5/1.5/2.5/3.5).
#
# **Data demo:** RetinaMNIST (foto retina, tingkat retinopati diabetik 0–4), 128 px.

# %%
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/ordinal_retina"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
CLIP_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/clip_b32"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "grade"
LOWEST_GRADE = 0
HIGHEST_GRADE = 4
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
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts().sort_index())

# %% [markdown]
# ## Embedding CLIP (beku)

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


X_all = embed_images(list(train[PATH_COL]))
X_test = embed_images(list(test[PATH_COL]))
y_all = train[TARGET].to_numpy()
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
print("train:", len(X_train), "| validasi:", len(X_val))

# %% [markdown]
# ## Model regresi
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.
# Klasifikasi biasa (mengabaikan urutan) sebagai pembanding: LogisticRegression di bawah,
# lalu pakai `model.predict` langsung tanpa ambang.

# %%
model = make_pipeline(StandardScaler(), Ridge(alpha=3000))
# model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
model.fit(X_train, y_train)

val_values = model.predict(X_val)
rounded = np.clip(np.round(val_values), LOWEST_GRADE, HIGHEST_GRADE).astype(int)
print("dibulatkan QWK:", round(cohen_kappa_score(y_val, rounded, weights="quadratic"), 3),
      "| MAE:", round(mean_absolute_error(y_val, rounded), 3))

# %% [markdown]
# ## Cari ambang (cutpoint)
# Ambang dicari pada tebakan out-of-fold data latih (bukan di validasi, supaya validasi jujur).
# Cara: geser satu ambang sekali, pilih posisi dengan QWK terbaik, ulangi 3 putaran.
# `np.digitize(nilai, ambang)` = nomor kelas: nilai < ambang[0] → 0, dst.

# %%
folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_values = cross_val_predict(model, X_train, y_train, cv=folds)

thresholds = [0.5, 1.5, 2.5, 3.5]     # titik awal = pembulatan biasa
for round_number in range(3):
    for position in range(len(thresholds)):
        best_score = -1
        best_value = thresholds[position]
        for candidate in np.arange(thresholds[position] - 1.0, thresholds[position] + 1.0, 0.05):
            trial = list(thresholds)
            trial[position] = candidate
            guess = np.digitize(oof_values, sorted(trial))
            score = cohen_kappa_score(y_train, guess, weights="quadratic")
            if score > best_score:
                best_score = score
                best_value = candidate
        thresholds[position] = best_value
thresholds = sorted(thresholds)
print("ambang:", np.round(thresholds, 2))

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = np.digitize(val_values, thresholds)
print("QWK:", round(cohen_kappa_score(y_val, val_predictions, weights="quadratic"), 3))
print("MAE:", round(mean_absolute_error(y_val, val_predictions), 3))
print("akurasi:", round(accuracy_score(y_val, val_predictions), 3))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X_all, y_all)
predictions = np.digitize(model.predict(X_test), thresholds)
grade_counts = pd.Series(predictions).value_counts()
print(grade_counts.sort_index())

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["grade"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
print("QWK test:", round(cohen_kappa_score(test_key, predictions, weights="quadratic"), 3))
print("MAE test:", round(mean_absolute_error(test_key, predictions), 3))
print("akurasi test:", round(accuracy_score(test_key, predictions), 3))
