# %% [markdown]
# # Vision — Klasifikasi Klasik (HOG + histogram warna + model sklearn)
# **Kapan dipakai:** tanpa GPU / waktu sangat sedikit, gambar kecil dan sederhana, atau sebagai
# pembanding cepat sebelum memakai embedding pretrained (`vision/klasifikasi_embedding`).
#
# **Langkah:** tiap gambar → fitur HOG (bentuk/tepi) + histogram warna → model sklearn biasa.
#
# **Data demo:** CIFAR-10 kecil (gambar 32x32, 10 kelas).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
COLOR_BINS = 16       # jumlah kotak histogram per warna
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Lihat contoh gambar

# %%
fig, axes = plt.subplots(2, 6, figsize=(10, 4))
for ax, path, label in zip(axes.flat, train[PATH_COL][:12], train[TARGET][:12]):
    ax.imshow(Image.open(f"{DATA_FOLDER}/{path}"))
    ax.set_title(label)
    ax.axis("off")
plt.show()

# %% [markdown]
# ## Fitur: HOG + histogram warna
# HOG = arah tepi di tiap kotak 8×8 piksel (bentuk). Histogram = sebaran nilai tiap warna R/G/B.

# %%
def image_features(path):
    image = np.asarray(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB"))
    shape_part = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                     channel_axis=-1)
    parts = [shape_part]
    for channel in range(3):
        counts, edges = np.histogram(image[:, :, channel], bins=COLOR_BINS, range=(0, 256))
        parts.append(counts)
    return np.concatenate(parts)


print("panjang fitur satu gambar:", len(image_features(train.loc[0, PATH_COL])))

train_features = []
for path in train[PATH_COL]:
    train_features.append(image_features(path))
test_features = []
for path in test[PATH_COL]:
    test_features.append(image_features(path))
X_all = np.array(train_features)
X_test = np.array(test_features)
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_all, train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = make_pipeline(StandardScaler(), LinearSVC(C=0.1))
# model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
# model = RandomForestClassifier(n_estimators=300, random_state=SEED)
# model = ExtraTreesClassifier(n_estimators=300, random_state=SEED)
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = model.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X_all, train[TARGET])
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
