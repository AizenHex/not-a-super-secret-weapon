# %% [markdown]
# # Vision — Analisis Kesalahan (gambar mana yang salah, dan kenapa?)
# **Kapan dipakai:** **setelah** model jadi, untuk bahan perbaikan (augmentasi, data tambahan)
# dan slide presentasi.
#
# **Langkah:** prediksi out-of-fold (tiap gambar train punya tebakan "jujur") → lihat:
# - kelas mana yang tertukar;
# - gambar "yakin tapi salah" (curiga label salah) dan gambar paling ragu;
# - irisan gambar (gelap/terang, kontras rendah) yang paling sering salah.
#
# **Data demo:** CIFAR-10 kecil, embedding CLIP + LogisticRegression.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
CLIP_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/clip_b32"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
N_EXAMPLES = 16       # gambar yang ditampilkan di tiap grid
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
print(train.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Embedding CLIP + ukuran sederhana tiap gambar (kecerahan, kontras)

# %%
clip = CLIPModel.from_pretrained(CLIP_FOLDER, local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()
processor = CLIPProcessor.from_pretrained(CLIP_FOLDER, local_files_only=True)

batches = []
brightness = []
contrast = []
for start in range(0, len(train), BATCH_SIZE):
    images = []
    for path in train[PATH_COL][start:start + BATCH_SIZE]:
        image = Image.open(f"{DATA_FOLDER}/{path}").convert("RGB")
        gray = np.asarray(image.convert("L"), dtype=float)
        brightness.append(gray.mean())
        contrast.append(gray.std())
        images.append(image)
    encoded = processor(images=images, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = clip.get_image_features(**encoded)
    batches.append(output.pooler_output.cpu().numpy())
X = np.concatenate(batches)
train["kecerahan"] = brightness
train["kontras"] = contrast

# %% [markdown]
# ## Prediksi out-of-fold (tiap gambar ditebak model yang tidak melihatnya)

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
proba = cross_val_predict(model, X, train[TARGET], cv=folds, method="predict_proba")
classes = np.array(sorted(train[TARGET].unique()))
train["tebakan"] = classes[proba.argmax(axis=1)]
train["yakin"] = proba.max(axis=1)
train["salah"] = train["tebakan"] != train[TARGET]
print(classification_report(train[TARGET], train["tebakan"]))

# %% [markdown]
# ## Kelas mana yang tertukar?

# %%
ConfusionMatrixDisplay.from_predictions(train[TARGET], train["tebakan"], xticks_rotation=45,
                                        normalize="true", values_format=".2f")
plt.savefig("confusion.png", dpi=150, bbox_inches="tight")
plt.show()

wrong = train[train["salah"]]
pair_counts = wrong.groupby([TARGET, "tebakan"]).size()
pair_counts = pair_counts.sort_values(ascending=False)
print("pasangan paling sering tertukar (label asli, tebakan):")
print(pair_counts.head(10))

# %% [markdown]
# ## Yakin tapi salah (curiga label salah) dan paling ragu

# %%
confident_wrong = wrong.sort_values("yakin", ascending=False).head(N_EXAMPLES)
unsure = train.sort_values("yakin").head(N_EXAMPLES)

for title, table in [("yakin_tapi_salah", confident_wrong), ("paling_ragu", unsure)]:
    fig, axes = plt.subplots(2, N_EXAMPLES // 2, figsize=(16, 4.6))
    for ax, path, label, guess, confidence in zip(axes.ravel(), table[PATH_COL], table[TARGET],
                                                   table["tebakan"], table["yakin"]):
        ax.imshow(Image.open(f"{DATA_FOLDER}/{path}"))
        ax.set_title(f"{label}\n→ {guess} ({confidence:.2f})", fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=150)
    plt.show()

# %% [markdown]
# ## Irisan: apakah gambar gelap / kontras rendah lebih sering salah?
# `pd.qcut` membagi gambar jadi 5 kelompok sama banyak, dari paling gelap ke paling terang.

# %%
train["kelompok_kecerahan"] = pd.qcut(train["kecerahan"], 5)
train["kelompok_kontras"] = pd.qcut(train["kontras"], 5)
error_by_brightness = train.groupby("kelompok_kecerahan", observed=True)["salah"].mean()
error_by_contrast = train.groupby("kelompok_kontras", observed=True)["salah"].mean()
print(error_by_brightness.round(3))
print(error_by_contrast.round(3))
print("rata-rata kesalahan:", round(train["salah"].mean(), 3))

# %% [markdown]
# ## Ringkasan per kelas (untuk slide)

# %%
error_by_class = train.groupby(TARGET)["salah"].mean()
error_by_class = error_by_class.sort_values(ascending=False)
error_by_class.plot.barh(title="Proporsi salah per kelas")
plt.savefig("salah_per_kelas.png", dpi=150, bbox_inches="tight")
plt.show()
train[[PATH_COL, TARGET, "tebakan", "yakin", "salah"]].to_csv("oof.csv", index=False)
