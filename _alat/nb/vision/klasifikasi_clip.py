# %% [markdown]
# # Vision — Klasifikasi CLIP Zero-shot
# **Kapan dipakai:** klasifikasi gambar dengan label sedikit/tidak ada, kelasnya benda umum yang
# bisa ditulis dengan kata (kucing, mobil, pesawat).
#
# **Langkah:** tiap kelas diubah jadi kalimat (`TEMPLATE`) → CLIP mengubah gambar dan kalimat
# jadi vektor → gambar diberi kelas yang kalimatnya paling mirip. Tidak ada latih.
#
# **Data demo:** CIFAR-10 kecil (gambar 32x32, 10 kelas). Bobot: `pretrained/clip_b32`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import functional
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
CLIP_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/clip_b32"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
BATCH_SIZE = 64
SEED = 42
TEMPLATE = "a photo of a {}"
# TEMPLATE = "a photo of the small {}"
# TEMPLATE = "a low-resolution photo of a {}"
# TEMPLATE = "a blurry picture of a {}"
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
# ## Split (zero-shot tidak butuh latih, validasi hanya untuk mengukur)

# %%
train_part, val_part = train_test_split(train, test_size=0.2, stratify=train[TARGET],
                                        random_state=SEED)
classes = sorted(train[TARGET].unique())
print("kelas:", classes)

# %% [markdown]
# ## Model CLIP + vektor kalimat tiap kelas

# %%
clip = CLIPModel.from_pretrained(CLIP_FOLDER, local_files_only=True)
processor = CLIPProcessor.from_pretrained(CLIP_FOLDER, local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()

prompts = []
for name in classes:
    prompts.append(TEMPLATE.format(name))
print("prompt:", prompts[:3])

with torch.no_grad():
    encoded = processor(text=prompts, return_tensors="pt", padding=True).to(DEVICE)
    text_vectors = clip.get_text_features(**encoded).pooler_output
text_vectors = functional.normalize(text_vectors, dim=-1)
text_vectors = text_vectors.cpu().numpy()

# %% [markdown]
# ## Vektor gambar + tebak kelas
# `predict_classes`: kelas dengan kemiripan (perkalian vektor) tertinggi.

# %%
def image_vectors(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = []
        for path in paths[start:start + BATCH_SIZE]:
            images.append(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB"))
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            batches.append(clip.get_image_features(**encoded).pooler_output.cpu())
    vectors = torch.cat(batches)
    return functional.normalize(vectors, dim=-1).numpy()


def predict_classes(paths):
    similarity = image_vectors(paths) @ text_vectors.T
    return np.array(classes)[similarity.argmax(axis=1)]


print(predict_classes(list(val_part[PATH_COL][:5])), list(val_part[TARGET][:5]))

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = predict_classes(list(val_part[PATH_COL]))
print("F1 macro:", round(f1_score(val_part[TARGET], val_predictions, average="macro"), 4))
print(classification_report(val_part[TARGET], val_predictions))
ConfusionMatrixDisplay.from_predictions(val_part[TARGET], val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Prediksi test + submission (tanpa latih ulang)

# %%
predictions = predict_classes(list(test[PATH_COL]))

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
