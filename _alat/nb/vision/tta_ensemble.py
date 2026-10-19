# %% [markdown]
# # Vision — TTA + Ensemble (gabung beberapa backbone dan versi gambar)
# **Kapan dipakai:** model gambar sudah jalan, ingin skor naik sedikit tanpa melatih ulang
# jaringan.
# - **TTA** (test-time augmentation): tebak gambar asli dan gambar yang dibalik kiri-kanan, lalu
#   rata-ratakan probabilitasnya.
# - **Ensemble**: rata-rata probabilitas dari beberapa backbone (ResNet18 + CLIP).
#
# **Data demo:** CIFAR-10 kecil. Model tiap backbone: embedding beku + LogisticRegression.

# %%
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
from transformers import CLIPModel, CLIPProcessor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
PRETRAINED = "../../pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data + split

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train_part, val_part = train_test_split(train, test_size=0.2, random_state=SEED,
                                        stratify=train[TARGET])
print("train:", train_part.shape, "| validasi:", val_part.shape, "| test:", test.shape)

# %% [markdown]
# ## Baca gambar
# `flip=True` membalik gambar kiri-kanan (untuk TTA).

# %%
def load_images(paths, flip):
    images = []
    for path in paths:
        image = Image.open(f"{DATA_FOLDER}/{path}").convert("RGB")
        if flip:
            image = ImageOps.mirror(image)
        images.append(image)
    return images


print(len(load_images(list(train_part[PATH_COL][:2]), flip=True)))

# %% [markdown]
# ## Backbone 1: ResNet18 (beku)

# %%
resnet = models.resnet18(weights=None)
resnet.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
resnet.fc = nn.Identity()
resnet = resnet.to(DEVICE)
resnet.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406])
MEAN = MEAN.view(1, 3, 1, 1).to(DEVICE)    # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225])
STD = STD.view(1, 3, 1, 1).to(DEVICE)


def resnet_embed(paths, flip):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        arrays = []
        for image in load_images(paths[start:start + BATCH_SIZE], flip):
            arrays.append(np.asarray(image.resize((224, 224))))
        batch = torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2)
        batch = batch.float().to(DEVICE) / 255
        with torch.no_grad():
            batches.append(resnet((batch - MEAN) / STD).cpu().numpy())
    return np.concatenate(batches)


print(resnet_embed(list(train_part[PATH_COL][:2]), flip=True).shape)

# %% [markdown]
# ## Backbone 2: CLIP (beku)

# %%
clip = CLIPModel.from_pretrained(f"{PRETRAINED}/clip_b32", local_files_only=True)
clip = clip.to(DEVICE)
clip.eval()
processor = CLIPProcessor.from_pretrained(f"{PRETRAINED}/clip_b32", local_files_only=True)


def clip_embed(paths, flip):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        images = load_images(paths[start:start + BATCH_SIZE], flip)
        encoded = processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(**encoded)
        batches.append(output.pooler_output.cpu().numpy())
    return np.concatenate(batches)


print(clip_embed(list(train_part[PATH_COL][:2]), flip=True).shape)

# %% [markdown]
# ## Model per backbone + TTA di validasi
# `tta_proba`: rata-rata probabilitas gambar asli dan gambar dibalik.

# %%
def tta_proba(model, embed, paths):
    normal = model.predict_proba(embed(paths, flip=False))
    flipped = model.predict_proba(embed(paths, flip=True))
    return (normal + flipped) / 2


train_paths = list(train_part[PATH_COL])
val_paths = list(val_part[PATH_COL])

resnet_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
resnet_model.fit(resnet_embed(train_paths, flip=False), train_part[TARGET])
resnet_val = tta_proba(resnet_model, resnet_embed, val_paths)

clip_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
clip_model.fit(clip_embed(train_paths, flip=False), train_part[TARGET])
clip_val = tta_proba(clip_model, clip_embed, val_paths)

classes = resnet_model.classes_
resnet_f1 = f1_score(val_part[TARGET], classes[resnet_val.argmax(axis=1)], average="macro")
clip_f1 = f1_score(val_part[TARGET], classes[clip_val.argmax(axis=1)], average="macro")
print("ResNet18 + TTA F1 macro:", round(resnet_f1, 4))
print("CLIP + TTA     F1 macro:", round(clip_f1, 4))

# %% [markdown]
# ## Ensemble
# Pilih SATU cara: hapus `#` di baris yang ingin dipakai.

# %%
RESNET_WEIGHT = 0.5
# RESNET_WEIGHT = 0.3     # bobot lebih besar ke backbone yang lebih kuat (CLIP)
ensemble_val = RESNET_WEIGHT * resnet_val + (1 - RESNET_WEIGHT) * clip_val
val_predictions = classes[ensemble_val.argmax(axis=1)]
print("ENSEMBLE + TTA F1 macro:", round(f1_score(val_part[TARGET], val_predictions,
                                                  average="macro"), 4))

# %% [markdown]
# ## Latih ulang di semua train + submission

# %%
all_paths = list(train[PATH_COL])
test_paths = list(test[PATH_COL])
resnet_model.fit(resnet_embed(all_paths, flip=False), train[TARGET])
clip_model.fit(clip_embed(all_paths, flip=False), train[TARGET])
resnet_test = tta_proba(resnet_model, resnet_embed, test_paths)
clip_test = tta_proba(clip_model, clip_embed, test_paths)
ensemble_test = RESNET_WEIGHT * resnet_test + (1 - RESNET_WEIGHT) * clip_test
predictions = classes[ensemble_test.argmax(axis=1)]

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
