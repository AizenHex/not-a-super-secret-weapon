# %% [markdown]
# # Klasifikasi gambar: embedding pretrained + model sklearn (baseline tercepat, coba DULU)
#
# **Butuh:** `train` (kolom `path`, `label`) dan `test` (kolom `path` + kolom id soal).
# **Menghasilkan:** `submission.csv`, `outputs/val_predictions.csv` (untuk analisis salah tebak).
#
# Kapan: satu label per gambar, data ratusan–ribuan, waktu sempit. Biasanya menang jauh dari fitur
# klasik dan hanya sedikit kalah dari fine-tune (`finetune.ipynb`), tapi 10× lebih cepat.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi, `open_rgb`, backbone, `embed_images` | **ya** |
# | 2 | vektor SEMUA train + test (+ simpan `.npy`) | **ya** |
# | 3 | validasi: split 80/20, laporan per kelas, salah tebak | **ya** |
# | 4 | cross-validation 5 fold (skor lebih bisa dipercaya) | disarankan |
# | 5 | latih di semua train → `submission.csv` | **ya** |
# | 6 | lihat gambar yang salah tebak | opsional |
#
# Sudah punya vektor dari `jembatan/gambar_ke_vektor.ipynb`? Lewati sel 1–2, muat `.npy`-nya.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
train["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + train["id"]
test = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/test.csv")
test["path"] = f"{DEMO_FOLDER}/vision/pets/test/" + test["id"]
print(train["label"].value_counts())

# %% [markdown]
# ## 1. Konfigurasi, `open_rgb`, backbone, `embed_images`
# Ganti backbone: lihat sel 2 `jembatan/gambar_ke_vektor.ipynb` (resnet50, efficientnet, ...).

# %%
import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

PRETRAINED_FOLDER = "pretrained"   # UBAH
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)

backbone = models.convnext_tiny(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"))
backbone.classifier[2] = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()
embed_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def open_rgb(path):   # versi lengkap (PNG transparan → latar putih): vision/dasar sel 1
    image = ImageOps.exif_transpose(Image.open(path))
    return image.convert("RGB")


print(open_rgb(train["path"].iloc[0]).size)

# %% [markdown]
# Fungsi `embed_images`: daftar path → array (jumlah gambar, 768).

# %%
def embed_images(paths):
    vectors = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            batch.append(embed_transform(open_rgb(path)))
        with torch.no_grad():
            output = backbone(torch.stack(batch).to(DEVICE))
        vectors.append(output.cpu().numpy())
    return np.concatenate(vectors)


print(embed_images(train["path"].tolist()[:2]).shape)

# %% [markdown]
# ## 2. Vektor SEMUA train + test → `X`, `X_test`, `y`

# %%
import numpy as np

X = embed_images(train["path"].tolist())
X_test = embed_images(test["path"].tolist())
y = train["label"].to_numpy()
np.save("outputs/X.npy", X)
np.save("outputs/X_test.npy", X_test)
print(X.shape, X_test.shape)

# %% [markdown]
# ## 3. Validasi 80/20 → skor, laporan per kelas, `outputs/val_predictions.csv`
# `stratify=y`: proporsi kelas di train dan val sama. Model: satu baris aktif, alternatif di `#`.
# Kelas timpang? `class_weight="balanced"` sudah aktif di LogisticRegression.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000,
                                                           class_weight="balanced"))
# from sklearn.svm import SVC
# model = make_pipeline(StandardScaler(), SVC(C=10, probability=True))   # sering sedikit lebih baik
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=5, metric="cosine")           # data sangat sedikit
# from lightgbm import LGBMClassifier
# model = LGBMClassifier(n_estimators=300, verbose=-1)                   # biasanya kalah di sini


index_train, index_val = train_test_split(range(len(y)), test_size=0.2, stratify=y,
                                          random_state=42)
model.fit(X[index_train], y[index_train])
val_pred = model.predict(X[index_val])
print("akurasi:", round(accuracy_score(y[index_val], val_pred), 4),
      "| F1 macro:", round(f1_score(y[index_val], val_pred, average="macro"), 4))
print(classification_report(y[index_val], val_pred))
val_table = train.iloc[index_val].copy()
val_table["pred"] = val_pred
val_table.to_csv("outputs/val_predictions.csv", index=False)

# %% [markdown]
# ## 4. Cross-validation 5 fold (opsional, skor lebih stabil daripada satu split)

# %%
from sklearn.model_selection import StratifiedKFold, cross_val_score

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=folds, scoring="f1_macro")
print("F1 macro per fold:", scores.round(4), "| rata-rata:", round(scores.mean(), 4))

# %% [markdown]
# ## 5. Latih di SEMUA train → `submission.csv`
# Kolom submission disalin dari `sample_submission.csv` soal.

# %%
import pandas as pd

model.fit(X, y)
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]          # UBAH: nama kolom dari sample_submission
submission["label"] = predictions      # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(submission["label"].value_counts())

# %% [markdown]
# ## 6. Lihat gambar yang salah tebak (dari sel 3)
# Pola salah tebak memberi petunjuk: kelas yang mirip, gambar terlalu gelap, label train salah.

# %%
import matplotlib.pyplot as plt

wrong = val_table[val_table["label"] != val_table["pred"]].head(8)
fig, axes = plt.subplots(1, len(wrong), figsize=(2.5 * len(wrong), 3), squeeze=False)
for ax, (_, row) in zip(axes[0], wrong.iterrows()):
    ax.imshow(open_rgb(row["path"]))
    ax.set_title(f"asli: {row['label']}\ntebak: {row['pred']}", fontsize=8)
    ax.axis("off")
plt.show()
