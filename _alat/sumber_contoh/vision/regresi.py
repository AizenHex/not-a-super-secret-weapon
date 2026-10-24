# %% [markdown]
# # Vision — Regresi dari Gambar (tebak angka dari foto)
# **Kapan dipakai:** usia dari foto wajah (Satria Data 2021), harga dari foto barang, berat
# ternak, luas kerusakan, jumlah orang di keramaian. Target berupa **angka**, bukan kelas.
#
# **Langkah:** tiap gambar diperkecil ke ukuran sama → embedding pretrained beku → regresi
# sklearn. Multi-output (mis. usia + gender sekaligus): `y` berisi beberapa kolom, `Ridge`
# langsung bisa, model lain dibungkus `MultiOutputRegressor(...)`.
#
# **Data demo:** PennFudan (foto pejalan kaki, ukuran tiap gambar berbeda), target `jumlah` orang.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from PIL import Image
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/regresi"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/pennfudan_train.csv"
TEST_CSV = f"{DATA_FOLDER}/pennfudan_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/pennfudan_test_labels.csv"   # hanya ada di data latihan
IMAGE_FOLDER = "../../../data_latihan/vision/detection/PennFudanPed"
PRETRAINED = "../../../pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap IMAGE_FOLDER
TARGET = "jumlah"
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = np.array([0.485, 0.456, 0.406])   # normalisasi warna ImageNet
STD = np.array([0.229, 0.224, 0.225])

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts().sort_index())
train.head()

# %% [markdown]
# ## EDA: ukuran gambar dan contoh

# %%
for path in train[PATH_COL][:5]:
    print(path, "ukuran (lebar, tinggi):", Image.open(f"{IMAGE_FOLDER}/{path}").size)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, path, count in zip(axes, train[PATH_COL][:4], train[TARGET][:4]):
    ax.imshow(Image.open(f"{IMAGE_FOLDER}/{path}"))
    ax.set_title(f"{TARGET} = {count}")
    ax.axis("off")
plt.show()

# %% [markdown]
# ## Backbone pretrained beku
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok resnet50.

# %%
backbone = models.resnet50(weights=None)
backbone.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth", map_location="cpu"))
backbone.fc = nn.Identity()

# backbone = models.convnext_tiny(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth",
#                                     map_location="cpu"))
# backbone.classifier[2] = nn.Identity()

# backbone = models.vit_b_16(weights=None)
# backbone.load_state_dict(torch.load(f"{PRETRAINED}/vit_b_16/model.pth", map_location="cpu"))
# backbone.heads = nn.Identity()

backbone = backbone.to(DEVICE)
backbone.eval()

# %% [markdown]
# ## Embedding semua gambar

# %%
def load_image(path):
    image = Image.open(f"{IMAGE_FOLDER}/{path}").convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    return (np.asarray(image) / 255 - MEAN) / STD


def embed_images(paths):
    batches = []
    for start in range(0, len(paths), BATCH_SIZE):
        arrays = []
        for path in paths[start:start + BATCH_SIZE]:
            arrays.append(load_image(path))
        batch = torch.tensor(np.stack(arrays), dtype=torch.float32)
        batch = batch.permute(0, 3, 1, 2).to(DEVICE)
        with torch.no_grad():
            batches.append(backbone(batch).cpu())
    return torch.cat(batches).numpy()


X_all = embed_images(list(train[PATH_COL]))
X_test = embed_images(list(test[PATH_COL]))
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(X_all, train[TARGET], test_size=0.25,
                                                  random_state=SEED)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = make_pipeline(StandardScaler(), Ridge(alpha=1000))
# model = make_pipeline(StandardScaler(), Ridge(alpha=100))
# model = make_pipeline(StandardScaler(), SVR(C=3))
# model = KNeighborsRegressor(n_neighbors=5, metric="cosine")
# model = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbose=-1, random_state=SEED)
# model = TransformedTargetRegressor(make_pipeline(StandardScaler(), Ridge(alpha=1000)),
#                                    func=np.log1p, inverse_func=np.expm1)   # target miring

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi, dibandingkan tebakan rata-rata

# %%
val_predictions = model.predict(X_val)
mean_guess = np.full(len(y_val), y_train.mean())
rounded = np.clip(np.round(val_predictions), 1, None)   # jumlah orang: bulat, minimal 1
print("MAE model:", round(mean_absolute_error(y_val, val_predictions), 4))
print("MAE dibulatkan:", round(mean_absolute_error(y_val, rounded), 4))
print("MAE tebak rata-rata:", round(mean_absolute_error(y_val, mean_guess), 4))
print("RMSE model:", round(root_mean_squared_error(y_val, val_predictions), 4))
plt.scatter(y_val, val_predictions)
plt.plot([1, 8], [1, 8], "r--")
plt.xlabel("asli")
plt.ylabel("prediksi")
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(X_all, train[TARGET])
predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["jumlah"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["jumlah"]
print("MAE test (kunci latihan):", round(mean_absolute_error(test_key, predictions), 4))
