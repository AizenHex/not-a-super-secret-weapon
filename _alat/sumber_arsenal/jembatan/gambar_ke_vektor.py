# %% [markdown]
# # Gambar → vektor angka (embedding), siap dipakai model tabular / kemiripan / clustering
#
# **Butuh:** tabel `train` dan `test` dengan kolom `path`. **Menghasilkan:** array
# `train_vectors`, `test_vectors` (satu baris per gambar) + `outputs/train_vectors.npy`,
# `outputs/test_vectors.npy`.
#
# Ide: backbone pretrained (dilatih di jutaan foto) dipakai **tanpa dilatih ulang**. Lapisan
# terakhirnya (penebak 1000 kelas ImageNet) dibuang, sisanya mengubah gambar jadi vektor
# 512–2048 angka. Gambar yang mirip → vektor yang dekat. Setelah itu soal gambar menjadi soal
# tabel biasa: LogisticRegression, LightGBM, kNN, KMeans.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `open_rgb` | **ya** |
# | 2 | pilih backbone (satu baris aktif) | **ya** |
# | 3 | fungsi `embed_images` (dicoba pada 3 gambar) | **ya** |
# | 4 | terapkan ke SEMUA train/test + simpan `.npy` | **ya** |
# | 5 | muat `.npy` (setelah kernel restart, lewati sel 2–4) | opsional |
# | 6 | alternatif: vektor CLIP (bisa dibandingkan dengan teks) | opsional |
# | 7 | vektor → kolom tabel, digabung dengan kolom tabular lain | opsional |
#
# Lanjutan: `vision/klasifikasi/embedding.ipynb`, `vision/kemiripan`, `vision/anomali`,
# `vision/pengelompokan`, `jembatan/vektor_ke_model.ipynb`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
train["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + train["id"]
test = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/test.csv")
test["path"] = f"{DEMO_FOLDER}/vision/pets/test/" + test["id"]
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + `open_rgb`
# `PRETRAINED_FOLDER` = folder hasil `download_model.bat` (berisi `resnet50/`, `convnext_tiny/`,
# `clip_b32/`, ...).

# %%
import os

import torch
from PIL import Image, ImageOps

PRETRAINED_FOLDER = "pretrained"   # UBAH
BATCH_SIZE = 64                    # UBAH: "CUDA out of memory" → 32 atau 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)


def open_rgb(path):
    image = ImageOps.exif_transpose(Image.open(path))
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


print(DEVICE, open_rgb(train["path"].iloc[0]).size)

# %% [markdown]
# ## 2. Pilih backbone → `backbone`
# Satu baris model aktif; ganti dengan menukar `#`. Baris `... = nn.Identity()` membuang
# penebak kelas ImageNet, jadi keluaran = vektor. Urutan dicoba: convnext_tiny → resnet50 →
# efficientnet_b0 (cepat) → resnet18 (paling cepat, CPU).

# %%
import torch
from torch import nn
from torchvision import models

backbone = models.convnext_tiny(weights=None)                     # 768 angka, sering terbaik
weights_path = f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"
backbone.load_state_dict(torch.load(weights_path, map_location="cpu"))
backbone.classifier[2] = nn.Identity()

# backbone = models.resnet50(weights=None)                        # 2048 angka
# backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/resnet50/model.pth"))
# backbone.fc = nn.Identity()
# backbone = models.efficientnet_b0(weights=None)                 # 1280 angka
# backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/efficientnet_b0/model.pth"))
# backbone.classifier = nn.Identity()
# backbone = models.resnet18(weights=None)                        # 512 angka, paling cepat
# backbone.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/resnet18/model.pth"))
# backbone.fc = nn.Identity()

backbone = backbone.to(DEVICE)
backbone.eval()

# %% [markdown]
# ## 3. Fungsi `embed_images(paths)` → array (jumlah gambar, panjang vektor)
# **Butuh:** `backbone`, `open_rgb`.
# `FLIP=True`: rata-rata vektor gambar asli + gambar dicerminkan (sedikit lebih stabil, 2× lama;
# jangan untuk teks). Urutan baris vektor = urutan baris tabel.

# %%
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

IMAGE_SIZE = 224   # UBAH
FLIP = False       # UBAH
embed_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def embed_images(paths):
    vectors = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            batch.append(embed_transform(open_rgb(path)))
        batch = torch.stack(batch).to(DEVICE)
        with torch.no_grad():
            output = backbone(batch)
            if FLIP:
                output = (output + backbone(torch.flip(batch, dims=[3]))) / 2
        vectors.append(output.cpu().numpy())
    return np.concatenate(vectors)


print(embed_images(train["path"].tolist()[:3]).shape)

# %% [markdown]
# ## 4. Terapkan ke SEMUA train dan test + simpan → `train_vectors`, `test_vectors`

# %%
import numpy as np

train_vectors = embed_images(train["path"].tolist())
test_vectors = embed_images(test["path"].tolist())
np.save("outputs/train_vectors.npy", train_vectors)
np.save("outputs/test_vectors.npy", test_vectors)
print(train_vectors.shape, test_vectors.shape)

# %% [markdown]
# ## 5. Muat vektor tersimpan (setelah kernel restart)
# Jalankan sel baca data + sel ini saja; sel 2–4 dilewati.

# %%
import numpy as np

train_vectors = np.load("outputs/train_vectors.npy")
test_vectors = np.load("outputs/test_vectors.npy")
print(train_vectors.shape, test_vectors.shape)

# %% [markdown]
# ## 6. Alternatif: vektor CLIP → `train_clip`, `test_clip`
# CLIP dilatih dengan pasangan gambar + kalimat, jadi vektornya bisa dibandingkan dengan vektor
# teks (lihat `vision/klasifikasi/zero_shot_clip.ipynb`). Vektor dibagi panjangnya (norma = 1)
# supaya perkalian titik = cosine similarity.

# %%
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained(f"{PRETRAINED_FOLDER}/clip_b32").to(DEVICE)
processor = CLIPProcessor.from_pretrained(f"{PRETRAINED_FOLDER}/clip_b32")
clip.eval()


def clip_images(paths):
    vectors = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE)):
        batch = []
        for path in paths[start:start + BATCH_SIZE]:
            batch.append(open_rgb(path))
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(DEVICE)
        with torch.no_grad():
            output = clip.get_image_features(pixel_values=pixels).pooler_output
        vectors.append(output.cpu().numpy())
    vectors = np.concatenate(vectors)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


train_clip = clip_images(train["path"].tolist())
test_clip = clip_images(test["path"].tolist())
print(train_clip.shape)

# %% [markdown]
# ## 7. Vektor → kolom tabel (digabung dengan kolom tabular lain)
# **Menghasilkan:** `train_with_vectors`, kolom `img_0`, `img_1`, ... Urutan baris harus sama
# dengan `train` (jangan sort/filter `train` setelah sel 4).
# Lanjut: `jembatan/vektor_ke_model`.

# %%
import pandas as pd

column_names = []
for i in range(train_vectors.shape[1]):
    column_names.append(f"img_{i}")
vector_table = pd.DataFrame(train_vectors, columns=column_names)
train_with_vectors = pd.concat([train.reset_index(drop=True), vector_table], axis=1)
print(train_with_vectors.shape)
print(train_with_vectors.iloc[:3, :6])
