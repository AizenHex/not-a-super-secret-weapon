# %% [markdown]
# # CNN penentu orientasi (0/90/180/270°): gambar terbalik/menyamping → diluruskan
# Ide: label tidak perlu dibuat tangan. Ambil gambar yang SUDAH tegak, putar sendiri 0/90/180/270°
# → itu labelnya. Latih CNN (resnet18 pretrained) menebak putaran, lalu luruskan semua gambar.
#
# **Kapan dipakai:** OCR sering salah memilih 0° vs 180° (teks terbalik terbaca jelek). CNN ini
# belajar sendiri bentuk "teks tegak", lalu menebak putaran tiap potongan.
#
# **Butuh:** `crops` dengan `fixed_path` (dari `potong_perbaiki`). Untuk sumber (b) juga `conf` dan
# `rotation` dari `ocr.ipynb` sel 4. Kernel mati? `crops = pd.read_csv("outputs/crops.csv")`.
#
# **Menghasilkan:** `crops["rotation_cnn"]` = berapa kali `np.rot90` supaya tegak, disimpan ke
# `outputs/crops.csv`.
#
# ## Cara menempel ke notebook-mu
# Salin **berurutan**, di bawah sel OCR "Baca SEMUA":
# sumber gambar tegak (**pilih (a) ATAU (b)**, bukan dua-duanya) → Dataset → Model → Latih →
# Tebak putaran. Setelah itu, untuk membaca ulang dengan putaran CNN, lihat sel terakhir.

# %% [markdown]
# ## Sumber gambar tegak, pilih SATU
# (a) Folder gambar yang sudah tegak (mis. train yang dicek mata, atau teks sintetis).

# %%
import glob

UPRIGHT_FOLDER = "data/upright"   # UBAH
upright_paths = sorted(glob.glob(f"{UPRIGHT_FOLDER}/*"))
print("gambar tegak:", len(upright_paths))

# %% [markdown]
# (b) Pseudo-label dari OCR: potongan yang skornya tinggi diputar ke rotasi pilihan OCR
# (`crops["rotation"]` dari `ocr.ipynb`) lalu dianggap tegak. Ubah: `MIN_CONF`, `UPRIGHT_FOLDER`.

# %%
import os

import numpy as np
from PIL import Image

UPRIGHT_FOLDER = "outputs/upright"   # UBAH
MIN_CONF = 0.5                       # UBAH: makin tinggi makin bersih, makin sedikit

os.makedirs(UPRIGHT_FOLDER, exist_ok=True)
upright_paths = []
for row in crops[crops["conf"] >= MIN_CONF].itertuples():
    image = np.array(Image.open(row.fixed_path))
    upright = np.ascontiguousarray(np.rot90(image, row.rotation))
    path = f"{UPRIGHT_FOLDER}/{os.path.basename(row.fixed_path)}"
    Image.fromarray(upright).save(path)
    upright_paths.append(path)
print("gambar tegak:", len(upright_paths), "dari", len(crops))

# %% [markdown]
# ## Dataset: tiap gambar tegak → `len(ROTATIONS)` contoh berlabel putaran
# Ubah: `ROTATIONS` ([0, 2] bila cukup tegak vs terbalik), `IMAGE_SIZE`. Gambar diberi tepi putih
# jadi persegi (rasio tidak gepeng).

# %%
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

ROTATIONS = [0, 1, 2, 3]   # UBAH
IMAGE_SIZE = 128           # UBAH


def to_tensor(image):
    # PIL → tensor 3×IMAGE_SIZE×IMAGE_SIZE, dinormalisasi seperti ImageNet
    image = ImageOps.pad(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), color=(255, 255, 255))
    array = np.asarray(image, dtype=np.float32) / 255
    array = (array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.tensor(array, dtype=torch.float32).permute(2, 0, 1)


class RotationDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths) * len(ROTATIONS)

    def __getitem__(self, index):
        path_index, label = divmod(index, len(ROTATIONS))
        image = Image.open(self.paths[path_index])
        return to_tensor(image.rotate(90 * ROTATIONS[label], expand=True)), label

# %% [markdown]
# ## Model + latih
# Ubah: `RESNET_WEIGHTS`, `EPOCHS`, `BATCH` (kecil = GPU aman).

# %%
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

RESNET_WEIGHTS = "pretrained/resnet18/model.pth"   # UBAH
EPOCHS = 5                                         # UBAH
BATCH = 32                                         # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_paths, val_paths = train_test_split(upright_paths, test_size=0.2, random_state=42)
train_loader = DataLoader(RotationDataset(train_paths), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(RotationDataset(val_paths), batch_size=BATCH)
model = models.resnet18(weights=None)
model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location="cpu"))
model.fc = nn.Linear(model.fc.in_features, len(ROTATIONS))   # kepala baru: satu kelas per putaran
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# %%
import torch


def accuracy(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(DEVICE)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return correct / total


for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        loss = loss_fn(model(images.to(DEVICE)), labels.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {epoch + 1} | loss {loss.item():.4f} | akurasi val {accuracy(val_loader):.4f}")

# %% [markdown]
# ## Tebak putaran tiap potongan → `crops["rotation_cnn"]` → gambar tegak = `np.rot90(image, k)`
# **Butuh:** `model` yang sudah dilatih, `crops["fixed_path"]`. **Menghasilkan:**
# `crops["rotation_cnn"]`, `outputs/crops.csv`. Gambar yang ditebak = gambar yang sama jenisnya
# dengan saat latih (potongan hasil `fix`).

# %%
import numpy as np
import torch
from PIL import Image


def upright_turns(image):
    # gambar diputar ROTATIONS[i] kali → perlu diputar balik (4 - putaran) kali
    model.eval()
    with torch.no_grad():
        logits = model(to_tensor(image).unsqueeze(0).to(DEVICE))
    return (4 - ROTATIONS[int(logits.argmax())]) % 4


turns = []
for fixed_path in crops["fixed_path"]:
    turns.append(upright_turns(Image.open(fixed_path)))
crops["rotation_cnn"] = turns
crops.to_csv("outputs/crops.csv", index=False)
print(crops["rotation_cnn"].value_counts())
if "rotation" in crops:
    print("sama dengan pilihan OCR:", (crops["rotation_cnn"] == crops["rotation"]).mean().round(3))

# %% [markdown]
# ## Pakai putaran CNN untuk OCR
# **Butuh:** `crops["rotation_cnn"]`, fungsi `read_easyocr` dari `ocr.ipynb` sel 2.
# Gambar diputar dulu sesuai tebakan CNN, lalu dibaca sekali saja (tidak perlu coba 0° dan 180°).

# %%
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

texts = []
for row in tqdm(crops.itertuples(), total=len(crops)):
    image = np.array(Image.open(row.fixed_path))
    upright = np.ascontiguousarray(np.rot90(image, row.rotation_cnn))
    text, _ = read_easyocr(upright)
    texts.append(text)
crops["text_cnn_rotation"] = texts
crops.to_csv("outputs/crops.csv", index=False)
print(crops[["text", "text_cnn_rotation"]].head(10).to_string())
