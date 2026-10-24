# %% [markdown]
# # Segmentasi semantik: tebak kelas untuk SETIAP piksel (latar / jalan / bangunan / ...)
#
# **Butuh:** tabel `pairs` (kolom `image_path`, `mask_path`) + tabel `test` (kolom
# `image_path`). Mask = gambar seukuran foto, nilai piksel = nomor kelas (atau warna per kelas).
# **Menghasilkan:** `outputs/best_seg.pt`, mask tebakan di `outputs/pred_masks/`,
# `submission.csv` (RLE).
#
# Hanya perlu kotak, bukan bentuk persis? Pakai deteksi (`vision/deteksi/deteksi_yolo`), lebih
# mudah dilabeli. Perlu memisahkan tiap objek (orang 1, orang 2)? `mask_kontur.ipynb` bagian
# YOLO-seg.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `load_mask` (nomor kelas / warna → nomor) + lihat overlay | **ya** |
# | 2 | `SegmentationDataset` (gambar dan mask di-resize/flip bersama) + loader | **ya** |
# | 3 | model LRASPP/DeepLabV3 pretrained, kepala `N_CLASSES` | **ya** |
# | 4 | `run_epoch` + IoU per kelas | **ya** |
# | 5 | loop training, simpan bobot terbaik | **ya** |
# | 6 | prediksi test → mask seukuran asli → PNG + lihat | **ya** |
# | 7 | mask → RLE → `submission.csv` (kalau format soal RLE) | sesuai soal |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Foto pejalan kaki + mask (0 = latar, 1, 2, 3... = orang ke-n → dijadikan 1 = orang).

# %%
import glob

import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

folder = f"{DEMO_FOLDER}/vision/detection/PennFudanPed"
pairs = pd.DataFrame()
pairs["image_path"] = sorted(glob.glob(f"{folder}/PNGImages/*.png"))
pairs["mask_path"] = pairs["image_path"].str.replace("PNGImages", "PedMasks")
pairs["mask_path"] = pairs["mask_path"].str.replace(".png", "_mask.png")
test = pairs.tail(20).reset_index(drop=True)          # pura-pura test (tanpa mask)
test["id"] = test["image_path"].str.extract(r"(\w+)\.png$", expand=False)
pairs = pairs.head(len(pairs) - 20)
print(len(pairs), "pasangan |", len(test), "test")

# %% [markdown]
# ## 1. Konfigurasi + `load_mask(path)` → array nomor kelas (tinggi, lebar)
# Mask berwarna (ekspor CVAT "Segmentation mask", LabelMe): isi `COLOR_TO_CLASS`.
# Mask abu-abu berisi nomor kelas: kosongkan `COLOR_TO_CLASS`.

# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PRETRAINED_FOLDER = "pretrained"   # UBAH
N_CLASSES = 2                      # UBAH: termasuk latar (kelas 0)
IMAGE_SIZE = 320                   # UBAH: kelipatan 32
BATCH_SIZE = 8
EPOCHS = 10                        # UBAH
COLOR_TO_CLASS = {}                # UBAH: {(0, 0, 0): 0, (250, 50, 83): 1, ...}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs/pred_masks", exist_ok=True)


def load_mask(path):
    mask = np.array(Image.open(path))
    if len(COLOR_TO_CLASS) > 0:
        classes = np.zeros(mask.shape[:2], dtype=np.int64)
        for color, number in COLOR_TO_CLASS.items():
            classes[(mask[:, :, :3] == color).all(axis=2)] = number
        return classes
    return (mask > 0).astype(np.int64)   # UBAH: nilai mask = nomor kelas → mask.astype(np.int64)


print(load_mask(pairs["mask_path"].iloc[0]).shape)

# %% [markdown]
# Cek: nilai di mask harus 0..N_CLASSES-1, dan warna menutup objek yang benar.

# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

example_mask = load_mask(pairs["mask_path"].iloc[0])
print("nilai di mask:", np.unique(example_mask))
plt.imshow(Image.open(pairs["image_path"].iloc[0]))
plt.imshow(example_mask, alpha=0.4, cmap="jet")   # mask ditumpuk transparan
plt.show()

# %% [markdown]
# ## 2. `SegmentationDataset` + loader
# Gambar dan mask HARUS diubah bersamaan (resize sama, flip sama). Mask di-resize dengan
# `NEAREST` supaya nomor kelas tidak tercampur (tidak ada kelas 0,5).

# %%
import random

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional


class SegmentationDataset(Dataset):
    def __init__(self, table, training):
        self.table, self.training = table.reset_index(drop=True), training

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        image = Image.open(self.table["image_path"][i]).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        mask = Image.fromarray(load_mask(self.table["mask_path"][i]).astype(np.uint8))
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        if self.training and random.random() < 0.5:
            image, mask = functional.hflip(image), functional.hflip(mask)
        tensor = functional.normalize(functional.to_tensor(image), [0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])
        return tensor, torch.from_numpy(np.array(mask, dtype=np.int64))


print(SegmentationDataset(pairs, True)[0][0].shape)

# %% [markdown]
# Split train/val + loader → `train_loader`, `val_loader`.

# %%
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

part_train, part_val = train_test_split(pairs, test_size=0.2, random_state=42)
train_loader = DataLoader(SegmentationDataset(part_train, True), batch_size=BATCH_SIZE,
                          shuffle=True)
val_loader = DataLoader(SegmentationDataset(part_val, False), batch_size=BATCH_SIZE)
images, masks = next(iter(train_loader))
print(images.shape, masks.shape, masks.unique())

# %% [markdown]
# ## 3. Model pretrained, kepala baru `N_CLASSES` → `network`

# %%
import torch
from torch import nn
from torchvision.models import segmentation

network = segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None,
                                                 num_classes=21)
network.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/lraspp_mobilenet/model.pth"))
network.classifier.low_classifier = nn.Conv2d(40, N_CLASSES, 1)
network.classifier.high_classifier = nn.Conv2d(128, N_CLASSES, 1)
# alternatif lebih akurat, lebih lambat:
# network = segmentation.deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None,
#                                                     num_classes=21, aux_loss=True)
# network.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/deeplabv3_mobilenet/model.pth"))
# network.classifier[4] = nn.Conv2d(256, N_CLASSES, 1)
network = network.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(network.parameters(), lr=3e-4)

# %% [markdown]
# ## 4. `run_epoch` + IoU per kelas
# IoU kelas k = (piksel tebak k DAN benar k) / (piksel tebak k ATAU benar k). 1 = sempurna.
# mIoU = rata-rata IoU semua kelas (metrik segmentasi yang paling umum).

# %%
import numpy as np
import torch
from tqdm import tqdm


def run_epoch(loader, training):
    network.train(training)
    total_loss, intersection, union = 0.0, np.zeros(N_CLASSES), np.zeros(N_CLASSES)
    for images, masks in tqdm(loader, leave=False):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        with torch.set_grad_enabled(training):
            logits = network(images)["out"]
            loss = loss_function(logits, masks)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(images)
        predicted = logits.argmax(dim=1)
        for number in range(N_CLASSES):
            intersection[number] += ((predicted == number) & (masks == number)).sum().item()
            union[number] += ((predicted == number) | (masks == number)).sum().item()
    return total_loss / len(loader.dataset), intersection / np.maximum(union, 1)


val_loss, val_iou = run_epoch(val_loader, training=False)
print("sebelum latih | IoU per kelas:", val_iou.round(3))

# %% [markdown]
# ## 5. Loop training → `outputs/best_seg.pt` (mIoU val terbaik)

# %%
import torch

best_miou = -1.0
for epoch in range(EPOCHS):
    train_loss, _ = run_epoch(train_loader, training=True)
    val_loss, val_iou = run_epoch(val_loader, training=False)
    print(f"epoch {epoch + 1}: train {train_loss:.3f} | val {val_loss:.3f} | "
          f"IoU {val_iou.round(3)} | mIoU {val_iou.mean():.4f}")
    if val_iou.mean() > best_miou:
        best_miou = val_iou.mean()
        torch.save(network.state_dict(), "outputs/best_seg.pt")
print("mIoU val terbaik:", round(best_miou, 4))

# %% [markdown]
# ## 6. Prediksi test → mask seukuran gambar asli → `outputs/pred_masks/<id>.png`
# **Menghasilkan:** `test_masks` (list array nomor kelas). Keluaran model berukuran
# `IMAGE_SIZE`; dikembalikan ke ukuran asli dengan `interpolate`.

# %%
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as nn_functional

network.load_state_dict(torch.load("outputs/best_seg.pt", map_location=DEVICE))
network.eval()
test_masks = []
for i in range(len(test)):
    original = Image.open(test["image_path"][i]).convert("RGB")
    resized = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = functional.normalize(functional.to_tensor(resized), [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
    with torch.no_grad():
        logits = network(tensor.unsqueeze(0).to(DEVICE))["out"]
        logits = nn_functional.interpolate(logits, size=(original.height, original.width))
    mask = logits.argmax(dim=1)[0].cpu()
    mask = mask.numpy().astype(np.uint8)

    test_masks.append(mask)
    Image.fromarray(mask).save(f"outputs/pred_masks/{test['id'][i]}.png")
plt.imshow(Image.open(test["image_path"][0]))
plt.imshow(test_masks[0], alpha=0.4, cmap="jet")
plt.show()

# %% [markdown]
# ## 7. Mask → RLE → `submission.csv`
# RLE (run-length encoding) = "mulai panjang mulai panjang ..." untuk piksel bernilai 1.
# Format paling umum (Kaggle): piksel dihitung **per kolom** (atas→bawah, lalu kolom berikutnya),
# mulai dari 1. Cek deskripsi soal: kalau per baris, ganti `order="F"` jadi `order="C"`.

# %%
import numpy as np
import pandas as pd


def mask_to_rle(mask):
    pixels = np.concatenate([[0], mask.flatten(order="F"), [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1   # posisi nilai berubah
    changes[1::2] = changes[1::2] - changes[::2]           # posisi akhir → panjang
    return " ".join(changes.astype(str))


print(mask_to_rle(np.array([[0, 1], [1, 1]])))   # kolom: 0,1,1,1 → "2 3"
rle_texts = []
for mask in test_masks:
    rle_texts.append(mask_to_rle((mask == 1).astype(np.uint8)))   # UBAH: kelas yang dikirim
submission = pd.DataFrame()
submission["id"] = test["id"]              # UBAH
submission["rle"] = rle_texts              # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head(3))
