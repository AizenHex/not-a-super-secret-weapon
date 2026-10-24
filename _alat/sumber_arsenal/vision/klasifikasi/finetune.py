# %% [markdown]
# # Klasifikasi gambar: fine-tune CNN pretrained (PyTorch)
#
# **Butuh:** `train` (kolom `path`, `label`), `test` (kolom `path` + id soal), GPU disarankan.
# **Menghasilkan:** `outputs/best_model.pt`, `outputs/test_probabilities.npy`, `submission.csv`.
#
# Kapan: `embedding.ipynb` sudah dicoba dan masih kurang, gambar soal jauh dari foto biasa
# (rontgen, satelit, aksara, tekstur), atau data ribuan+. Lebih lama, biasanya lebih akurat.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi, `open_rgb`, transform | **ya** |
# | 2 | `ImageDataset`, split train/val, DataLoader | **ya** |
# | 3 | model pretrained + kepala baru (jumlah kelas soal) | **ya** |
# | 4 | fungsi `run_epoch` (latih / prediksi satu putaran) | **ya** |
# | 5 | loop training, simpan bobot terbaik | **ya** |
# | 6 | kurva loss/F1 | disarankan |
# | 7 | muat bobot terbaik → prediksi test → `submission.csv` | **ya** |
#
# Kernel mati di tengah training? Jalankan sel 1–4 lalu langsung sel 7 (bobot terbaik tersimpan).
# Lanjut training dari checkpoint: `umum/simpan_muat.ipynb` bagian 4.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
train["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + train["id"]
test = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/test.csv")
test["path"] = f"{DEMO_FOLDER}/vision/pets/test/" + test["id"]

# %% [markdown]
# ## 1. Konfigurasi, `open_rgb`, transform
# Augmentasi: lihat `vision/dasar/gambar_dasar.ipynb` bagian 4 (teks → jangan flip).

# %%
import os

import torch
from PIL import Image, ImageOps
from torchvision import transforms

PRETRAINED_FOLDER = "pretrained"   # UBAH
IMAGE_SIZE = 224                   # UBAH
BATCH_SIZE = 32                    # UBAH: "CUDA out of memory" → 16
EPOCHS = 8                         # UBAH
LEARNING_RATE = 1e-4               # UBAH: 1e-4 aman untuk fine-tune
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("outputs", exist_ok=True)
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(), normalize])


def open_rgb(path):
    image = ImageOps.exif_transpose(Image.open(path))
    return image.convert("RGB")


print(DEVICE, open_rgb(train["path"].iloc[0]).size)

# %% [markdown]
# ## 2. Dataset, split train/val, DataLoader
# **Menghasilkan:** `CLASSES`, `train["target"]` (nomor kelas), `train_loader`, `val_loader`,
# `test_loader`. Test tidak punya label → nomor 0 sebagai pengisi (tidak dipakai).

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

CLASSES = sorted(train["label"].unique())
train["target"] = train["label"].map(dict(zip(CLASSES, range(len(CLASSES)))))


class ImageDataset(Dataset):
    def __init__(self, paths, targets, transform):
        self.paths, self.targets, self.transform = list(paths), list(targets), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(open_rgb(self.paths[i])), self.targets[i]


part_train, part_val = train_test_split(train, test_size=0.2, stratify=train["target"],
                                        random_state=42)
train_loader = DataLoader(ImageDataset(part_train["path"], part_train["target"], train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(ImageDataset(part_val["path"], part_val["target"], test_transform),
                        batch_size=BATCH_SIZE)
test_loader = DataLoader(ImageDataset(test["path"], np.zeros(len(test), dtype=int),
                                      test_transform), batch_size=BATCH_SIZE)
print(len(CLASSES), "kelas |", len(part_train), "train |", len(part_val), "val")

# %% [markdown]
# ## 3. Model pretrained + kepala baru → `network`
# Kepala lama menebak 1000 kelas ImageNet → diganti `nn.Linear(..., len(CLASSES))`.
# `FREEZE = True`: hanya kepala yang dilatih (cepat, data sangat sedikit, < 50 per kelas).

# %%
import torch
from torch import nn
from torchvision import models

FREEZE = False   # UBAH

network = models.convnext_tiny(weights=None)
network.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/convnext_tiny/model.pth"))
if FREEZE:
    for parameter in network.parameters():
        parameter.requires_grad = False
network.classifier[2] = nn.Linear(network.classifier[2].in_features, len(CLASSES))

# network = models.resnet50(weights=None)          # alternatif: kepala bernama fc
# network.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/resnet50/model.pth"))
# network.fc = nn.Linear(network.fc.in_features, len(CLASSES))
# network = models.efficientnet_b0(weights=None)   # alternatif ringan
# network.load_state_dict(torch.load(f"{PRETRAINED_FOLDER}/efficientnet_b0/model.pth"))
# network.classifier[1] = nn.Linear(network.classifier[1].in_features, len(CLASSES))

network = network.to(DEVICE)
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(network.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# %% [markdown]
# ## 4. Fungsi `run_epoch(loader, training)` → (loss rata-rata, probabilitas per kelas)
# `training=True`: bobot diperbarui. `False`: hanya menebak (val/test).

# %%
import numpy as np
import torch
from tqdm import tqdm


def run_epoch(loader, training):
    network.train(training)
    total_loss = 0.0
    probabilities = []
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        with torch.set_grad_enabled(training):
            logits = network(images)
            loss = loss_function(logits, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(images)
        probability = torch.softmax(logits.detach(), dim=1)
        probabilities.append(probability.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(probabilities)


val_loss, val_probabilities = run_epoch(val_loader, training=False)
print("sebelum latih | val loss:", round(val_loss, 3), "| bentuk:", val_probabilities.shape)

# %% [markdown]
# ## 5. Loop training → `outputs/best_model.pt` (bobot dengan F1 val terbaik)

# %%
import torch
from sklearn.metrics import f1_score

history = {"train_loss": [], "val_loss": [], "val_f1": []}
best_f1 = -1.0
for epoch in range(EPOCHS):
    train_loss, _ = run_epoch(train_loader, training=True)
    val_loss, val_probabilities = run_epoch(val_loader, training=False)
    scheduler.step()
    val_f1 = f1_score(part_val["target"], val_probabilities.argmax(axis=1), average="macro")
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_f1"].append(val_f1)
    print(f"epoch {epoch + 1}: train {train_loss:.3f} | val {val_loss:.3f} | F1 {val_f1:.4f}")
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(network.state_dict(), "outputs/best_model.pt")
print("F1 val terbaik:", round(best_f1, 4))

# %% [markdown]
# ## 6. Kurva loss / F1
# Val loss naik sementara train loss turun = overfitting → kurangi `EPOCHS`, tambah augmentasi.

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].legend()
axes[0].set_title("loss")
axes[1].plot(history["val_f1"])
axes[1].set_title("F1 val")
plt.show()

# %% [markdown]
# ## 7. Bobot terbaik → prediksi test → `submission.csv`
# Probabilitas disimpan untuk digabung dengan model lain (rata-rata probabilitas = ensemble).
# Mau memakai val juga untuk latihan? Setelah yakin dengan `EPOCHS`, latih ulang di semua train.

# %%
import numpy as np
import pandas as pd
import torch

network.load_state_dict(torch.load("outputs/best_model.pt", map_location=DEVICE))
_, test_probabilities = run_epoch(test_loader, training=False)
np.save("outputs/test_probabilities.npy", test_probabilities)
predicted_numbers = test_probabilities.argmax(axis=1)

submission = pd.DataFrame()
submission["id"] = test["id"]                                     # UBAH
submission["label"] = np.array(CLASSES)[predicted_numbers]        # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())
