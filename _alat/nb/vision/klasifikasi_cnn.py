# %% [markdown]
# # Vision — Klasifikasi CNN (fine-tune)
# **Kapan dipakai:** klasifikasi gambar dan embedding beku (`klasifikasi_embedding`) belum cukup
# bagus, ada GPU, dan waktu cukup untuk melatih beberapa epoch.
#
# **Langkah:** ResNet18 pretrained → lapisan terakhir diganti sesuai jumlah kelas → semua lapisan
# dilatih ulang dengan loop training torch biasa (tanpa Trainer).
#
# **Data demo:** CIFAR-10 kecil (gambar 32x32, 10 kelas).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.nn import functional
from torchvision import models

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
PRETRAINED = "Z:/GEMASTIK-DATABASE/pretrained"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "label"
IMAGE_SIZE = 64
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42
torch.manual_seed(SEED)
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406])
MEAN = MEAN.view(1, 3, 1, 1)      # normalisasi warna ImageNet
STD = torch.tensor([0.229, 0.224, 0.225])
STD = STD.view(1, 3, 1, 1)

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train.isna().sum())
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Gambar ke tensor (diperbesar ke IMAGE_SIZE, dinormalisasi ImageNet)

# %%
def load_images(paths):
    arrays = []
    for path in paths:
        arrays.append(np.asarray(Image.open(f"{DATA_FOLDER}/{path}").convert("RGB")))
    X = torch.tensor(np.stack(arrays)).permute(0, 3, 1, 2)
    X = X.float() / 255
    X = functional.interpolate(X, size=IMAGE_SIZE, mode="bilinear")
    return (X - MEAN) / STD


X_all = load_images(list(train[PATH_COL]))
X_test = load_images(list(test[PATH_COL]))
print("train:", X_all.shape, "| test:", X_test.shape)

# %% [markdown]
# ## Split train / validasi
# Label teks diubah jadi nomor kelas 0, 1, 2, … (torch butuh angka).

# %%
encoder = LabelEncoder()
y_numbers = encoder.fit_transform(train[TARGET])
y_all = torch.tensor(y_numbers)
train_rows, val_rows = train_test_split(np.arange(len(y_numbers)), test_size=0.2,
                                        stratify=y_numbers, random_state=SEED)
print("train:", len(train_rows), "| validasi:", len(val_rows))

# %% [markdown]
# ## Model
# Pilih SATU blok: hapus `#` di blok yang ingin dipakai, beri `#` pada blok resnet18.

# %%
def make_model():
    n_classes = len(encoder.classes_)
    net = models.resnet18(weights=None)
    net.load_state_dict(torch.load(f"{PRETRAINED}/resnet18/model.pth", map_location="cpu"))
    net.fc = nn.Linear(net.fc.in_features, n_classes)

    # net = models.efficientnet_b0(weights=None)
    # net.load_state_dict(torch.load(f"{PRETRAINED}/efficientnet_b0/model.pth"))
    # net.classifier[1] = nn.Linear(net.classifier[1].in_features, n_classes)

    # net = models.convnext_tiny(weights=None)
    # net.load_state_dict(torch.load(f"{PRETRAINED}/convnext_tiny/model.pth"))
    # net.classifier[2] = nn.Linear(net.classifier[2].in_features, n_classes)

    # net = models.resnet50(weights=None)
    # net.load_state_dict(torch.load(f"{PRETRAINED}/resnet50/model.pth"))
    # net.fc = nn.Linear(net.fc.in_features, n_classes)
    return net.to(DEVICE)


print(type(make_model()).__name__)

# %% [markdown]
# ## Latih (loop training torch)
# Augmentasi: setengah batch dibalik kiri-kanan. Kelas tidak seimbang: hapus `#` di baris
# `class_weight` supaya kelas yang jarang diberi bobot loss lebih besar.

# %%
class_weight = None
# class_weight = (1 / torch.bincount(y_all).float()).to(DEVICE)


def train_model(X_part, y_part):
    net = make_model()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        net.train()
        order = torch.randperm(len(X_part))
        for start in range(0, len(order), BATCH_SIZE):
            batch_rows = order[start:start + BATCH_SIZE]
            batch = X_part[batch_rows]
            if torch.rand(1).item() < 0.5:
                batch = batch.flip(3)     # balik kiri-kanan
            output = net(batch.to(DEVICE))
            loss = functional.cross_entropy(output, y_part[batch_rows].to(DEVICE),
                                            weight=class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch", epoch + 1, "loss", round(loss.item(), 4))
    return net


net = train_model(X_all[train_rows], y_all[train_rows])

# %% [markdown]
# ## Evaluasi di validasi

# %%
def predict_classes(net, X_part):
    net.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(X_part), 256):
            output = net(X_part[start:start + 256].to(DEVICE))
            batches.append(output.argmax(1).cpu())
    return torch.cat(batches).numpy()


val_predictions = predict_classes(net, X_all[val_rows])
y_val = y_numbers[val_rows]
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions, target_names=encoder.classes_))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, display_labels=encoder.classes_,
                                        xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
net = train_model(X_all, y_all)
predictions = encoder.inverse_transform(predict_classes(net, X_test))

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test:", round(f1_score(test_key, predictions, average="macro"), 4))
