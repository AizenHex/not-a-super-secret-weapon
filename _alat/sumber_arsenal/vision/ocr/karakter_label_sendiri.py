# %% [markdown]
# # OCR aksara non-Latin dengan label buatan sendiri (contoh: aksara Jawa)
#
# **Butuh:** gambar baris/kata + kotak per karakter berlabel (hasil CVAT → `anotasi_baca`),
# atau belum ada label sama sekali (bagian 3 membantu melabeli cepat).
# **Menghasilkan:** model pengenal karakter `outputs/char_cnn.pt`, teks per gambar,
# `submission.csv`.
#
# Kenapa per karakter: OCR pretrained (EasyOCR, TrOCR) tidak mengenal aksara Jawa, dan data
# lomba sedikit. Mengenali SATU karakter (±20–60 kelas) bisa dilatih dari beberapa ratus potongan
# saja. Teks = karakter-karakter yang diurutkan kiri → kanan.
#
# ## Alur
# 1. **Labeli** di CVAT (bagian 1): kotak per karakter, nama label = nama karakter.
# 2. **Potong** tiap kotak → gambar kecil 32×32 (bagian 2).
# 3. Label masih sedikit? **Kelompokkan** potongan tak berlabel, beri nama per kelompok (bagian 3).
# 4. **Latih** CNN kecil pengenal karakter (bagian 4).
# 5. **Pecah** gambar test jadi karakter (bagian 5) → tebak → gabung → teks (bagian 6).
#
# | sel | isi | wajib? |
# |---|---|---|
# | 2 | potong kotak → `crops` (+ simpan PNG) | **ya** |
# | 3 | pengelompokan potongan untuk melabeli cepat | kalau label sedikit |
# | 4 | CNN kecil + latih + akurasi val | **ya** |
# | 5 | `split_characters`: baris → kotak karakter (tanpa detektor) | **ya** (atau YOLO) |
# | 6 | baca SEMUA test → teks → CER → `submission.csv` | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Baris aksara Jawa sintetis dari font Windows "Javanese Text" + kotak per karakter, bentuknya
# sama dengan tabel `boxes` hasil `vision/deteksi/anotasi_baca.ipynb` (path, label, x1..y2).

# %%
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_PATH = "C:/Windows/Fonts/javatext.ttf"
AKSARA = {"ha": "ꦲ", "na": "ꦤ", "ca": "ꦕ", "ra": "ꦫ", "ka": "ꦏ", "da": "ꦢ", "ta": "ꦠ",
          "sa": "ꦱ", "wa": "ꦮ", "la": "ꦭ", "pa": "ꦥ", "dha": "ꦝ", "ja": "ꦗ", "ya": "ꦪ",
          "nya": "ꦚ", "ma": "ꦩ", "ga": "ꦒ", "ba": "ꦧ", "tha": "ꦛ", "nga": "ꦔ"}
random.seed(0)
os.makedirs("data/lines", exist_ok=True)


def make_line(path):
    font = ImageFont.truetype(FONT_PATH, random.randint(40, 56))
    image = Image.new("L", (900, 110), 255)
    draw = ImageDraw.Draw(image)
    left, rows = random.randint(5, 30), []
    for name in random.choices(list(AKSARA), k=random.randint(4, 9)):
        top = random.randint(5, 25)
        x1, y1, x2, y2 = draw.textbbox((left, top), AKSARA[name], font=font)
        draw.text((left, top), AKSARA[name], font=font, fill=0)
        rows.append({"path": path, "label": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        left = x2 + random.randint(6, 25)
    image = image.filter(ImageFilter.GaussianBlur(random.uniform(0, 1.2)))
    image.save(path)
    return rows

# %% [markdown]
# Buat 160 baris train (dengan kotak) dan 40 baris test (hanya teks kunci untuk CER).

# %%
rows = []
for i in range(160):
    rows.extend(make_line(f"data/lines/train_{i:03d}.png"))
boxes = pd.DataFrame(rows)
test_rows = []
for i in range(40):
    line_rows = make_line(f"data/lines/test_{i:03d}.png")
    test_rows.append({"id": f"test_{i:03d}", "path": line_rows[0]["path"],
                      "text": "".join(pd.DataFrame(line_rows)["label"])})
test = pd.DataFrame(test_rows)
print(boxes.head())
per_label = boxes["label"].value_counts()
print("kotak per label:", per_label.min(), "–", per_label.max())

# %% [markdown]
# ## 1. Melabeli di CVAT (dikerjakan di browser, bukan di notebook)
# 1. *Projects → +*: isi **Labels** = semua nama karakter (`ha`, `na`, `ca`, ..., juga
#    sandhangan seperti `wulu`, `suku`, `taling`, `pangkon` kalau muncul). Tipe: *Rectangle*.
# 2. *Tasks → +*: unggah gambar (zip). Gambar ribuan? Labeli 20–50 gambar dulu.
# 3. Gambar kotak rapat per karakter (shortcut **N** = kotak baru, **F** = gambar berikutnya,
#    ganti label sebelum menggambar supaya tidak perlu diubah satu per satu).
#    Sandhangan di atas/bawah huruf: kotak sendiri dengan labelnya.
# 4. Lebih cepat: semua kotak diberi SATU label `karakter`, lalu nama kelas diberikan per
#    kelompok di bagian 3 notebook ini.
# 5. *Actions → Export task dataset* → format **CVAT for images 1.1** (xml) atau **COCO 1.0**.
# 6. Baca dengan `vision/deteksi/anotasi_baca.ipynb` → tabel `boxes` (path, label, x1, y1, x2, y2).

# %% [markdown]
# ## 2. Potong kotak → `crops` (gambar abu-abu 32×32, karakter di tengah) + simpan PNG
# `crop_char` merapatkan kotak ke tinta dulu: kotak label manusia kadang longgar, kotak dari
# `split_characters` (bagian 5) selalu rapat. Tanpa langkah ini potongan train dan test beda
# bentuk → tebakan meleset. `pad_square` menjaga bentuk (huruf tidak gepeng).

# %%
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

CHAR_SIZE = 32       # UBAH
MARGIN = 2           # UBAH: piksel putih di sekeliling huruf
INK_LEVEL = 128      # UBAH: piksel lebih gelap dari ini = tinta (latar terang, tinta gelap)
os.makedirs("outputs/chars", exist_ok=True)


def pad_square(image, size=CHAR_SIZE):
    side = max(image.width, image.height)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((size, size))


def crop_char(gray, x1, y1, x2, y2):
    piece = gray.crop((x1, y1, x2, y2))
    ys, xs = np.where(np.array(piece) < INK_LEVEL)
    if len(xs) > 0:
        piece = piece.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    return pad_square(ImageOps.expand(piece, border=MARGIN, fill=255))


gray = ImageOps.grayscale(Image.open(boxes["path"][0]))
row = boxes.iloc[0]
print(np.array(crop_char(gray, row["x1"], row["y1"], row["x2"], row["y2"])).shape)

# %% [markdown]
# Terapkan ke SEMUA kotak → `crops` (crop_path, path, label) + `outputs/crops_char.csv`.

# %%
from tqdm import tqdm

crop_paths = []
for path, group in tqdm(boxes.groupby("path", sort=False)):
    gray = ImageOps.grayscale(Image.open(path))       # buka sekali per gambar
    for index, row in group.iterrows():
        crop_path = f"outputs/chars/{index:06d}.png"
        crop_char(gray, row["x1"], row["y1"], row["x2"], row["y2"]).save(crop_path)
        crop_paths.append((index, crop_path))
crops = boxes.copy()
crops["crop_path"] = pd.Series(dict(crop_paths))
crops.to_csv("outputs/crops_char.csv", index=False)
print(len(crops), "potongan")

# %% [markdown]
# ## 3. (Label sedikit) Kelompokkan potongan → beri nama per kelompok
# Vektor = piksel 32×32 diratakan (untuk karakter hitam-putih ini lebih cocok daripada
# backbone foto). `N_CLUSTERS` ±1,5–2× jumlah karakter: satu huruf boleh terbagi 2 kelompok,
# yang penting satu kelompok tidak berisi 2 huruf. Lihat grid → isi `CLUSTER_NAMES`.

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

N_CLUSTERS = 30   # UBAH

pixels = []
for crop_path in crops["crop_path"]:
    pixels.append(np.array(Image.open(crop_path), dtype=np.float32).reshape(-1) / 255)
pixels = np.stack(pixels)
reduced = PCA(n_components=40, random_state=42).fit_transform(pixels)
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
crops["cluster"] = kmeans.fit_predict(reduced)
fig, axes = plt.subplots(N_CLUSTERS, 8, figsize=(8, N_CLUSTERS * 1.0), squeeze=False)
for cluster in range(N_CLUSTERS):
    paths = crops.loc[crops["cluster"] == cluster, "crop_path"].head(8).tolist()
    for col in range(8):
        axes[cluster, col].axis("off")
        if col < len(paths):
            axes[cluster, col].imshow(Image.open(paths[col]), cmap="gray")
    axes[cluster, 0].set_title(str(cluster), fontsize=7, loc="left")
plt.show()

# %% [markdown]
# Isi nama per kelompok. Di data contoh label asli ada, jadi kemurnian kelompok bisa dicek:
# kelompok yang berisi >1 label = perlu dipecah (naikkan `N_CLUSTERS`) atau dicek manual.

# %%
CLUSTER_NAMES = {}   # UBAH: {0: "ka", 1: "na", 2: "ka", ...}

crops["label_cluster"] = crops["cluster"].map(CLUSTER_NAMES)
purity = crops.groupby("cluster")["label"].nunique()   # jumlah label berbeda per kelompok
print("kelompok berisi >1 label:", (purity > 1).sum(), "dari", N_CLUSTERS)

# %% [markdown]
# ## 4. CNN kecil pengenal karakter → `outputs/char_cnn.pt`
# Augmentasi: geser/putar/skala sedikit, **tanpa flip** (huruf cermin = huruf lain/salah).

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSES = sorted(crops["label"].unique())
crops["target"] = crops["label"].map(dict(zip(CLASSES, range(len(CLASSES)))))
train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.85, 1.1), fill=255),
    transforms.ToTensor()])
test_transform = transforms.ToTensor()


class CharDataset(Dataset):
    def __init__(self, images, targets, transform):
        self.images, self.targets, self.transform = images, list(targets), transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


print(len(CLASSES), "kelas:", CLASSES)

# %% [markdown]
# Model: 3 blok konvolusi kecil (cukup untuk 32×32, latih dari nol dalam hitungan detik).

# %%
from torch import nn


class CharCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, images):
        return self.head(self.features(images))


network = CharCNN(len(CLASSES)).to(DEVICE)
print(network(torch.zeros(2, 1, CHAR_SIZE, CHAR_SIZE).to(DEVICE)).shape)

# %% [markdown]
# Split train/val → `train_loader`, `val_loader` (semua potongan dimuat ke memori, kecil).

# %%
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

images = []
for crop_path in crops["crop_path"]:
    images.append(Image.open(crop_path).copy())   # copy(): file langsung ditutup
targets = crops["target"].to_numpy()
index_train, index_val = train_test_split(np.arange(len(crops)), test_size=0.2,
                                          stratify=targets, random_state=42)
train_images, val_images = [], []
for i in index_train:
    train_images.append(images[i])
for i in index_val:
    val_images.append(images[i])
train_loader = DataLoader(CharDataset(train_images, targets[index_train], train_transform),
                          batch_size=64, shuffle=True)
val_loader = DataLoader(CharDataset(val_images, targets[index_val], test_transform),
                        batch_size=256)
print(len(train_images), "train |", len(val_images), "val")

# %% [markdown]
# Latih + akurasi val tiap epoch. Akurasi rendah untuk huruf tertentu → lihat
# `classification_report`, tambah label huruf itu.

# %%
import numpy as np
import torch
from sklearn.metrics import classification_report

EPOCHS = 20   # UBAH

optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3)
loss_function = nn.CrossEntropyLoss()
for epoch in range(EPOCHS):
    network.train()
    for batch, batch_targets in train_loader:
        loss = loss_function(network(batch.to(DEVICE)), batch_targets.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    predicted = []
    with torch.no_grad():
        for batch, _ in val_loader:
            predicted.extend(network(batch.to(DEVICE)).argmax(dim=1).tolist())
    accuracy = (np.array(predicted) == targets[index_val]).mean()
    print(f"epoch {epoch + 1}: akurasi val {accuracy:.4f}")
torch.save(network.state_dict(), "outputs/char_cnn.pt")
print(classification_report(targets[index_val], predicted, target_names=CLASSES, digits=3))

# %% [markdown]
# ## 5. `split_characters(path)` → kotak karakter di gambar test (tanpa detektor)
# Threshold Otsu → komponen terhubung → gabung komponen yang jarak horizontalnya kurang dari
# `MERGE_GAP` piksel (negatif = bertumpuk: sandhangan di atas/bawah, huruf dari 2 bagian) →
# urut kiri ke kanan. Cara memilih `MERGE_GAP`: lebih kecil dari jarak antarhuruf terkecil
# di data, lihat hasil cetak sel ini.
# Foto berantakan / latar ramai: latih YOLO 1 kelas `karakter` dari kotak CVAT
# (`vision/deteksi/deteksi_yolo.ipynb`), lalu pakai kotak YOLO menggantikan fungsi ini.

# %%
import cv2
import numpy as np

MIN_AREA = 15   # UBAH: buang titik noise
MERGE_GAP = 3   # UBAH: piksel


def split_characters(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    found = []
    for left, top, width, height, area in stats[1:]:
        if area >= MIN_AREA:
            found.append([int(left), int(top), int(left + width), int(top + height)])
    found.sort()
    merged = []
    for box in found:
        if merged and box[0] - merged[-1][2] < MERGE_GAP:
            last = merged[-1]
            merged[-1] = [min(last[0], box[0]), min(last[1], box[1]),
                          max(last[2], box[2]), max(last[3], box[3])]
        else:
            merged.append(box)
    return merged


print(split_characters(test["path"][0]), test["text"][0])

# %% [markdown]
# ## 6. Baca SEMUA test → teks → CER → `submission.csv`
# Teks = nama karakter digabung (`SEPARATOR = ""` → `hanacaraka`; `"-"` → `ha-na-ca`).
# CER = jarak edit / panjang teks benar (0 = sempurna); butuh teks kunci (data latihan saja).

# %%
import pandas as pd
import torch
from PIL import Image, ImageOps

SEPARATOR = ""   # UBAH


def read_line(path):
    gray = ImageOps.grayscale(Image.open(path))
    batch = []
    for x1, y1, x2, y2 in split_characters(path):
        batch.append(test_transform(crop_char(gray, x1, y1, x2, y2)))
    if len(batch) == 0:
        return ""
    with torch.no_grad():
        numbers = network(torch.stack(batch).to(DEVICE)).argmax(dim=1)
    names = []
    for number in numbers.tolist():
        names.append(CLASSES[number])
    return SEPARATOR.join(names)


network.eval()
test["pred"] = test["path"].apply(read_line)
print(test[["text", "pred"]].head())

# %% [markdown]
# CER (character error rate) dengan jarak edit buatan sendiri, lalu `submission.csv`.
# Jarak edit = jumlah sisip/hapus/ganti minimum untuk mengubah teks tebakan jadi teks benar.

# %%
import pandas as pd


def edit_distance(first, second):
    previous = list(range(len(second) + 1))
    for i in range(1, len(first) + 1):
        current = [i]
        for j_index in range(1, len(second) + 1):
            cost = int(first[i - 1] != second[j_index - 1])
            current.append(min(previous[j_index] + 1, current[-1] + 1,
                               previous[j_index - 1] + cost))
        previous = current
    return previous[-1]


errors, total = 0, 0
for truth, guess in zip(test["text"], test["pred"]):
    errors += edit_distance(guess, truth)
    total += len(truth)
exact = (test["text"] == test["pred"]).mean()
print("CER:", round(errors / total, 4), "| baris persis benar:", round(exact, 4))

submission = pd.DataFrame()
submission["id"] = test["id"]       # UBAH
submission["text"] = test["pred"]   # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())
