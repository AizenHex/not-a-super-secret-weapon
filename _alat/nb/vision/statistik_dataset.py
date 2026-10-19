# %% [markdown]
# # Vision — Statistik Dataset Gambar + Cari Gambar Kembar (deduplikasi hash)
# **Kapan dipakai:** awal lomba, sebelum melatih apa pun. Menjawab: ukuran gambar seragam atau
# tidak, kelas seimbang atau tidak, kelas mana yang gelap/berwarna aneh, dan **ada gambar kembar
# antara train dan test?** (kebocoran: skor validasi terlalu bagus, skor leaderboard jelek).
#
# **Langkah:** baca tiap gambar → ukuran, rasio, kecerahan, warna rata-rata → ringkasan per
# kelas. Lalu `imagehash` phash → gambar dengan hash hampir sama = kembar / hampir kembar.
#
# **Data demo:** CIFAR-10 kecil. Beberapa gambar test sengaja disisipkan (diedit) ke train.

# %%
import os

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/classification/cifar10_small"  # GANTI
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"

PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
LABEL_COL = "label"
MAX_HAMMING = 6       # jarak hash <= angka ini dianggap kembar (0 = sama persis, maks 64)
N_INSERTED = 10       # HANYA DEMO: jumlah kembar buatan

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Ukur tiap gambar
# Satu baris per gambar: ukuran, rasio lebar/tinggi, kecerahan (0–255), rata-rata R G B, hash.

# %%
train["source"] = "train"
test["source"] = "test"
test[LABEL_COL] = ""      # test tidak berlabel
all_images = pd.concat([train, test], ignore_index=True)

rows = []
for path in all_images[PATH_COL]:
    file = f"{DATA_FOLDER}/{path}"
    image = Image.open(file).convert("RGB")
    pixels = np.asarray(image, dtype=float)
    rows.append({
        "width": image.width,
        "height": image.height,
        "ratio": image.width / image.height,
        "file_kb": os.path.getsize(file) / 1024,
        "brightness": pixels.mean(),
        "R": pixels[:, :, 0].mean(),
        "G": pixels[:, :, 1].mean(),
        "B": pixels[:, :, 2].mean(),
        "phash": str(imagehash.phash(image)),
    })
all_images = pd.concat([all_images, pd.DataFrame(rows)], axis=1)

train_stats = all_images[all_images["source"] == "train"].reset_index(drop=True)
test_stats = all_images[all_images["source"] == "test"].reset_index(drop=True)
print(train_stats[["width", "height", "ratio", "brightness"]].describe().round(2))

# %% [markdown]
# ## Ringkasan: ukuran, jumlah per kelas, warna rata-rata per kelas

# %%
print("ukuran unik (lebar, tinggi):")
print(train_stats.groupby(["width", "height"]).size())

summary = train_stats.groupby(LABEL_COL)[["brightness", "R", "G", "B"]].mean()
summary = summary.round(1)
summary["count"] = train_stats[LABEL_COL].value_counts()
print(summary)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
summary["count"].plot.bar(ax=axes[0], title="jumlah gambar per kelas")
axes[1].hist(train_stats["brightness"], bins=30)
axes[1].set_title("sebaran kecerahan (train)")
colors = summary[["R", "G", "B"]].to_numpy() / 255
axes[2].imshow(colors.reshape(len(summary), 1, 3), aspect=0.3)
axes[2].set_yticks(range(len(summary)), list(summary.index))
axes[2].set_title("warna rata-rata per kelas")
plt.tight_layout()
plt.savefig("statistik.png", dpi=110)
plt.show()

# %% [markdown]
# ## Sisipkan kembar buatan (HANYA DEMO, hapus sel ini saat lomba)
# Ambil beberapa gambar test, edit sedikit (lebih terang, diperkecil), taruh di tabel train.
# Gambar editan disimpan di folder notebook supaya bisa digambar nanti.

# %%
inserted = []
for number in range(N_INSERTED):
    original = Image.open(f"{DATA_FOLDER}/{test[PATH_COL][number * 90]}").convert("RGB")
    edited = ImageEnhance.Brightness(original).enhance(1.2)
    edited = edited.resize((24, 24))
    file_name = f"SISIPAN_{number}.png"
    edited.save(file_name)
    inserted.append({
        PATH_COL: file_name,
        "source": "train",
        "phash": str(imagehash.phash(edited)),
        LABEL_COL: "sisipan",
    })
train_stats = pd.concat([train_stats, pd.DataFrame(inserted)], ignore_index=True)
print(train_stats.shape)

# %% [markdown]
# ## Cari kembar train ↔ test
# Hash teks diubah jadi 64 bit. Jarak Hamming = jumlah bit berbeda. Tiap gambar test dibandingkan
# dengan semua gambar train. Kembar sama persis saja:
# `train_stats["phash"].isin(test_stats["phash"]).sum()`.

# %%
train_bits = []
for text in train_stats["phash"]:
    train_bits.append(imagehash.hex_to_hash(text).hash.flatten())
train_bits = np.array(train_bits)

twins = []
for row in range(len(test_stats)):
    test_bits = imagehash.hex_to_hash(test_stats["phash"][row]).hash.flatten()
    distances = (train_bits != test_bits).sum(axis=1)       # jarak ke semua gambar train
    nearest = distances.argmin()
    if distances[nearest] <= MAX_HAMMING:
        twins.append({
            "test": test_stats[PATH_COL][row],
            "train": train_stats[PATH_COL][nearest],
            "distance": distances[nearest],
            "train_label": train_stats[LABEL_COL][nearest],
        })
twins = pd.DataFrame(twins)
print("gambar test yang punya kembar di train:", len(twins), "dari", len(test_stats))
print(twins.sort_values("distance").head(15))
twins.to_csv("kembar_train_test.csv", index=False)

# %% [markdown]
# ## Kembar di dalam train, dan kembar yang labelnya beda (label berisik?)

# %%
doubles = train_stats[train_stats.duplicated("phash", keep=False)]
print("gambar train dengan phash persis sama dengan gambar train lain:", len(doubles))
label_variety = doubles.groupby("phash")[LABEL_COL].nunique()
print("grup hash dengan label berbeda:", (label_variety > 1).sum())
print(doubles.sort_values("phash")[[PATH_COL, LABEL_COL, "phash"]].head(10))

# %% [markdown]
# ## Contoh pasangan kembar (atas: test, bawah: train)

# %%
examples = twins.sort_values("distance").head(6)
examples = examples.reset_index(drop=True)
fig, axes = plt.subplots(2, len(examples), figsize=(2 * len(examples), 4.5), squeeze=False)
for column in range(len(examples)):
    axes[0, column].imshow(Image.open(f"{DATA_FOLDER}/{examples['test'][column]}"))
    axes[0, column].set_title("test", fontsize=8)
    train_path = examples["train"][column]
    if train_path.startswith("SISIPAN"):
        axes[1, column].imshow(Image.open(train_path))      # gambar editan di folder notebook
    else:
        axes[1, column].imshow(Image.open(f"{DATA_FOLDER}/{train_path}"))
    axes[1, column].set_title(f"train (jarak {examples['distance'][column]})", fontsize=8)
    axes[0, column].axis("off")
    axes[1, column].axis("off")
plt.tight_layout()
plt.savefig("kembar_contoh.png", dpi=110)
plt.show()

# %% [markdown]
# ## Simpan tabel ukuran semua gambar (untuk notebook lain / laporan)

# %%
all_stats = pd.concat([train_stats, test_stats], ignore_index=True)
all_stats.to_csv("ukuran_semua_gambar.csv", index=False)
print(all_stats.shape)
