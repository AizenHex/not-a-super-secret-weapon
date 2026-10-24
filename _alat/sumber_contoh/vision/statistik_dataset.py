# %% [markdown]
# # Vision — Statistik Dataset Gambar + Cari Gambar Kembar (deduplikasi hash)
# **Kapan dipakai:** awal lomba, sebelum melatih apa pun. Menjawab: format file benar atau tidak
# (`.jpg` yang isinya GIF/PNG?), mode warna (grayscale/RGBA?), ukuran seragam atau tidak, kelas
# seimbang atau tidak, kelas mana yang gelap/berwarna aneh, dan **ada gambar kembar antara train
# dan test?** (kebocoran: skor validasi terlalu bagus, skor leaderboard jelek).
#
# **Bentuk data yang didukung** (pilih satu sel di bagian Baca data):
# 1. CSV `id,label` + gambar dalam satu folder (aktif)
# 2. folder per kelas: `train/<kelas>/*.jpg`
# 3. label tertulis di nama file: `Abyssinian_12.jpg` → `Abyssinian`
#
# Ditambah folder lain yang ingin dicek kembarnya (misalnya gambar tanpa label) di `EXTRA_FOLDER`.
#
# **Langkah:** baca tiap gambar → format, mode, ukuran, rasio, kecerahan, warna rata-rata →
# ringkasan per kelas. Lalu `imagehash` phash → gambar dengan hash hampir sama = kembar.
#
# **Data demo:** `pets` (10 ras kucing/anjing, Oxford-IIIT Pet). Dataset aslinya memang berisi
# foto beruntun/kembar: ada gambar test yang sama persis dengan gambar train.

# %%
import glob
import os

import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/pets"    # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"              # tidak ada? pakai CSV contoh jawaban
TRAIN_IMAGE_FOLDER = f"{DATA_FOLDER}/train"
TEST_IMAGE_FOLDER = f"{DATA_FOLDER}/test"
EXTRA_FOLDER = f"{DATA_FOLDER}/tanpa_label"       # folder lain yang dicek, "" bila tidak ada
OUTPUT_FOLDER = "../../../outputs/vision/statistik_dataset"   # di luar repo
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ID_COL = "id"           # kolom nama file gambar
TARGET = "label"        # kolom kelas
IMAGE_EXTENSION = ""    # isi ".jpg" bila kolom id TIDAK memuat ekstensi (misalnya "img_01")
MAX_HAMMING = 8         # jarak hash <= angka ini dianggap kembar (0 = sama persis, maks 64)

# %% [markdown]
# ## Baca data — bentuk 1: CSV `id,label` + folder gambar (aktif)

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train["path"] = TRAIN_IMAGE_FOLDER + "/" + train[ID_COL].astype(str) + IMAGE_EXTENSION
test["path"] = TEST_IMAGE_FOLDER + "/" + test[ID_COL].astype(str) + IMAGE_EXTENSION
print("train:", train.shape, "| test:", test.shape)
print("contoh path:", train["path"].iloc[0])

# %% [markdown]
# ## (Alternatif) bentuk 2: folder per kelas `train/<kelas>/*.jpg`

# %%
# rows = []
# for class_name in sorted(os.listdir(TRAIN_IMAGE_FOLDER)):
#     for path in sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/{class_name}/*")):
#         rows.append([os.path.basename(path), class_name, path])
# train = pd.DataFrame(rows, columns=[ID_COL, TARGET, "path"])

# %% [markdown]
# ## (Alternatif) bentuk 3: label di nama file `Abyssinian_12.jpg`

# %%
# train = pd.DataFrame()
# train["path"] = sorted(glob.glob(f"{TRAIN_IMAGE_FOLDER}/*"))
# train[ID_COL] = train["path"].apply(os.path.basename)
# train[TARGET] = train[ID_COL].str.rsplit("_", n=1).str[0]

# %% [markdown]
# ## Gabungkan semua gambar dalam satu tabel
# Kolom `source` = asal gambar (train / test / extra). Test dan extra tidak berlabel.

# %%
extra = pd.DataFrame()
if EXTRA_FOLDER != "":
    extra["path"] = sorted(glob.glob(f"{EXTRA_FOLDER}/*"))
    extra[ID_COL] = extra["path"].apply(os.path.basename)
train["source"] = "train"
test["source"] = "test"
extra["source"] = "extra"
all_images = pd.concat([train, test, extra], ignore_index=True)
all_images["path"] = all_images["path"].str.replace("\\", "/")
all_images[TARGET] = all_images[TARGET].fillna("")
print(all_images["source"].value_counts())

# %% [markdown]
# ## Ukur tiap gambar
# Satu baris per gambar. `format` = isi file sebenarnya (JPEG/PNG/GIF), bukan ekstensinya.
# `mode` = RGB (warna), L (grayscale), RGBA (ada transparansi), P (palet, biasanya GIF).
# Kecerahan dan warna diukur setelah `convert("RGB")`, jadi semua mode bisa dibandingkan.

# %%
rows = []
for path in all_images["path"]:
    original = Image.open(path)
    image = original.convert("RGB")
    pixels = np.asarray(image, dtype=float)
    rows.append({
        "format": original.format,
        "mode": original.mode,
        "width": image.width,
        "height": image.height,
        "ratio": image.width / image.height,
        "file_kb": os.path.getsize(path) / 1024,
        "brightness": pixels.mean(),
        "R": pixels[:, :, 0].mean(),
        "G": pixels[:, :, 1].mean(),
        "B": pixels[:, :, 2].mean(),
        "phash": str(imagehash.phash(image)),
    })
all_images = pd.concat([all_images, pd.DataFrame(rows)], axis=1)
train_stats = all_images[all_images["source"] == "train"].reset_index(drop=True)
print(all_images[["width", "height", "ratio", "file_kb", "brightness"]].describe().round(1))

# %% [markdown]
# ## Format dan mode: ada yang aneh?
# Ekstensi `.jpg` tapi format GIF/PNG → `cv2.imread` bisa gagal (GIF), PIL aman. Mode selain RGB
# → wajib `convert("RGB")` sebelum masuk model (grayscale = 1 kanal, RGBA = 4 kanal).

# %%
print(pd.crosstab(all_images["source"], all_images["format"]))
print(pd.crosstab(all_images["source"], all_images["mode"]))
extension = all_images["path"].str.rsplit(".", n=1).str[-1]
extension = extension.str.lower()
extension = extension.replace({"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "gif": "GIF"})
mismatch = all_images[extension != all_images["format"]]
print("ekstensi tidak cocok dengan isi file:", len(mismatch))
mismatch[["source", ID_COL, "format", "mode"]]

# %% [markdown]
# ## Ringkasan: ukuran, jumlah per kelas, warna rata-rata per kelas
# Ukuran sangat beragam → perlu `ImageOps.fit` (potong) atau `ImageOps.pad` (beri tepi).

# %%
print("jumlah ukuran (lebar, tinggi) berbeda:", train_stats.groupby(["width", "height"]).ngroups)
summary = train_stats.groupby(TARGET)[["brightness", "R", "G", "B", "width"]].mean()
summary = summary.round(1)
summary["count"] = train_stats[TARGET].value_counts()
print(summary)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
summary["count"].plot.bar(ax=axes[0], title="jumlah gambar per kelas")
axes[1].hist(train_stats["brightness"], bins=30)
axes[1].set_title("sebaran kecerahan (train)")
axes[2].scatter(train_stats["width"], train_stats["height"], s=5)
axes[2].set_title("lebar vs tinggi (train)")
colors = summary[["R", "G", "B"]].to_numpy() / 255
axes[3].imshow(colors.reshape(len(summary), 1, 3), aspect=0.3)
axes[3].set_yticks(range(len(summary)), list(summary.index))
axes[3].set_title("warna rata-rata per kelas")
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/statistik.png", dpi=110)
plt.show()

# %% [markdown]
# ## Cari kembar: tiap gambar test/extra dibandingkan dengan semua gambar train
# Hash teks diubah jadi 64 bit. Jarak Hamming = jumlah bit berbeda. Kembar sama persis saja:
# `all_images["phash"].duplicated(keep=False).sum()`.

# %%
def hash_bits(hash_texts):
    bits = []
    for text in hash_texts:
        bits.append(imagehash.hex_to_hash(text).hash.flatten())
    return np.array(bits)


train_bits = hash_bits(train_stats["phash"])
others = all_images[all_images["source"] != "train"].reset_index(drop=True)
other_bits = hash_bits(others["phash"])
twins = []
for row in range(len(others)):
    distances = (train_bits != other_bits[row]).sum(axis=1)    # jarak ke semua gambar train
    nearest = distances.argmin()
    if distances[nearest] <= MAX_HAMMING:
        twins.append({"source": others["source"][row], "other": others["path"][row],
                      "train": train_stats["path"][nearest], "distance": distances[nearest],
                      "train_label": train_stats[TARGET][nearest]})
twins = pd.DataFrame(twins, columns=["source", "other", "train", "distance", "train_label"])
print("gambar yang punya kembar di train:", len(twins), "dari", len(others))
twins.to_csv(f"{OUTPUT_FOLDER}/kembar_dengan_train.csv", index=False)
twins.sort_values("distance")

# %% [markdown]
# ## Kembar di dalam train, dan kembar yang labelnya beda (label berisik?)
# Kembar di dalam train membuat validasi acak terlalu optimis (satu di latih, kembarannya di
# validasi). Solusi: kelompokkan kembar lalu pakai `GroupKFold`.

# %%
same_class = 0
different_class = 0
for row in range(len(train_stats)):
    distances = (train_bits[row + 1:] != train_bits[row]).sum(axis=1)
    for other_row in np.where(distances <= MAX_HAMMING)[0] + row + 1:
        if train_stats[TARGET][row] == train_stats[TARGET][other_row]:
            same_class += 1
        else:
            different_class += 1
            print("label beda:", train_stats[ID_COL][row], "↔", train_stats[ID_COL][other_row])
print("pasangan kembar di train | label sama:", same_class, "| label beda:", different_class)

# %% [markdown]
# ## Contoh pasangan kembar (atas: test/extra, bawah: train)

# %%
examples = twins.sort_values("distance").head(6)
examples = examples.reset_index(drop=True)
fig, axes = plt.subplots(2, max(len(examples), 1), figsize=(2.5 * max(len(examples), 1), 5),
                         squeeze=False)
for ax in axes.ravel():
    ax.axis("off")
for column in range(len(examples)):
    axes[0, column].imshow(Image.open(examples["other"][column]).convert("RGB"))
    axes[0, column].set_title(os.path.basename(examples["other"][column]), fontsize=8)
    axes[1, column].imshow(Image.open(examples["train"][column]).convert("RGB"))
    title = f"{examples['train_label'][column]} (jarak {examples['distance'][column]})"
    axes[1, column].set_title(title, fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}/kembar_contoh.png", dpi=110)
plt.show()

# %% [markdown]
# ## Simpan tabel statistik semua gambar (untuk notebook lain / laporan)

# %%
all_images.to_csv(f"{OUTPUT_FOLDER}/statistik_semua_gambar.csv", index=False)
print(all_images.shape)

# %% [markdown]
# ## Cara membaca hasil → tindakan
# - **Gambar test kembar dengan train:** labelnya hampir pasti sama dengan label train kembarannya.
#   Boleh dipakai bila aturan lomba mengizinkan (tanyakan panitia); minimal sebutkan di laporan.
# - **Kembar di dalam train:** validasi acak terlalu optimis → `GroupKFold` dengan grup kembar.
# - **Kembar dengan label beda:** label berisik; lihat `label_berisik`.
# - **Format/mode campur:** selalu `Image.open(path).convert("RGB")`.
# - **`MAX_HAMMING` terlalu besar** (≥ 10) → gambar berbeda ikut dianggap kembar. Selalu lihat
#   grid contoh sebelum percaya angkanya.
