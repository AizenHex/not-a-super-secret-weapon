# %% [markdown]
# # Dasar gambar: buka aman, cek ukuran, resize/padding, augmentasi, Dataset + DataLoader
#
# **Butuh:** tabel `images` dengan kolom `path` (+ `label` untuk bagian 5), dari
# `umum/baca_folder.ipynb`. **Menghasilkan:** fungsi `open_rgb`, tabel ukuran, `train_loader`.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | `open_rgb`: semua gambar → RGB tegak (GIF, PNG transparan, grayscale, foto HP) | **ya** |
# | 2 | cek ukuran + mode warna semua gambar | disarankan |
# | 3 | `pad_to_square`: jadikan persegi tanpa gepeng (teks, objek panjang) | opsional |
# | 4 | augmentasi + normalisasi (torchvision) | untuk training CNN |
# | 5 | `ImageDataset` + `DataLoader` (batch tensor untuk PyTorch) | untuk training CNN |
#
# Hanya butuh vektor fitur untuk sklearn/LightGBM? Pakai `jembatan/gambar_ke_vektor.ipynb`
# (cukup sel 1 dari sini).

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Foto ras kucing/anjing. Sengaja kotor: ukuran beragam, GIF/PNG berekstensi `.jpg`, grayscale.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

images = pd.read_csv(f"{DEMO_FOLDER}/vision/pets/train.csv")
images["path"] = f"{DEMO_FOLDER}/vision/pets/train/" + images["id"]
images = images.head(120)
print(images.head())

# %% [markdown]
# ## 1. Buka gambar dengan aman → `open_rgb(path)`
# **Butuh:** path gambar. **Menghasilkan:** gambar PIL RGB, sudah diputar sesuai EXIF.
# Foto HP sering tersimpan miring + catatan EXIF "putar 90°"; `exif_transpose` menerapkannya.
# PNG transparan (RGBA) ditempel di atas latar putih supaya bagian transparan tidak jadi hitam.

# %%
from PIL import Image, ImageOps


def open_rgb(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


example = open_rgb(images["path"].iloc[0])
print(example.size, example.mode)

# %% [markdown]
# ## 2. Cek ukuran dan mode warna semua gambar → tabel `sizes`
# Lihat: ukuran terkecil (terlalu kecil → buram saat diperbesar), rasio lebar/tinggi (sangat
# lebar → pakai `pad_to_square`), mode selain RGB, format asli berbeda dari ekstensi.

# %%
from PIL import Image
from tqdm import tqdm

rows = []
for path in tqdm(images["path"]):
    with Image.open(path) as image:
        rows.append({"path": path, "width": image.width, "height": image.height,
                     "mode": image.mode, "format": image.format})
sizes = pd.DataFrame(rows)
sizes["ratio"] = sizes["width"] / sizes["height"]
print(sizes[["width", "height", "ratio"]].describe().round(2))
print(sizes["mode"].value_counts())
print(sizes["format"].value_counts())
images = images.merge(sizes[["path", "width", "height"]], on="path", how="left")

# %% [markdown]
# ## 3. Jadikan persegi tanpa gepeng → `pad_to_square(image, size)`
# `resize((224, 224))` biasa menggepengkan gambar yang tidak persegi. Fungsi ini mengecilkan
# dengan rasio tetap, lalu menambah pinggiran warna `fill`.

# %%
import matplotlib.pyplot as plt
from PIL import Image


def pad_to_square(image, size, fill=(0, 0, 0)):
    scale = size / max(image.width, image.height)
    new_width = max(1, round(image.width * scale))
    new_height = max(1, round(image.height * scale))
    resized = image.resize((new_width, new_height))
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


example = open_rgb(images["path"].iloc[3])
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(example)
axes[1].imshow(example.resize((224, 224)))
axes[2].imshow(pad_to_square(example, 224))
for ax, title in zip(axes, ["asli", "resize (gepeng)", "pad_to_square"]):
    ax.set_title(title)
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 4. Augmentasi + normalisasi (torchvision)
# **Menghasilkan:** `train_transform` (acak, hanya untuk train) dan `test_transform` (tetap).
# Pilih augmentasi yang masuk akal untuk soal: teks/angka → **jangan** `RandomHorizontalFlip`
# (huruf jadi cermin); foto atas-bawah tidak penting (satelit, sel) → boleh `RandomVerticalFlip`.
# `Normalize` dengan angka ImageNet wajib kalau memakai backbone pretrained ImageNet.

# %%
import matplotlib.pyplot as plt
from torchvision import transforms

IMAGE_SIZE = 224   # UBAH

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),   # potong acak lalu resize
    transforms.RandomHorizontalFlip(),                            # UBAH: hapus untuk teks
    transforms.RandomRotation(10),                                # putar acak ±10°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),                                        # 0..255 → 0..1, (3, H, W)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

example = open_rgb(images["path"].iloc[0])
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax in axes:
    tensor = train_transform(example)
    shown = tensor.permute(1, 2, 0) * 0.226 + 0.449   # (3,H,W) → (H,W,3), kira-kira balik ke 0..1
    ax.imshow(shown.clamp(0, 1))
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 5. Dataset + DataLoader → `train_loader`, `CLASSES`
# **Butuh:** `images` (kolom `path`, `label`), `open_rgb`, transform dari sel 4.
# `CLASSES` = daftar nama kelas; model menebak **nomor** kelas, `CLASSES[nomor]` = namanya.
# Test tanpa label: buat `ImageDataset(test_table, test_transform)` dengan kolom `label` diisi
# kelas apa saja (mis. `CLASSES[0]`), labelnya tidak dipakai.

# %%
import torch
from torch.utils.data import DataLoader, Dataset

CLASSES = sorted(images["label"].unique())
class_index = {}
for i in range(len(CLASSES)):
    class_index[CLASSES[i]] = i


class ImageDataset(Dataset):
    def __init__(self, table, transform):
        self.paths = table["path"].tolist()
        self.labels = table["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = self.transform(open_rgb(self.paths[i]))
        return image, class_index[self.labels[i]]


train_loader = DataLoader(ImageDataset(images, train_transform), batch_size=32, shuffle=True,
                          num_workers=0)   # Windows + Jupyter: num_workers=0 paling aman
batch_images, batch_labels = next(iter(train_loader))
print(batch_images.shape, batch_labels[:8], len(CLASSES), "kelas")
