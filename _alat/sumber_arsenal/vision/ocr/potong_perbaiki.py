# %% [markdown]
# # Potong kotak → perbaiki potongan (mendatar, tajam, lurus) → simpan, siap OCR
#
# **Butuh:** tabel `best_boxes` (kolom path, label, x1, y1, x2, y2) dari `deteksi_yolo.ipynb`
# sel "Kotak terbaik". Kernel sempat mati? Tabel itu juga tersimpan di `outputs/best_boxes.csv`.
#
# **Menghasilkan (tersimpan di disk, aman kalau kernel mati):**
# - `outputs/crops/*.png`: potongan mentah
# - `outputs/crops_fixed/*.png`: potongan yang sudah diperbaiki → ini yang dibaca OCR
# - `outputs/crops.csv`: tabel `crops` (crop_path, path, label, width, height, fixed_path)
#
# ## Cara menempel ke notebook-mu
# Salin sel kode di bawah **berurutan**, tepat di bawah sel "Kotak terbaik" milikmu:
#
# | urutan | sel di notebook ini | wajib? | gunanya |
# |---|---|---|---|
# | 1 | Potong | wajib | kotak → file potongan + tabel `crops` |
# | 2 | Fungsi A: mendatar + pertajam | wajib | membuat `make_horizontal`, `enhance` |
# | 3 | Fungsi B: lurus + `fix` | wajib | membuat `deskew`, `fix` |
# | 4 | Lihat sebelum vs sesudah | dianjurkan | cek mata dulu sebelum proses semua |
# | 5 | Terapkan ke SEMUA + simpan | **wajib** | tanpa sel ini, belum ada potongan yang diperbaiki |
# | - | 4 titik | opsional | hanya untuk stiker/dokumen yang difoto miring perspektif |
#
# Sel 2 dan 3 hanya **membuat** fungsi dan mencobanya pada satu gambar. Perbaikan baru benar-benar
# dikerjakan untuk semua potongan di sel 5.
#
# Contoh susunan di notebook training (Vignate):
# ```
# ... sel Latih / Evaluasi / Prediksi / Kotak terbaik   (dari deteksi_yolo.ipynb)
# sel Potong                                             (mulai dari notebook ini)
# sel Fungsi A
# sel Fungsi B
# sel Lihat sebelum vs sesudah
# sel Terapkan ke SEMUA + simpan
# ... sel OCR                                            (dari ocr.ipynb)
# ```

# %% [markdown]
# ## 1. Potong dan simpan
# **Butuh:** `best_boxes`.
# **Menghasilkan:** `outputs/crops/*.png`, tabel `crops`, `outputs/crops.csv`.
#
# Ubah: `BOX_TABLE` (tabel kotak), `CROP_CLASS` (kelas yang dipotong), `PAD` (tepi tambahan supaya
# huruf di pinggir tidak terpotong; 0.05 = 5% lebar/tinggi kotak).

# %%
import os

import pandas as pd
from PIL import Image

BOX_TABLE = best_boxes            # UBAH: best_boxes (prediksi) / boxes (anotasi train)
# BOX_TABLE = pd.read_csv("outputs/best_boxes.csv")   # kalau kernel sempat mati
CROP_CLASS = "entete"             # UBAH
CROP_FOLDER = "outputs/crops"     # UBAH
PAD = 0.05                        # UBAH

os.makedirs(CROP_FOLDER, exist_ok=True)
crop_rows = []
selected = BOX_TABLE[BOX_TABLE["label"] == CROP_CLASS]
for number, box in enumerate(selected.itertuples()):
    image = Image.open(box.path).convert("RGB")
    pad_x = (box.x2 - box.x1) * PAD
    pad_y = (box.y2 - box.y1) * PAD
    left, top = max(0, box.x1 - pad_x), max(0, box.y1 - pad_y)
    right, bottom = min(image.width, box.x2 + pad_x), min(image.height, box.y2 + pad_y)
    crop = image.crop((int(left), int(top), int(right), int(bottom)))
    stem = os.path.splitext(os.path.basename(box.path))[0]
    crop_path = f"{CROP_FOLDER}/{stem}_{number}.png"
    crop.save(crop_path)
    crop_rows.append([crop_path, box.path, box.label, crop.width, crop.height])
crops = pd.DataFrame(crop_rows, columns=["crop_path", "path", "label", "width", "height"])
crops.to_csv("outputs/crops.csv", index=False)
print("potongan:", len(crops), "| tinggi > lebar:", (crops["height"] > crops["width"]).sum())

# %% [markdown]
# **Kernel mati setelah sel 1?** Tidak perlu memotong ulang. Jalankan baris ini, lalu lanjut ke
# sel 2: `crops = pd.read_csv("outputs/crops.csv")`

# %% [markdown]
# ## 2. Fungsi A: mendatarkan + mempertajam
# **Butuh:** `crops`. **Menghasilkan:** fungsi `make_horizontal`, `enhance` (belum dipakai ke semua
# data; itu di sel 5).
#
# - `make_horizontal`: teks berdiri (tinggi > lebar) diputar 90° searah jarum jam.
#   Dari gambar saja tidak bisa ditebak harus diputar ke kiri atau ke kanan, jadi **kira-kira
#   separuh hasilnya terbalik 180°. Itu normal.** Yang terbalik dibereskan saat OCR:
#   `ocr.ipynb` membaca di 0° dan 180° lalu memilih yang terbaca (atau pakai `orientasi_cnn`).
# - `enhance`: abu-abu → perbesar 2× → CLAHE (kontras lokal) → pertajam.

# %%
import cv2
import numpy as np
from PIL import Image


def make_horizontal(image):
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)   # UBAH: fx, fy
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)


# coba pada satu potongan
example_image = np.array(Image.open(crops["crop_path"].iloc[0]).convert("RGB"))
print("asli:", example_image.shape, "→ diperbaiki:", enhance(make_horizontal(example_image)).shape)

# %% [markdown]
# ## 3. Fungsi B: meluruskan teks yang miring sedikit (`deskew`) + `fix` = semua langkah
# **Butuh:** sel 2. **Menghasilkan:** fungsi `deskew`, `fix`.
#
# `deskew` mencoba memutar gambar dari -15° sampai 15° (tiap 0,5°) dan memilih sudut di mana baris
# teks paling "rapi": jumlah piksel gelap per baris naik-turun paling tajam (baris teks penuh,
# celah antarbaris kosong). Cara ini tidak terganggu barcode atau latar gelap di pinggir potongan.
#
# `fix` = urutan perbaikan yang dipakai di sel 4 dan 5. Mau menambah/membuang langkah? Ubah di sini.

# %%
import cv2
import numpy as np

MAX_ANGLE = 15   # UBAH: miring terbesar yang dicoba (derajat)


def rotate_gray(gray, angle):
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def deskew(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.resize(mask, None, fx=0.5, fy=0.5)   # kecilkan supaya cepat
    best_angle, best_score = 0, -1
    for angle in np.arange(-MAX_ANGLE, MAX_ANGLE + 0.5, 0.5):
        score = np.var(rotate_gray(mask, angle).sum(axis=1))
        if score > best_score:
            best_angle, best_score = angle, score
    return rotate_gray(gray, best_angle)


def fix(image):
    # UBAH: urutan langkah. Dibaca dari dalam ke luar: mendatar → pertajam → luruskan
    return deskew(enhance(make_horizontal(image)))


print("asli:", example_image.shape, "→ fix:", fix(example_image).shape)

# %% [markdown]
# ## 4. Lihat: potongan asli (atas) vs hasil `fix` (bawah)
# **Butuh:** `crops`, `fix`. Jalankan beberapa kali dengan `random_state` lain. Hasil jelek?
# Ubah `fix` di sel 3, jalankan sel 3 lagi, lalu sel ini lagi.
# Teks terbalik 180° di baris bawah = normal (dibereskan saat OCR).

# %%
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

N_SHOW = 6   # UBAH

preview_paths = crops["crop_path"].sample(N_SHOW, random_state=0)   # UBAH: random_state
fig, axes = plt.subplots(2, N_SHOW, figsize=(3 * N_SHOW, 5))
for column, crop_path in enumerate(preview_paths):
    image = np.array(Image.open(crop_path).convert("RGB"))
    axes[0, column].imshow(image)
    axes[1, column].imshow(fix(image), cmap="gray")
    axes[0, column].axis("off")
    axes[1, column].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Terapkan `fix` ke SEMUA potongan + simpan
# **Butuh:** `crops`, `fix`. **Menghasilkan:** `outputs/crops_fixed/*.png`, kolom
# `crops["fixed_path"]`, `outputs/crops.csv` diperbarui. Sekitar 0,02 detik per potongan.
#
# Setelah sel ini, OCR cukup membaca file di `crops["fixed_path"]`, tidak perlu memanggil `fix`
# lagi.
# Kernel mati? `crops = pd.read_csv("outputs/crops.csv")` sudah berisi `fixed_path`.

# %%
import os

import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

FIXED_FOLDER = "outputs/crops_fixed"   # UBAH

os.makedirs(FIXED_FOLDER, exist_ok=True)
fixed_paths = []
for crop_path in tqdm(crops["crop_path"]):
    image = np.array(Image.open(crop_path).convert("RGB"))
    fixed_path = f"{FIXED_FOLDER}/{os.path.basename(crop_path)}"
    cv2.imwrite(fixed_path, fix(image))
    fixed_paths.append(fixed_path)
crops["fixed_path"] = fixed_paths
crops.to_csv("outputs/crops.csv", index=False)
print("tersimpan:", len(fixed_paths), "file di", FIXED_FOLDER)
print(crops[["crop_path", "fixed_path"]].head())

# %% [markdown]
# ## (Opsional) Lurus dari 4 titik (perspektif): stiker/dokumen difoto miring
# Untuk vignette biasanya **tidak perlu** (hasilnya hampir sama dengan potongan aslinya).
# `find_four_points`: tepi terbesar di potongan → 4 sudut. Punya 4 titik dari model (seg/pose/OBB)?
# Langsung `warp_four_points(image, points)`.
#
# Cara memakai di `fix`: jalankan sel ini dan sel berikutnya, lalu di sel 3 ganti
# `make_horizontal(image)` menjadi
# `make_horizontal(warp_four_points(image, find_four_points(image)))`.
# Jalankan sel 3 lagi, cek di sel 4, lalu jalankan sel 5.

# %%
import cv2
import numpy as np


def find_four_points(image, min_fraction=0.3):
    # tepi → ditebalkan supaya tertutup → kontur terbesar → 4 sudut; terlalu kecil = pakai pojok
    height, width = image.shape[:2]
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return corners
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < min_fraction * width * height:
        return corners
    approx = cv2.approxPolyDP(biggest, 0.02 * cv2.arcLength(biggest, True), True)
    if len(approx) == 4:
        return approx.reshape(4, 2)
    return cv2.boxPoints(cv2.minAreaRect(biggest))   # cadangan: persegi terkecil yang memuat


print("4 titik:", find_four_points(example_image).round().tolist())

# %%
import cv2
import numpy as np


def warp_four_points(image, points):
    points = np.array(points, dtype="float32")
    total = points.sum(axis=1)
    diff = np.diff(points, axis=1).ravel()
    # urutan: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah
    ordered = np.array([points[np.argmin(total)], points[np.argmin(diff)],
                        points[np.argmax(total)], points[np.argmax(diff)]], dtype="float32")
    top_left, top_right, bottom_right, bottom_left = ordered
    width = int(max(np.linalg.norm(top_right - top_left),
                    np.linalg.norm(bottom_right - bottom_left)))
    height = int(max(np.linalg.norm(bottom_left - top_left),
                     np.linalg.norm(bottom_right - top_right)))
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                      dtype="float32")
    matrix = cv2.getPerspectiveTransform(ordered, target)
    return cv2.warpPerspective(image, matrix, (width, height))


print("lurus:", warp_four_points(example_image, find_four_points(example_image)).shape)
