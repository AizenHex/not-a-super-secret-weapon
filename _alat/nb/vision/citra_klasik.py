# %% [markdown]
# # Vision — Pengolahan Citra Klasik (tanpa deep learning)
# **Kapan dipakai:** objek mudah dipisah dari latar lewat **terang/gelap** atau **warna**, data
# berlabel sedikit/tidak ada, atau butuh fitur cepat (jumlah objek, luas, persentase warna).
#
# **Langkah:** ambang (adaptif / Otsu / rentang warna HSV) → rapikan masker (morfologi) →
# connected components (tiap gumpalan = satu objek) → hitung, ukur luas, saring yang kecil.
#
# **Data demo:** `skimage.data` (koin dan jaringan sel berwarna, ikut terpasang = offline).
# Saat lomba: ganti dengan `cv2.imread(path)` (OpenCV membaca warna sebagai BGR).

# %%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import data

# %% [markdown]
# ## Konfigurasi

# %%
MIN_AREA = 100       # gumpalan lebih kecil dari ini (piksel) dianggap noise
KERNEL_SIZE = 5      # ukuran kernel morfologi (besar = lebih banyak bintik terbuang)
# rentang warna HSV OpenCV: H 0-179, S 0-255, V 0-255. Coklat pekat ≈ H 8-20, S tinggi.
HSV_LOW = np.array([8, 120, 60])
HSV_HIGH = np.array([20, 255, 170])

# %% [markdown]
# ## 1. Hitung koin: ambang + morfologi + connected components
# Pilih SATU cara ambang: hapus `#` di baris yang ingin dipakai.

# %%
coins = data.coins()                      # abu-abu. Saat lomba: cv2.imread(path, 0)
smooth = cv2.GaussianBlur(coins, (5, 5), 0)
# ambang adaptif: lebih tahan cahaya tidak rata daripada satu ambang global
mask = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 51, -5)
# mask = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]   # Otsu global
kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # buang bintik
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)      # tutup lubang kecil

count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
objects = pd.DataFrame(stats[1:], columns=["x", "y", "lebar", "tinggi", "luas"])  # 0 = latar
objects["pusat_x"] = centers[1:, 0]
objects["pusat_y"] = centers[1:, 1]
objects = objects[objects["luas"] >= MIN_AREA]
print("jumlah objek:", len(objects))
objects.describe().round(1)

# %% [markdown]
# ## Gambar kotak tiap objek

# %%
boxed = cv2.cvtColor(coins, cv2.COLOR_GRAY2RGB)
for left, top, width, height in zip(objects["x"], objects["y"], objects["lebar"],
                                    objects["tinggi"]):
    top_left = (int(left), int(top))
    bottom_right = (int(left + width), int(top + height))
    cv2.rectangle(boxed, top_left, bottom_right, (255, 0, 0), 2)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].imshow(coins, cmap="gray")
axes[0].set_title("asli")
axes[1].imshow(mask, cmap="gray")
axes[1].set_title("masker")
axes[2].imshow(boxed)
axes[2].set_title(f"{len(objects)} objek")
plt.savefig("koin.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Segmentasi warna (HSV): berapa persen jaringan yang berwarna coklat?
# HSV memisahkan **jenis warna** (H) dari terang-gelap (V), jadi lebih tahan bayangan daripada
# RGB. Cara mencari rentang: lihat histogram H di bagian yang diinginkan, lalu coba-coba
# `HSV_LOW` / `HSV_HIGH`.

# %%
tissue = data.immunohistochemistry()      # RGB. Saat lomba cv2.imread → pakai COLOR_BGR2HSV
hsv = cv2.cvtColor(tissue, cv2.COLOR_RGB2HSV)
brown_mask = cv2.inRange(hsv, HSV_LOW, HSV_HIGH)
brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

brown_percent = (brown_mask > 0).mean() * 100
brown_count, brown_labels, brown_stats, brown_centers = cv2.connectedComponentsWithStats(
    brown_mask)
blob_areas = brown_stats[1:, cv2.CC_STAT_AREA]
print("persen area coklat:", round(brown_percent, 2), "%")
print("gumpalan coklat (luas >= MIN_AREA):", (blob_areas >= MIN_AREA).sum())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(tissue)
axes[0].set_title("asli")
axes[1].hist(hsv[:, :, 0].ravel(), bins=180)
axes[1].set_title("histogram H (jenis warna)")
axes[2].imshow(brown_mask, cmap="gray")
axes[2].set_title(f"coklat: {brown_percent:.1f}% area")
plt.savefig("warna.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Fitur bentuk dari kontur (untuk dipakai model tabular)
# Tiap objek → luas, keliling, kebulatan (1 = lingkaran sempurna).

# %%
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = []
perimeters = []
roundness = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        continue
    perimeter = cv2.arcLength(contour, True)
    areas.append(area)
    perimeters.append(perimeter)
    roundness.append(4 * np.pi * area / (perimeter ** 2))

shape_features = pd.DataFrame()
shape_features["luas"] = areas
shape_features["keliling"] = perimeters
shape_features["kebulatan"] = roundness
shape_features.to_csv("fitur_bentuk.csv", index=False)
shape_features.describe().round(2)
