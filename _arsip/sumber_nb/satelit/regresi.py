# %% [markdown]
# # Satelit — Regresi per Petak dari Indeks Vegetasi (GeoTIFF + zonal statistics)
# **Kapan dipakai:** soal memberi citra satelit (GeoTIFF multi-band) + batas petak/lahan
# (poligon), lalu minta menebak angka per petak (hasil panen, biomassa, kadar air).
#
# **Idenya:** citra dibaca dengan `rasterio`. Untuk tiap petak dihitung rata-rata tiap band di
# dalam poligonnya (**zonal statistics**), lalu indeks vegetasi (NDVI dkk.). Hasilnya satu baris
# per petak → regresi biasa.
#
# **Langkah:**
# 1. Baca citra dan poligon petak.
# 2. Zonal statistics + indeks vegetasi per petak.
# 3. Split spasial, latih model, buat submission.
#
# **Data demo:** Sentinel-2 sawah Indramayu, 2 tanggal (29 Jul dan 18 Agu 2024). Petak = kotak
# 200 m × 200 m. Feature dari citra 29 Jul, target = NDVI petak pada 18 Agu (pengganti data
# panen). Saat lomba: target dibaca dari CSV panitia.

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMRegressor
from rasterio.mask import mask
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/vision/sentinel2_indramayu"   # GANTI saat lomba
FEATURE_IMAGE = f"{DATA_FOLDER}/20240729.tif"    # citra untuk feature
TARGET_IMAGE = f"{DATA_FOLDER}/20240818.tif"     # hanya demo: dari sini target dibuat
PLOTS_FILE = "petak.geojson"     # demo: dibuat di sel bawah. Saat lomba: file poligon panitia

ID_COL = "plot_id"
BAND_NAMES = ["blue", "green", "red", "nir", "swir"]   # urutan band di file (band 6 = SCL)
PLOT_PIXELS = 20          # demo: sisi petak 20 piksel x 10 m = 200 m
TEST_FRACTION = 0.25      # petak paling timur dipakai sebagai uji (uji spasial)
SEED = 42

# %% [markdown]
# ## Baca GeoTIFF
# `crs` = sistem koordinat (meter), `transform` = kaitan piksel ↔ koordinat.

# %%
source = rasterio.open(FEATURE_IMAGE)
HEIGHT = source.height
WIDTH = source.width
transform = source.transform
crs = source.crs
print("band:", source.count, "| ukuran:", HEIGHT, "x", WIDTH)
print("crs:", crs)

# %% [markdown]
# ## Langkah 1 — petak (poligon)
# Demo: kotak beraturan disimpan sebagai GeoJSON lalu dibaca lagi seperti file dari panitia.
# Saat lomba: hapus bagian pembuatan kotak, cukup `gpd.read_file(PLOTS_FILE)`.

# %%
polygons = []
for row in range(0, HEIGHT - PLOT_PIXELS + 1, PLOT_PIXELS):
    for column in range(0, WIDTH - PLOT_PIXELS + 1, PLOT_PIXELS):
        left, top = transform * (column, row)
        right, bottom = transform * (column + PLOT_PIXELS, row + PLOT_PIXELS)
        polygons.append(box(left, bottom, right, top))
demo_plots = gpd.GeoDataFrame(geometry=polygons, crs=crs)
demo_plots[ID_COL] = np.arange(len(polygons))
demo_plots.to_file(PLOTS_FILE, driver="GeoJSON")

plots = gpd.read_file(PLOTS_FILE)
plots = plots.to_crs(crs)                  # samakan sistem koordinat dengan citra
print(len(plots), "petak")

# %% [markdown]
# ## Langkah 2a — zonal statistics
# `mask(..., crop=True)` memotong citra sebesar poligon. Satu baris per petak: rata-rata tiap
# band + simpangan baku band NIR (tekstur lahan).

# %%
def zonal_statistics(image_path, prefix):
    rows = []
    image = rasterio.open(image_path)
    for geometry in plots.geometry:
        pieces, _ = mask(image, [geometry], crop=True, filled=False)
        row = {}
        for i in range(len(BAND_NAMES)):
            row[prefix + BAND_NAMES[i]] = float(pieces[i].mean())
        row[prefix + "nir_std"] = float(pieces[3].std())
        rows.append(row)
    return pd.DataFrame(rows)


start_stats = zonal_statistics(FEATURE_IMAGE, "")
end_stats = zonal_statistics(TARGET_IMAGE, "end_")
start_stats.describe().round(1)

# %% [markdown]
# ## Langkah 2b — indeks vegetasi
# - NDVI = (NIR − merah)/(NIR + merah): tinggi bila tanaman sehat.
# - NDWI = (hijau − NIR)/(hijau + NIR): tinggi bila ada air.
# - NDMI = (NIR − SWIR)/(NIR + SWIR): kadar air daun.

# %%
data = start_stats.copy()
data["ndvi"] = (data["nir"] - data["red"]) / (data["nir"] + data["red"])
data["ndwi"] = (data["green"] - data["nir"]) / (data["green"] + data["nir"])
data["ndmi"] = (data["nir"] - data["swir"]) / (data["nir"] + data["swir"])
FEATURE_COLS = list(data.columns)

# target demo: NDVI petak di tanggal akhir
end_nir = end_stats["end_nir"]
end_red = end_stats["end_red"]
data["target"] = (end_nir - end_red) / (end_nir + end_red)
data[ID_COL] = plots[ID_COL]
data["center_x"] = plots.geometry.centroid.x
print(data.shape)
print(data["target"].describe())

# %% [markdown]
# ## Langkah 3a — split spasial
# Petak di sisi timur jadi uji, supaya tetangga dekat petak uji tidak ikut dilatih.

# %%
border_x = data["center_x"].quantile(1 - TEST_FRACTION)
train = data[data["center_x"] <= border_x]
test = data[data["center_x"] > border_x]
print("latih:", len(train), "| uji:", len(test))

# %% [markdown]
# ## Langkah 3b — model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.
# Pembanding: regresi linear dari NDVI saja.

# %%
model = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=SEED, n_jobs=1,
                      verbose=-1)
# model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=4)

model.fit(train[FEATURE_COLS], train["target"])
pred_test = model.predict(test[FEATURE_COLS])
print("R2 semua feature  :", round(r2_score(test["target"], pred_test), 4))
print("RMSE semua feature:", round(root_mean_squared_error(test["target"], pred_test), 4))

baseline = LinearRegression()
baseline.fit(train[["ndvi"]], train["target"])
print("R2 NDVI saja      :", round(r2_score(test["target"], baseline.predict(test[["ndvi"]])), 4))

# %% [markdown]
# ## Peta nilai asli vs prediksi

# %%
result_map = plots.copy()
result_map["truth"] = data["target"]
result_map["prediction"] = model.predict(data[FEATURE_COLS])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
result_map.plot(column="truth", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[0])
result_map.plot(column="prediction", cmap="RdYlGn", vmin=0, vmax=1, ax=axes[1])
axes[0].set_title("NDVI 18 Agu (asli)")
axes[1].set_title("prediksi dari citra 29 Jul")
fig.savefig("peta_prediksi.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Latih ulang di semua petak berlabel + submission
# Saat lomba: latih di petak yang ada targetnya, prediksi petak yang diminta.

# %%
model.fit(data[FEATURE_COLS], data["target"])
predictions = model.predict(test[FEATURE_COLS])

submission = pd.DataFrame()
submission["plot_id"] = test[ID_COL]
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
