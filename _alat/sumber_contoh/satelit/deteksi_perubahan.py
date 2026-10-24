# %% [markdown]
# # Satelit — Deteksi Perubahan (dua citra beda waktu → "berubah / tidak" per piksel)
# **Kapan dipakai:** ada citra daerah yang sama di dua waktu (sebelum/sesudah) dan diminta
# menandai bagian yang berubah (bangunan baru, deforestasi, banjir, lahan terbuka).
#
# **Idenya:** tiap piksel diberi feature: nilai band sebelum, sesudah, selisihnya, selisih
# indeks NDVI/NDBI, dan besar selisih yang dihaluskan. RandomForest memilih "berubah / tidak"
# per piksel. Ambang peluang dipilih di pasangan validasi (kelas sangat tidak seimbang).
#
# **Langkah:**
# 1. Hitung feature per piksel, ambil sampel piksel dari pasangan latih.
# 2. Latih model, pilih ambang dengan F1 terbaik di pasangan validasi.
# 3. Buat peta perubahan untuk pasangan uji, simpan sebagai PNG.
#
# **Data demo:** OSCD (Sentinel-2, 13 band, 10 m, perubahan perkotaan, lisensi non-komersial).
# Latih 4 pasang, validasi 1, uji 2 (kota yang belum pernah dilihat).

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMClassifier
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/oscd_sentinel2"   # GANTI saat lomba
BEFORE_NAME = "pasangan{}_sebelum.tif"     # {} diganti nomor pasangan
AFTER_NAME = "pasangan{}_sesudah.tif"
LABEL_NAME = "pasangan{}_label.tif"

TRAIN_PAIRS = ["00", "01", "03", "04"]
VALID_PAIRS = ["05"]          # dipakai memilih ambang peluang
TEST_PAIRS = ["07", "09"]
PIXELS_PER_PAIR = 40000       # sampel piksel latih per pasangan (citra utuh terlalu besar)
SMALL_WINDOW = 5              # jendela perataan selisih (piksel), meredam noise
LARGE_WINDOW = 15             # jendela lebih lebar, perubahan biasanya berupa bidang
SEED = 42
# Urutan 13 band Sentinel-2: B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
RED = 3                       # nomor band (mulai 0)
NIR = 7
SWIR = 11

# %% [markdown]
# ## Langkah 1a — baca satu pasangan
# Nilai band Sentinel-2 dibagi 10000 = pantulan 0–1.

# %%
def read_pair(number):
    before_file = rasterio.open(f"{DATA_FOLDER}/{BEFORE_NAME.format(number)}")
    after_file = rasterio.open(f"{DATA_FOLDER}/{AFTER_NAME.format(number)}")
    label_file = rasterio.open(f"{DATA_FOLDER}/{LABEL_NAME.format(number)}")
    before = before_file.read().astype("float32") / 10000
    after = after_file.read().astype("float32") / 10000
    label = label_file.read(1)
    return before, after, label


# coba pada pasangan latih pertama
before, after, label = read_pair(TRAIN_PAIRS[0])
print("band x tinggi x lebar:", before.shape, "| porsi piksel berubah:", label.mean().round(3))

# %% [markdown]
# ## Langkah 1b — feature per piksel
# Indeks dihitung dari nilai asli. Lalu tiap band disamakan skalanya (rata-rata 0, simpangan 1)
# per citra, supaya beda cuaca/musim antar tanggal teredam.
# Hasil: (tinggi, lebar, jumlah feature).

# %%
def pixel_features(before, after):
    ndvi_before = (before[NIR] - before[RED]) / (before[NIR] + before[RED] + 1e-6)
    ndvi_after = (after[NIR] - after[RED]) / (after[NIR] + after[RED] + 1e-6)
    ndbi_before = (before[SWIR] - before[NIR]) / (before[SWIR] + before[NIR] + 1e-6)
    ndbi_after = (after[SWIR] - after[NIR]) / (after[SWIR] + after[NIR] + 1e-6)
    before = before - before.mean(axis=(1, 2), keepdims=True)
    before = before / before.std(axis=(1, 2), keepdims=True)
    after = after - after.mean(axis=(1, 2), keepdims=True)
    after = after / after.std(axis=(1, 2), keepdims=True)
    difference = after - before
    size = np.sqrt((difference ** 2).sum(axis=0))
    extra = np.stack([ndvi_after - ndvi_before, ndbi_after - ndbi_before, size,
                      uniform_filter(size, SMALL_WINDOW), uniform_filter(size, LARGE_WINDOW)])
    features = np.concatenate([before, after, difference, extra])
    return features.transpose(1, 2, 0)


# coba pada pasangan tadi
features = pixel_features(before, after)
print(features.shape)

# %% [markdown]
# ## Langkah 1c — sampel piksel dari pasangan latih

# %%
rng = np.random.default_rng(SEED)
feature_parts = []
label_parts = []
for number in TRAIN_PAIRS:
    before, after, label = read_pair(number)
    features = pixel_features(before, after)
    rows = rng.choice(label.size, PIXELS_PER_PAIR, replace=False)
    feature_parts.append(features.reshape(-1, features.shape[2])[rows])
    label_parts.append(label.reshape(-1)[rows])
X_train = np.concatenate(feature_parts)
y_train = np.concatenate(label_parts)
print(X_train.shape, "| porsi piksel berubah:", y_train.mean().round(3))

# %% [markdown]
# ## Langkah 2a — model
# Piksel berubah hanya sekitar 5%, jadi kelas itu diberi bobot lebih (`class_weight`).
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = RandomForestClassifier(n_estimators=200, min_samples_leaf=3,
                               class_weight="balanced_subsample", n_jobs=4, random_state=SEED)
# model = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
#                        n_jobs=4, verbose=-1)
model.fit(X_train, y_train)

# %% [markdown]
# ## Langkah 2b — pilih ambang peluang di pasangan validasi
# Dengan kelas tidak seimbang, ambang 0.5 jarang terbaik. Dicari ambang dengan F1 tertinggi.

# %%
before, after, label = read_pair(VALID_PAIRS[0])
features = pixel_features(before, after)
probabilities = model.predict_proba(features.reshape(-1, features.shape[2]))[:, 1]
precision, recall, candidates = precision_recall_curve(label.reshape(-1), probabilities)
candidate_f1 = 2 * precision * recall / (precision + recall + 1e-9)
THRESHOLD = candidates[candidate_f1[:-1].argmax()]
print("ambang terpilih:", THRESHOLD.round(3), "| F1 validasi:", candidate_f1.max().round(3))

# %% [markdown]
# ## Langkah 3 — peta perubahan pasangan uji
# Tiap peta disimpan sebagai `mask_<nomor>.png` (putih = berubah) di folder notebook.

# %%
change_maps = {}
for number in TEST_PAIRS:
    before, after, label = read_pair(number)
    features = pixel_features(before, after)
    probabilities = model.predict_proba(features.reshape(-1, features.shape[2]))[:, 1]
    change_map = (probabilities > THRESHOLD).reshape(label.shape)
    change_maps[number] = change_map
    Image.fromarray(change_map.astype("uint8") * 255).save(f"mask_{number}.png")

after_file = rasterio.open(f"{DATA_FOLDER}/{AFTER_NAME.format(TEST_PAIRS[-1])}")
rgb = after_file.read([4, 3, 2]).transpose(1, 2, 0)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.clip(rgb / 3000, 0, 1))
axes[0].set_title("citra sesudah")
axes[1].imshow(change_maps[TEST_PAIRS[-1]], cmap="gray")
axes[1].set_title("prediksi perubahan")
fig.savefig("perubahan.png", dpi=100, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Simpan ringkasan
# Keluaran utama = file `mask_*.png`. CSV ini hanya ringkasan per pasangan (ganti sesuai format
# soal, misalnya satu baris per piksel atau per patch).

# %%
changed_percent = []
for number in TEST_PAIRS:
    changed_percent.append(change_maps[number].mean() * 100)

submission = pd.DataFrame()
submission["pair"] = TEST_PAIRS
submission["changed_percent"] = changed_percent
submission.to_csv("submission.csv", index=False)
submission

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
for number in TEST_PAIRS:
    before, after, label = read_pair(number)
    truth = label.reshape(-1)
    predicted = change_maps[number].reshape(-1)
    print(number, "| F1:", round(f1_score(truth, predicted), 3),
          "| precision:", round(precision_score(truth, predicted), 3),
          "| recall:", round(recall_score(truth, predicted), 3))
