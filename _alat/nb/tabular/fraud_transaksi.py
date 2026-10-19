# %% [markdown]
# # Tabular — Deteksi Fraud Transaksi (feature kecepatan transaksi + split waktu)
# **Kapan dipakai:** tabel transaksi kartu/dompet digital dengan kolom waktu, id kartu,
# nominal, dan label fraud yang sangat jarang (<1%).
#
# **Idenya:** fraud biasanya terlihat dari **pola kecepatan**: tiba-tiba banyak transaksi
# dalam sejam, nominal jauh di atas kebiasaan, jeda sangat pendek.
#
# **Langkah:**
# 1. Untuk tiap transaksi, hitung feature dari transaksi SEBELUMNYA milik kartu yang sama
#    (jumlah dan total dalam 1 jam / 24 jam / 7 hari, jam sejak transaksi terakhir, rasio
#    nominal vs rata-rata kartu).
# 2. Split berdasarkan WAKTU (bukan acak).
# 3. LGBMClassifier → PR-AUC → pilih threshold.
#
# **Aturan emas:** feature hanya boleh memakai masa lalu (`closed="left"`, `shift(1)`). Kalau
# tidak, skor validasi bagus palsu.
#
# **Data demo:** 133 ribu transaksi dari 120 kartu (Sparkov, data kartu SINTETIS, Apache-2.0).

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/fraud_transaksi"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/transaksi_sparkov.csv"

TIME_COL = "waktu_transaksi"
CARD_COL = "cc_num"
AMOUNT_COL = "amt"
TARGET = "is_fraud"
TRAIN_END = "2019-11-01"        # sebelum tanggal ini = train
VALIDATION_END = "2020-01-01"   # sampai sebelum tanggal ini = validasi, sisanya = test
WINDOWS = ["1h", "24h", "7d"]   # jendela waktu feature kecepatan
EARTH_RADIUS_KM = 6371
SEED = 42

# %% [markdown]
# ## Baca data
# Data diurutkan per kartu lalu per waktu, supaya "transaksi sebelumnya" benar.

# %%
data = pd.read_csv(DATA_CSV, parse_dates=[TIME_COL])
data = data.sort_values([CARD_COL, TIME_COL])
data = data.reset_index(drop=True)
print(data.shape, "| tingkat fraud:", round(data[TARGET].mean(), 4))
print(data[TIME_COL].min(), "->", data[TIME_COL].max())
print("kartu:", data[CARD_COL].nunique())
data.head()

# %% [markdown]
# ## Feature kecepatan per kartu (hanya masa lalu)
# Tiap kartu diproses sendiri. `closed="left"` = jendela TIDAK menyertakan transaksi saat ini.
# `shift(1)` = rata-rata hanya dari transaksi sebelumnya.

# %%
for window in WINDOWS:
    data["jumlah_" + window] = 0.0
    data["total_" + window] = 0.0
data["jam_sejak_terakhir"] = np.nan
data["rata2_kartu_sebelumnya"] = np.nan

for card, group in data.groupby(CARD_COL):
    for window in WINDOWS:
        rolling = group.rolling(window, on=TIME_COL, closed="left")[AMOUNT_COL]
        data.loc[group.index, "jumlah_" + window] = rolling.count().to_numpy()
        window_total = rolling.sum().fillna(0)
        data.loc[group.index, "total_" + window] = window_total.to_numpy()
    gap = group[TIME_COL].diff()
    data.loc[group.index, "jam_sejak_terakhir"] = gap.dt.total_seconds().to_numpy() / 3600
    past_mean = group[AMOUNT_COL].expanding().mean()
    data.loc[group.index, "rata2_kartu_sebelumnya"] = past_mean.shift(1).to_numpy()

print(data[["jumlah_1h", "jumlah_24h", "total_24h", "jam_sejak_terakhir"]].describe().round(2))

# %% [markdown]
# ## Feature lain: rasio nominal, jarak lokasi, waktu, kategori
# Jarak rumah pemegang kartu ke merchant (km) dihitung dengan rumus haversine.

# %%
data["rasio_nominal"] = data[AMOUNT_COL] / data["rata2_kartu_sebelumnya"]   # kosong di awal
data["jam_hari"] = data[TIME_COL].dt.hour
data["hari_minggu"] = data[TIME_COL].dt.dayofweek

home_lat = np.radians(data["lat"])
shop_lat = np.radians(data["merch_lat"])
lon_gap = np.radians(data["merch_long"] - data["long"])
part = np.sin((shop_lat - home_lat) / 2) ** 2
part = part + np.cos(home_lat) * np.cos(shop_lat) * np.sin(lon_gap / 2) ** 2
data["jarak_km"] = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(part))

data["kategori"] = data["category"].astype("category").cat.codes
data["jenis_kelamin"] = (data["gender"] == "F").astype(int)

VELOCITY_FEATURES = ["jumlah_1h", "total_1h", "jumlah_24h", "total_24h", "jumlah_7d",
                     "total_7d", "jam_sejak_terakhir", "rata2_kartu_sebelumnya", "rasio_nominal"]
BASE_FEATURES = [AMOUNT_COL, "jam_hari", "hari_minggu", "jarak_km", "kategori", "jenis_kelamin",
                 "city_pop"]
print(data[["rasio_nominal", "jarak_km"]].describe().round(2))

# %% [markdown]
# ## Split berdasarkan waktu
# Train = masa lalu, validasi = berikutnya, test = paling baru. Split acak akan bocor:
# transaksi kartu yang sama di hari yang sama muncul di train dan validasi.

# %%
train = data[data[TIME_COL] < TRAIN_END]
validation = data[(data[TIME_COL] >= TRAIN_END) & (data[TIME_COL] < VALIDATION_END)]
test = data[data[TIME_COL] >= VALIDATION_END]
print("train   :", train.shape, "| fraud:", train[TARGET].sum())
print("validasi:", validation.shape, "| fraud:", validation[TARGET].sum())
print("test    :", test.shape, "| fraud:", test[TARGET].sum())

# %% [markdown]
# ## Model
# `scale_pos_weight` menaikkan bobot kelas fraud yang jarang.
# Baris `# features = ...` = tanpa feature kecepatan, untuk melihat seberapa besar sumbangannya.

# %%
features = BASE_FEATURES + VELOCITY_FEATURES
# features = BASE_FEATURES

normal_count = (train[TARGET] == 0).sum()
fraud_count = (train[TARGET] == 1).sum()
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                       scale_pos_weight=normal_count / fraud_count / 5, random_state=SEED,
                       n_jobs=4, verbose=-1)
model.fit(train[features], train[TARGET])

val_proba = model.predict_proba(validation[features])[:, 1]
print("PR-AUC validasi:", round(average_precision_score(validation[TARGET], val_proba), 4))

# %% [markdown]
# ## Threshold keputusan
# PR-AUC tidak butuh threshold, tetapi label akhir butuh. Pilih threshold dengan F1 fraud
# terbaik di validasi.

# %%
best_f1 = 0
threshold = 0.5
for candidate in np.arange(0.05, 0.96, 0.01):
    candidate_f1 = f1_score(validation[TARGET], val_proba > candidate)
    if candidate_f1 > best_f1:
        best_f1 = candidate_f1
        threshold = candidate

val_guess = val_proba > threshold
print("threshold terbaik:", round(threshold, 2), "| F1 fraud validasi:", round(best_f1, 3))
print("precision:", round(precision_score(validation[TARGET], val_guess), 3),
      "| recall:", round(recall_score(validation[TARGET], val_guess), 3))

importance = pd.Series(model.feature_importances_, index=features)
print(importance.sort_values(ascending=False).head(8))

# %% [markdown]
# ## Test (data paling baru) dan submission

# %%
test_proba = model.predict_proba(test[features])[:, 1]
predictions = (test_proba > threshold).astype(int)
print("PR-AUC test:", round(average_precision_score(test[TARGET], test_proba), 4))
print("F1 fraud test:", round(f1_score(test[TARGET], predictions), 3),
      "| precision:", round(precision_score(test[TARGET], predictions), 3),
      "| recall:", round(recall_score(test[TARGET], predictions), 3))

submission = pd.DataFrame()
submission["trans_num"] = test["trans_num"].to_numpy()
submission["proba_fraud"] = test_proba
submission["is_fraud"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
