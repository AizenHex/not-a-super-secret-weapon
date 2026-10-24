# %% [markdown]
# # Trajektori — Feature GPS dan Klasifikasi Moda Transport
# **Kapan dipakai:** data berupa titik GPS (lat, lon, waktu) per perjalanan dan targetnya per
# perjalanan (moda transport, jenis kendaraan, perilaku pengemudi).
#
# **Idenya:** titik mentah tidak bisa langsung masuk model. Hitung dulu jarak antar titik,
# kecepatan, percepatan, belokan, dan titik berhenti, lalu ringkas jadi **satu baris per
# perjalanan** (rata-rata, maksimum, persentil) → LightGBM.
#
# **Langkah:**
# 1. Feature per titik (jarak haversine, kecepatan, percepatan, belokan, berhenti).
# 2. Ringkas jadi satu baris per perjalanan.
# 3. Validasi per pengguna (GroupKFold), latih ulang, submission.
#
# **Data demo:** potongan GeoLife (Beijing), 5 moda: jalan, sepeda, bus, mobil, kereta.
# Orang yang sama tidak ada di latih dan test.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/trajektori/geolife_mini"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/gps_latih.csv"
TEST_CSV = f"{DATA_FOLDER}/gps_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/gps_test_labels.csv"   # hanya ada di data latihan

TRIP_COL = "trip_id"      # satu nilai = satu perjalanan
USER_COL = "user"         # untuk validasi per pengguna
TIME_COL = "waktu"
TARGET = "moda"
EARTH_RADIUS_M = 6371000
STOP_SPEED = 0.5          # kecepatan di bawah ini (m/s) dianggap berhenti
SHARP_TURN = 45           # perubahan arah di atas ini (derajat) dianggap belokan tajam
MAX_SPEED = 70            # di atas ini (m/s) dianggap GPS loncat dan dibuang
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV, dtype={USER_COL: str}, parse_dates=[TIME_COL])
test = pd.read_csv(TEST_CSV, dtype={USER_COL: str}, parse_dates=[TIME_COL])
print(train.shape, test.shape)
print(train.groupby(TARGET)[TRIP_COL].nunique())
train.head()

# %% [markdown]
# ## Langkah 1a — jarak, kecepatan, percepatan
# Train dan test digabung supaya feature dibuat sekali. Titik "sebelumnya" = `shift(1)` di dalam
# tiap perjalanan.

# %%
def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    inner = (np.sin((lat2 - lat1) / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(inner))


# coba: Monas ke Bundaran HI, sekitar 2.200 m
print(round(haversine_m(-6.1754, 106.8272, -6.1950, 106.8231)), "m")

points = pd.concat([train, test], ignore_index=True)
points = points.sort_values([TRIP_COL, TIME_COL])
per_trip = points.groupby(TRIP_COL)
points["prev_lat"] = per_trip["lat"].shift(1)
points["prev_lon"] = per_trip["lon"].shift(1)
points["distance_m"] = haversine_m(points["prev_lat"], points["prev_lon"],
                                   points["lat"], points["lon"])
points["seconds"] = per_trip[TIME_COL].diff().dt.total_seconds()
points["speed"] = points["distance_m"] / points["seconds"]

# buang titik dengan selisih waktu 0 atau loncatan GPS yang mustahil
points = points[(points["seconds"] > 0) & (points["speed"] < MAX_SPEED)].copy()
per_trip = points.groupby(TRIP_COL)
points["acceleration"] = per_trip["speed"].diff() / points["seconds"]
points["acceleration_abs"] = points["acceleration"].abs()

# %% [markdown]
# ## Langkah 1b — arah, belokan, berhenti
# Belokan = perubahan arah antar segmen (0–180 derajat). "Mulai berhenti" = tadi bergerak,
# sekarang diam.

# %%
lat1 = np.radians(points["prev_lat"])
lat2 = np.radians(points["lat"])
lon_gap = np.radians(points["lon"] - points["prev_lon"])
north = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon_gap)
east = np.sin(lon_gap) * np.cos(lat2)
points["heading"] = np.degrees(np.arctan2(east, north))
heading_change = points.groupby(TRIP_COL)["heading"].diff()
heading_change = heading_change.abs()
points["turn"] = np.minimum(heading_change, 360 - heading_change)
points["is_sharp_turn"] = (points["turn"] > SHARP_TURN).astype(int)

points["is_stop"] = (points["speed"] < STOP_SPEED).astype(int)
stop_before = points.groupby(TRIP_COL)["is_stop"].shift(1)
points["stop_start"] = ((points["is_stop"] == 1) & (stop_before == 0)).astype(int)
points[["distance_m", "speed", "acceleration", "turn"]].describe()

# %% [markdown]
# ## Langkah 2 — satu baris per perjalanan

# %%
per_trip = points.groupby(TRIP_COL)
features = pd.DataFrame()
features["distance_km"] = per_trip["distance_m"].sum() / 1000
features["duration_min"] = per_trip["seconds"].sum() / 60
features["n_points"] = per_trip["seconds"].count()
features["speed_mean"] = per_trip["speed"].mean()
features["speed_median"] = per_trip["speed"].median()
features["speed_std"] = per_trip["speed"].std()
features["speed_max"] = per_trip["speed"].max()
features["speed_p85"] = per_trip["speed"].quantile(0.85)
features["accel_abs_mean"] = per_trip["acceleration_abs"].mean()
features["accel_std"] = per_trip["acceleration"].std()
features["turn_mean"] = per_trip["turn"].mean()
features["sharp_turns_per_km"] = per_trip["is_sharp_turn"].sum() / features["distance_km"]
features["stop_share"] = per_trip["is_stop"].mean()
features["n_stops"] = per_trip["stop_start"].sum()
features = features.fillna(0)

features["mode"] = per_trip[TARGET].first()        # kosong untuk trip test
features["user"] = per_trip[USER_COL].first()
print(features.shape)
features.head()

# %% [markdown]
# ## Langkah 3a — model + validasi per pengguna (GroupKFold)
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
train_features = features[features["mode"].notna()]
test_features = features[features["mode"].isna()]
X = train_features.drop(columns=["mode", "user"])
y = train_features["mode"]
groups = train_features["user"]
X_test = test_features.drop(columns=["mode", "user"])

model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       random_state=SEED, verbose=-1, n_jobs=4)
# model = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=SEED,
#                                n_jobs=4)

oof_predictions = cross_val_predict(model, X, y, cv=GroupKFold(5), groups=groups)
print("F1 macro GroupKFold:", round(f1_score(y, oof_predictions, average="macro"), 4))
print(classification_report(y, oof_predictions))

# %% [markdown]
# ## Langkah 3b — latih ulang + submission (satu baris per perjalanan)

# %%
model.fit(X, y)
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values()
importance.tail(10).plot.barh(figsize=(6, 4))
plt.show()

predictions = model.predict(X_test)

submission = pd.DataFrame()
submission["trip_id"] = X_test.index
submission["moda"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
test_key = pd.read_csv(TEST_KEY_CSV)
true_modes = pd.Series(test_key["moda"].to_numpy(), index=test_key["trip_id"])
truth = submission["trip_id"].map(true_modes)
print("F1 macro test:", round(f1_score(truth, predictions, average="macro"), 4))
ConfusionMatrixDisplay.from_predictions(truth, predictions)
plt.show()
