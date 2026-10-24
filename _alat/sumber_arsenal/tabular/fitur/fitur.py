# %% [markdown]
# # Rekayasa fitur tabel: kategori → angka, tanggal, agregasi per kelompok, jarak, kombinasi
#
# **Butuh:** `train` dan `test` (sudah bersih, lihat `umum/tabel_bersih`), kolom `fold`
# (`tabular/dasar/validasi`). **Menghasilkan:** kolom fitur baru di `train`/`test` +
# tabel perbandingan skor CV per kelompok fitur (fitur yang tidak menaikkan skor → buang).
#
# Aturan emas: **hitung fitur di train dan test dengan cara yang SAMA**. Paling aman: gabung
# dulu `full = pd.concat([train, test])`, buat fitur, lalu pisah lagi. Fitur yang memakai
# TARGET (target encoding) wajib dihitung out-of-fold supaya tidak bocor.
#
# Playbook Kaggle Grandmaster (NVIDIA 2025): kombinasi pasangan kategori dan agregasi
# groupby adalah sumber skor terbesar di data tabel; GBDT (LightGBM) tidak butuh scaling.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | kapan |
# |---|---|---|
# | 1 | gabung train+test → `full` | **selalu** |
# | 2 | kategori: kode, frekuensi (count), one-hot | **selalu** |
# | 3 | target encoding out-of-fold | kategori banyak nilai (> 30) |
# | 4 | kombinasi pasangan kategori | coba |
# | 5 | fitur tanggal (jam, hari, siklus sin/cos) | ada tanggal |
# | 6 | agregasi per kelompok (rata-rata, rasio, selang waktu) | ada id pelanggan/toko |
# | 7 | jarak geografis (haversine) | ada lat/long |
# | 8 | log, binning, penanda kosong | angka miring / ada NaN |
# | 9 | pisah lagi + bandingkan skor per kelompok fitur | **selalu** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Transaksi kartu kredit (`is_fraud`). Demo: train = 70% waktu awal, test = 30% sisanya.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

data = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/fraud_transaksi/transaksi_sparkov.csv")
data["waktu_transaksi"] = pd.to_datetime(data["waktu_transaksi"])
cut = data["waktu_transaksi"].quantile(0.7)
train = data[data["waktu_transaksi"] < cut].reset_index(drop=True)
test = data[data["waktu_transaksi"] >= cut].reset_index(drop=True)
test_labels = test[["trans_num", "is_fraud"]].copy()
test = test.drop(columns="is_fraud")
print(train.shape, test.shape, "| porsi fraud:", round(train["is_fraud"].mean(), 4))

# %% [markdown]
# ## 1. Gabung train + test → `full` (kolom `is_test` untuk memisah lagi di sel 9)

# %%
import pandas as pd

ID = "trans_num"       # UBAH
TARGET = "is_fraud"    # UBAH
DATE = "waktu_transaksi"   # UBAH: None kalau tidak ada

train["is_test"] = 0
test["is_test"] = 1
full = pd.concat([train, test], ignore_index=True)
full[DATE] = pd.to_datetime(full[DATE])
print(full.shape, "| target kosong di baris test:", full[TARGET].isna().sum())

# %% [markdown]
# ## 2. Kategori → angka
# - **kode** (`cat.codes`): cukup untuk LightGBM/XGBoost/CatBoost.
# - **frekuensi**: berapa kali nilai itu muncul; nilai langka sering menarik (fraud, anomali).
# - **one-hot**: untuk model linear/NN, hanya kategori dengan sedikit nilai (< 15).

# %%
import pandas as pd

CATEGORICAL = ["category", "gender", "state", "merchant"]   # UBAH
ONE_HOT = ["category", "gender"]                             # UBAH: yang nilainya sedikit

for column in CATEGORICAL:
    full[column] = full[column].fillna("kosong").astype(str)
    full[f"{column}_code"] = full[column].astype("category").cat.codes
    full[f"{column}_count"] = full[column].map(full[column].value_counts())
one_hot = pd.get_dummies(full[ONE_HOT], prefix=ONE_HOT, dtype=int)
full = pd.concat([full, one_hot], axis=1)
print(full[CATEGORICAL].nunique())
print(one_hot.shape[1], "kolom one-hot")

# %% [markdown]
# ## 3. Target encoding out-of-fold: kategori → rata-rata target kategori itu
# Baris train: rata-rata dihitung dari fold LAIN (tidak melihat targetnya sendiri). Baris test:
# dari seluruh train. `SMOOTHING` menarik kategori kecil ke rata-rata global (anti-overfit).

# %%
import numpy as np
from sklearn.model_selection import KFold

SMOOTHING = 20   # UBAH: makin besar makin hati-hati


def smoothed_mean(source, column, target):
    stats = source.groupby(column)[target].agg(["sum", "count"])
    prior = source[target].mean()
    return (stats["sum"] + prior * SMOOTHING) / (stats["count"] + SMOOTHING), prior


def target_encode(table, column, target, n_splits=5):
    is_train = (table["is_test"] == 0).to_numpy()
    encoded = np.full(len(table), np.nan)
    train_index = np.where(is_train)[0]
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fit_part, fill_part in splitter.split(train_index):
        means, prior = smoothed_mean(table.iloc[train_index[fit_part]], column, target)
        mapped = table[column].iloc[train_index[fill_part]].map(means)
        encoded[train_index[fill_part]] = mapped.fillna(prior).to_numpy()
    means, prior = smoothed_mean(table[is_train], column, target)
    mapped = table.loc[~is_train, column].map(means)
    encoded[~is_train] = mapped.fillna(prior).to_numpy()
    return encoded


for column in ["merchant", "state"]:   # UBAH
    full[f"{column}_te"] = target_encode(full, column, TARGET)
print(full[["merchant", "merchant_te"]].drop_duplicates().head())

# %% [markdown]
# ## 4. Kombinasi pasangan kategori (gabung teks) → kode + frekuensi
# Pola "kategori A DAN kategori B" sulit ditemukan pohon sendiri. Coba semua pasangan, pertahankan
# yang menaikkan skor CV (sel 9). Pasangan dengan nilai sangat banyak → pakai target encoding.

# %%
from itertools import combinations

PAIR_SOURCE = ["category", "gender", "state"]   # UBAH

for left, right in combinations(PAIR_SOURCE, 2):
    name = f"{left}_x_{right}"
    full[name] = full[left] + "_" + full[right]
    full[f"{name}_code"] = full[name].astype("category").cat.codes
    full[f"{name}_count"] = full[name].map(full[name].value_counts())
    print(name, full[name].nunique(), "nilai")

# %% [markdown]
# ## 5. Fitur tanggal: jam, hari dalam minggu, akhir pekan, malam, siklus sin/cos
# JANGAN pakai tanggal mentah/nomor urut waktu sebagai fitur kalau test = masa depan (lihat
# adversarial validation di `tabular/dasar/validasi`). Sin/cos: jam 23 dekat jam 0.

# %%
import numpy as np

when = full[DATE]
full["hour"] = when.dt.hour
full["day_of_week"] = when.dt.dayofweek          # 0 = Senin
full["day_of_month"] = when.dt.day
full["is_weekend"] = (full["day_of_week"] >= 5).astype(int)
full["is_night"] = ((full["hour"] >= 22) | (full["hour"] <= 4)).astype(int)
full["hour_sin"] = np.sin(2 * np.pi * full["hour"] / 24)
full["hour_cos"] = np.cos(2 * np.pi * full["hour"] / 24)
night_rate = full.groupby("is_night")[TARGET].mean()
print(night_rate.round(4))   # malam lebih banyak fraud?

# %% [markdown]
# ## 6. Agregasi per kelompok (pelanggan/kartu): rata-rata, rasio, banyak transaksi, selang waktu
# "Transaksi ini 10× rata-rata biasanya" lebih bermakna daripada nominalnya saja.
# Tanpa target → boleh dihitung di train+test gabungan.

# %%
import numpy as np

GROUP = "cc_num"   # UBAH
AMOUNT = "amt"     # UBAH

full = full.sort_values([GROUP, DATE]).reset_index(drop=True)
by_group = full.groupby(GROUP)[AMOUNT]
full["group_amt_mean"] = by_group.transform("mean")
full["group_amt_std"] = by_group.transform("std").fillna(0)
full["group_count"] = by_group.transform("count")
full["amt_to_group_mean"] = full[AMOUNT] / full["group_amt_mean"]
full["amt_zscore"] = (full[AMOUNT] - full["group_amt_mean"]) / (full["group_amt_std"] + 1)
gap = full.groupby(GROUP)[DATE].diff()
full["hours_since_previous"] = (gap.dt.total_seconds() / 3600).fillna(-1)
full["previous_amt"] = by_group.shift(1).fillna(-1)
full["amt_rank_in_group"] = by_group.rank(pct=True)
print(full[[GROUP, AMOUNT, "amt_to_group_mean", "hours_since_previous"]].head())

# %% [markdown]
# Jendela waktu: banyak transaksi & total nominal **24 jam terakhir** per kartu (termasuk baris
# itu sendiri). Fraud sering berupa rentetan transaksi dalam waktu singkat.

# %%
import pandas as pd

WINDOW = "24h"   # UBAH: "1h", "7D", ...

timed = full.set_index(DATE)
rolling = timed.groupby(GROUP)[AMOUNT].rolling(WINDOW)
window_count = rolling.count().reset_index(drop=True)
window_sum = rolling.sum().reset_index(drop=True)
full["count_last_24h"] = window_count.to_numpy()   # urutan sama: full sudah diurut kelompok+waktu
full["amt_sum_last_24h"] = window_sum.to_numpy()
burst = full.groupby(TARGET)["count_last_24h"].mean()
print(burst.round(2))   # fraud: lebih banyak transaksi dalam 24 jam

# %% [markdown]
# ## 7. Jarak geografis (haversine, km) antara dua titik lat/long

# %%
import numpy as np


def haversine_km(lat_a, long_a, lat_b, long_b):
    lat_a, long_a, lat_b, long_b = np.radians([lat_a, long_a, lat_b, long_b])
    inner = np.sin((lat_b - lat_a) / 2) ** 2
    inner = inner + np.cos(lat_a) * np.cos(lat_b) * np.sin((long_b - long_a) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(inner))


full["distance_km"] = haversine_km(full["lat"], full["long"], full["merch_lat"],
                                   full["merch_long"])   # UBAH: kolom lat/long
print(full["distance_km"].describe().round(1))

# %% [markdown]
# ## 8. Angka: log (nilai miring), binning (kuantil), penanda kosong
# Log & binning terutama membantu model linear/NN; GBDT kebal terhadap skala.
# Penanda kosong: "data tidak diisi" sering informatif → simpan SEBELUM NaN diisi.

# %%
import numpy as np
import pandas as pd

SKEWED = ["amt", "city_pop"]   # UBAH
for column in SKEWED:
    full[f"{column}_log"] = np.log1p(full[column].clip(lower=0))
full["amt_bin"] = pd.qcut(full["amt"], q=10, labels=False, duplicates="drop")
for column in full.columns:
    if full[column].isna().any() and column != TARGET:
        full[f"{column}_missing"] = full[column].isna().astype(int)
print(full[["amt", "amt_log", "amt_bin"]].head(3))

# %% [markdown]
# ## 9. Pisah lagi → `train`, `test` + bandingkan skor CV tiap kelompok fitur
# Tambah kelompok satu per satu; kelompok yang tidak menaikkan skor dibuang.
# Data berwaktu → val = 25% waktu terakhir train (lihat `tabular/dasar/validasi`).

# %%
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score

train = full[full["is_test"] == 0].sort_values(DATE).reset_index(drop=True)
test = full[full["is_test"] == 1].sort_values(DATE).reset_index(drop=True)
split = int(len(train) * 0.75)
part_fit, part_val = train.iloc[:split], train.iloc[split:]
GROUPS = {"dasar": ["amt", "city_pop", "category_code", "gender_code"],                  # UBAH
          "+tanggal": ["hour", "day_of_week", "is_night"],
          "+kelompok": ["amt_to_group_mean", "amt_zscore", "hours_since_previous",
                        "count_last_24h", "amt_sum_last_24h", "amt_rank_in_group"],
          "+frek/te/pasangan": ["merchant_count", "merchant_te", "category_x_gender_code"],
          "+jarak": ["distance_km"]}
features = []
for name, columns in GROUPS.items():
    features = features + columns
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1)
    model.fit(part_fit[features], part_fit[TARGET])
    score = average_precision_score(part_val[TARGET], model.predict_proba(part_val[features])[:, 1])
    print(f"{name:20s} {len(features):3d} fitur | PR-AUC val {score:.4f}")

# %% [markdown]
# Latih dengan fitur terpilih di SEMUA train → prediksi test. Cek ke kunci (hanya demo).

# %%
import pandas as pd
from sklearn.metrics import average_precision_score

features.remove("distance_km")   # UBAH: buang kelompok yang menurunkan skor val
model = LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1)
model.fit(train[features], train[TARGET])
submission = pd.DataFrame()
submission["trans_num"] = test[ID]
submission["is_fraud"] = model.predict_proba(test[features])[:, 1]
submission.to_csv("submission.csv", index=False)
answer = submission[ID].map(dict(zip(test_labels[ID], test_labels[TARGET])))
print("PR-AUC test:", round(average_precision_score(answer, submission["is_fraud"]), 4))
