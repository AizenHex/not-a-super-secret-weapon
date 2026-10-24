# %% [markdown]
# # Unsupervised — Imputasi (mengisi nilai kosong)
# **Kapan dipakai:** banyak sel kosong (NaN) dan diminta mengisinya, atau model butuh data
# lengkap. Juga untuk memilih cara isi terbaik sebelum modeling.
#
# **Idenya:** sebagian sel yang sebenarnya diketahui sengaja dihapus, lalu diisi oleh imputer.
# Tebakan dibandingkan dengan nilai aslinya (RMSE), jadi cara isi bisa dinilai tanpa label.
#
# **Langkah:**
# 1. Samakan skala, hapus 10% sel secara acak.
# 2. Isi dengan imputer, hitung RMSE pada sel yang dihapus.
# 3. Isi data asli dengan imputer yang sama, simpan.
#
# **Data demo:** Wine (13 kolom angka, tanpa missing asli). RMSE isi-rata-rata kira-kira 1.0,
# makin kecil makin baik.

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer   # wajib sebelum IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/wine.csv"

DROP_COLS = ["cultivar"]    # kolom yang tidak ikut diisi (label, id)
MISSING_FRACTION = 0.10     # porsi sel yang sengaja dihapus untuk menilai imputer
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
X = data.drop(columns=DROP_COLS)
missing_per_column = X.isna().sum()
print(X.shape)
print("missing asli:", missing_per_column.sum())
X.head()

# %% [markdown]
# ## Langkah 1 — hapus sel secara acak
# `hidden` = tabel True/False, True = sel yang dihapus. Skala disamakan supaya RMSE tiap kolom
# setara.

# %%
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
rng = np.random.default_rng(SEED)
hidden = rng.random(X_scaled.shape) < MISSING_FRACTION
X_missing = X_scaled.mask(hidden)
print("sel dihapus:", hidden.sum(), "dari", hidden.size)

# %% [markdown]
# ## Langkah 2 — imputer
# Pilih SATU imputer: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
imputer = SimpleImputer(strategy="mean")
# imputer = SimpleImputer(strategy="median")                  # tahan nilai ekstrem
# imputer = KNNImputer(n_neighbors=5)                         # rata-rata 5 baris termirip
# imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=SEED)
# imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=100, random_state=SEED),
#                            max_iter=5, random_state=SEED)   # paling teliti, paling lambat

X_filled = imputer.fit_transform(X_missing)
true_values = X_scaled.to_numpy()[hidden]
guesses = X_filled[hidden]
rmse = np.sqrt(np.mean((guesses - true_values) ** 2))
print("RMSE pada sel yang dihapus:", round(rmse, 3))

# %% [markdown]
# ## Langkah 3 — isi data asli dan simpan
# Imputer dilatih ulang pada data asli (skala asli), lalu kolom yang dibuang dikembalikan.

# %%
X_final = imputer.fit_transform(X)

submission = pd.DataFrame()
for i in range(len(X.columns)):
    submission[X.columns[i]] = X_final[:, i]
for name in DROP_COLS:
    submission[name] = data[name]
submission.to_csv("submission.csv", index=False)
submission.head()
