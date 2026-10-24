# %% [markdown]
# # Tabular — Regresi
# **Kapan dipakai:** data tabel (CSV), dan kita diminta menebak ANGKA untuk tiap baris test.
# Contoh: harga rumah, jumlah penjualan, nilai klaim.
#
# **Langkah:**
# 1. Baca data, lihat sebaran target.
# 2. Bagi train menjadi train dan validasi.
# 3. Siapkan kolom, latih model, ukur RMSE/MAE/R2 di validasi.
# 4. Latih ulang di semua data train, tebak test, simpan submission.
#
# **Data demo:** Diamonds (target `price`). Ganti path dan nama kolom di sel Konfigurasi.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/diamonds_train.csv"
TEST_CSV = f"{DATA_FOLDER}/diamonds_test.csv"

TARGET = "price"                                       # kolom angka yang ditebak
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]  # kolom angka
CAT_COLS = ["cut", "color", "clarity"]                 # kolom kategori (teks)
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA
# - Sebaran target: kalau sangat miring ke kanan (banyak kecil, sedikit sangat besar),
#   pertimbangkan `np.log1p(target)`.
# - Korelasi kolom angka dengan target: makin jauh dari 0, makin berpengaruh.

# %%
print(train[TARGET].describe())
train[TARGET].hist(bins=50)
plt.title(TARGET)
plt.show()

correlation = train[NUM_COLS + [TARGET]].corr()
print(correlation[TARGET].sort_values())

# %% [markdown]
# ## Split train / validasi

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("train:", X_train.shape, "| validasi:", X_val.shape)

# %% [markdown]
# ## Model
# `preprocess`: kolom angka diisi median + StandardScaler, kolom kategori diisi nilai terbanyak
# + OneHot. Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = Ridge(alpha=1.0)                                  # regresi linear + penalti
# model = Lasso(alpha=1.0)
# model = ElasticNet(alpha=0.1, l1_ratio=0.5)
# model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
# model = ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
# model = HistGradientBoostingRegressor(random_state=SEED)
# model = XGBRegressor(n_estimators=1000, learning_rate=0.03, random_state=SEED)
# model = CatBoostRegressor(iterations=1000, random_seed=SEED, verbose=0)
# model = KNeighborsRegressor(n_neighbors=10)
# model = DecisionTreeRegressor(max_depth=10, random_state=SEED)
# model = GradientBoostingRegressor(random_state=SEED)
# model = AdaBoostRegressor(n_estimators=300, random_state=SEED)
# model = BayesianRidge()
# model = HuberRegressor(max_iter=1000)                     # tahan terhadap outlier
# model = LinearSVR(C=1.0, max_iter=10000, random_state=SEED)
# model = MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=SEED)

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# RMSE dan MAE dalam satuan target (makin kecil makin bagus). R2: 1.0 = sempurna.
# Gambar: titik makin dekat garis diagonal, makin tepat.

# %%
val_predictions = pipe.predict(X_val)
print("RMSE:", round(np.sqrt(mean_squared_error(y_val, val_predictions)), 2))
print("MAE :", round(mean_absolute_error(y_val, val_predictions), 2))
print("R2  :", round(r2_score(y_val, val_predictions), 4))

plt.scatter(y_val, val_predictions, s=5, alpha=0.5)
plt.xlabel("asli")
plt.ylabel("prediksi")
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[NUM_COLS + CAT_COLS])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["price"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
