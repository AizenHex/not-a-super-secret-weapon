# %% [markdown]
# # Tabular — Feature Engineering (Regresi)
# **Kapan dipakai:** model sudah jalan, dan kita ingin menambah kolom baru (feature) yang
# membantu model, misalnya volume = panjang × lebar × tinggi.
#
# **Langkah:** tulis feature baru di fungsi `add_features`, lalu bandingkan skor CV sebelum
# dan sesudah. Tambah/hapus baris di fungsi itu untuk mencoba feature lain.
#
# **Data demo:** Diamonds (target `price`).

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/diamonds_train.csv"
TEST_CSV = f"{DATA_FOLDER}/diamonds_test.csv"

TARGET = "price"
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS = ["cut", "color", "clarity"]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Feature baru
# - `volume`, `density`, `xy_ratio`: hitungan dari kolom ukuran. Angka 0 diganti NaN agar
#   tidak terjadi pembagian dengan nol.
# - `log_carat`: log membuat angka yang sangat besar tidak terlalu mendominasi.
# - `carat_vs_clarity`: carat dibanding rata-rata carat di kelompok clarity yang sama.
#   Rata-rata kelompok dihitung dari TRAIN saja, lalu dipakai juga untuk test.
# - `cut_color`: gabungan dua kategori menjadi satu kategori baru.

# %%
def add_features(table, carat_mean_per_clarity):
    table = table.copy()
    volume = table["x"] * table["y"] * table["z"]
    table["volume"] = volume.replace(0, np.nan)
    table["density"] = table["carat"] / table["volume"]
    table["xy_ratio"] = table["x"] / table["y"].replace(0, np.nan)
    table["log_carat"] = np.log1p(table["carat"])
    table["carat_vs_clarity"] = table["carat"] / table["clarity"].map(carat_mean_per_clarity)
    table["cut_color"] = table["cut"] + "_" + table["color"]
    return table


carat_mean_per_clarity = train.groupby("clarity")["carat"].mean()
train_fe = add_features(train, carat_mean_per_clarity)
test_fe = add_features(test, carat_mean_per_clarity)

NEW_NUM_COLS = NUM_COLS + ["volume", "density", "xy_ratio", "log_carat", "carat_vs_clarity"]
NEW_CAT_COLS = CAT_COLS + ["cut_color"]
train_fe[NEW_NUM_COLS + NEW_CAT_COLS].head()

# %% [markdown]
# ## Model
# Pilih SATU `model` dan SATU `encoder` (cara mengubah kategori menjadi angka):
# hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.
# - OneHot: satu kolom 0/1 per kategori.
# - Ordinal: satu angka per kategori (cocok untuk model pohon).
# - Target: rata-rata target per kategori (dihitung dengan CV agar tidak bocor).

# %%
model = Ridge(alpha=1.0)
# model = LGBMRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)

encoder = OneHotEncoder(handle_unknown="ignore")
# encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
# encoder = TargetEncoder(random_state=SEED)

num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"), encoder)

# %% [markdown]
# ## Evaluasi: CV sebelum vs sesudah feature baru
# RMSE dihitung di 5 fold. Kalau "sesudah" lebih kecil, feature baru membantu.

# %%
preprocess_before = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])
preprocess_after = ColumnTransformer([
    ("num", num_steps, NEW_NUM_COLS),
    ("cat", cat_steps, NEW_CAT_COLS),
])
pipe_before = make_pipeline(preprocess_before, model)
pipe_after = make_pipeline(preprocess_after, model)

cv = KFold(5, shuffle=True, random_state=SEED)
y = train[TARGET]
scores_before = -cross_val_score(pipe_before, train, y, cv=cv,
                                 scoring="neg_root_mean_squared_error")
scores_after = -cross_val_score(pipe_after, train_fe, y, cv=cv,
                                scoring="neg_root_mean_squared_error")
print("RMSE sebelum:", round(scores_before.mean(), 1), "±", round(scores_before.std(), 1))
print("RMSE sesudah:", round(scores_after.mean(), 1), "±", round(scores_after.std(), 1))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe_after.fit(train_fe, y)
predictions = pipe_after.predict(test_fe)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["price"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
