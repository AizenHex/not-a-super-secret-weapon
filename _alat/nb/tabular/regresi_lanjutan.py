# %% [markdown]
# # Tabular — Regresi Lanjutan
# **Kapan dipakai:**
# 1. Target sangat miring ke kanan (banyak nilai kecil, sedikit nilai sangat besar) atau
#    banyak outlier → pakai log-target, Tweedie, atau Huber.
# 2. Diminta rentang tebakan (bawah–atas), bukan satu angka → regresi kuantil.
#
# **Data demo:** Diamonds (target `price`, miring ke kanan).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Ridge, TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/diamonds_train.csv"
TEST_CSV = f"{DATA_FOLDER}/diamonds_test.csv"

TARGET = "price"
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS = ["cut", "color", "clarity"]
LOWER_QUANTILE = 0.1     # batas bawah rentang: 10% nilai asli di bawahnya
UPPER_QUANTILE = 0.9     # batas atas rentang: 90% nilai asli di bawahnya
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## EDA — apakah target miring?
# Kiri: target asli. Kanan: setelah `log1p` (biasanya lebih simetris).
# Skewness > 1 = sangat miring.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
train[TARGET].hist(bins=50, ax=axes[0])
np.log1p(train[TARGET]).hist(bins=50, ax=axes[1])
axes[0].set_title(TARGET)
axes[1].set_title(f"log1p({TARGET})")
plt.show()
print("skewness:", round(train[TARGET].skew(), 2))

# %% [markdown]
# ## Split train / validasi

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %% [markdown]
# ## Model
# - `TransformedTargetRegressor(..., func=np.log1p, inverse_func=np.expm1)`: model belajar
#   log(target), lalu tebakannya otomatis dikembalikan ke satuan asli.
# - `objective="tweedie"/"poisson"/"gamma"`: cocok untuk target positif yang miring.
# - `objective="huber"` / `HuberRegressor`: tahan terhadap outlier.
#
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

lgbm = LGBMRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED, verbose=-1)
model = TransformedTargetRegressor(lgbm, func=np.log1p, inverse_func=np.expm1)
# model = LGBMRegressor(objective="tweedie", n_estimators=800, learning_rate=0.03, verbose=-1)
# model = LGBMRegressor(objective="huber", alpha=1000, n_estimators=800, verbose=-1)
# model = TweedieRegressor(power=2, link="log", max_iter=1000)
# model = HuberRegressor(max_iter=1000)
# model = LGBMRegressor(objective="poisson", n_estimators=800, learning_rate=0.03, verbose=-1)
# model = LGBMRegressor(objective="gamma", n_estimators=800, learning_rate=0.03, verbose=-1)
# model = TransformedTargetRegressor(Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1)

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("RMSE:", round(np.sqrt(mean_squared_error(y_val, val_predictions)), 2))
print("MAE :", round(mean_absolute_error(y_val, val_predictions), 2))

# %% [markdown]
# ## Rentang tebakan (regresi kuantil)
# Dua model: satu menebak batas bawah, satu menebak batas atas.
# Coverage = persen nilai asli yang masuk rentang. Targetnya `UPPER - LOWER` (di sini 80%).

# %%
lower_model = LGBMRegressor(objective="quantile", alpha=LOWER_QUANTILE, n_estimators=500,
                            verbose=-1)
upper_model = LGBMRegressor(objective="quantile", alpha=UPPER_QUANTILE, n_estimators=500,
                            verbose=-1)
pipe_lower = make_pipeline(preprocess, lower_model)
pipe_upper = make_pipeline(preprocess, upper_model)
pipe_lower.fit(X_train, y_train)
pipe_upper.fit(X_train, y_train)

val_lower = pipe_lower.predict(X_val)
val_upper = pipe_upper.predict(X_val)
inside = (y_val >= val_lower) & (y_val <= val_upper)
print("coverage:", round(inside.mean(), 3), "| target:", UPPER_QUANTILE - LOWER_QUANTILE)
print("lebar rentang rata-rata:", round((val_upper - val_lower).mean(), 1))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
pipe_lower.fit(X, y)
pipe_upper.fit(X, y)
X_test = test[NUM_COLS + CAT_COLS]

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["price"] = pipe.predict(X_test)
submission["price_bawah"] = pipe_lower.predict(X_test)
submission["price_atas"] = pipe_upper.predict(X_test)
submission.to_csv("submission.csv", index=False)
submission.head()
