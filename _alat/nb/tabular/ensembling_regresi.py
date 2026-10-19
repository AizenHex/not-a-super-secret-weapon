# %% [markdown]
# # Tabular — Ensembling (Regresi)
# **Kapan dipakai:** sudah punya beberapa model regresi, ingin menggabungkannya agar
# tebakan lebih stabil.
#
# **Dua cara menggabungkan:**
# - **Voting:** rata-rata tebakan semua model.
# - **Stacking:** model "meta" belajar dari tebakan model-model dasar (dibuat dengan CV).
#
# **Data demo:** Diamonds (target `price`).

# %%
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# ## Split train / validasi

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %% [markdown]
# ## Model
# `base_models` = daftar model dasar, tiap model diberi nama pendek.
# Pilih SATU `ensemble`: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

base_models = [
    ("ridge", Ridge(alpha=1.0)),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ("lgbm", LGBMRegressor(n_estimators=800, learning_rate=0.03, verbose=-1)),
    ("cat", CatBoostRegressor(iterations=800, random_seed=SEED, verbose=0,
                              allow_writing_files=False)),
]

ensemble = StackingRegressor(base_models, final_estimator=RidgeCV(), cv=5)
# ensemble = VotingRegressor(base_models)                        # rata-rata tebakan
# ensemble = StackingRegressor(base_models, final_estimator=LGBMRegressor(verbose=-1), cv=5)
# ensemble = StackingRegressor(base_models, final_estimator=RidgeCV(), cv=5, passthrough=True)
# ensemble = VotingRegressor(base_models, weights=[1, 1, 2, 2])

pipe = make_pipeline(preprocess, ensemble)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("RMSE:", round(np.sqrt(mean_squared_error(y_val, val_predictions)), 2))
print("MAE :", round(mean_absolute_error(y_val, val_predictions), 2))

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
