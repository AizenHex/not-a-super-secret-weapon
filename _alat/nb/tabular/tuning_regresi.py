# %% [markdown]
# # Tabular — Tuning Hyperparameter (Regresi, Optuna)
# **Kapan dipakai:** model sudah dipilih, dan kita ingin mencari setelan (hyperparameter)
# yang lebih baik, misalnya jumlah pohon dan learning rate.
#
# **Cara kerja Optuna:** coba `N_TRIALS` kombinasi setelan. Tiap kombinasi dinilai dengan
# RMSE cross validation. Kombinasi berikutnya dipilih dekat kombinasi yang sudah bagus.
#
# **Data demo:** Diamonds (target `price`).

# %%
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)   # sembunyikan log tiap percobaan

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/diamonds_train.csv"
TEST_CSV = f"{DATA_FOLDER}/diamonds_test.csv"

TARGET = "price"
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS = ["cut", "color", "clarity"]
N_TRIALS = 20          # jumlah kombinasi yang dicoba (makin banyak makin lama)
SEED = 42

# %% [markdown]
# ## Baca data, split, siapkan kolom

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)

X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

# %% [markdown]
# ## Model + ruang pencarian
# `make_model(trial)` membuat satu model dengan setelan yang dipilih Optuna.
# `trial.suggest_int("nama", bawah, atas)` = Optuna memilih angka bulat di rentang itu.
# Ganti model: pindahkan `#` sehingga hanya satu `return` yang aktif.

# %%
def make_model(trial):
    # return Ridge(alpha=trial.suggest_float("alpha", 1e-3, 100, log=True))
    # return RandomForestRegressor(n_estimators=trial.suggest_int("n_estimators", 100, 500),
    #     max_depth=trial.suggest_int("max_depth", 5, 30), random_state=SEED, n_jobs=-1)
    # return XGBRegressor(n_estimators=trial.suggest_int("n_estimators", 200, 1500),
    #     max_depth=trial.suggest_int("max_depth", 3, 10),
    #     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True))
    # return CatBoostRegressor(iterations=trial.suggest_int("iterations", 200, 1500),
    #     depth=trial.suggest_int("depth", 4, 10), verbose=0, allow_writing_files=False)
    return LGBMRegressor(
        n_estimators=trial.suggest_int("n_estimators", 200, 1500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 8, 128),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=SEED, verbose=-1,
    )


def objective(trial):
    pipe = make_pipeline(preprocess, make_model(trial))
    scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring="neg_root_mean_squared_error")
    return -scores.mean()


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)
print("CV RMSE terbaik:", round(study.best_value, 2))
print(study.best_params)

# %% [markdown]
# ## Evaluasi setelan terbaik di validasi

# %%
pipe = make_pipeline(preprocess, make_model(study.best_trial))
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("RMSE validasi:", round(np.sqrt(mean_squared_error(y_val, val_predictions)), 2))
print("MAE validasi :", round(mean_absolute_error(y_val, val_predictions), 2))

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
