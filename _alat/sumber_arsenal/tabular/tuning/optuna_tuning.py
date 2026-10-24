# %% [markdown]
# # Tuning hyperparameter dengan Optuna (LightGBM / XGBoost / CatBoost)
#
# **Butuh:** `train`, `FEATURES`, target, kolom `fold` (atau dibuat di sel 1).
# **Menghasilkan:** `best_params` (dict siap ditempel ke `LGB_PARAMS` di notebook model),
# riwayat percobaan `outputs/optuna_trials.csv`, parameter yang paling berpengaruh.
#
# Kapan tuning: SETELAH fitur & validasi beres. Biasanya hanya memberi 1–3% → beri batas
# waktu (`TIMEOUT_SECONDS`). Fitur baru hampir selalu lebih menguntungkan daripada tuning.
# Optuna memakai TPE (Bergstra dkk. 2011): percobaan berikutnya diarahkan ke daerah
# parameter yang sudah terbukti bagus → jauh lebih hemat daripada grid search.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi: data, fold, metrik | **ya** |
# | 2 | `cv_score(params)`: skor CV satu set parameter | **ya** |
# | 3 | `objective(trial)`: ruang pencarian LightGBM | **ya** |
# | 4 | jalankan study (n_trials / timeout) | **ya** |
# | 5 | hasil: parameter terbaik + paling berpengaruh + simpan | **ya** |
# | 6 | ruang pencarian XGBoost / CatBoost (ganti sel 3) | opsional |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Harga berlian (regresi, RMSLE). Klasifikasi: ganti model + metrik di sel 1–3.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_test_labels.csv")
for column in ["cut", "color", "clarity"]:
    train[column] = train[column].astype("category")
    test[column] = pd.Categorical(test[column], categories=train[column].cat.categories)
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi: fitur, target (log), fold
# Fold dibuat SEKALI dan dipakai semua percobaan → perbandingan adil.

# %%
import numpy as np
from sklearn.model_selection import KFold

TARGET = "price"   # UBAH
FEATURES = ["carat", "cut", "color", "clarity", "depth", "table", "x", "y", "z"]   # UBAH
TIMEOUT_SECONDS = 300   # UBAH: batas waktu total tuning
N_TRIALS = 40           # UBAH

y = np.log1p(train[TARGET].to_numpy(dtype=float))   # RMSE di skala log = RMSLE
folds = np.zeros(len(train), dtype=int)
for fold, (_, index_val) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
    folds[index_val] = fold
print(len(FEATURES), "fitur |", folds.max() + 1, "fold")

# %% [markdown]
# ## 2. `cv_score(params)` → rata-rata RMSE val di semua fold (+ putaran terbaik)
# Early stopping per fold; `n_estimators` besar dan dibiarkan dipotong otomatis.

# %%
import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_error


def cv_score(params):
    errors, rounds = [], []
    for fold in range(folds.max() + 1):
        is_val = folds == fold
        model = lgb.LGBMRegressor(n_estimators=5000, verbose=-1, random_state=42, **params)
        model.fit(train.loc[~is_val, FEATURES], y[~is_val],
                  eval_X=train.loc[is_val, FEATURES], eval_y=y[is_val],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        guess = model.predict(train.loc[is_val, FEATURES])
        errors.append(np.sqrt(mean_squared_error(y[is_val], guess)))
        rounds.append(model.best_iteration_)
    return float(np.mean(errors)), int(np.mean(rounds))


default_score, default_rounds = cv_score({"learning_rate": 0.05})
print("setelan bawaan: RMSLE CV", round(default_score, 5), "| putaran", default_rounds)

# %% [markdown]
# ## 3. `objective(trial)`: ruang pencarian LightGBM
# Rentang di bawah = rentang umum yang dipakai di Kaggle. `log=True` untuk parameter yang
# bekerja di skala perkalian (learning rate, regularisasi).

# %%
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True),
    }
    score, rounds = cv_score(params)
    trial.set_user_attr("rounds", rounds)
    return score


print("ruang pencarian siap: 8 parameter")

# %% [markdown]
# ## 4. Jalankan study (berhenti di `N_TRIALS` atau `TIMEOUT_SECONDS`, mana duluan)
# Setelan bawaan dimasukkan sebagai percobaan pertama (`enqueue_trial`) → hasil tuning
# minimal sama dengan bawaan. Lomba: jalankan sel ini sambil mengerjakan fitur lain.

# %%
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.enqueue_trial({"learning_rate": 0.05, "num_leaves": 31, "max_depth": 12,
                     "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                     "reg_alpha": 1e-8, "reg_lambda": 1e-8})
study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=False)
print(len(study.trials), "percobaan | terbaik RMSLE CV:", round(study.best_value, 5),
      "| bawaan:", round(default_score, 5))

# %% [markdown]
# ## 5. Hasil: parameter terbaik, parameter paling berpengaruh, simpan riwayat
# `best_params` → tempel ke `LGB_PARAMS` (notebook `tabular/klasifikasi|regresi/gbdt`).
# Latih final di semua train dengan `n_estimators` = putaran terbaik × 1,1.

# %%
import os

import optuna

best_params = dict(study.best_params)
best_params["subsample_freq"] = 1
best_rounds = study.best_trial.user_attrs["rounds"]
print("best_params =", best_params)
print("n_estimators final =", int(best_rounds * 1.1))
importance = optuna.importance.get_param_importances(study)
for name, value in importance.items():
    print(f"  {name:18s} {value:.3f}")
os.makedirs("outputs", exist_ok=True)
study.trials_dataframe().to_csv("outputs/optuna_trials.csv", index=False)

# %% [markdown]
# Cek ke kunci jawaban (hanya demo): bawaan vs hasil tuning di test.

# %%
import lightgbm as lgb
import numpy as np

truth = np.log1p(test_labels["price"].to_numpy(dtype=float))
settings = {"bawaan": ({"learning_rate": 0.05}, default_rounds),
            "tuning": (best_params, best_rounds)}
for name, (params, rounds) in settings.items():
    model = lgb.LGBMRegressor(n_estimators=int(rounds * 1.1), verbose=-1, random_state=42,
                              **params)
    model.fit(train[FEATURES], y)
    error = np.sqrt(np.mean((model.predict(test[FEATURES]) - truth) ** 2))
    print(f"{name:7s} RMSLE test: {error:.5f}")

# %% [markdown]
# ## 6. (Opsional) Ruang pencarian XGBoost dan CatBoost
# Ganti isi `params` di `objective` + model di `cv_score` dengan salah satu di bawah.
#
# ```python
# # XGBoost (XGBRegressor / XGBClassifier, early_stopping_rounds=100)
# "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
# "max_depth": trial.suggest_int("max_depth", 3, 10),
# "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
# "subsample": trial.suggest_float("subsample", 0.5, 1.0),
# "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
# "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
# "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
#
# # CatBoost (CatBoostRegressor / CatBoostClassifier, early_stopping_rounds=100)
# "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
# "depth": trial.suggest_int("depth", 4, 10),
# "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
# "random_strength": trial.suggest_float("random_strength", 1e-3, 10, log=True),
# "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
# ```
