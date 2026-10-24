# %% [markdown]
# # Regresi tabel: baseline → transformasi target → LightGBM / XGBoost / CatBoost OOF
#
# **Butuh:** `train` (fitur + target angka), `test`. **Menghasilkan:** `submission.csv`,
# `outputs/oof_<model>.npy` + `outputs/test_<model>.npy` (untuk `tabular/ensembling`),
# tabel RMSE/MAE/RMSLE per model, plot sisaan (residual).
#
# Kunci regresi: **latih pada skala yang sesuai metrik**.
#
# | metrik soal | latih pada | kembalikan |
# |---|---|---|
# | RMSE | target asli (atau log kalau sangat miring, cek di CV) | — |
# | RMSLE / MAPE / harga, jumlah | `log1p(target)` | `expm1(prediksi)` |
# | MAE | target asli, `objective="l1"` (median) | — |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi, kategori ordinal → angka, fold | **ya** |
# | 2 | `regression_scores` + `cross_validate_regression` | **ya** |
# | 3 | baseline: rata-rata + Ridge | **ya** |
# | 4 | LightGBM target asli vs target log | **ya** |
# | 5–6 | XGBoost, CatBoost | min. 1 |
# | 7 | bandingkan + rata-rata | **ya** |
# | 8 | plot sisaan: di mana model paling meleset? | disarankan |
# | 9 | simpan + `submission.csv` (clip ke rentang wajar) | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Harga berlian dari karat, potongan, warna, kejernihan, ukuran.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular/regression/diamonds_test_labels.csv")
print(train.shape, test.shape)
print(train["price"].describe().round(1))

# %% [markdown]
# ## 1. Konfigurasi: kategori BERURUTAN → angka sesuai urutannya, sisanya → `category`
# Urutan bermakna (buruk → bagus) membantu semua model, terutama linear.

# %%
import numpy as np
from sklearn.model_selection import KFold

ID = "id"          # UBAH
TARGET = "price"   # UBAH
ORDINAL = {"cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],             # UBAH
           "color": ["J", "I", "H", "G", "F", "E", "D"],
           "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]}

for column, order in ORDINAL.items():
    mapping = dict(zip(order, range(len(order))))
    train[column] = train[column].map(mapping)
    test[column] = test[column].map(mapping)
FEATURES = []
for column in train.columns:
    if column not in [ID, TARGET]:
        FEATURES.append(column)
y = train[TARGET].to_numpy(dtype=float)
folds = np.zeros(len(train), dtype=int)
for fold, (_, index_val) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
    folds[index_val] = fold
missing = train[FEATURES].isna()
print(FEATURES, "| kosong setelah map (harus 0):", int(missing.to_numpy().sum()))

# %% [markdown]
# ## 2. `regression_scores` + `cross_validate_regression(fit_function, ..., log_target)`
# `log_target=True`: model dilatih pada `log1p(y)`, prediksi dikembalikan dengan `expm1`.

# %%
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_scores(name, truth, guess):
    guess = np.clip(guess, 0, None)
    return {"model": name, "rmse": round(float(np.sqrt(mean_squared_error(truth, guess))), 2),
            "mae": round(float(mean_absolute_error(truth, guess)), 2),
            "rmsle": round(float(np.sqrt(np.mean((np.log1p(guess) - np.log1p(truth)) ** 2))), 4)}


def cross_validate_regression(fit_function, X, y, X_test, folds, log_target=False):
    fit_y = y
    if log_target:
        fit_y = np.log1p(y)
    oof, test_guess = np.zeros(len(X)), np.zeros(len(X_test))
    for fold in range(folds.max() + 1):
        is_val = folds == fold
        model = fit_function(X[~is_val], fit_y[~is_val], X[is_val], fit_y[is_val])
        oof[is_val] = model.predict(X[is_val])
        test_guess += model.predict(X_test) / (folds.max() + 1)
    if log_target:
        oof, test_guess = np.expm1(oof), np.expm1(test_guess)
    return oof, test_guess


results, oofs, test_guesses = [], {}, {}

# %% [markdown]
# ## 3. Baseline: tebak rata-rata + Ridge (linear)

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

results.append(regression_scores("rata_rata_train", y, np.full(len(y), y.mean())))


def fit_ridge(X_fit, y_fit, X_val, y_val):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_fit, y_fit)


for name, log_target in {"ridge": False, "ridge_log": True}.items():
    oofs[name], test_guesses[name] = cross_validate_regression(
        fit_ridge, train[FEATURES], y, test[FEATURES], folds, log_target)
    results.append(regression_scores(name, y, oofs[name]))
print(pd.DataFrame(results))

# %% [markdown]
# Ridge + log bisa meledak: `expm1` dari tebakan linear yang kebablasan → RMSE besar. Linear
# pada target log butuh fitur log juga (mis. `log(carat)`).

# %% [markdown]
# ## 4. LightGBM: target asli vs target log (pilih yang lebih baik untuk METRIK SOAL)

# %%
import lightgbm as lgb
import pandas as pd

LGB_PARAMS = {"n_estimators": 5000, "learning_rate": 0.03, "num_leaves": 31,   # UBAH
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "min_child_samples": 20, "verbose": -1, "random_state": 42}


def fit_lightgbm(X_fit, y_fit, X_val, y_val):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    return model.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val,
                     callbacks=[lgb.early_stopping(200, verbose=False)])


for name, log_target in {"lightgbm": False, "lightgbm_log": True}.items():
    oofs[name], test_guesses[name] = cross_validate_regression(
        fit_lightgbm, train[FEATURES], y, test[FEATURES], folds, log_target)
    results.append(regression_scores(name, y, oofs[name]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 5. XGBoost (target log)

# %%
import pandas as pd
from xgboost import XGBRegressor

XGB_PARAMS = {"n_estimators": 5000, "learning_rate": 0.03, "max_depth": 6,   # UBAH
              "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
              "early_stopping_rounds": 200, "random_state": 42}


def fit_xgboost(X_fit, y_fit, X_val, y_val):
    model = XGBRegressor(**XGB_PARAMS)
    return model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)


oofs["xgboost_log"], test_guesses["xgboost_log"] = cross_validate_regression(
    fit_xgboost, train[FEATURES], y, test[FEATURES], folds, log_target=True)
results.append(regression_scores("xgboost_log", y, oofs["xgboost_log"]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 6. CatBoost (target log)

# %%
import pandas as pd
from catboost import CatBoostRegressor

CAT_PARAMS = {"iterations": 3000, "learning_rate": 0.05, "depth": 6,   # UBAH
              "loss_function": "RMSE", "early_stopping_rounds": 200,
              "random_seed": 42, "verbose": 0}


def fit_catboost(X_fit, y_fit, X_val, y_val):
    return CatBoostRegressor(**CAT_PARAMS).fit(X_fit, y_fit, eval_set=(X_val, y_val))


oofs["catboost_log"], test_guesses["catboost_log"] = cross_validate_regression(
    fit_catboost, train[FEATURES], y, test[FEATURES], folds, log_target=True)
results.append(regression_scores("catboost_log", y, oofs["catboost_log"]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 7. Bandingkan + rata-rata model terbaik

# %%
import numpy as np
import pandas as pd

BLEND = ["lightgbm_log", "xgboost_log", "catboost_log"]   # UBAH
blend_oof = np.zeros(len(train))
blend_test = np.zeros(len(test))
for name in BLEND:
    blend_oof += oofs[name] / len(BLEND)
    blend_test += test_guesses[name] / len(BLEND)
oofs["rata_rata"], test_guesses["rata_rata"] = blend_oof, blend_test
results.append(regression_scores("rata_rata", y, blend_oof))
table = pd.DataFrame(results)
print(table.sort_values("rmse").to_string(index=False))

# %% [markdown]
# ## 8. Plot sisaan: tebakan vs asli + sisaan per kelompok nilai target
# Sisaan membesar di harga tinggi → wajar untuk harga (pakai log/RMSLE). Pola melengkung
# → ada fitur/transformasi yang kurang.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CHOSEN = "rata_rata"   # UBAH
residual = y - oofs[CHOSEN]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y, oofs[CHOSEN], s=3, alpha=0.4)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], color="red")
axes[0].set_xlabel("asli")
axes[0].set_ylabel("tebakan OOF")
axes[1].scatter(oofs[CHOSEN], residual, s=3, alpha=0.4)
axes[1].axhline(0, color="red")
axes[1].set_xlabel("tebakan OOF")
axes[1].set_ylabel("sisaan (asli - tebakan)")
plt.tight_layout()
plt.show()
bins = pd.qcut(y, 5)
absolute_error = pd.Series(np.abs(residual))
error_per_bin = absolute_error.groupby(bins, observed=True).mean()
print("rata-rata |sisaan| per kelompok harga:\n", error_per_bin.round(1))

# %% [markdown]
# ## 9. Simpan prediksi + `submission.csv`
# Clip ke rentang train (sedikit dilebarkan): mencegah tebakan negatif / ekstrem.

# %%
import os

import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)
for name in oofs:
    np.save(f"outputs/oof_{name}.npy", oofs[name])
    np.save(f"outputs/test_{name}.npy", test_guesses[name])
submission = pd.DataFrame()
submission["id"] = test[ID]                                                  # UBAH
submission["price"] = np.clip(test_guesses[CHOSEN], y.min() * 0.9, y.max() * 1.1)   # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
import pandas as pd

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
checks = []
for name in test_guesses:
    checks.append(regression_scores(name, test_labels["price"].to_numpy(), test_guesses[name]))
checks = pd.DataFrame(checks).sort_values("rmse")
print(checks.to_string(index=False))
