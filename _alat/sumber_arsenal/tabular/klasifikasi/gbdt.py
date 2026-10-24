# %% [markdown]
# # Klasifikasi tabel: baseline → LightGBM / XGBoost / CatBoost out-of-fold → submission
#
# **Butuh:** `train` (fitur + kolom target kelas), `test` (fitur + id). **Menghasilkan:**
# `submission.csv`, `outputs/oof_<model>.npy` + `outputs/test_<model>.npy` (proba, dipakai
# `tabular/ensembling`), tabel skor CV per model, fitur terpenting.
#
# Kenapa GBDT: studi Grinsztajn dkk. (NeurIPS 2022, 45 dataset) → model pohon (XGBoost, random
# forest) masih mengalahkan deep learning pada data tabel ukuran sedang (~10 ribu baris).
# Playbook Kaggle Grandmaster: buat **baseline beragam** dulu, simpan prediksi OOF semuanya.
#
# | model | kelebihan | kategori |
# |---|---|---|
# | LogisticRegression | baseline cepat, beda "sudut pandang" (bagus untuk ensemble) | one-hot |
# | LightGBM | paling cepat, biasanya terkuat | kolom `category` langsung |
# | XGBoost | kuat, banyak dipakai | `enable_categorical=True` |
# | CatBoost | paling kuat untuk banyak kategori, setelan bawaan sudah bagus | `cat_features` |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi, kelas → nomor, kategori → `category`, fold | **ya** |
# | 2 | `cross_validate(fit_function, ...)` → OOF + test proba | **ya** |
# | 3 | baseline: kelas terbanyak + LogisticRegression | **ya** |
# | 4–6 | LightGBM, XGBoost, CatBoost (early stopping per fold) | min. 1 |
# | 7 | bandingkan + rata-rata proba | **ya** |
# | 8 | fitur terpenting | disarankan |
# | 9 | simpan OOF/test proba + `submission.csv` | **ya** |
# | 10 | label berisik: cari baris yang labelnya kemungkinan salah | kalau curiga |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# KDD: koneksi jaringan → `normal` / jenis serangan (`dos`, `probe`, `r2l`, `u2r`).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ids/kdd_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ids/kdd_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ids/kdd_test_labels.csv")
print(train.shape, test.shape)
print(train["kategori"].value_counts())

# %% [markdown]
# ## 1. Konfigurasi: kelas → nomor 0..K-1, kolom kategori → tipe `category`, fold stratified
# Kategori train dan test diberi daftar nilai yang SAMA (gabungan), supaya kodenya cocok.

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ID = "id"                 # UBAH
TARGET = "kategori"       # UBAH
CATEGORICAL = ["protocol_type", "service", "flag"]   # UBAH
N_SPLITS = 5              # UBAH

CLASSES = sorted(train[TARGET].unique())
y = train[TARGET].map(dict(zip(CLASSES, range(len(CLASSES))))).to_numpy()
for column in CATEGORICAL:
    values = sorted(set(train[column].astype(str)) | set(test[column].astype(str)))
    train[column] = pd.Categorical(train[column].astype(str), categories=values)
    test[column] = pd.Categorical(test[column].astype(str), categories=values)
FEATURES = []
for column in train.columns:
    if column not in [ID, TARGET]:
        FEATURES.append(column)
folds = np.zeros(len(train), dtype=int)
splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for fold, (_, index_val) in enumerate(splitter.split(train, y)):
    folds[index_val] = fold
print(len(FEATURES), "fitur |", dict(zip(CLASSES, np.bincount(y).tolist())))

# %% [markdown]
# ## 2. `cross_validate(fit_function, ...)` → `oof` (proba train) + `test_proba` (rata-rata fold)
# `fit_function(X_fit, y_fit, X_val, y_val)` mengembalikan model yang sudah dilatih
# (val dipakai untuk early stopping). Prediksi test = rata-rata model dari semua fold.

# %%
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def cross_validate(fit_function, X, y, X_test, folds):
    oof = np.zeros((len(X), len(CLASSES)))
    test_proba = np.zeros((len(X_test), len(CLASSES)))
    for fold in range(folds.max() + 1):
        is_val = folds == fold
        model = fit_function(X[~is_val], y[~is_val], X[is_val], y[is_val])
        oof[is_val] = model.predict_proba(X[is_val])
        test_proba += model.predict_proba(X_test) / (folds.max() + 1)
    return oof, test_proba


def report(name, oof):
    guess = oof.argmax(axis=1)
    oof = np.clip(oof, 1e-7, 1)
    oof = oof / oof.sum(axis=1, keepdims=True)
    return {"model": name, "f1_macro": round(f1_score(y, guess, average="macro"), 4),
            "accuracy": round(accuracy_score(y, guess), 4),
            "logloss": round(log_loss(y, oof), 4)}


results, oofs, test_probas = [], {}, {}

# %% [markdown]
# ## 3. Baseline: tebak kelas terbanyak + LogisticRegression (one-hot + standarisasi)
# Model rumit yang tidak jauh mengalahkan baseline = ada yang salah (fitur/target/kebocoran).

# %%
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

majority = np.zeros((len(train), len(CLASSES)))
majority[:, np.bincount(y).argmax()] = 1
results.append(report("kelas_terbanyak", majority))
NUMERIC = []
for column in FEATURES:
    if column not in CATEGORICAL:
        NUMERIC.append(column)


def fit_logistic(X_fit, y_fit, X_val, y_val):
    columns = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
                                      (StandardScaler(), NUMERIC))
    model = make_pipeline(columns, LogisticRegression(C=1.0, max_iter=2000))
    return model.fit(X_fit, y_fit)


oofs["logreg"], test_probas["logreg"] = cross_validate(fit_logistic, train[FEATURES], y,
                                                       test[FEATURES], folds)
results.append(report("logreg", oofs["logreg"]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 4. LightGBM + early stopping (berhenti kalau logloss val tidak membaik 100 putaran)
# Setelan awal yang aman: learning_rate 0,05, num_leaves 31, subsample/colsample 0,8.
# Data kecil (< 2000 baris): num_leaves 15, min_child_samples 30 → tidak overfit.

# %%
import lightgbm as lgb
import pandas as pd

LGB_PARAMS = {"n_estimators": 3000, "learning_rate": 0.05, "num_leaves": 31,   # UBAH
              "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "min_child_samples": 20, "verbose": -1, "random_state": 42}
best_rounds = []


def fit_lightgbm(X_fit, y_fit, X_val, y_val):
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_fit, y_fit, eval_X=X_val, eval_y=y_val,
              callbacks=[lgb.early_stopping(100, verbose=False)])
    best_rounds.append(model.best_iteration_)
    return model


oofs["lightgbm"], test_probas["lightgbm"] = cross_validate(fit_lightgbm, train[FEATURES], y,
                                                           test[FEATURES], folds)
results.append(report("lightgbm", oofs["lightgbm"]))
print("putaran terbaik per fold:", best_rounds)
print(pd.DataFrame(results))

# %% [markdown]
# ## 5. XGBoost (`enable_categorical`, `tree_method="hist"`; GPU: `device="cuda"`)

# %%
import pandas as pd
from xgboost import XGBClassifier

XGB_PARAMS = {"n_estimators": 3000, "learning_rate": 0.05, "max_depth": 6,   # UBAH
              "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
              "enable_categorical": True, "early_stopping_rounds": 100,
              "eval_metric": "mlogloss", "random_state": 42}


def fit_xgboost(X_fit, y_fit, X_val, y_val):
    model = XGBClassifier(**XGB_PARAMS)
    return model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)


oofs["xgboost"], test_probas["xgboost"] = cross_validate(fit_xgboost, train[FEATURES], y,
                                                         test[FEATURES], folds)
results.append(report("xgboost", oofs["xgboost"]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 6. CatBoost (`cat_features` = nama kolom kategori; GPU: `task_type="GPU"`)
# Paling lambat dari ketiganya; turunkan `iterations` kalau waktu sempit.

# %%
import pandas as pd
from catboost import CatBoostClassifier

CAT_PARAMS = {"iterations": 2000, "learning_rate": 0.08, "depth": 6,   # UBAH
              "loss_function": "MultiClass", "early_stopping_rounds": 100,
              "random_seed": 42, "verbose": 0}


def fit_catboost(X_fit, y_fit, X_val, y_val):
    model = CatBoostClassifier(**CAT_PARAMS, cat_features=CATEGORICAL)
    return model.fit(X_fit, y_fit, eval_set=(X_val, y_val))


oofs["catboost"], test_probas["catboost"] = cross_validate(fit_catboost, train[FEATURES], y,
                                                           test[FEATURES], folds)
results.append(report("catboost", oofs["catboost"]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 7. Bandingkan + rata-rata proba model terbaik
# Rata-rata sederhana beberapa model GBDT hampir selalu ≥ model terbaik sendirian.
# Bobot/stacking lebih lanjut: `tabular/ensembling`.

# %%
import numpy as np
import pandas as pd

BLEND = ["lightgbm", "xgboost", "catboost"]   # UBAH
blend_oof = np.zeros_like(oofs[BLEND[0]])
blend_test = np.zeros_like(test_probas[BLEND[0]])
for name in BLEND:
    blend_oof += oofs[name] / len(BLEND)
    blend_test += test_probas[name] / len(BLEND)
oofs["rata_rata"], test_probas["rata_rata"] = blend_oof, blend_test
results.append(report("rata_rata", blend_oof))
table = pd.DataFrame(results)
print(table.sort_values("f1_macro", ascending=False).to_string(index=False))

# %% [markdown]
# ## 8. Fitur terpenting (gain LightGBM, dilatih di semua train)
# Fitur dengan gain 0 → kandidat dibuang. Penjelasan lebih jujur (SHAP, permutation):
# `tabular/penjelasan`.

# %%
import lightgbm as lgb
import numpy as np
import pandas as pd

final_params = dict(LGB_PARAMS)
final_params["n_estimators"] = int(np.mean(best_rounds) * 1.1)   # +10%: data lebih banyak
final_params["importance_type"] = "gain"
final_model = lgb.LGBMClassifier(**final_params).fit(train[FEATURES], y)
importance = pd.Series(final_model.feature_importances_, index=FEATURES)
importance = (importance / importance.sum()).sort_values(ascending=False)
print(importance.head(10).round(4))
print("fitur dengan gain 0:", list(importance[importance == 0].index))

# %% [markdown]
# ## 9. Simpan proba OOF/test (untuk ensembling) + `submission.csv`
# `CHOSEN` = model dengan skor CV terbaik (tabel sel 7). Kelas dikembalikan ke nama asli.

# %%
import os

import numpy as np
import pandas as pd

CHOSEN = "rata_rata"   # UBAH
os.makedirs("outputs", exist_ok=True)
for name in oofs:
    np.save(f"outputs/oof_{name}.npy", oofs[name])
    np.save(f"outputs/test_{name}.npy", test_probas[name])
submission = pd.DataFrame()
submission["id"] = test[ID]                                            # UBAH
submission["kategori"] = np.array(CLASSES)[test_probas[CHOSEN].argmax(axis=1)]   # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["kategori"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
for name in test_probas:
    guess = np.array(CLASSES)[test_probas[name].argmax(axis=1)]
    score = f1_score(test_labels["kategori"], guess, average="macro")
    print(f"{name:10s} F1 macro test: {score:.4f}")

# %% [markdown]
# ## 10. Label berisik: baris yang labelnya kemungkinan salah
# Data dikumpulkan manusia (survei, anotasi) sering salah label. Cara "confident learning"
# sederhana: proba OOF untuk label yang tertulis sangat rendah → curiga. Buang (atau periksa)
# lalu latih ulang; nilai di val/test yang BERSIH. Data contoh: covertype, ±15% label diacak.

# %%
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict

noisy = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/label_noise/covertype_noisy_SINTETIS_train.csv")
truth = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/label_noise/covertype_noisy_SINTETIS_kunci.csv")
NOISY_TARGET = "Cover_Type"   # UBAH
SUSPECT_SHARE = 0.10          # UBAH: porsi baris paling mencurigakan
noisy_features = noisy.drop(columns=[ID, NOISY_TARGET])
noisy_classes = sorted(noisy[NOISY_TARGET].unique())
noisy_y = noisy[NOISY_TARGET].map(dict(zip(noisy_classes, range(len(noisy_classes))))).to_numpy()
model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
proba = cross_val_predict(model, noisy_features, noisy_y, cv=5, method="predict_proba")
given_label_proba = proba[np.arange(len(noisy)), noisy_y]
suspect = given_label_proba < np.quantile(given_label_proba, SUSPECT_SHARE)
really_wrong = (noisy[NOISY_TARGET] != truth["label_asli"]).to_numpy()
print("baris curiga:", suspect.sum(), "| yang memang salah label:", (suspect & really_wrong).sum())
print("porsi salah label: semua", round(really_wrong.mean(), 3),
      "→ setelah buang curiga", round(really_wrong[~suspect].mean(), 3))

# %% [markdown]
# Latih ulang tanpa baris curiga → bandingkan di test bersih.

# %%
import pandas as pd
from sklearn.metrics import accuracy_score

clean_test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/multiclass/covertype_test.csv")
clean_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/multiclass/covertype_test_labels.csv")
X_clean_test = clean_test[noisy_features.columns]
keep_rows = {"semua baris": np.ones(len(noisy), dtype=bool), "buang curiga": ~suspect}
for name, keep in keep_rows.items():
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1)
    model.fit(noisy_features[keep], noisy_y[keep])
    guess = np.array(noisy_classes)[model.predict(X_clean_test)]
    print(f"{name:12s}: akurasi test {accuracy_score(clean_labels[NOISY_TARGET], guess):.4f}")
