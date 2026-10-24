# %% [markdown]
# # Ensembling: rata-rata, hill climbing, stacking, rata-rata seed, pseudo-label
#
# **Butuh:** proba OOF + proba test beberapa model dengan fold yang SAMA (dari notebook
# `tabular/klasifikasi|regresi/gbdt`: `outputs/oof_<model>.npy`, `outputs/test_<model>.npy`),
# atau dibuat di sel 2. **Menghasilkan:** `submission.csv` dari gabungan terbaik.
#
# Urutan teknik dari playbook Kaggle Grandmaster (NVIDIA 2025):
#
# | teknik | ide | sel |
# |---|---|---|
# | model beragam | linear, pohon, KNN: kesalahannya berbeda → saling menutup | 2–3 |
# | rata-rata | rata-rata proba; hampir selalu ≥ model terbaik | 4 |
# | hill climbing | mulai model terbaik, tambah model yang paling membantu OOF | 5 |
# | stacking | proba OOF jadi fitur model level 2 | 6 |
# | rata-rata seed | model sama, seed berbeda → varians turun | 7 |
# | pseudo-label | tebakan test yang sangat yakin dipakai sebagai data latih tambahan | 8 |
#
# Bobot/ensemble HANYA dipilih dari skor OOF (bukan test). Hill climbing & stacking bisa
# overfit OOF → bandingkan dengan rata-rata sederhana; selisih tipis = pilih yang sederhana.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `cross_validate` + `score` | **ya** |
# | 2 | latih model beragam → `oofs`, `test_probas` (atau muat `.npy`) | **ya** |
# | 3 | skor + korelasi antarmodel | **ya** |
# | 4 | rata-rata | **ya** |
# | 5 | `hill_climb` | disarankan |
# | 6 | stacking | coba |
# | 7 | rata-rata seed | kalau waktu cukup |
# | 8 | pseudo-label | coba, nilai di CV |
# | 9 | pilih → `submission.csv` | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Covertype: jenis tutupan hutan dari elevasi, jarak, jenis tanah (7 kelas).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/multiclass/covertype_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/multiclass/covertype_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/multiclass/covertype_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + `cross_validate` (sama dengan notebook gbdt) + `score` (logloss)
# Metrik untuk memilih gabungan: logloss (memakai proba penuh, lebih halus dari akurasi).

# %%
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

ID = "id"               # UBAH
TARGET = "Cover_Type"   # UBAH
FEATURES = list(train.columns.drop([ID, TARGET]))
CLASSES = sorted(train[TARGET].unique())
y = train[TARGET].map(dict(zip(CLASSES, range(len(CLASSES))))).to_numpy()
folds = np.zeros(len(train), dtype=int)
for fold, (_, index_val) in enumerate(StratifiedKFold(5, shuffle=True,
                                                      random_state=42).split(train, y)):
    folds[index_val] = fold


def score(proba, truth=None):
    if truth is None:
        truth = y
    proba = np.clip(proba, 1e-7, 1)
    proba = proba / proba.sum(axis=1, keepdims=True)
    loss = round(log_loss(truth, proba, labels=range(len(CLASSES))), 4)
    return loss, round(accuracy_score(truth, proba.argmax(axis=1)), 4)


print(len(FEATURES), "fitur |", len(CLASSES), "kelas | skor tebak rata:",
      score(np.full((len(y), len(CLASSES)), 1 / len(CLASSES))))

# %% [markdown]
# `cross_validate(fit_function, X, X_test)` → proba OOF + rata-rata proba test semua fold.

# %%
import numpy as np


def cross_validate(fit_function, X, X_test, target=None):
    if target is None:
        target = y
    oof = np.zeros((len(X), len(CLASSES)))
    test_proba = np.zeros((len(X_test), len(CLASSES)))
    for fold in range(folds.max() + 1):
        is_val = folds == fold
        model = fit_function(X[~is_val], target[~is_val])
        oof[is_val] = model.predict_proba(X[is_val])
        test_proba += model.predict_proba(X_test) / (folds.max() + 1)
    return oof, test_proba


oofs, test_probas = {}, {}
print("siap")

# %% [markdown]
# ## 2. Latih model beragam → `oofs[nama]`, `test_probas[nama]`
# Sudah punya file dari notebook lain? Ganti sel ini dengan:
# `oofs[name] = np.load(f"outputs/oof_{name}.npy")` dan `test_probas[name] = np.load(...)`.

# %%
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logreg(X, target):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=3000)).fit(X, target)


def fit_knn(X, target):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(15, weights="distance")).fit(
        X, target)


def fit_extra_trees(X, target):
    return ExtraTreesClassifier(500, min_samples_leaf=1, n_jobs=-1, random_state=42).fit(
        X, target)


print("model sederhana: logreg, knn, extra_trees")

# %% [markdown]
# Tiga model GBDT. Tanpa early stopping di sini → jumlah pohon dibuat sedang.

# %%
import lightgbm as lgb
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def fit_lightgbm(X, target, seed=42):
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                               colsample_bytree=0.5, verbose=-1, random_state=seed)
    return model.fit(X, target)


def fit_xgboost(X, target):
    return XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=7,
                         colsample_bytree=0.5, random_state=42).fit(X, target)


def fit_catboost(X, target):
    return CatBoostClassifier(iterations=800, learning_rate=0.1, depth=6, verbose=0,
                              random_seed=42).fit(X, target)


print("GBDT: lightgbm, xgboost, catboost")

# %% [markdown]
# Latih semua (± 1–2 menit). Proba OOF tiap model disimpan untuk dipakai sel berikut.

# %%
import os

import numpy as np

FIT_FUNCTIONS = {"logreg": fit_logreg, "knn": fit_knn, "extra_trees": fit_extra_trees,
                 "lightgbm": fit_lightgbm, "xgboost": fit_xgboost,
                 "catboost": fit_catboost}   # UBAH
os.makedirs("outputs", exist_ok=True)
for name, fit_function in FIT_FUNCTIONS.items():
    oofs[name], test_probas[name] = cross_validate(fit_function, train[FEATURES],
                                                   test[FEATURES])
    np.save(f"outputs/oof_{name}.npy", oofs[name])
    np.save(f"outputs/test_{name}.npy", test_probas[name])
    print(f"{name:12s} logloss, akurasi OOF: {score(oofs[name])}")

# %% [markdown]
# ## 3. Korelasi antarmodel (proba kelas benar): makin RENDAH makin berguna digabung
# Model kuat tapi berkorelasi 0,99 dengan model lain hampir tidak menambah apa-apa.

# %%
import numpy as np
import pandas as pd

true_class_proba = {}
for name in oofs:
    true_class_proba[name] = oofs[name][np.arange(len(y)), y]
correlation = pd.DataFrame(true_class_proba).corr()
print(correlation.round(3))


# %% [markdown]
# ## 4. Rata-rata sederhana (semua model, dan hanya model kuat)

# %%
import numpy as np

STRONG = ["extra_trees", "lightgbm", "xgboost", "catboost"]   # UBAH
average_groups = {"rata_semua": list(FIT_FUNCTIONS), "rata_kuat": STRONG}
for name, members in average_groups.items():
    oofs[name] = np.zeros_like(oofs[members[0]])
    test_probas[name] = np.zeros_like(test_probas[members[0]])
    for member in members:
        oofs[name] += oofs[member] / len(members)
        test_probas[name] += test_probas[member] / len(members)
    print(f"{name:10s} logloss, akurasi OOF: {score(oofs[name])}")

# %% [markdown]
# ## 5. `hill_climb(candidates)`: ensemble selection (Caruana dkk. 2004)
# Mulai dari model terbaik. Tiap langkah: coba tambah SETIAP model (boleh model yang sama
# lagi = bobot naik), ambil yang paling menurunkan logloss OOF. Berhenti kalau tidak membaik.
# Bobot akhir = berapa kali model terpilih / jumlah langkah.

# %%
import numpy as np
import pandas as pd


def hill_climb(candidates, max_steps=30):
    losses = {}
    for name in candidates:
        losses[name] = score(oofs[name])[0]
    chosen = [min(losses, key=losses.get)]
    total, best = oofs[chosen[0]].copy(), min(losses.values())
    for _ in range(max_steps):
        trials = {}
        for name in candidates:
            trials[name] = score((total + oofs[name]) / (len(chosen) + 1))[0]
        pick = min(trials, key=trials.get)
        if trials[pick] >= best:
            break
        chosen.append(pick)
        total, best = total + oofs[pick], trials[pick]
    return pd.Series(chosen).value_counts(normalize=True)


hill_weights = hill_climb(list(FIT_FUNCTIONS))
print("bobot hill climbing:\n", hill_weights.round(3))

# %% [markdown]
# Terapkan bobot ke OOF dan test.

# %%
import numpy as np

oofs["hill_climb"] = np.zeros_like(oofs["lightgbm"])
test_probas["hill_climb"] = np.zeros_like(test_probas["lightgbm"])
for name, weight in hill_weights.items():
    oofs["hill_climb"] += oofs[name] * weight
    test_probas["hill_climb"] += test_probas[name] * weight
print("hill_climb logloss, akurasi OOF:", score(oofs["hill_climb"]))

# %% [markdown]
# ## 6. Stacking: proba OOF semua model → fitur → LogisticRegression (level 2)
# Model level 2 dinilai dengan CV KEDUA di atas OOF (fold berbeda). Fitur = log proba
# (lebih linear). Level 2 harus SEDERHANA (LogReg / Ridge / LightGBM dangkal).

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

BASE_MODELS = list(FIT_FUNCTIONS)   # UBAH
train_columns, test_columns = [], []
for name in BASE_MODELS:
    train_columns.append(np.log(np.clip(oofs[name], 1e-6, 1)))
    test_columns.append(np.log(np.clip(test_probas[name], 1e-6, 1)))
stack_train, stack_test = np.hstack(train_columns), np.hstack(test_columns)
level_two = LogisticRegression(C=0.1, max_iter=3000)
second_folds = StratifiedKFold(5, shuffle=True, random_state=7)
oofs["stacking"] = cross_val_predict(level_two, stack_train, y, cv=second_folds,
                                     method="predict_proba")
test_probas["stacking"] = level_two.fit(stack_train, y).predict_proba(stack_test)
print("stacking logloss, akurasi OOF:", score(oofs["stacking"]))

# %% [markdown]
# ## 7. Rata-rata seed: model sama, `random_state` berbeda → prediksi lebih stabil
# Paling berguna untuk model yang acak (colsample/subsample < 1, NN). Biaya: N× waktu latih.

# %%
import numpy as np

SEEDS = [0, 1, 2]   # UBAH


def fit_lightgbm_seeded(X, target):
    return fit_lightgbm(X, target, seed=SEED)   # SEED diambil dari loop di bawah


seed_oofs, seed_tests = [], []
for SEED in SEEDS:
    oof, test_proba = cross_validate(fit_lightgbm_seeded, train[FEATURES], test[FEATURES])
    seed_oofs.append(oof)
    seed_tests.append(test_proba)
    print(f"seed {SEED}: {score(oof)}")
oofs["lightgbm_3seed"] = np.mean(seed_oofs, axis=0)
test_probas["lightgbm_3seed"] = np.mean(seed_tests, axis=0)
print("rata-rata seed:", score(oofs["lightgbm_3seed"]))

# %% [markdown]
# ## 8. Pseudo-label: tebakan test yang sangat yakin → data latih tambahan
# Supaya tidak bocor, pseudo-label dibuat ULANG di tiap fold dari model yang hanya melihat
# bagian latih fold itu (playbook NVIDIA: "k set pseudo-label untuk k fold").
# Paling berguna kalau test jauh lebih besar dari train atau distribusinya sedikit bergeser.

# %%
import numpy as np

CONFIDENCE = 0.95   # UBAH


def cross_validate_pseudo(fit_function, X, X_test):
    oof = np.zeros((len(X), len(CLASSES)))
    test_proba = np.zeros((len(X_test), len(CLASSES)))
    for fold in range(folds.max() + 1):
        is_val = folds == fold
        first = fit_function(X[~is_val], y[~is_val]).predict_proba(X_test)
        sure = first.max(axis=1) >= CONFIDENCE
        X_more = np.vstack([X[~is_val].to_numpy(), X_test[sure].to_numpy()])
        y_more = np.r_[y[~is_val], first[sure].argmax(axis=1)]
        model = fit_function(X_more, y_more)
        oof[is_val] = model.predict_proba(X[is_val].to_numpy())
        test_proba += model.predict_proba(X_test.to_numpy()) / (folds.max() + 1)
    return oof, test_proba


oofs["lightgbm_pseudo"], test_probas["lightgbm_pseudo"] = cross_validate_pseudo(
    fit_lightgbm, train[FEATURES], test[FEATURES])
print("lightgbm biasa :", score(oofs["lightgbm"]))
print("lightgbm pseudo:", score(oofs["lightgbm_pseudo"]))
print("→ pakai hanya kalau skor OOF membaik (sering TIDAK membaik bila test kecil)")

# %% [markdown]
# ## 9. Pilih gabungan dengan logloss OOF terbaik → `submission.csv`
# Selisih tipis (< 0,005) antara hill climbing/stacking dan rata-rata → pilih rata-rata (lebih
# tahan overfit). Opsional (playbook): latih ulang model terpilih di 100% train.

# %%
import numpy as np
import pandas as pd

rows = []
for name in oofs:
    loss, accuracy = score(oofs[name])
    rows.append({"cara": name, "logloss_oof": loss, "akurasi_oof": accuracy})
board = pd.DataFrame(rows).sort_values("logloss_oof")
print(board.to_string(index=False))
CHOSEN = board["cara"].iloc[0]   # UBAH: atau tulis nama cara sendiri
submission = pd.DataFrame()
submission["id"] = test[ID]                                                       # UBAH
submission["Cover_Type"] = np.array(CLASSES)[test_probas[CHOSEN].argmax(axis=1)]  # UBAH
submission.to_csv("submission.csv", index=False)
print("dipilih:", CHOSEN)

# %% [markdown]
# Cek ke kunci jawaban (hanya demo): urutan skor test mirip urutan skor OOF?

# %%
import pandas as pd

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
truth = test_labels["Cover_Type"].map(dict(zip(CLASSES, range(len(CLASSES))))).to_numpy()
rows = []
for name in test_probas:
    loss, accuracy = score(test_probas[name], truth)
    rows.append({"cara": name, "logloss_test": loss, "akurasi_test": accuracy})
checks = pd.DataFrame(rows).sort_values("logloss_test")
print(checks.to_string(index=False))
