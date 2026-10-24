# %% [markdown]
# # Metrik: ditulis sendiri (numpy) + cara mengoptimalkan tebakan untuk metrik soal
#
# **Butuh:** kunci + tebakan (proba / angka / kelas). **Menghasilkan:** fungsi metrik yang bisa
# dipakai saat dokumentasi offline, ambang QWK terbaik, pengali kelas untuk F1 macro.
#
# Baca metrik soal DULU, lalu pilih cara latih dan cara mengubah proba → jawaban:
#
# | metrik soal | latih dengan | ubah keluaran |
# |---|---|---|
# | accuracy | logloss | argmax |
# | F1 biner | logloss | ambang dicari di OOF (`tabular/imbalanced`) |
# | F1 macro | logloss | pengali per kelas dicari di OOF (sel 8) |
# | ROC-AUC / PR-AUC / Gini | logloss | kirim PROBA (urutan saja yang dinilai) |
# | logloss | logloss | proba, clip 1e-6 (jangan 0/1) |
# | RMSE / MAE | l2 / l1 | — |
# | RMSLE / MAPE | l2 di `log1p(y)` | `expm1` |
# | QWK (kelas berurutan) | regresi | ambang dicari di OOF (sel 6) |
# | MAP@k / NDCG@k | lambdarank (`tabular/ranking`) | urutkan skor per query |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi |
# |---|---|
# | 1 | klasifikasi: confusion, precision/recall/F1 (biner, macro, micro, weighted) |
# | 2 | ROC-AUC (cara peringkat), PR-AUC, logloss, Gini |
# | 3 | regresi: RMSE, MAE, RMSLE, MAPE, R² |
# | 4 | QWK (quadratic weighted kappa) |
# | 5 | ranking: precision@k, MAP@k, NDCG@k |
# | 6–7 | optimasi ambang QWK → submission |
# | 8 | optimasi pengali kelas untuk F1 macro |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Kualitas anggur merah: skor 3–8 (kelas berurutan → QWK).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ordinal/wine_quality_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ordinal/wine_quality_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/ordinal/wine_quality_test_labels.csv")
print(train.shape, test.shape)
print(train["quality"].value_counts().sort_index())

# %% [markdown]
# ## 1. Klasifikasi: `confusion`, `f1_scores` (biner per kelas, macro, micro, weighted)
# - macro: rata-rata F1 tiap kelas (kelas kecil sama penting) → paling sering di lomba.
# - micro: dihitung dari total TP/FP/FN (= accuracy untuk multikelas satu label).
# - weighted: rata-rata F1 diberi bobot jumlah baris kelas itu.

# %%
import numpy as np
from sklearn.metrics import f1_score


def confusion(truth, guess, classes):
    table = np.zeros((len(classes), len(classes)), dtype=int)
    position = dict(zip(classes, range(len(classes))))
    for real, said in zip(truth, guess):
        table[position[real], position[said]] += 1
    return table   # baris = asli, kolom = tebakan


print(confusion([0, 1, 1], [0, 1, 0], [0, 1]))

# %% [markdown]
# `f1_scores(truth, guess)` → F1 per kelas + macro, micro, weighted.

# %%
import numpy as np
from sklearn.metrics import f1_score


def f1_scores(truth, guess):
    classes = sorted(set(truth) | set(guess))
    table = confusion(truth, guess, classes)
    hit = np.diag(table).astype(float)
    precision = hit / np.clip(table.sum(axis=0), 1, None)
    recall = hit / np.clip(table.sum(axis=1), 1, None)
    per_class = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    support = table.sum(axis=1)
    return {"per_kelas": dict(zip(classes, per_class.round(4))),
            "macro": per_class.mean(), "micro": hit.sum() / table.sum(),
            "weighted": (per_class * support).sum() / support.sum()}


truth = np.array([0, 0, 1, 1, 2, 2, 2])
guess = np.array([0, 1, 1, 1, 2, 0, 2])
mine = f1_scores(truth, guess)
print(confusion(truth, guess, [0, 1, 2]))
print("macro sendiri", round(mine["macro"], 4), "| sklearn",
      round(f1_score(truth, guess, average="macro"), 4))

# %% [markdown]
# ## 2. ROC-AUC lewat peringkat (Mann–Whitney), PR-AUC (average precision), logloss, Gini
# AUC = peluang baris positif acak mendapat skor lebih tinggi dari baris negatif acak.
# Gini (lomba kredit) = 2·AUC − 1.

# %%
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def roc_auc(truth, proba):
    ranks = rankdata(proba)                        # nilai sama → peringkat rata-rata
    positive = truth == 1
    n_positive, n_negative = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


truth = np.array([0, 0, 1, 1, 0, 1, 0, 1])
proba = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7])
print("AUC", round(roc_auc(truth, proba), 4), round(roc_auc_score(truth, proba), 4),
      "| Gini", round(2 * roc_auc(truth, proba) - 1, 4))

# %% [markdown]
# `average_precision` (= PR-AUC versi sklearn) dan `binary_log_loss`.

# %%
import numpy as np
from sklearn.metrics import average_precision_score, log_loss


def average_precision(truth, proba):
    order = np.argsort(-proba, kind="stable")
    hits = truth[order]
    precision_at = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return (precision_at * hits).sum() / hits.sum()


def binary_log_loss(truth, proba):
    proba = np.clip(proba, 1e-15, 1 - 1e-15)
    return -np.mean(truth * np.log(proba) + (1 - truth) * np.log(1 - proba))


print("AP ", round(average_precision(truth, proba), 4),
      round(average_precision_score(truth, proba), 4))
print("logloss", round(binary_log_loss(truth, proba), 4), round(log_loss(truth, proba), 4))

# %% [markdown]
# ## 3. Regresi: RMSE, MAE, RMSLE, MAPE, R²

# %%
import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(truth, guess):
    truth, guess = np.asarray(truth, dtype=float), np.asarray(guess, dtype=float)
    error = guess - truth
    return {"rmse": np.sqrt(np.mean(error ** 2)), "mae": np.mean(np.abs(error)),
            "rmsle": np.sqrt(np.mean((np.log1p(np.clip(guess, 0, None)) - np.log1p(truth)) ** 2)),
            "mape": np.mean(np.abs(error) / np.clip(np.abs(truth), 1e-9, None)) * 100,
            "r2": 1 - (error ** 2).sum() / ((truth - truth.mean()) ** 2).sum()}


truth = np.array([3.0, 5.0, 2.5, 7.0])
guess = np.array([2.5, 5.0, 4.0, 8.0])
for name, value in regression_metrics(truth, guess).items():
    print(f"{name:5s} {value:.4f}")
print("r2 sklearn", round(r2_score(truth, guess), 4))

# %% [markdown]
# ## 4. QWK (quadratic weighted kappa): kesepakatan untuk kelas berurutan
# Salah 1 tingkat dihukum ringan, salah jauh dihukum berat (kuadrat jarak). 1 = sempurna,
# 0 = setara acak. Dipakai untuk nilai esai, tingkat penyakit, rating.

# %%
import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(truth, guess, classes):
    observed = confusion(truth, guess, classes).astype(float)
    size = len(classes)
    distance = np.zeros((size, size))
    for row in range(size):
        for column in range(size):
            distance[row, column] = (row - column) ** 2 / (size - 1) ** 2
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    return 1 - (distance * observed).sum() / (distance * expected).sum()


truth = np.array([3, 4, 5, 5, 6, 7])
guess = np.array([3, 5, 5, 6, 6, 5])
print("QWK sendiri", round(quadratic_weighted_kappa(truth, guess, [3, 4, 5, 6, 7]), 4),
      "| sklearn", round(cohen_kappa_score(truth, guess, weights="quadratic"), 4))

# %% [markdown]
# ## 5. Ranking: precision@k, AP@k → MAP@k, NDCG@k
# `relevant` = daftar item benar; `ranked` = daftar tebakan urut dari skor tertinggi.
# NDCG memakai relevansi bertingkat (0, 1, 2, ...): item sangat relevan di atas = skor tinggi.

# %%
import numpy as np
from sklearn.metrics import ndcg_score


def average_precision_at_k(relevant, ranked, top_k):
    hits, total = 0, 0.0
    for position, item in enumerate(ranked[:top_k]):
        if item in relevant:
            hits += 1
            total += hits / (position + 1)
    return total / max(1, min(len(relevant), top_k))


queries = [(["a", "c"], ["a", "b", "c", "d"]), (["x"], ["y", "z", "x"])]
scores = []
for relevant, ranked in queries:
    scores.append(average_precision_at_k(relevant, ranked, top_k=3))
print("MAP@3:", round(np.mean(scores), 4))

# %% [markdown]
# `ndcg_at_k(relevansi_urut_tebakan, top_k)`: relevansi item dalam urutan yang ditebak.

# %%
import numpy as np
from sklearn.metrics import ndcg_score


def ndcg_at_k(relevance_in_ranked_order, top_k, exponential=True):
    gains = np.asarray(relevance_in_ranked_order, dtype=float)
    if exponential:                                  # 2^rel - 1 (LightGBM, kebanyakan lomba)
        gains = 2 ** gains - 1
    discount = 1 / np.log2(np.arange(2, top_k + 2))
    dcg = (gains[:top_k] * discount[:len(gains[:top_k])]).sum()
    ideal = np.sort(gains)[::-1][:top_k]
    idcg = (ideal * discount[:len(ideal)]).sum()
    if idcg == 0:
        return 0.0
    return dcg / idcg


print("NDCG@3 gain eksponensial:", round(ndcg_at_k([2, 0, 1, 0], 3), 4))
print("NDCG@3 gain linear", round(ndcg_at_k([2, 0, 1, 0], 3, exponential=False), 4),
      "| sklearn (linear)", round(ndcg_score([[2, 0, 1, 0]], [[4, 3, 2, 1]], k=3), 4))

# %% [markdown]
# ## 6. QWK: regresi + ambang yang dioptimalkan (cara pemenang lomba Kaggle Petfinder/PANDA)
# 1) latih REGRESI → tebakan angka OOF. 2) cari ambang antar-kelas yang memaksimalkan QWK di OOF
# (koordinat demi koordinat). Biasanya jauh lebih baik daripada `round()` atau klasifikasi.

# %%
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

TARGET = "quality"   # UBAH
FEATURES = list(train.columns.drop(["id", TARGET]))
LEVELS = sorted(train[TARGET].unique())
y = train[TARGET].to_numpy()
folds = StratifiedKFold(5, shuffle=True, random_state=42)
regressor = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              verbose=-1, random_state=42)
oof_value = cross_val_predict(regressor, train[FEATURES], y, cv=folds)
classifier = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15,
                                verbose=-1, random_state=42)
oof_class = cross_val_predict(classifier, train[FEATURES], y, cv=folds)
rounded = np.clip(np.round(oof_value), LEVELS[0], LEVELS[-1])
print("QWK klasifikasi:", round(quadratic_weighted_kappa(y, oof_class, LEVELS), 4))
print("QWK regresi + round:", round(quadratic_weighted_kappa(y, rounded, LEVELS), 4))

# %% [markdown]
# `apply_cuts` + `optimize_cuts`: geser tiap ambang dalam rentang kecil, simpan yang terbaik.

# %%
import numpy as np


def apply_cuts(values, cuts):
    return np.array(LEVELS)[np.searchsorted(cuts, values)]


def optimize_cuts(values, truth, rounds=3, step=0.02):
    cuts = np.array(LEVELS[:-1], dtype=float) + 0.5          # mulai dari round()
    best = quadratic_weighted_kappa(truth, apply_cuts(values, cuts), LEVELS)
    for _ in range(rounds):
        for i in range(len(cuts)):
            for candidate in np.arange(cuts[i] - 0.5, cuts[i] + 0.5, step):
                trial = cuts.copy()
                trial[i] = candidate
                trial = np.sort(trial)
                kappa = quadratic_weighted_kappa(truth, apply_cuts(values, trial), LEVELS)
                if kappa > best:
                    best, cuts = kappa, trial
    return cuts, best


cuts, best = optimize_cuts(oof_value, y)
print("ambang:", cuts.round(3), "| QWK OOF:", round(best, 4))

# %% [markdown]
# ## 7. Terapkan ambang OOF ke test → `submission.csv`

# %%
import lightgbm as lgb
import numpy as np
import pandas as pd

regressor.fit(train[FEATURES], y)
test_value = regressor.predict(test[FEATURES])
submission = pd.DataFrame()
submission["id"] = test["id"]                              # UBAH
submission["quality"] = apply_cuts(test_value, cuts)      # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["quality"].value_counts().sort_index())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
import numpy as np

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
answer = test_labels["quality"].to_numpy()
plain = np.clip(np.round(test_value), LEVELS[0], LEVELS[-1])
print("QWK test round()   :", round(quadratic_weighted_kappa(answer, plain, LEVELS), 4))
print("QWK test ambang OOF:", round(quadratic_weighted_kappa(answer, submission["quality"],
                                                             LEVELS), 4))

# %% [markdown]
# ## 8. F1 macro: kalikan proba tiap kelas dengan pengali, cari pengali terbaik di OOF
# argmax proba cenderung jarang memilih kelas kecil → F1 macro rendah. Pengali > 1 untuk
# kelas kecil menaikkan recall-nya. Dicari koordinat demi koordinat seperti ambang QWK.

# %%
import numpy as np
from sklearn.metrics import f1_score


def optimize_class_weights(proba, truth, rounds=2):
    weights = np.ones(proba.shape[1])
    best = f1_score(truth, proba.argmax(axis=1), average="macro")
    for _ in range(rounds):
        for i in range(len(weights)):
            for candidate in [0.5, 0.75, 1, 1.5, 2, 3, 5, 8]:
                trial = weights.copy()
                trial[i] = candidate
                score = f1_score(truth, (proba * trial).argmax(axis=1), average="macro")
                if score > best:
                    best, weights = score, trial
    return weights, best


class_index = np.searchsorted(LEVELS, y)
oof_proba = cross_val_predict(classifier, train[FEATURES], class_index, cv=folds,
                              method="predict_proba")
print("F1 macro argmax biasa:", round(f1_score(class_index, oof_proba.argmax(axis=1),
                                               average="macro"), 4))
class_weights, best = optimize_class_weights(oof_proba, class_index)
print("pengali:", class_weights, "| F1 macro OOF:", round(best, 4))
