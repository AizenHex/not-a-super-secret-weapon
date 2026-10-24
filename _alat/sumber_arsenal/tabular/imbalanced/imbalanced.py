# %% [markdown]
# # Kelas tidak seimbang: bobot kelas, ambang, SMOTE (yang benar), metrik yang tepat
#
# **Butuh:** `train` (fitur + target 0/1, kelas positif langka), `test`.
# **Menghasilkan:** tabel perbandingan cara, ambang terbaik, `submission.csv`.
#
# Akurasi menyesatkan: data 98% negatif → tebak "negatif" semua = akurasi 98%. Pakai
# **PR-AUC** (average precision), F1 kelas positif, atau metrik yang diminta soal.
#
# Temuan penting dari literatur (mis. van den Goorbergh dkk. 2022; Elor & Averbuch-Elor 2022
# "To SMOTE, or not to SMOTE?"): untuk model kuat (GBDT), **menggeser ambang** sering setara
# atau lebih baik daripada SMOTE; resampling merusak kalibrasi proba. Jadi urutan coba:
# ambang → bobot kelas → baru resampling. SMOTE **wajib di dalam fold** (di luar = bocor).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | sebaran kelas + `scores` (PR-AUC, ROC-AUC, F1 @ambang) | **ya** |
# | 2 | `oof_proba(model)`: proba out-of-fold | **ya** |
# | 3 | tanpa penanganan vs bobot kelas | **ya** |
# | 4 | `best_threshold`: ambang F1 terbaik dari OOF | **ya** (metrik F1/recall) |
# | 5 | SMOTE / undersampling di dalam CV (imblearn Pipeline) | coba |
# | 6 | BUKTI: SMOTE di luar CV memberi skor palsu | baca |
# | 7 | bandingkan → submission dengan ambang terpilih | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Mammografi: 6 fitur angka, `label` 1 = kalsifikasi berbahaya (langka).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/imbalanced/mammography_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/imbalanced/mammography_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/imbalanced/mammography_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Sebaran kelas + `scores(name, truth, proba, threshold)`

# %%
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

ID = "id"         # UBAH
TARGET = "label"  # UBAH
FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6"]   # UBAH
y = train[TARGET].to_numpy()
print("positif:", int(y.sum()), "dari", len(y), "→", round(y.mean() * 100, 2), "%")


def scores(name, truth, proba, threshold=0.5):
    guess = (proba >= threshold).astype(int)
    return {"cara": name, "pr_auc": round(average_precision_score(truth, proba), 4),
            "roc_auc": round(roc_auc_score(truth, proba), 4),
            "f1": round(f1_score(truth, guess), 4), "ambang": round(threshold, 3),
            "ditebak_positif": int(guess.sum())}


print("acuan PR-AUC model acak = porsi positif =", round(y.mean(), 4))

# %% [markdown]
# ## 2. `oof_proba(model)`: proba out-of-fold, StratifiedKFold (porsi positif sama tiap fold)

# %%
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def oof_proba(model, X=None, target=None):
    if X is None:
        X, target = train[FEATURES], y
    proba = cross_val_predict(model, X, target, cv=folds, method="predict_proba")
    return proba[:, 1]


results, probas = [], {}

# %% [markdown]
# ## 3. Tanpa penanganan vs bobot kelas
# - LightGBM: `scale_pos_weight` = negatif / positif (atau akarnya, lebih lembut).
# - sklearn (LogReg, RF): `class_weight="balanced"`.

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

BASE = {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "verbose": -1,
        "random_state": 42}   # UBAH
ratio = (y == 0).sum() / (y == 1).sum()
weights = {"tanpa_penanganan": 1.0, "bobot_akar": float(np.sqrt(ratio)), "bobot_penuh": ratio}
for name, weight in weights.items():
    probas[name] = oof_proba(LGBMClassifier(**BASE, scale_pos_weight=weight))
    results.append(scores(name, y, probas[name]))
print(pd.DataFrame(results))

# %% [markdown]
# ## 4. `best_threshold(truth, proba)`: ambang dengan F1 tertinggi (dicari di OOF)
# Ambang 0,5 hampir tidak pernah terbaik untuk data langka. Ambang dicari di OOF lalu
# dipakai apa adanya di test (jangan dicari di test!).

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def best_threshold(truth, proba):
    precision, recall, thresholds = precision_recall_curve(truth, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


thresholds = {}
for name in list(weights):
    thresholds[name] = best_threshold(y, probas[name])
    results.append(scores(name + "+ambang", y, probas[name], thresholds[name]))
print(pd.DataFrame(results).to_string(index=False))

# %% [markdown]
# ## 5. Resampling DI DALAM CV: `imblearn.pipeline.Pipeline` (resample hanya saat fit)
# - SMOTE: buat contoh positif sintetis (interpolasi tetangga).
# - Undersampling: buang sebagian negatif (cepat, cocok data besar).
# Pipeline imblearn memastikan fold val TIDAK ikut di-resample.

# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier

SAMPLERS = {"smote": SMOTE(sampling_strategy=0.3, random_state=42),            # UBAH
            "undersample": RandomUnderSampler(sampling_strategy=0.3, random_state=42)}
for name, sampler in SAMPLERS.items():
    model = Pipeline([("sampler", sampler), ("model", LGBMClassifier(**BASE))])
    probas[name] = oof_proba(model)
    thresholds[name] = best_threshold(y, probas[name])
    results.append(scores(name + "+ambang", y, probas[name], thresholds[name]))
print(pd.DataFrame(results).to_string(index=False))

# %% [markdown]
# ## 6. BUKTI: SMOTE SEBELUM CV = skor palsu
# Titik sintetis dibuat dari tetangga yang nanti masuk fold val → model "melihat" val.

# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

X_wrong, y_wrong = SMOTE(sampling_strategy=0.3, random_state=42).fit_resample(train[FEATURES], y)
wrong_proba = oof_proba(LGBMClassifier(**BASE), X_wrong, y_wrong)
wrong = scores("SMOTE_di_luar_CV (SALAH)", y_wrong, wrong_proba,
               best_threshold(y_wrong, wrong_proba))
print(pd.DataFrame([wrong, results[-2]]).to_string(index=False))
print("→ skor jauh lebih tinggi tapi PALSU: tidak akan terulang di test.")

# %% [markdown]
# ## 7. Bandingkan → latih cara terbaik di semua train → submission dengan ambang OOF

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

table = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print(table.to_string(index=False))
CHOSEN = "bobot_akar"   # UBAH: nama cara (tanpa "+ambang")
model = LGBMClassifier(**BASE, scale_pos_weight=weights[CHOSEN])   # UBAH bila pilih sampler
model.fit(train[FEATURES], y)
test_proba = model.predict_proba(test[FEATURES])[:, 1]
submission = pd.DataFrame()
submission["id"] = test[ID]                                                  # UBAH
submission["label"] = (test_proba >= thresholds[CHOSEN]).astype(int)        # UBAH
submission.to_csv("submission.csv", index=False)
print("ambang:", round(thresholds[CHOSEN], 3), "| ditebak positif:", submission["label"].sum())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo). Positif di test hanya ±50 baris → F1 per ambang naik
# turun beberapa poin karena kebetulan; PR-AUC lebih stabil. Jangan menyetel ambang di test.

# %%
import pandas as pd

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
truth = test_labels["label"].to_numpy()
checks = [scores("ambang 0,5", truth, test_proba),
          scores("ambang OOF", truth, test_proba, thresholds[CHOSEN])]
print(pd.DataFrame(checks).to_string(index=False))
