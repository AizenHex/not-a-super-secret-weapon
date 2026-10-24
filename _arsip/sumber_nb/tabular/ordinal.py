# %% [markdown]
# # Tabular — Klasifikasi Ordinal (kelas bertingkat: rating 1–5, golongan UKT 0–7)
# **Kapan dipakai:** kelasnya berurutan, jadi salah 1 tingkat lebih ringan daripada salah
# 5 tingkat.
#
# **Cara:**
# 1. Latih model **regresi** (tebakan berupa angka pecahan, misalnya 5.7).
# 2. Cari batas antar kelas dari tebakan OOF agar metrik paling tinggi
#    (misalnya < 5.4 → kelas 5, 5.4–6.6 → kelas 6, ...).
#
# **Metrik:** QWK (quadratic weighted kappa). Mau F1 macro (seperti UKT 2024)? Ganti di
# fungsi `score`.
#
# **Data demo:** Wine Quality merah (`quality` 3–8, kelas ujung langka).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import ConfusionMatrixDisplay, cohen_kappa_score, f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular_extra/ordinal"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/wine_quality_train.csv"
TEST_CSV = f"{DATA_FOLDER}/wine_quality_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/wine_quality_test_labels.csv"   # hanya ada di data latihan

TARGET = "quality"
ID_COL = "id"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
FEATURES = test.columns.drop(ID_COL).tolist()
CLASSES = sorted(train[TARGET].unique())
print("train:", train.shape, "| test:", test.shape, "| kelas:", CLASSES)
train.head()

# %% [markdown]
# ## EDA
# Korelasi Spearman cocok untuk target berurutan.

# %%
class_counts = train[TARGET].value_counts()
class_counts.sort_index().plot.bar()
plt.title("Jumlah per kelas")
plt.show()

correlation = train[FEATURES + [TARGET]].corr(method="spearman")
print(correlation[TARGET].sort_values().round(3))

# %% [markdown]
# ## Split train / validasi

# %%
X = train[FEATURES]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# %% [markdown]
# ## Metrik dan batas antar kelas
# - `score`: pilih SATU `return` (QWK atau F1 macro).
# - `to_class`: angka pecahan → kelas, memakai daftar batas.
# - `find_thresholds`: mulai dari batas x.5, lalu tiap batas digeser satu per satu ke nilai
#   yang skornya terbaik (diulang 3 putaran).

# %%
def score(y_true, predicted_class):
    return cohen_kappa_score(y_true, predicted_class, weights="quadratic")
    # return f1_score(y_true, predicted_class, average="macro")


def to_class(values, thresholds):
    return np.array(CLASSES)[np.digitize(values, thresholds)]


def find_thresholds(values, y_true):
    thresholds = np.array(CLASSES[:-1], dtype=float) + 0.5
    for round_number in range(3):
        for i in range(len(thresholds)):
            candidates = np.linspace(thresholds[i] - 0.5, thresholds[i] + 0.5, 41)
            candidate_scores = []
            for value in candidates:
                trial = thresholds.copy()
                trial[i] = value
                candidate_scores.append(score(y_true, to_class(values, np.sort(trial))))
            thresholds[i] = candidates[np.argmax(candidate_scores)]
    return np.sort(thresholds)


# coba: dengan batas x.5, nilai 5.7 menjadi kelas 6
print("contoh:", to_class([5.7, 3.2, 7.6], np.array(CLASSES[:-1]) + 0.5))

# %% [markdown]
# ## Model regresi
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.
# Batas dicari dari tebakan OOF (5-fold) di data train.

# %%
model = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, random_state=SEED,
                      verbose=-1)
# model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1)
# model = HistGradientBoostingRegressor(random_state=SEED)
# model = CatBoostRegressor(iterations=500, random_seed=SEED, verbose=0)
# model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

oof_values = cross_val_predict(model, X_train, y_train, cv=5)
thresholds = find_thresholds(oof_values, y_train)
model.fit(X_train, y_train)
print("batas antar kelas:", thresholds.round(3))

# %% [markdown]
# ## Evaluasi di validasi
# Baris `#` di bawah = pembulatan biasa (tanpa batas yang dicari), untuk pembanding.
# Data kecil → hasil kedua cara bisa terbalik; pastikan dengan CV sebelum memilih.

# %%
val_values = model.predict(X_val)
val_predictions = to_class(val_values, thresholds)
# val_predictions = np.clip(np.round(val_values), CLASSES[0], CLASSES[-1]).astype(int)

print("skor    :", round(score(y_val, val_predictions), 4))
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print("MAE     :", round(mean_absolute_error(y_val, val_predictions), 4))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
oof_values = cross_val_predict(model, X, y, cv=5)
thresholds = find_thresholds(oof_values, y)
model.fit(X, y)
predictions = to_class(model.predict(test[FEATURES]), thresholds)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["quality"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["quality"]
print("skor test (kunci latihan):", round(score(test_key, predictions), 4))
