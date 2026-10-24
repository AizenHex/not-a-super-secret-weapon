# %% [markdown]
# # Tabular — PU Learning (hanya positif yang berlabel)
# **Kapan dipakai:** data hanya punya contoh **positif** yang diketahui (misalnya penipuan yang
# terbukti, pelanggan yang membeli). Sisanya **tidak berlabel**: campuran positif dan negatif.
#
# **Kesalahan umum:** menganggap semua yang tak berlabel = negatif (model "naif"). Akibatnya
# peluang positif terlalu kecil.
#
# **Metode Elkan-Noto:** latih model "berlabel vs tidak", lalu bagi skornya dengan
# `label_frequency` = peluang sebuah positif ikut berlabel. Alternatif: bagging PU.
#
# **Data demo:** Breast Cancer, positif = malignant, hanya 50% positif yang dibuka labelnya.

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/breast_cancer.csv"

TARGET = "diagnosis"
POSITIVE = "malignant"
SHOWN_SHARE = 0.5         # demo: berapa bagian positif yang berlabel
THRESHOLD = 0.5           # tebak positif kalau peluang ≥ angka ini
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
FEATURES = data.columns.drop(TARGET).tolist()
print(data.shape)
print(data[TARGET].value_counts())

# %% [markdown]
# ## Buat situasi PU (HANYA DEMO)
# Kolom `s`: 1 = berlabel positif, 0 = tidak berlabel. Saat lomba, `s` sudah diberikan soal
# (misalnya kolom "terbukti_fraud" yang hanya berisi 1 atau kosong).

# %%
random_generator = np.random.default_rng(SEED)
true_label = (data[TARGET] == POSITIVE).astype(int)
shown = (true_label == 1) & (random_generator.random(len(data)) < SHOWN_SHARE)
data["s"] = 0
data.loc[shown, "s"] = 1
print("berlabel positif:", data["s"].sum(), "| tidak berlabel:", (data["s"] == 0).sum())

train, test = train_test_split(data, test_size=0.3, random_state=SEED, stratify=true_label)
y_test = (test[TARGET] == POSITIVE).astype(int)

# %% [markdown]
# ## Model naif: berlabel vs tidak berlabel
# Positif yang tak berlabel dianggap negatif, sehingga peluang positif terlalu kecil.

# %%
naive_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
naive_model.fit(train[FEATURES], train["s"])
naive_proba = naive_model.predict_proba(test[FEATURES])[:, 1]
print("F1 naif:", round(f1_score(y_test, naive_proba >= THRESHOLD), 4))

# %% [markdown]
# ## Elkan-Noto
# 1. Sisihkan sebagian train sebagai hold-out.
# 2. Latih model g(x) = peluang baris ini berlabel.
# 3. `label_frequency` = rata-rata g(x) pada positif berlabel di hold-out.
# 4. Peluang positif sebenarnya = g(x) / `label_frequency`.
#
# LogisticRegression dipakai karena peluangnya lebih tepat (terkalibrasi) daripada boosting.

# %%
fit_part, holdout = train_test_split(train, test_size=0.2, random_state=SEED,
                                     stratify=train["s"])
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
# model = LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1)   # peluang meleset
model.fit(fit_part[FEATURES], fit_part["s"])

holdout_positive = holdout[holdout["s"] == 1]
label_frequency = model.predict_proba(holdout_positive[FEATURES])[:, 1].mean()
print("label_frequency:", round(label_frequency, 3), "| sebenarnya:", SHOWN_SHARE)

pu_proba = model.predict_proba(test[FEATURES])[:, 1] / label_frequency
pu_proba = np.clip(pu_proba, 0, 1)
print("F1 Elkan-Noto:", round(f1_score(y_test, pu_proba >= THRESHOLD), 4))
print("perkiraan proporsi positif:", round(pu_proba.mean(), 3))
print("proporsi positif sebenarnya:", round(y_test.mean(), 3))

# %% [markdown]
# ## Alternatif: bagging PU
# Ulangi: ambil sampel acak dari yang tak berlabel sebagai "negatif sementara", latih vs
# positif, lalu rata-ratakan skor test. Hapus `#` di semua baris sel ini untuk mencoba.

# %%
# N_BAGS = 30
# positives = train[train["s"] == 1]
# unlabeled = train[train["s"] == 0]
# bag_scores = np.zeros(len(test))
# for bag in range(N_BAGS):
#     negatives = unlabeled.sample(len(positives), random_state=SEED + bag)
#     bag_data = pd.concat([positives, negatives])
#     bag_model = LGBMClassifier(n_estimators=100, random_state=SEED + bag, verbose=-1)
#     bag_model.fit(bag_data[FEATURES], bag_data["s"])
#     bag_scores = bag_scores + bag_model.predict_proba(test[FEATURES])[:, 1] / N_BAGS
# print("F1 bagging PU:", f1_score(y_test, bag_scores >= THRESHOLD))

# %% [markdown]
# ## Ringkasan (cek dengan label asli — HAPUS saat lomba)
# AUC hampir sama (urutan skor sama). Yang diperbaiki Elkan-Noto adalah peluang/threshold.

# %%
print("naif       AUC:", round(roc_auc_score(y_test, naive_proba), 3))
print("Elkan-Noto AUC:", round(roc_auc_score(y_test, pu_proba), 3))

# %% [markdown]
# ## Submission
# Di demo ini test = potongan data; saat lomba ganti `test` dengan file test soal.

# %%
predictions = (pu_proba >= THRESHOLD).astype(int)

submission = pd.DataFrame()
submission["id"] = test.index
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
