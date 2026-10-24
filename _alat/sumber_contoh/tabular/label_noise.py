# %% [markdown]
# # Tabular — Label Berisik (confident learning)
# **Kapan dipakai:** data hasil input manual/survei, dan sebagian label kemungkinan salah.
#
# **Idenya:**
# 1. Tiap baris ditebak oleh model yang TIDAK melihat baris itu (probabilitas OOF, 5-fold).
# 2. Baris yang labelnya diberi probabilitas rendah, padahal model yakin kelas lain,
#    ditandai "curiga".
# 3. Buang baris curiga, latih ulang.
#
# **Data demo:** Covertype dengan 15% label **sengaja diacak** (`covertype_noisy_SINTETIS`).
# File `_kunci.csv` berisi label asli, hanya untuk mengukur ketepatan deteksi.

# %%
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra/label_noise"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/covertype_noisy_SINTETIS_train.csv"
KEY_CSV = f"{DATA_FOLDER}/covertype_noisy_SINTETIS_kunci.csv"     # hanya ada di data latihan
TEST_FOLDER = "../../../data_latihan/tabular_extra/multiclass"
TEST_CSV = f"{TEST_FOLDER}/covertype_test.csv"

TARGET = "Cover_Type"
ID_COL = "id"
SEED = 42

# %% [markdown]
# ## Baca data
# `LabelEncoder` mengubah label menjadi nomor 0, 1, 2, ... (dibutuhkan untuk deteksi).

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
FEATURES = test.columns.drop(ID_COL).tolist()

encoder = LabelEncoder()
y = encoder.fit_transform(train[TARGET])
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Probabilitas OOF
# `proba` = tabel probabilitas: satu baris per baris train, satu kolom per kelas.
# `label_confidence` = probabilitas yang diberikan model untuk label yang TERTULIS di data.

# %%
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
proba = cross_val_predict(model, train[FEATURES], y, cv=cv, method="predict_proba")

label_confidence = np.zeros(len(y))
for row in range(len(y)):
    label_confidence[row] = proba[row, y[row]]
print("rata-rata probabilitas label tertulis:", round(label_confidence.mean(), 3))

# %% [markdown]
# ## Deteksi baris curiga
# Cara manual: ambang tiap kelas = rata-rata keyakinan model pada baris berlabel kelas itu.
# Baris curiga = keyakinannya di bawah ambang DAN tebakan model berbeda dari labelnya.
#
# Pilih SATU `suspicious`: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
thresholds = np.zeros(len(encoder.classes_))
for class_number in range(len(encoder.classes_)):
    thresholds[class_number] = label_confidence[y == class_number].mean()

model_guess = proba.argmax(axis=1)
suspicious = (label_confidence < thresholds[y]) & (model_guess != y)
# suspicious = find_label_issues(labels=y, pred_probs=proba)          # pustaka cleanlab
# suspicious = find_label_issues(labels=y, pred_probs=proba, filter_by="prune_by_noise_rate")
print("baris curiga:", suspicious.sum(), "(", round(suspicious.mean() * 100, 1), "% )")

# %% [markdown]
# ## Ketepatan deteksi
# **HAPUS sel ini saat lomba**: label asli (`_kunci.csv`) hanya ada di data latihan.
# Precision = dari yang dicurigai, berapa yang memang salah. Recall = dari yang salah, berapa
# yang tertangkap.

# %%
true_label = pd.read_csv(KEY_CSV)["label_asli"]
really_wrong = (true_label != train[TARGET]).to_numpy()
print("label yang benar-benar salah:", really_wrong.sum())
print("precision deteksi:", round(precision_score(really_wrong, suspicious), 3))
print("recall deteksi   :", round(recall_score(really_wrong, suspicious), 3))

# %% [markdown]
# ## Evaluasi: latih dengan semua baris vs buang baris curiga
# Validasi juga berisik, jadi selisihnya cenderung terlihat lebih kecil dari aslinya.

# %%
all_rows = np.arange(len(y))
train_rows, val_rows = train_test_split(all_rows, test_size=0.2, stratify=y,
                                        random_state=SEED)
clean_rows = train_rows[~suspicious[train_rows]]
X_val = train[FEATURES].iloc[val_rows]

model.fit(train[FEATURES].iloc[train_rows], y[train_rows])
score_all = f1_score(y[val_rows], model.predict(X_val), average="macro")
print("semua baris  F1 macro validasi:", round(score_all, 4))

model.fit(train[FEATURES].iloc[clean_rows], y[clean_rows])
score_clean = f1_score(y[val_rows], model.predict(X_val), average="macro")
print("buang curiga F1 macro validasi:", round(score_clean, 4))

# %% [markdown]
# ## Latih ulang tanpa baris curiga + submission
# Baris curiga juga disimpan ke `baris_curiga.csv` untuk dicek manual.

# %%
model.fit(train[FEATURES][~suspicious], y[~suspicious])
predictions = encoder.inverse_transform(model.predict(test[FEATURES]))
train[suspicious].to_csv("baris_curiga.csv", index=False)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["Cover_Type"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: `covertype_test_labels.csv` hanya ada di data latihan.

# %%
test_key = pd.read_csv(f"{TEST_FOLDER}/covertype_test_labels.csv")["Cover_Type"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
