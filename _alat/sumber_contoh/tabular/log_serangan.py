# %% [markdown]
# # Tabular — Log Jaringan / Deteksi Serangan (IDS), multikelas dengan kelas langka
# **Kapan dipakai:** dari log koneksi jaringan, tebak tipe serangan (pola soal final
# GEMASTIK 2023).
#
# **Metrik:** F1 macro (semua kelas sama penting, termasuk kelas yang sangat langka).
#
# **Data demo:** KDD Cup 99 (10%), target `kategori`: normal, dos, probe, r2l, u2r
# (u2r sangat langka).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra/ids"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/kdd_train.csv"
TEST_CSV = f"{DATA_FOLDER}/kdd_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/kdd_test_labels.csv"   # hanya ada di data latihan

TARGET = "kategori"
ID_COL = "id"
CAT_COLS = ["protocol_type", "service", "flag"]
BYTE_COLS = ["src_bytes", "dst_bytes", "count"]       # kolom untuk dilihat di EDA
SEED = 42

# %% [markdown]
# ## Baca data
# `NUM_COLS` = semua kolom test kecuali id dan kolom kategori.

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
NUM_COLS = test.columns.drop([ID_COL] + CAT_COLS).tolist()
print("train:", train.shape, "| test:", test.shape, "|", len(NUM_COLS), "kolom angka")
missing_per_column = train.isna().sum()
print("sel kosong:", missing_per_column.sum())
train.head()

# %% [markdown]
# ## EDA
# - Jumlah per kategori serangan.
# - Proporsi serangan per protokol.
# - Median ukuran data per kategori (skala log, karena angkanya sangat lebar).
# - Baris kembar: koneksi dengan feature sama persis (umum di log jaringan).

# %%
print(train[TARGET].value_counts())
print(pd.crosstab(train["protocol_type"], train[TARGET], normalize="index").round(3))

median_bytes = train.groupby(TARGET)[BYTE_COLS].median()
print(np.log1p(median_bytes).round(2))
print("baris kembar:", train.duplicated(subset=NUM_COLS + CAT_COLS).sum())

# %% [markdown]
# ## Feature domain
# Tambah/hapus baris di fungsi ini untuk mencoba feature lain.
# - log ukuran data, rasio data kirim/terima;
# - total tingkat error;
# - `tanpa_balasan` = 1 kalau server tidak mengirim data balik.

# %%
def add_features(table):
    table = table.copy()
    table["log_src_bytes"] = np.log1p(table["src_bytes"])
    table["log_dst_bytes"] = np.log1p(table["dst_bytes"])
    table["rasio_bytes"] = table["src_bytes"] / (table["dst_bytes"] + 1)
    table["error_total"] = table["serror_rate"] + table["rerror_rate"]
    table["tanpa_balasan"] = (table["dst_bytes"] == 0).astype(int)
    return table


train = add_features(train)
test = add_features(test)
NUM_COLS = NUM_COLS + ["log_src_bytes", "log_dst_bytes", "rasio_bytes", "error_total",
                       "tanpa_balasan"]
train[NUM_COLS[-5:]].describe().round(2)

# %% [markdown]
# ## Split train / validasi

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED
)

# %% [markdown]
# ## Model
# `class_weight="balanced"` = kelas langka diberi bobot lebih besar.
# Kolom kategori diubah dengan OneHot; kategori yang muncul < 10 kali digabung jadi satu.
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
preprocess = ColumnTransformer([("cat", encoder, CAT_COLS)], remainder="passthrough")

model = LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced",
                       random_state=SEED, verbose=-1)
# model = LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1)
# model = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
# model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced",
#                                                            max_iter=3000))

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# Confusion matrix dinormalisasi per baris: angka = proporsi tiap label asli.

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions, digits=3))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, normalize="true",
                                        values_format=".2f")
plt.show()

# %% [markdown]
# ## Feature paling berpengaruh (permutation importance, bahan presentasi)

# %%
result = permutation_importance(pipe, X_val, y_val, scoring="f1_macro", n_repeats=3,
                                random_state=SEED, n_jobs=-1)
importance = pd.Series(result.importances_mean, index=X_val.columns)
importance = importance.sort_values()
importance.tail(15).plot.barh(figsize=(6, 5))
plt.title("Turunnya F1 macro bila kolom diacak")
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[NUM_COLS + CAT_COLS])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["kategori"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["kategori"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
