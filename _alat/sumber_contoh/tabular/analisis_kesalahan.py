# %% [markdown]
# # Tabular — Analisis Kesalahan Prediksi (di mana dan kenapa model salah?)
# **Kapan dipakai:** SETELAH model jadi, sebelum tuning. Hasilnya bahan perbaikan feature dan
# slide presentasi.
#
# **Langkah:**
# 1. Prediksi OOF 5-fold: tiap baris train ditebak oleh model yang tidak melihat baris itu.
# 2. Kelas mana yang sering tertukar?
# 3. Irisan data (kategori, rentang angka) mana yang paling sering salah?
# 4. Baris "yakin tapi salah" (curiga label salah) dan baris ragu (dekat batas kelas).
# 5. Feature apa yang paling berbeda antara baris salah dan benar?
#
# **Data demo:** KDD Cup 99 (kategori serangan jaringan).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra/ids"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/kdd_train.csv"

TARGET = "kategori"
ID_COL = "id"
CAT_COLS = ["protocol_type", "service", "flag"]
MIN_ROWS = 30            # irisan dengan baris lebih sedikit dari ini tidak ditampilkan
CONFIDENT = 0.9          # "yakin" = probabilitas tebakan ≥ angka ini
DOUBT_GAP = 0.1          # "ragu" = selisih dua kelas teratas < angka ini
SEED = 42

# %% [markdown]
# ## Baca data
# `NUM_COLS` = semua kolom selain id, target, dan kolom kategori.

# %%
train = pd.read_csv(TRAIN_CSV)
NUM_COLS = train.columns.drop([ID_COL, TARGET] + CAT_COLS).tolist()
print(train.shape)
train.head()

# %% [markdown]
# ## Model + prediksi OOF 5-fold
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.
# `result` = salinan train + kolom tebakan, keyakinan, dan salah/benar.

# %%
encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
preprocess = ColumnTransformer([("cat", encoder, CAT_COLS)], remainder="passthrough")
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                       random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1)

pipe = make_pipeline(preprocess, model)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
proba = cross_val_predict(pipe, train[CAT_COLS + NUM_COLS], train[TARGET], cv=cv,
                          method="predict_proba")

CLASSES = np.sort(train[TARGET].unique())
result = train.copy()
result["prediction"] = CLASSES[proba.argmax(axis=1)]
result["confidence"] = proba.max(axis=1)
result["wrong"] = result["prediction"] != result[TARGET]
print("jumlah salah:", result["wrong"].sum(), "dari", len(result))

# %% [markdown]
# ## 1. Kelas mana yang tertukar?
# Confusion matrix dinormalisasi per baris: angka = proporsi tiap label asli.

# %%
print(classification_report(result[TARGET], result["prediction"], digits=3))
ConfusionMatrixDisplay.from_predictions(result[TARGET], result["prediction"], normalize="true",
                                        values_format=".2f")
plt.show()

wrong_rows = result[result["wrong"]]
swapped = wrong_rows.groupby([TARGET, "prediction"]).size()
print(swapped.sort_values(ascending=False).head(8))

# %% [markdown]
# ## 2. Irisan data yang paling sering salah
# Kolom kategori dipakai langsung. Kolom angka dipotong menjadi 5 kelompok sama banyak.

# %%
slices = []
for col in CAT_COLS + NUM_COLS:
    values = result[col]
    if col in NUM_COLS:
        values = pd.qcut(values, 5, duplicates="drop").astype(str)
    summary = result.groupby(values)["wrong"].agg(["mean", "size"])
    summary = summary[summary["size"] >= MIN_ROWS]
    for value, row in summary.iterrows():
        slices.append([col, value, row["size"], row["mean"]])

slices = pd.DataFrame(slices, columns=["kolom", "nilai", "jumlah", "persen_salah"])
print("rata-rata salah:", round(result["wrong"].mean(), 4))
worst_slices = slices.sort_values("persen_salah", ascending=False)
worst_slices.head(12).round(4)

# %% [markdown]
# ## 3. Yakin tapi salah, dan ragu
# - Yakin tapi salah: curiga labelnya salah, atau kasus unik.
# - Ragu: selisih probabilitas dua kelas teratas kecil (dekat batas kelas).

# %%
sorted_proba = np.sort(proba, axis=1)
result["top2_gap"] = sorted_proba[:, -1] - sorted_proba[:, -2]
confident_wrong = result[result["wrong"] & (result["confidence"] >= CONFIDENT)]
doubtful = result[result["top2_gap"] < DOUBT_GAP]
print("yakin tapi salah:", len(confident_wrong))
print("ragu:", len(doubtful), "| persen salah di baris ragu:", round(doubtful["wrong"].mean(), 3))
confident_wrong[[TARGET, "prediction", "confidence"] + CAT_COLS].head(10)

# %% [markdown]
# ## 4. Feature yang paling berbeda: baris salah vs benar
# Selisih rata-rata dibagi simpangan baku, supaya kolom dengan satuan berbeda bisa dibanding.

# %%
wrong_mean = result.loc[result["wrong"], NUM_COLS].mean()
right_mean = result.loc[~result["wrong"], NUM_COLS].mean()
difference = (wrong_mean - right_mean) / (result[NUM_COLS].std() + 1e-9)
difference = difference.abs().sort_values()
difference.tail(10).plot.barh()
plt.title("Feature paling beda: baris salah vs benar")
plt.show()

# %% [markdown]
# ## Simpan tabel kesalahan (bahan laporan / cek manual)
# Langkah lanjut umum: tambah feature untuk irisan terburuk, beri bobot kelas yang tertukar,
# periksa ulang label baris "yakin tapi salah", atau atur threshold per kelas.

# %%
error_table = pd.DataFrame()
error_table["id"] = result[ID_COL]
error_table["label"] = result[TARGET]
error_table["prediksi"] = result["prediction"]
error_table["yakin"] = result["confidence"].round(4)
error_table["salah"] = result["wrong"]
error_table.to_csv("analisis_kesalahan.csv", index=False)
slices.to_csv("irisan_kesalahan.csv", index=False)
error_table[error_table["salah"]].head()
