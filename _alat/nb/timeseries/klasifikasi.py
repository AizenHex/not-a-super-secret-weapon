# %% [markdown]
# # Timeseries — Klasifikasi Deret Waktu
# **Kapan dipakai:** tiap baris = satu deret pendek (sinyal gerakan, detak jantung, sensor) dan
# diminta menebak kelasnya. Kolom `t0`, `t1`, ... = nilai di tiap waktu.
#
# **Langkah:** plot rata-rata deret per kelas → split dari data latih → model → evaluasi →
# latih ulang di semua data latih → submission.
#
# **Data demo:** GunPoint (`label`, 150 titik per deret). Dataset lain: ItalyPower =
# `italy_power_train.csv` / `italy_power_test.csv` dengan `SERIES_LENGTH = 24`.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/timeseries/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/gunpoint_train.csv"
TEST_CSV = f"{DATA_FOLDER}/gunpoint_test.csv"

LABEL_COL = "label"
SERIES_LENGTH = 150       # titik per deret: GunPoint 150, ItalyPower 24
FEATURE_COLS = []
for i in range(SERIES_LENGTH):
    FEATURE_COLS.append(f"t{i}")
SEED = 42

# %% [markdown]
# ## Baca data dan EDA

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print(train.shape, test.shape)
train_missing = train.isna().sum()
test_missing = test.isna().sum()
print("missing latih:", train_missing.sum(), "| uji:", test_missing.sum())
print(train[LABEL_COL].value_counts())

for label, group in train.groupby(LABEL_COL):
    plt.plot(group[FEATURE_COLS].mean().to_numpy(), label=f"kelas {label}")
plt.title("Rata-rata deret per kelas (data latih)")
plt.legend()
plt.show()

# %% [markdown]
# ## Split
# Tiap baris = satu deret yang berdiri sendiri (tidak ada urutan waktu antar baris), jadi boleh
# diacak. Data latih GunPoint hanya 50 baris, skor validasi masih naik-turun.

# %%
X = train[FEATURE_COLS].to_numpy()
y = train[LABEL_COL].to_numpy()
X_test = test[FEATURE_COLS].to_numpy()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=SEED)
print("latih:", X_train.shape, "| validasi:", X_val.shape)

# %% [markdown]
# ## Alternatif: model berbasis fitur (tsfresh)
# Tiap deret diringkas jadi ratusan fitur (rata-rata, puncak, autokorelasi, entropi, FFT, ...),
# lalu model tabular biasa. Hapus `#` di sel ini, X_train/X_val/X_test diganti matriks fitur.

# %%
# from tsfresh import extract_features
# from tsfresh.feature_extraction import EfficientFCParameters
# from tsfresh.utilities.dataframe_functions import impute
#
#
# def tsfresh_features(matrix):
#     # matriks (deret x waktu) → tabel panjang: series_id, time, value → fitur per deret
#     long = pd.DataFrame(matrix)
#     long["series_id"] = range(len(long))
#     long = long.melt(id_vars="series_id", var_name="time", value_name="value")
#     features = extract_features(long, column_id="series_id", column_sort="time",
#                                 default_fc_parameters=EfficientFCParameters(),
#                                 n_jobs=0, disable_progressbar=True)
#     return impute(features)
#
#
# X_train = tsfresh_features(X_train)
# X_val = tsfresh_features(X_val)[X_train.columns]
# X_test = tsfresh_features(X_test)[X_train.columns]
# print("jumlah fitur:", X_train.shape[1])

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris RandomForest.
# KNN-DTW = tetangga terdekat dengan jarak yang boleh bergeser di sumbu waktu.
# Rocket (MiniRocket) = ribuan filter konvolusi acak + model linear.

# %%
model = RandomForestClassifier(n_estimators=500, random_state=SEED)
# model = KNeighborsClassifier(n_neighbors=1)
# model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
# model = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="dtw")
# model = RocketClassifier(rocket_transform="minirocket", random_state=SEED)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)
# model = TimeSeriesForestClassifier(n_estimators=200, random_state=SEED)
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)
model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi (data validasi)

# %%
val_predictions = model.predict(X_val)
print("akurasi:", round(accuracy_score(y_val, val_predictions), 4))
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data latih + submission
# File test tidak punya kolom id, jadi `id` = nomor baris test.

# %%
final_model = clone(model)
final_model.fit(X, y)
predictions = final_model.predict(X_test)

submission = pd.DataFrame()
submission["id"] = range(len(test))
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# File test demo masih memuat kolom `label`.

# %%
print("akurasi test:", round(accuracy_score(test[LABEL_COL], predictions), 4))
print("F1 macro test:", round(f1_score(test[LABEL_COL], predictions, average="macro"), 4))
