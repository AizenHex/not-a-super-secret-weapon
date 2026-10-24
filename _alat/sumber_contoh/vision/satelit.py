# %% [markdown]
# # Vision — Citra Satelit Multispektral: Klasifikasi Tutupan Lahan
# **Kapan dipakai:** tema pangan / ekonomi hijau: sawah vs tanah kosong, deforestasi,
# kekeringan, pemetaan kebun. Citra satelit punya **band** di luar RGB (inframerah dekat/NIR).
# Tanaman sehat memantulkan NIR kuat dan menyerap merah, maka indeks seperti NDVI informatif.
#
# **Langkah:** fitur per band (rata-rata, tengah, sebaran) + indeks vegetasi/air → model tabular.
#
# **Data demo:** Statlog Landsat (satimage). Satu baris = jendela 3×3 piksel × 4 band = 36 kolom,
# urut piksel kiri-atas → kanan-bawah, tiap piksel 4 band (hijau, merah, NIR1, NIR2). Target =
# tutupan lahan piksel tengah. Data demo sudah distandarkan (rata-rata 0), jadi indeks ditulis
# sebagai **selisih** band. Data asli 0–255: NDVI = (NIR − merah)/(NIR + merah).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/vision/satelit"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/satimage_train.csv"
TEST_CSV = f"{DATA_FOLDER}/satimage_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/satimage_test_labels.csv"   # hanya ada di data latihan

TARGET = "kelas"
BANDS = ["hijau", "merah", "nir1", "nir2"]
N_PIXELS = 9          # jendela 3×3
CENTER_PIXEL = 4      # piksel ke-5 (mulai 0) = tengah
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
PIXEL_COLS = list(test.columns.drop("id"))   # 36 kolom: piksel 1 band 1..4, piksel 2, dst
print("train:", train.shape, "| test:", test.shape)
missing = train.isna().sum()
print("missing:", missing.sum())
train.head()

# %% [markdown]
# ## Fitur band: rata-rata 9 piksel, piksel tengah, sebaran (tekstur), indeks vegetasi

# %%
def add_features(table):
    table = table.copy()
    for band_number in range(len(BANDS)):
        name = BANDS[band_number]
        band_cols = []
        for pixel in range(N_PIXELS):
            band_cols.append(PIXEL_COLS[len(BANDS) * pixel + band_number])
        table[name + "_rata"] = table[band_cols].mean(axis=1)
        table[name + "_tengah"] = table[band_cols[CENTER_PIXEL]]
        table[name + "_std"] = table[band_cols].std(axis=1)
    table["indeks_vegetasi"] = table["nir2_rata"] - table["merah_rata"]
    table["indeks_vegetasi_tengah"] = table["nir2_tengah"] - table["merah_tengah"]
    table["indeks_air"] = table["hijau_rata"] - table["nir2_rata"]
    table["kecerahan"] = table[["hijau_rata", "merah_rata", "nir1_rata", "nir2_rata"]].mean(axis=1)
    return table


train = add_features(train)
test = add_features(test)
NEW_FEATURES = list(train.columns.drop(["id", TARGET] + PIXEL_COLS))
FEATURES = PIXEL_COLS + NEW_FEATURES
print(NEW_FEATURES)

# %% [markdown]
# ## EDA: tiap kelas punya "sidik spektral" berbeda

# %%
print(train[TARGET].value_counts())
profile = train.groupby(TARGET)[["hijau_rata", "merah_rata", "nir1_rata", "nir2_rata"]].mean()
profile.T.plot(marker="o", figsize=(7, 4))
plt.title("Rata-rata nilai band per kelas (sidik spektral)")
plt.show()
train.boxplot(column="indeks_vegetasi", by=TARGET, rot=45, figsize=(8, 4))
plt.show()

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[FEATURES], train[TARGET], test_size=0.25, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = make_pipeline(StandardScaler(), SVC(C=10, random_state=SEED))
# model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))

model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = model.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Apakah fitur indeks membantu? (CV 5-fold di data latih)

# %%
raw_scores = cross_val_score(model, train[PIXEL_COLS], train[TARGET], cv=5, scoring="f1_macro")
all_scores = cross_val_score(model, train[FEATURES], train[TARGET], cv=5, scoring="f1_macro")
print("36 piksel mentah F1 macro CV:", round(raw_scores.mean(), 4))
print("piksel + indeks  F1 macro CV:", round(all_scores.mean(), 4))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(train[FEATURES], train[TARGET])
predictions = model.predict(test[FEATURES])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["kelas"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["kelas"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
