# %% [markdown]
# # Tabular — Survei Kotor, Multikelas, F1 macro (template utama final)
# **Kapan dipakai:** pola soal final GEMASTIK 2024 (penggolongan UKT): kuesioner kotor, kolom
# teks bebas, 8 kelas berurutan, kelas langka, metrik F1 macro.
#
# **Langkah:**
# 1. Profil kolom → cari angka yang tertulis sebagai teks dan ejaan kategori yang berbeda.
# 2. Bersihkan: "Rp 1.500.000" → angka, "S-1"/"SARJANA" → "S1", teks bebas → n-gram.
# 3. Bobot kelas seimbang → 5-fold OOF → penskalaan probabilitas per kelas → submission.
#
# OOF dan proba test disimpan (`oof_<Model>.npy`, `test_<Model>.npy`) untuk notebook
# `tabular/blending`.
#
# **Data demo:** `ukt_SINTETIS` (dibuat oleh `_alat/buat_data_survei.py`).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

# %% [markdown]
# ## Konfigurasi
# Isi `CATEGORY_MAP` setelah melihat `value_counts` di sel profil
# (kunci = teks setelah strip + huruf besar).

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/survei_kotor"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_train.csv"
TEST_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_test.csv"

TARGET = "kelas_ukt"
ID_COL = "id"
MONEY_COLS = ["penghasilan_ayah", "penghasilan_ibu"]         # "Rp 1.500.000", "1,5 jt"
NUMBER_TEXT_COLS = ["jumlah_tanggungan", "daya_listrik"]     # "3 orang", "900 VA"
NUM_COLS = ["jumlah_kendaraan"]
CAT_COLS = ["pendidikan_ayah", "status_rumah", "kota", "program_studi", "jalur_masuk",
            "riwayat_beasiswa"]
TEXT_COL = "pekerjaan_ayah"                                  # teks bebas -> feature n-gram
CATEGORY_MAP = {"S-1": "S1", "S 1": "S1", "SARJANA": "S1", "S-2": "S2", "MAGISTER": "S2",
                "S-3": "S3", "S- 3": "S3", "DOKTOR": "S3", "D-3": "D3", "DIPLOMA": "D3",
                "SLTA": "SMA", "SMA/SMK": "SMA", "SMK": "SMA", "SLTP": "SMP",
                "TIDAK TAMAT SD": "SD", "KONTRAK": "SEWA", "NUMPANG ORANG TUA": "MENUMPANG",
                "KIP": "KIP-K", "KIP KULIAH": "KIP-K", "TIDAK ADA": "TIDAK",
                "BELUM PERNAH": "TIDAK", "-": "TIDAK"}
N_FOLD = 5
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.sample(5, random_state=SEED)

# %% [markdown]
# ## EDA 1 — profil kolom
# Cari: angka yang terbaca sebagai teks (`object`), kategori dengan ejaan berbeda, missing.

# %%
profile = pd.DataFrame()
profile["tipe"] = train.dtypes
profile["unik"] = train.nunique()
profile["missing_%"] = (train.isna().mean() * 100).round(1)
profile["contoh"] = train.iloc[0]
print(profile)

for col in CAT_COLS + [TEXT_COL]:
    print(train[col].value_counts().head(12), "\n")

# %% [markdown]
# ## Ubah teks uang menjadi angka
# "Rp 1.500.000" / "1,5 jt" / "2.0 juta" → 1500000.0. Selain angka → NaN.

# %%
def to_number(text):
    text = str(text).lower()
    text = text.replace("rp", "")
    text = text.replace(",-", "").strip()
    is_million = "jt" in text or "juta" in text
    text = text.replace("juta", "")
    text = text.replace("jt", "").strip()
    if is_million:
        return pd.to_numeric(text.replace(",", "."), errors="coerce") * 1_000_000
    return pd.to_numeric(text.replace(".", "").replace(",", ""), errors="coerce")


# coba pada beberapa contoh
for example in ["Rp 1.500.000", "1,5 jt", "2.0 juta", "Rp1.500.000,-", "-"]:
    print(example, "->", to_number(example))

# %% [markdown]
# ## Bersihkan tabel
# - Kategori: strip + huruf besar + spasi ganda dirapikan + `CATEGORY_MAP`.
# - Uang: `to_number`. Angka-teks ("3 orang"): ambil digitnya saja.
# - Dua feature turunan (contoh; sesuaikan dengan data). Outlier tidak dibuang: model pohon tahan.

# %%
def clean_table(table):
    table = table.copy()
    for col in CAT_COLS:
        values = table[col].astype(str).str.strip()
        values = values.str.upper()
        values = values.str.replace(r"\s+", " ", regex=True)
        table[col] = values.replace(CATEGORY_MAP)
    for col in MONEY_COLS:
        table[col] = table[col].map(to_number)
    for col in NUMBER_TEXT_COLS:
        digits = table[col].astype(str).str.replace(r"\D", "", regex=True)
        table[col] = pd.to_numeric(digits, errors="coerce")
    table[TEXT_COL] = table[TEXT_COL].fillna("").str.lower()
    table["penghasilan_total"] = table[MONEY_COLS].fillna(0).sum(axis=1)
    people = table["jumlah_tanggungan"].fillna(1)
    table["penghasilan_per_orang"] = table["penghasilan_total"] / people.clip(lower=1)
    return table


train = clean_table(train)
test = clean_table(test)
for col in CAT_COLS:
    print(col, train[col].nunique(), "nilai:", sorted(train[col].unique())[:10])

# %% [markdown]
# ## EDA 2 — keputusan
# Distribusi target (kelas langka?), duplikat, dan kolom yang hampir menentukan target.

# %%
target_share = train[TARGET].value_counts(normalize=True)
print(target_share.sort_index().round(3))
without_id = train.drop(columns=ID_COL)
print("duplikat (tanpa id):", without_id.duplicated().sum())
print(train.groupby("riwayat_beasiswa")[TARGET].agg(["mean", "count"]))
print(train.groupby(TARGET)["penghasilan_total"].median())

# %% [markdown]
# ## Siapkan X, y, fold
# `CLASSES` = urutan kolom hasil `predict_proba`.

# %%
FEATURE_NUM = MONEY_COLS + NUMBER_TEXT_COLS + NUM_COLS
FEATURE_NUM = FEATURE_NUM + ["penghasilan_total", "penghasilan_per_orang"]
X = train[FEATURE_NUM + CAT_COLS + [TEXT_COL]]
y = train[TARGET]
CLASSES = np.sort(y.unique())
folds = StratifiedKFold(N_FOLD, shuffle=True, random_state=SEED)

# %% [markdown]
# ## Model
# Semua pilihan memakai bobot kelas seimbang, seperti tim juara 2024.
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
text_encoder = CountVectorizer(ngram_range=(1, 3), min_df=5, max_features=300, binary=True)
cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
preprocess = ColumnTransformer([
    ("num", "passthrough", FEATURE_NUM),
    ("cat", cat_encoder, CAT_COLS),
    ("teks", text_encoder, TEXT_COL),
], sparse_threshold=0)

model = LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced",
                       random_state=SEED, verbose=-1)
# model = CatBoostClassifier(iterations=800, auto_class_weights="Balanced", random_seed=SEED,
#                            verbose=0, allow_writing_files=False)
# model = RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=3,
#                                random_state=SEED, n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=3,
#                              random_state=SEED, n_jobs=-1)

pipe = make_pipeline(preprocess, model)
MODEL_NAME = type(model).__name__   # dipakai untuk nama file oof_/test_
print(MODEL_NAME)

# %% [markdown]
# ## Evaluasi — 5-fold OOF
# Tiap baris train diprediksi oleh model yang tidak melihatnya. Skor ini lebih bisa dipercaya
# daripada public leaderboard.

# %%
oof = cross_val_predict(pipe, X, y, cv=folds, method="predict_proba")
oof_predictions = CLASSES[oof.argmax(axis=1)]
oof_score = f1_score(y, oof_predictions, average="macro")
print(MODEL_NAME, "F1 macro OOF:", round(oof_score, 4))
print(classification_report(y, oof_predictions))
ConfusionMatrixDisplay.from_predictions(y, oof_predictions, normalize="true",
                                        values_format=".2f")
plt.show()
np.save(f"oof_{MODEL_NAME}.npy", oof)

# %% [markdown]
# ## Penskalaan probabilitas per kelas (opsional, biasanya +0.005–0.02 F1 macro)
# Probabilitas tiap kelas dikali satu pengali. Pengali dicari satu kelas per langkah: coba
# 0.5–3.0, simpan yang F1 macro OOF-nya tertinggi. Hanya 8 angka dari 30 ribu baris, jadi
# risiko overfit kecil.

# %%
def find_class_weights(proba, y_true):
    weights = np.ones(proba.shape[1])
    for round_number in range(2):
        for class_index in range(len(weights)):
            best_score = -1
            for value in np.arange(0.5, 3.01, 0.1):
                trial = weights.copy()
                trial[class_index] = value
                guess = CLASSES[(proba * trial).argmax(axis=1)]
                score = f1_score(y_true, guess, average="macro")
                if score > best_score:
                    best_score = score
                    weights[class_index] = value
    return weights


class_weights = find_class_weights(oof, y)
scaled_predictions = CLASSES[(oof * class_weights).argmax(axis=1)]
print("pengali per kelas:", class_weights.round(2))
print("F1 macro OOF sebelum:", round(oof_score, 4),
      "| sesudah:", round(f1_score(y, scaled_predictions, average="macro"), 4))

# %% [markdown]
# ## Latih ulang di semua data + feature terpenting

# %%
pipe.fit(X, y)
feature_names = pipe[0].get_feature_names_out()
importance = pd.Series(pipe[-1].feature_importances_, index=feature_names)
importance = importance.sort_values()
importance.tail(20).plot.barh(figsize=(7, 6))
plt.title(f"20 feature terpenting ({MODEL_NAME})")
plt.show()

test_proba = pipe.predict_proba(test[X.columns])
np.save(f"test_{MODEL_NAME}.npy", test_proba)

# %% [markdown]
# ## Submission

# %%
predictions = CLASSES[(test_proba * class_weights).argmax(axis=1)]

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["kelas_ukt"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
