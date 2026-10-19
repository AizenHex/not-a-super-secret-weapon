# %% [markdown]
# # Tabular — Semi-supervised (label sedikit) dengan Pseudo-label
# **Kapan dipakai:** hanya sebagian baris train yang berlabel, sisanya kosong. Atau data latih
# kecil tapi data test besar (pola final "data latih ±50%").
#
# **Langkah pseudo-label:**
# 1. Latih model di baris berlabel.
# 2. Tebak baris tanpa label.
# 3. Di TIAP kelas, ambil tebakan paling yakin, jadikan label tambahan.
# 4. Latih ulang. Ulangi beberapa putaran.
#
# Ambang global (misalnya probabilitas ≥ 0.9) justru merusak: hampir semua pseudo-label jatuh
# ke kelas besar. Karena itu diambil persentase per kelas.
#
# **Data demo:** Covertype, 90% label **sengaja dihapus** (sel Baca data). Saat lomba hapus
# baris simulasi itu.

# %%
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/multiclass"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/covertype_train.csv"
TEST_CSV = f"{DATA_FOLDER}/covertype_test.csv"

TARGET = "Cover_Type"
ID_COL = "id"
PERCENT_PER_CLASS = 0.30   # ambil 30% tebakan paling yakin di TIAP kelas
N_ROUNDS = 3               # jumlah putaran pseudo-label
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
FEATURES = test.columns.drop(ID_COL).tolist()

# HAPUS 3 baris ini saat lomba: simulasi 90% label hilang
random_generator = np.random.default_rng(SEED)
hidden = random_generator.random(len(train)) < 0.90
train.loc[hidden, TARGET] = np.nan

print("train:", train.shape, "| test:", test.shape)
print("berlabel:", train[TARGET].notna().sum(), "| tanpa label:", train[TARGET].isna().sum())

# %% [markdown]
# ## Split
# - `labeled` = baris berlabel. Validasi diambil dari sini.
# - `pool` = semua baris tanpa label (train kosong + test). Dari sinilah pseudo-label diambil.

# %%
labeled = train[train[TARGET].notna()]
unlabeled = train[train[TARGET].isna()]
pool = pd.concat([unlabeled[FEATURES], test[FEATURES]], ignore_index=True)

X_labeled, X_val, y_labeled, y_val = train_test_split(
    labeled[FEATURES], labeled[TARGET], test_size=0.25, stratify=labeled[TARGET],
    random_state=SEED
)
print("latih:", len(X_labeled), "| validasi:", len(X_val), "| pool tanpa label:", len(pool))

# %% [markdown]
# ## Model dasar
# `class_weight="balanced"`: kelas langka diberi bobot lebih besar (data ini sangat timpang).
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = LGBMClassifier(n_estimators=300, learning_rate=0.05, class_weight="balanced",
                       random_state=SEED, verbose=-1)
# model = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)

# %% [markdown]
# ## Pseudo-label + evaluasi
# Fungsi `pseudo_label` menjalankan langkah 1–4 di atas, lalu langsung dicoba:
# skor validasi tanpa pseudo-label dibanding dengan pseudo-label.

# %%
def pseudo_label(model, X_labeled, y_labeled, X_pool):
    X_now = X_labeled
    y_now = y_labeled
    for round_number in range(N_ROUNDS):
        model.fit(X_now, y_now)
        proba = model.predict_proba(X_pool)
        guess = pd.Series(model.classes_[proba.argmax(axis=1)], index=X_pool.index)
        confidence = pd.Series(proba.max(axis=1), index=X_pool.index)
        chosen = pd.Series(False, index=X_pool.index)
        for label in guess.unique():
            in_class = guess == label
            cutoff = confidence[in_class].quantile(1 - PERCENT_PER_CLASS)
            chosen = chosen | (in_class & (confidence >= cutoff))
        # chosen = confidence >= 0.90   # ambang global: kelas besar mendominasi
        X_now = pd.concat([X_labeled, X_pool[chosen]])
        y_now = pd.concat([y_labeled, guess[chosen]])
        print("putaran", round_number + 1, ": tambah", chosen.sum(), "pseudo-label")
    model.fit(X_now, y_now)
    return model


model.fit(X_labeled, y_labeled)
base_predictions = model.predict(X_val)
print("F1 macro hanya berlabel:", round(f1_score(y_val, base_predictions, average="macro"), 4))

model = pseudo_label(model, X_labeled, y_labeled, pool)
val_predictions = model.predict(X_val)
print("F1 macro + pseudo-label:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Latih ulang (semua baris berlabel + pseudo-label) + submission

# %%
model = pseudo_label(model, labeled[FEATURES], labeled[TARGET], pool)
predictions = model.predict(test[FEATURES])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["Cover_Type"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: `covertype_test_labels.csv` hanya ada di data latihan.

# %%
test_key = pd.read_csv(f"{DATA_FOLDER}/covertype_test_labels.csv")["Cover_Type"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
