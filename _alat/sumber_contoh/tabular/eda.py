# %% [markdown]
# # Tabular — EDA Lengkap Sebelum Modeling (checklist 30 menit pertama)
# **Isi:** profil kolom, sel kosong, duplikat, target, outlier, kategori langka, kategori yang
# hanya ada di test, **kebocoran** (kolom yang terlalu mirip target), dan **drift** (train dan
# test berbeda sebaran).
#
# **Data demo:** Titanic. Sengaja ada kolom bocor (`alive` = salinan target) dan kolom kembar
# (`class` = `pclass`). Ringkasan disimpan ke `eda_ringkasan.csv` (bahan laporan/presentasi).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"
ID_COL = "id"
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_COLS = ["sex", "embarked", "class", "who", "adult_male", "deck", "embark_town", "alive",
            "alone"]
RARE_COUNT = 10           # kategori yang muncul < 10 kali dianggap langka
PSI_BINS = 10
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
FEATURES = NUM_COLS + CAT_COLS
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## 1. Profil kolom
# Tipe, persen sel kosong di train dan test, jumlah nilai unik, satu contoh isi.

# %%
profile = pd.DataFrame()
profile["tipe"] = train.dtypes.astype(str)
profile["missing_train_%"] = (train.isna().mean() * 100).round(1)
profile["missing_test_%"] = (test.isna().mean() * 100).round(1)
profile["n_unik"] = train.nunique()
profile["contoh"] = train.iloc[0]
profile

# %% [markdown]
# ## 2. Duplikat dan target
# "Kembar tapi target beda" = baris dengan feature sama persis tetapi labelnya berbeda
# (tanda label berisik, atau feature kurang lengkap).

# %%
print("baris kembar (semua feature sama):", train.duplicated(subset=FEATURES).sum())

duplicated_rows = train[train.duplicated(subset=FEATURES, keep=False)]
targets_per_group = duplicated_rows.groupby(FEATURES, dropna=False)[TARGET].nunique()
print("kembar tapi target beda:", (targets_per_group > 1).sum())
print("ID kembar:", train[ID_COL].duplicated().sum())
print(train[TARGET].value_counts(normalize=True).round(3))

# %% [markdown]
# ## 3. Kolom angka: sebaran dan outlier (aturan IQR)
# Outlier atas = nilai > Q3 + 1.5 × (Q3 − Q1).

# %%
train[NUM_COLS].hist(figsize=(10, 5), bins=30)
plt.show()

q1 = train[NUM_COLS].quantile(0.25)
q3 = train[NUM_COLS].quantile(0.75)
upper_limit = q3 + 1.5 * (q3 - q1)
print("jumlah outlier atas per kolom:")
print((train[NUM_COLS] > upper_limit).sum())

# %% [markdown]
# ## 4. Kategori: langka dan yang hanya muncul di test
# Kategori yang hanya ada di test tidak pernah dilihat model → perlu `handle_unknown="ignore"`.

# %%
for col in CAT_COLS:
    counts = train[col].value_counts()
    only_in_test = set(test[col].dropna()) - set(train[col].dropna())
    rare = (counts < RARE_COUNT).sum()
    print(col, "| unik", len(counts), "| langka", rare, "| hanya di test:", only_in_test)

# %% [markdown]
# ## 5. Kebocoran: skor satu kolom saja terhadap target
# Kolom kategori diubah ke kode angka dulu. AUC ≈ 1.0 dari satu kolom saja = curiga bocor
# (isi kolom baru diketahui setelah target terjadi).

# %%
codes = pd.DataFrame()
for col in FEATURES:
    if col in CAT_COLS:
        codes[col] = train[col].astype("category").cat.codes
    else:
        codes[col] = train[col].fillna(-1)

leak_auc = pd.Series(dtype=float)
for col in FEATURES:
    auc = roc_auc_score(train[TARGET], codes[col])
    leak_auc[col] = max(auc, 1 - auc)
leak_auc = leak_auc.sort_values()
print(leak_auc.tail(6).round(3))
print("korelasi ID dengan target:", round(train[ID_COL].corr(train[TARGET]), 3))

# %% [markdown]
# ## 6. Kolom kembar / sangat berkorelasi (salah satunya bisa dibuang)
# Korelasi Spearman antar semua pasangan kolom; tampilkan yang > 0.9.

# %%
correlation = codes.corr(method="spearman").abs()
for i in range(len(FEATURES)):
    for other in range(i + 1, len(FEATURES)):
        value = correlation.iloc[i, other]
        if value > 0.9:
            print(FEATURES[i], "~", FEATURES[other], ":", round(value, 3))

# %% [markdown]
# ## 7. Drift train vs test: uji KS (kolom angka) dan PSI
# - KS p-value kecil (< 0.05) = sebaran train dan test berbeda.
# - PSI < 0.1 aman, 0.1–0.25 waspada, > 0.25 bergeser jauh.
#   PSI membandingkan persen data di tiap kelompok (bin) train vs test.

# %%
def psi(train_values, test_values):
    edges = np.unique(np.nanquantile(train_values, np.linspace(0, 1, PSI_BINS + 1)))
    test_values = np.clip(test_values, edges[0], edges[-1])
    train_share = np.histogram(train_values, edges)[0] / len(train_values) + 1e-4
    test_share = np.histogram(test_values, edges)[0] / len(test_values) + 1e-4
    return float(np.sum((train_share - test_share) * np.log(train_share / test_share)))


drift = pd.DataFrame(index=NUM_COLS)
for col in NUM_COLS:
    train_values = train[col].dropna()
    test_values = test[col].dropna()
    drift.loc[col, "ks_pvalue"] = stats.ks_2samp(train_values, test_values).pvalue
    drift.loc[col, "psi"] = psi(train_values, test_values)
drift.round(4)

# %% [markdown]
# ## 8. Hubungan feature dengan target (mutual information)
# Makin panjang batangnya, makin banyak informasi kolom itu tentang target.

# %%
mutual_info = mutual_info_classif(codes.fillna(-1), train[TARGET], random_state=SEED)
mutual_info = pd.Series(mutual_info, index=FEATURES)
mutual_info.sort_values().plot.barh(figsize=(6, 4))
plt.title("Mutual information dengan target")
plt.show()

# %% [markdown]
# ## Ringkasan → CSV

# %%
summary = profile.copy()
summary["auc_satu_kolom"] = leak_auc
summary["psi_train_test"] = drift["psi"]
summary.to_csv("eda_ringkasan.csv")
print("kolom curiga bocor:", list(leak_auc[leak_auc > 0.95].index))
summary
