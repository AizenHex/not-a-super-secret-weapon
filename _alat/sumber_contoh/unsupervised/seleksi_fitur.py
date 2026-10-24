# %% [markdown]
# # Unsupervised — Seleksi Feature (buang kolom yang tidak berguna)
# **Kapan dipakai:** kolom sangat banyak, sebagian mungkin acak atau duplikat, dan diminta
# memilih kolom penting atau mempercepat model.
#
# **Idenya:** pemilih feature diletakkan DI DALAM pipeline, jadi tiap fold CV memilih
# kolomnya sendiri (tanpa bocor). Skor CV dengan semua kolom dibandingkan dengan skor CV
# setelah seleksi.
#
# **Langkah:**
# 1. Skor CV dengan semua kolom.
# 2. Skor CV dengan pemilih feature, lihat kolom yang terpilih.
# 3. Permutation importance (info tambahan), simpan daftar kolom.
#
# **Data demo:** Breast cancer + kolom palsu: `noise_*` = acak, `dup_*` = salinan kolom asli.

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/unsupervised"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/breast_cancer_noisy.csv"

TARGET_COL = "diagnosis"
POSITIVE_LABEL = "malignant"   # kelas yang dihitung F1-nya
N_FEATURES = 10                # jumlah kolom yang dipilih
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
y = (data[TARGET_COL] == POSITIVE_LABEL).astype(int)
X = data.drop(columns=[TARGET_COL])
print("jumlah kolom:", X.shape[1])
print(y.value_counts())

# %% [markdown]
# ## Langkah 1 — skor CV dengan semua kolom

# %%
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
all_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
all_scores = cross_val_score(all_model, X, y, cv=folds, scoring="f1")
print("semua kolom, F1 CV:", round(all_scores.mean(), 3))

# %% [markdown]
# ## Langkah 2 — pemilih feature
# Pilih SATU: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
selector = SelectKBest(f_classif, k=N_FEATURES)          # uji statistik per kolom, cepat
# selector = SelectKBest(mutual_info_classif, k=N_FEATURES)   # menangkap hubungan tak linear
# selector = RFE(LogisticRegression(max_iter=5000),
#                n_features_to_select=N_FEATURES)        # buang kolom terlemah berulang
# selector = SelectFromModel(RandomForestClassifier(n_estimators=300, random_state=SEED),
#                            max_features=N_FEATURES, threshold=-1)   # pentingnya pohon

model = make_pipeline(StandardScaler(), selector, LogisticRegression(max_iter=5000))
selected_scores = cross_val_score(model, X, y, cv=folds, scoring="f1")
print(N_FEATURES, "kolom terpilih, F1 CV:", round(selected_scores.mean(), 3))

model.fit(X, y)
is_selected = model[1].get_support()
selected_cols = X.columns[is_selected]
print("terpilih:", list(selected_cols))

# %% [markdown]
# ## Langkah 3 — permutation importance (info tambahan)
# Kolom diacak satu per satu di data uji: makin turun skornya, makin penting kolom itu.

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                    random_state=SEED)
all_model.fit(X_train, y_train)
permutation = permutation_importance(all_model, X_test, y_test, scoring="f1", n_repeats=10,
                                     random_state=SEED)
importance = pd.Series(permutation.importances_mean, index=X.columns)
importance = importance.sort_values(ascending=False)
importance.head(N_FEATURES).round(4)

# %% [markdown]
# ## Simpan daftar kolom

# %%
submission = pd.DataFrame()
submission["feature"] = X.columns
submission["selected"] = is_selected
submission["permutation_importance"] = permutation.importances_mean
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)
# Di data demo, kolom `noise_*` seharusnya tidak terpilih.

# %%
noise_selected = selected_cols.str.startswith("noise")
print("kolom noise yang terpilih:", noise_selected.sum())
