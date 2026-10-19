# %% [markdown]
# # Tabular — Data Tidak Seimbang (imbalanced)
# **Kapan dipakai:** salah satu kelas sangat jarang (misalnya penipuan 2%, penyakit langka).
# Model biasa cenderung selalu menebak kelas besar, jadi akurasi tinggi tapi tidak berguna.
#
# **Cara menyeimbangkan:**
# - Oversampling (SMOTE dkk.): buat contoh buatan untuk kelas kecil.
# - Undersampling: buang sebagian contoh kelas besar.
# - Class weight: kesalahan pada kelas kecil diberi hukuman lebih besar.
# - Geser threshold: tebak positif kalau probabilitas ≥ threshold (tidak harus 0.5).
#
# **Metrik:** F1 kelas positif dan PR-AUC (bukan akurasi).
#
# **Data demo:** Mammography (target `label`, kelas positif ±2%).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/imbalanced"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/mammography_train.csv"
TEST_CSV = f"{DATA_FOLDER}/mammography_test.csv"

TARGET = "label"
FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6"]
THRESHOLD = 0.5          # tebak positif kalau probabilitas ≥ angka ini
THRESHOLDS_TO_TRY = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
missing_per_column = train.isna().sum()
print("sel kosong:", missing_per_column.sum())
train.head()

# %% [markdown]
# ## EDA
# Lihat seberapa jarang kelas positif, dan apakah rata-rata feature berbeda antar kelas.

# %%
print(train[TARGET].value_counts())
print("proporsi positif:", round(train[TARGET].mean(), 4))
train.groupby(TARGET)[FEATURES].mean()

# %% [markdown]
# ## Split train / validasi
# `stratify=y` penting di sini: tanpa itu, validasi bisa hampir tidak berisi kelas positif.

# %%
X = train[FEATURES]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# %% [markdown]
# ## Model
# `make_pipeline` dari `imblearn`: sampler (SMOTE dkk.) hanya dipakai saat `fit`, tidak saat
# `predict`, jadi data validasi/test tidak ikut diubah.
# Pilih SATU `pipe`: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
pipe = make_pipeline(StandardScaler(), SMOTE(random_state=SEED),
                     LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(StandardScaler(), SMOTE(random_state=SEED),
#                      LogisticRegression(max_iter=2000))
# pipe = make_pipeline(StandardScaler(), BorderlineSMOTE(random_state=SEED),
#                      RandomForestClassifier(n_estimators=300, random_state=SEED))
# pipe = make_pipeline(StandardScaler(), SMOTEENN(random_state=SEED),
#                      LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(RandomUnderSampler(random_state=SEED),
#                      RandomForestClassifier(n_estimators=300, random_state=SEED))
# pipe = make_pipeline(StandardScaler(),
#                      LogisticRegression(class_weight="balanced", max_iter=2000))
# pipe = make_pipeline(LGBMClassifier(class_weight="balanced", verbose=-1))
# pipe = make_pipeline(XGBClassifier(scale_pos_weight=40, n_estimators=300))
# pipe = make_pipeline(BalancedRandomForestClassifier(n_estimators=300, random_state=SEED))
# pipe = make_pipeline(EasyEnsembleClassifier(n_estimators=10, random_state=SEED))
# pipe = make_pipeline(StandardScaler(), ADASYN(random_state=SEED),
#                      LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(StandardScaler(), SVMSMOTE(random_state=SEED),
#                      LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(StandardScaler(), SMOTETomek(random_state=SEED),
#                      LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(TomekLinks(), LGBMClassifier(n_estimators=300, verbose=-1))
# pipe = make_pipeline(CatBoostClassifier(auto_class_weights="Balanced", verbose=0,
#                                         allow_writing_files=False, random_seed=SEED))
# pipe = make_pipeline(RUSBoostClassifier(n_estimators=200, random_state=SEED))
# pipe = make_pipeline(BalancedBaggingClassifier(n_estimators=50, random_state=SEED))

pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# `[:, 1]` = kolom probabilitas kelas positif.

# %%
val_proba = pipe.predict_proba(X_val)[:, 1]
val_predictions = (val_proba >= THRESHOLD).astype(int)
print("F1 positif:", round(f1_score(y_val, val_predictions), 4))
print("PR-AUC    :", round(average_precision_score(y_val, val_proba), 4))
print(classification_report(y_val, val_predictions))
PrecisionRecallDisplay.from_predictions(y_val, val_proba)
plt.show()

# %% [markdown]
# ## Coba beberapa threshold di validasi
# Kalau ada threshold dengan F1 lebih tinggi, tulis angkanya di `THRESHOLD` (Konfigurasi).

# %%
for threshold in THRESHOLDS_TO_TRY:
    guess = (val_proba >= threshold).astype(int)
    print("threshold", threshold, "-> F1", round(f1_score(y_val, guess), 4))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
test_proba = pipe.predict_proba(test[FEATURES])[:, 1]
predictions = (test_proba >= THRESHOLD).astype(int)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission["label"].value_counts()
