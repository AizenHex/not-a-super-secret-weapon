# %% [markdown]
# # Tabular — Klasifikasi
# **Kapan dipakai:** data berbentuk tabel (CSV), dan kita diminta menebak kategori (label)
# untuk tiap baris test. Contoh: selamat/tidak, churn/tidak, kelas A/B/C.
#
# **Langkah:**
# 1. Baca data, lihat isinya.
# 2. Bagi train menjadi train dan validasi untuk mengukur model.
# 3. Siapkan kolom (isi yang kosong, samakan skala, ubah kategori jadi angka), lalu latih model.
# 4. Latih ulang di semua data train, tebak test, simpan submission.
#
# **Data demo:** Titanic (target `survived`). Ganti path dan nama kolom di sel Konfigurasi.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# %% [markdown]
# ## Konfigurasi
# Semua yang boleh diubah ada di sel ini.

# %%
DATA_FOLDER = "../../../data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"                                    # kolom yang ditebak
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]   # kolom angka
CAT_COLS = ["sex", "embarked", "who", "deck"]            # kolom kategori (teks)
SEED = 42

# %% [markdown]
# ## Baca data
# `isna().sum()` = jumlah sel kosong per kolom.

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA
# - Proporsi tiap label: kalau sangat timpang, pakai F1 macro, jangan akurasi.
# - Histogram kolom angka.
# - Rata-rata target per kategori: kategori mana yang paling berpengaruh.

# %%
print(train[TARGET].value_counts(normalize=True))
train[NUM_COLS].hist(figsize=(10, 6))
plt.show()

for col in CAT_COLS:
    print(train.groupby(col)[TARGET].agg(["mean", "count"]), "\n")

# %% [markdown]
# ## Split train / validasi
# 20% data train disisihkan untuk mengukur model. `stratify=y` menjaga proporsi label sama.

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("train:", X_train.shape, "| validasi:", X_val.shape)

# %% [markdown]
# ## Model
# `preprocess` menyiapkan kolom:
# - kolom angka: isi yang kosong dengan median, lalu samakan skala (StandardScaler);
# - kolom kategori: isi yang kosong dengan nilai terbanyak, lalu ubah jadi kolom 0/1 (OneHot).
#
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

model = LogisticRegression(C=1.0, max_iter=2000)                      # cepat, mudah dijelaskan
# model = RandomForestClassifier(n_estimators=500, random_state=SEED)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED)
# model = HistGradientBoostingClassifier(random_state=SEED)
# model = LGBMClassifier(n_estimators=500, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = XGBClassifier(n_estimators=500, learning_rate=0.03, random_state=SEED)
# model = CatBoostClassifier(iterations=500, random_seed=SEED, verbose=0)
# model = SVC(C=1.0, random_state=SEED)
# model = KNeighborsClassifier(n_neighbors=15)
# model = DecisionTreeClassifier(max_depth=6, random_state=SEED)
# model = GradientBoostingClassifier(random_state=SEED)
# model = AdaBoostClassifier(n_estimators=300, random_state=SEED)
# model = BaggingClassifier(n_estimators=100, random_state=SEED)
# model = RidgeClassifier(alpha=1.0)
# model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=SEED)

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# F1 macro = rata-rata F1 semua label (label kecil dihitung sama penting).
# Confusion matrix: baris = label asli, kolom = tebakan.

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission
# Nama kolom submission disalin dari format soal.

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[NUM_COLS + CAT_COLS])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["survived"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
