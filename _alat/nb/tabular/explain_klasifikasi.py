# %% [markdown]
# # Tabular — Menjelaskan Model (Klasifikasi)
# **Kapan dipakai:** juri bertanya "kolom apa yang paling berpengaruh?" atau "kenapa baris ini
# ditebak selamat?". Biasanya untuk bahan presentasi.
#
# **Tiga cara:**
# 1. Permutation importance: seberapa turun skor kalau satu kolom diacak.
# 2. SHAP: sumbangan tiap kolom ke tiap tebakan.
# 3. Partial dependence (PDP/ICE): kalau satu kolom dinaikkan, peluang berubah bagaimana.
#
# **Data demo:** Titanic (target `survived`). SHAP `TreeExplainer` butuh model pohon.

# %%
import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_COLS = ["sex", "embarked", "who", "deck"]
PDP_COLS = ["age", "fare"]         # kolom angka yang digambar efeknya (partial dependence)
SEED = 42

# %% [markdown]
# ## Baca data dan split

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)

X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
train.head()

# %% [markdown]
# ## Model
# Kolom angka tidak di-scale (model pohon tidak butuh), supaya nilai di gambar SHAP mudah dibaca.
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False))
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
], verbose_feature_names_out=False)

model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=300, random_state=SEED)
# model = XGBClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED)
# model = CatBoostClassifier(iterations=300, random_seed=SEED, verbose=0,
#                            allow_writing_files=False)
# model = ExtraTreesClassifier(n_estimators=300, random_state=SEED)

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("F1 macro validasi:", round(f1_score(y_val, val_predictions, average="macro"), 4))

# %% [markdown]
# ## 1. Permutation importance
# Tiap kolom diacak 20 kali di data validasi. Batang panjang = skor turun banyak = kolom penting.

# %%
result = permutation_importance(pipe, X_val, y_val, scoring="f1_macro",
                                n_repeats=20, random_state=SEED)
importance = pd.Series(result.importances_mean, index=X_val.columns)
importance = importance.sort_values()
importance.plot.barh()
plt.title("Permutation importance")
plt.show()

# %% [markdown]
# ## 2. SHAP
# - Beeswarm: tiap titik = satu baris. Kanan = menaikkan peluang kelas 1, kiri = menurunkan.
#   Merah = nilai kolom tinggi, biru = rendah.
# - Waterfall: rincian satu tebakan (baris validasi pertama).
# RandomForest/ExtraTrees memberi SHAP untuk tiap kelas: hapus `#` pada baris kelas 1.

# %%
X_val_ready = pd.DataFrame(preprocess.transform(X_val),
                           columns=preprocess.get_feature_names_out())
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_val_ready)
# shap_values = shap_values[:, :, 1]   # RandomForest/ExtraTrees: ambil SHAP kelas 1
shap.plots.beeswarm(shap_values, max_display=12)
shap.plots.waterfall(shap_values[0])

# %% [markdown]
# ## 3. Partial dependence + ICE
# Garis tebal = rata-rata efek kolom terhadap peluang. Garis tipis = efek untuk tiap baris.

# %%
PartialDependenceDisplay.from_estimator(pipe, X_val, PDP_COLS, kind="both",
                                        subsample=100, random_state=SEED)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[NUM_COLS + CAT_COLS])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["survived"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
