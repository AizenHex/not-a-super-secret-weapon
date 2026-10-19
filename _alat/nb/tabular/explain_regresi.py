# %% [markdown]
# # Tabular — Menjelaskan Model (Regresi)
# **Kapan dipakai:** juri bertanya "kolom apa yang paling berpengaruh?" atau "kenapa model
# menebak harga ini?". Biasanya untuk bahan presentasi.
#
# **Tiga cara:**
# 1. Permutation importance: seberapa naik error kalau satu kolom diacak.
# 2. SHAP: sumbangan tiap kolom ke tiap tebakan (naik/turun berapa).
# 3. Partial dependence: kalau satu kolom dinaikkan, tebakan berubah bagaimana.
#
# **Data demo:** Diamonds (target `price`). SHAP `TreeExplainer` butuh model pohon.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/diamonds_train.csv"
TEST_CSV = f"{DATA_FOLDER}/diamonds_test.csv"

TARGET = "price"
NUM_COLS = ["carat", "depth", "table", "x", "y", "z"]
CAT_COLS = ["cut", "color", "clarity"]
PDP_COLS = ["carat", "depth"]      # kolom angka yang digambar efeknya (partial dependence)
SEED = 42

# %% [markdown]
# ## Baca data dan split

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)

X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
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

model = LGBMRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED, verbose=-1)
# model = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
# model = XGBRegressor(n_estimators=800, learning_rate=0.03, random_state=SEED)
# model = CatBoostRegressor(iterations=800, random_seed=SEED, verbose=0,
#                           allow_writing_files=False)
# model = ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)

pipe = make_pipeline(preprocess, model)
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("RMSE validasi:", round(np.sqrt(mean_squared_error(y_val, val_predictions)), 2))

# %% [markdown]
# ## 1. Permutation importance
# Tiap kolom diacak 5 kali di data validasi. Batang panjang = RMSE naik banyak = kolom penting.

# %%
result = permutation_importance(pipe, X_val, y_val, scoring="neg_root_mean_squared_error",
                                n_repeats=5, random_state=SEED)
importance = pd.Series(result.importances_mean, index=X_val.columns)
importance = importance.sort_values()
importance.plot.barh()
plt.title("Permutation importance (kenaikan RMSE)")
plt.show()

# %% [markdown]
# ## 2. SHAP
# - Beeswarm: tiap titik = satu baris. Kanan = menaikkan tebakan, kiri = menurunkan.
#   Merah = nilai kolom tinggi, biru = rendah.
# - Waterfall: rincian satu tebakan (baris validasi pertama).

# %%
X_val_ready = pd.DataFrame(preprocess.transform(X_val),
                           columns=preprocess.get_feature_names_out())
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_val_ready)
shap.plots.beeswarm(shap_values, max_display=12)
shap.plots.waterfall(shap_values[0])

# %% [markdown]
# ## 3. Partial dependence + ICE
# Garis tebal = rata-rata efek kolom terhadap tebakan. Garis tipis = efek untuk tiap baris.

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
submission["price"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
