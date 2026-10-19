# %% [markdown]
# # Multimodal — Regresi (teks + kolom angka)
# **Kapan dipakai:** tiap baris punya kolom teks (judul, deskripsi) DAN kolom angka, dan
# diminta menebak sebuah angka (harga, rating, jumlah penonton).
#
# **Idenya:** teks diubah jadi angka dengan TF-IDF (bobot tiap kata), kolom angka diisi
# median lalu disamakan skalanya. Keduanya digabung jadi satu tabel, lalu model regresi biasa.
#
# **Langkah:**
# 1. Baca data, split train/validasi.
# 2. Olah teks + angka, latih model, evaluasi RMSE/MAE/R².
# 3. Latih ulang di semua data, buat submission.
#
# **Data demo:** judul film (`title`) + tahun rilis (`release_year`), target angka `target`.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/multimodal/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/train.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"

ID_COL = "id"
TARGET_COL = "target"
TEXT_COL = "title"
NUMBER_COLS = ["release_year"]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print(train.shape, test.shape)
print(train.isna().sum())
print(train[TARGET_COL].describe())
train.head()

# %% [markdown]
# ## Langkah 1 — split train/validasi

# %%
X = train[[TEXT_COL] + NUMBER_COLS]
y = train[TARGET_COL]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %% [markdown]
# ## Langkah 2 — olah teks + angka, lalu model
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
text_steps = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
number_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
preprocess = ColumnTransformer([
    ("text", text_steps, TEXT_COL),
    ("number", number_steps, NUMBER_COLS),
])

model = Ridge(alpha=1.0)
# model = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)

pipeline = make_pipeline(preprocess, model)
pipeline.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di data validasi

# %%
pred_val = pipeline.predict(X_val)
print("RMSE:", round(root_mean_squared_error(y_val, pred_val), 4))
print("MAE :", round(mean_absolute_error(y_val, pred_val), 4))
print("R2  :", round(r2_score(y_val, pred_val), 4))
plt.scatter(y_val, pred_val, s=8, alpha=0.5)
plt.xlabel("aktual")
plt.ylabel("prediksi")
plt.show()

# %% [markdown]
# ## Langkah 3 — latih ulang di semua data + submission

# %%
pipeline.fit(X, y)
predictions = pipeline.predict(test[[TEXT_COL] + NUMBER_COLS])

submission = pd.DataFrame()
submission["id"] = test[ID_COL]
submission["target"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
