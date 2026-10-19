# %% [markdown]
# # Teks — Regresi dari Teks (tebak angka: rating bintang, harga, skor esai)
# **Kapan dipakai:** target berupa angka, bukan kelas.
#
# **Langkah:** TF-IDF → model regresi linear → batasi hasil ke rentang target → RMSE/MAE.
# Kalau target bertingkat (1–5) dan dinilai sebagai kelas, bulatkan prediksi lalu pakai QWK.
#
# **Data demo:** ulasan Yelp (Inggris), target `stars` 1–5.

# %%
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/regression"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/yelp_train.csv"
TEST_CSV = f"{DATA_FOLDER}/yelp_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/yelp_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "text"
TARGET = "stars"
MIN_VALUE = 1      # prediksi dibatasi ke rentang target
MAX_VALUE = 5
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train.isna().sum())
train.head()

# %% [markdown]
# ## EDA: sebaran target dan panjang teks

# %%
star_counts = train[TARGET].value_counts()
print(star_counts.sort_index())
word_counts = train[TEXT_COL].str.split().str.len()
print("korelasi panjang teks vs target:", round(word_counts.corr(train[TARGET]), 3))
word_counts.hist(bins=50)
plt.title("Jumlah kata per ulasan")
plt.show()

# %% [markdown]
# ## Bersihkan teks
# Tanda `'`, `!`, `?` dipertahankan karena membawa nada.

# %%
def clean_text(text):
    text = text.lower()
    text = text.replace("\\n", " ")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9'!?\s]", " ", text)
    return text


print(clean_text("GREAT food!!\\nSee http://yelp.com :)"))

train[TEXT_COL] = train[TEXT_COL].fillna("").apply(clean_text)
test[TEXT_COL] = test[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGET], test_size=0.2, random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                        sublinear_tf=True))
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

model = Ridge(alpha=3.0)                          # cepat, kuat untuk TF-IDF
# model = LinearSVR(C=0.1, max_iter=5000)
# model = SGDRegressor(alpha=1e-5, random_state=SEED)
# model = LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=SEED, verbose=-1)

pipe = make_pipeline(vectorizer, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi: RMSE/MAE (angka) dan QWK (setelah dibulatkan ke kelas)

# %%
val_values = np.clip(pipe.predict(X_val), MIN_VALUE, MAX_VALUE)
val_rounded = np.round(val_values).astype(int)
print("RMSE:", round(root_mean_squared_error(y_val, val_values), 4))
print("MAE :", round(mean_absolute_error(y_val, val_values), 4))
print("QWK (dibulatkan):", round(cohen_kappa_score(y_val, val_rounded, weights="quadratic"), 4))
print("akurasi tepat bintang:", round((val_rounded == y_val).mean(), 4))

true_stars = y_val.to_numpy()
groups = []
for star in range(MIN_VALUE, MAX_VALUE + 1):
    groups.append(val_values[true_stars == star])
plt.boxplot(groups)
plt.xlabel("bintang asli")
plt.ylabel("prediksi")
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission
# Butuh bilangan bulat? Tambah baris `predictions = np.round(predictions).astype(int)`.

# %%
pipe.fit(train[TEXT_COL], train[TARGET])
predictions = np.clip(pipe.predict(test[TEXT_COL]), MIN_VALUE, MAX_VALUE)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["stars"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["stars"]
print("RMSE test (kunci latihan):", round(root_mean_squared_error(test_key, predictions), 4))
