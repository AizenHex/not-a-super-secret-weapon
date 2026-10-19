# %% [markdown]
# # Multimodal — Klasifikasi (teks + kolom angka)
# **Kapan dipakai:** tiap baris punya kolom teks (judul, deskripsi, ulasan) DAN kolom angka
# (harga, tahun, jumlah), dan diminta menebak kelas.
#
# **Idenya:** teks diubah jadi angka dengan TF-IDF (bobot tiap kata), kolom angka diisi
# median lalu disamakan skalanya. Keduanya digabung jadi satu tabel, lalu model sklearn biasa.
# `ColumnTransformer` mengatur "kolom mana diolah dengan cara apa".
#
# **Langkah:**
# 1. Baca data, split train/validasi.
# 2. Olah teks + angka, latih model, evaluasi F1.
# 3. Latih ulang di semua data, buat submission.
#
# **Data demo:** judul film (`title`) + tahun rilis (`release_year`), target biner `target`.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/multimodal/classification"   # GANTI saat lomba
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
print(train[TARGET_COL].value_counts(normalize=True))
train.head()

# %% [markdown]
# ## Langkah 1 — split train/validasi

# %%
X = train[[TEXT_COL] + NUMBER_COLS]
y = train[TARGET_COL]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y,
                                                  random_state=SEED)

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

model = LogisticRegression(C=1.0, max_iter=2000)
# model = LinearSVC(C=0.5)                     # sering paling kuat untuk teks
# model = RandomForestClassifier(n_estimators=500, random_state=SEED)
# model = LGBMClassifier(n_estimators=300, learning_rate=0.03, random_state=SEED, verbose=-1)

pipeline = make_pipeline(preprocess, model)
pipeline.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di data validasi

# %%
pred_val = pipeline.predict(X_val)
print("F1 macro:", round(f1_score(y_val, pred_val, average="macro"), 4))
print(classification_report(y_val, pred_val))
ConfusionMatrixDisplay.from_predictions(y_val, pred_val)
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
