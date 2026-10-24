# %% [markdown]
# # Klasifikasi teks: TF-IDF + model linear (baseline tercepat, coba DULU)
#
# **Butuh:** `train` (kolom teks + label) dan `test` (kolom teks + kolom id soal).
# **Menghasilkan:** `submission.csv`, `outputs/val_predictions.csv`, kata terpenting per kelas.
# Bagian 7: versi **regresi** (target angka, mis. bintang 1–5) dengan Ridge.
#
# Kapan: hampir selalu jadi langkah pertama. Detik-an di CPU, sering hanya kalah 2–5 poin dari
# fine-tune BERT, dan kadang menang (teks panjang, banyak kata kunci, bahasa campur/gaul).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `make_model` (TF-IDF kata + huruf → LogisticRegression) | **ya** |
# | 2 | validasi 80/20: skor, laporan per kelas, salah tebak | **ya** |
# | 3 | cross-validation 5 fold + bandingkan beberapa model | disarankan |
# | 4 | kata paling berpengaruh per kelas | presentasi |
# | 5 | latih di SEMUA train → `submission.csv` | **ya** |
# | 6 | probabilitas per kelas (untuk ensembling / ambang) | opsional |
# | 7 | regresi teks (Ridge) | kalau target angka |
#
# Teks gaul/berantakan: bersihkan dulu dengan `clean_text` dari `teks/dasar/praproses.ipynb`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Ulasan berlabel sentimen; `test_labels` = kunci jawaban (hanya ada di demo).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/sentiment/smsa_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/sentiment/smsa_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/sentiment/smsa_test_labels.csv")
print(train.shape, test.shape)
print(train["label"].value_counts())

# %% [markdown]
# ## 1. Konfigurasi + `make_model()`
# `FeatureUnion` = gabungan dua TF-IDF (kata 1–2 + potongan huruf 2–5). Dibungkus `Pipeline`
# supaya TF-IDF ikut di-`fit` ulang di tiap fold (tidak bocor dari val).

# %%
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

TEXT = "text"     # UBAH
TARGET = "label"  # UBAH
os.makedirs("outputs", exist_ok=True)
train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)


def make_model(classifier):
    words = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    features = FeatureUnion([("kata", words), ("huruf", chars)])
    return make_pipeline(features, classifier)


model = make_model(LogisticRegression(C=5, max_iter=3000, class_weight="balanced"))   # UBAH: C
print(model.steps[-1])

# %% [markdown]
# ## 2. Validasi 80/20 → skor, laporan per kelas, `outputs/val_predictions.csv`

# %%
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

X = train[TEXT].to_numpy()
y = train[TARGET].to_numpy()
index_train, index_val = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                          random_state=42)
model.fit(X[index_train], y[index_train])
val_pred = model.predict(X[index_val])
print("akurasi:", round(accuracy_score(y[index_val], val_pred), 4),
      "| F1 macro:", round(f1_score(y[index_val], val_pred, average="macro"), 4))
print(classification_report(y[index_val], val_pred))
val_table = train.iloc[index_val].copy()
val_table["pred"] = val_pred
val_table.to_csv("outputs/val_predictions.csv", index=False)
wrong = val_table[val_table[TARGET] != val_table["pred"]]
print(wrong[[TEXT, TARGET, "pred"]].head(5).to_string(max_colwidth=70))

# %% [markdown]
# ## 3. Cross-validation 5 fold + bandingkan model
# LinearSVC sering sedikit lebih baik dari LogisticRegression untuk TF-IDF (tapi tanpa
# probabilitas). ComplementNB: sangat cepat, bagus untuk data kecil.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

candidates = {
    "logreg": make_model(LogisticRegression(C=5, max_iter=3000, class_weight="balanced")),
    "linear_svc": make_model(LinearSVC(C=0.5, class_weight="balanced")),
    "complement_nb": make_model(ComplementNB(alpha=0.3)),
}   # UBAH
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, candidate in candidates.items():
    scores = cross_val_score(candidate, X, y, cv=folds, scoring="f1_macro")
    print(f"{name:14s} F1 macro: {scores.mean():.4f} ± {scores.std():.4f}")

# %% [markdown]
# ## 4. Kata/potongan paling berpengaruh per kelas (dari model sel 2)
# Koefisien positif besar = fitur itu mendorong tebakan ke kelas tersebut.

# %%
import numpy as np

feature_names = np.array(model[0].get_feature_names_out())
classifier = model[-1]
for row, label in enumerate(classifier.classes_):
    top = np.argsort(-classifier.coef_[row])[:12]
    print(f"{label}: {', '.join(feature_names[top])}")

# %% [markdown]
# ## 5. Latih di SEMUA train → `submission.csv`

# %%
import pandas as pd

model.fit(X, y)
predictions = model.predict(test[TEXT].to_numpy())
submission = pd.DataFrame()
submission["id"] = test["id"]          # UBAH: nama kolom dari sample_submission
submission["label"] = predictions      # UBAH
submission.to_csv("submission.csv", index=False)
print(submission.head())
print(submission["label"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo; saat lomba tidak ada).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
print("F1 macro test:", round(f1_score(test_labels["label"], submission["label"],
                                       average="macro"), 4))

# %% [markdown]
# ## 6. Probabilitas per kelas → `outputs/test_proba.csv`
# Untuk rata-rata dengan model lain (ensembling) atau mengubah ambang. LinearSVC tidak punya
# `predict_proba`; bungkus dengan `CalibratedClassifierCV(LinearSVC())` kalau perlu.

# %%
import pandas as pd

proba = model.predict_proba(test[TEXT].to_numpy())
proba_table = pd.DataFrame(proba, columns=model.classes_)
proba_table.insert(0, "id", test["id"].to_numpy())
proba_table.to_csv("outputs/test_proba.csv", index=False)
print(proba_table.head().round(3))

# %% [markdown]
# ## 7. Regresi teks (target angka) → `submission_regresi.csv`
# Data contoh: ulasan Yelp + bintang 1–5. Metrik: RMSE / MAE. Tebakan dipotong ke rentang
# target (`clip`), dan dibulatkan kalau soal minta bilangan bulat.

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

reviews = pd.read_csv(f"{DEMO_FOLDER}/text/regression/yelp_train.csv")      # UBAH: train-mu
reviews_test = pd.read_csv(f"{DEMO_FOLDER}/text/regression/yelp_test.csv")  # UBAH: test-mu
NUMBER = "stars"                                                           # UBAH
regressor = make_model(Ridge(alpha=1.0))                                   # UBAH: alpha
part_train, part_val = train_test_split(reviews, test_size=0.2, random_state=42)
regressor.fit(part_train[TEXT], part_train[NUMBER])
low, high = reviews[NUMBER].min(), reviews[NUMBER].max()
val_guess = np.clip(regressor.predict(part_val[TEXT]), low, high)
print("RMSE:", round(mean_squared_error(part_val[NUMBER], val_guess) ** 0.5, 4),
      "| MAE:", round(mean_absolute_error(part_val[NUMBER], val_guess), 4),
      "| tebak rata-rata saja:", round(part_val[NUMBER].std(), 4))
regressor.fit(reviews[TEXT], reviews[NUMBER])
submission = pd.DataFrame()
submission["id"] = reviews_test["id"]                                                 # UBAH
submission[NUMBER] = np.clip(regressor.predict(reviews_test[TEXT]), low, high)  # UBAH: round()
submission.to_csv("submission_regresi.csv", index=False)
print(submission.head())
