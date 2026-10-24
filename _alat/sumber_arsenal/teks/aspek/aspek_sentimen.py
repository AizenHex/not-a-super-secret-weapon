# %% [markdown]
# # Sentimen per aspek (ABSA): satu ulasan → sentimen untuk TIAP aspek (kebersihan, wifi, ...)
#
# **Butuh:** `train` (kolom teks + satu kolom per aspek berisi kelas, mis. `pos`/`neg`/`neut`),
# `test` (teks + id). **Menghasilkan:** `submission.csv` (satu kolom per aspek).
#
# Beda dengan multilabel (`teks/multilabel`): tiap aspek bukan 0/1 tetapi **beberapa kelas**.
# Cara paling sederhana yang kuat: satu model klasifikasi per aspek. Acuan IndoNLU HoASA
# (F1 macro): IndoBERT-base 92,1.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + sebaran kelas per aspek | **ya** |
# | 2 | `make_model` + cross-validation per aspek | **ya** |
# | 3 | latih per aspek di SEMUA train → `submission.csv` | **ya** |
# | 4 | kata kunci aspek: aspek tidak disebut → paksa kelas netral | opsional, cek di CV |
#
# Format "panjang" (satu baris per ulasan × aspek)? Ubah ke lebar dulu:
# `table.pivot(index="id", columns="aspek", values="sentimen")`.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# HoASA: ulasan hotel, 10 aspek, kelas `pos` / `neg` / `neut` / `neg_pos`.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/absa/hoasa_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/absa/hoasa_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/absa/hoasa_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + sebaran kelas per aspek
# Kelas `neut` biasanya berarti "aspek tidak dibahas" dan mendominasi → F1 macro penting.
# Kelas yang sangat jarang (`neg_pos`, 1 contoh) menurunkan F1 macro per aspek; baca metrik soal:
# rata-rata per aspek, atau semua aspek digabung jadi satu daftar (sel cek di bawah sel 3).

# %%
import pandas as pd

TEXT = "review"   # UBAH
ASPECTS = ["ac", "air_panas", "bau", "general", "kebersihan", "linen", "service",
           "sunrise_meal", "tv", "wifi"]   # UBAH

train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)
spread = {}
for aspect in ASPECTS:
    spread[aspect] = train[aspect].value_counts()
spread = pd.DataFrame(spread).fillna(0)
print(spread.astype(int).T)

# %% [markdown]
# ## 2. `make_model()` + cross-validation 5 fold per aspek
# Satu pipeline TF-IDF + LogisticRegression per aspek. Kelas yang sangat jarang (< 5 contoh)
# membuat StratifiedKFold mengeluh → `KFold` biasa dipakai.

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline


def make_model():
    features = FeatureUnion([
        ("kata", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("huruf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                  sublinear_tf=True))])
    return make_pipeline(features, LogisticRegression(C=10, max_iter=3000,
                                                      class_weight="balanced"))


folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = {}
for aspect in ASPECTS:
    oof[aspect] = cross_val_predict(make_model(), train[TEXT], train[aspect], cv=folds)
    print(f"{aspect:13s} F1 macro: {f1_score(train[aspect], oof[aspect], average='macro'):.4f}")
scores = []
for aspect in ASPECTS:
    scores.append(f1_score(train[aspect], oof[aspect], average="macro"))
print("rata-rata:", round(np.mean(scores), 4))

# %% [markdown]
# ## 3. Latih per aspek di SEMUA train → `submission.csv`

# %%
import pandas as pd

submission = pd.DataFrame()
submission["id"] = test["id"]                 # UBAH
models = {}
for aspect in ASPECTS:
    models[aspect] = make_model()
    models[aspect].fit(train[TEXT], train[aspect])
    submission[aspect] = models[aspect].predict(test[TEXT])
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
import numpy as np
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
scores = []
for aspect in ASPECTS:
    scores.append(f1_score(test_labels[aspect], submission[aspect], average="macro"))
print("F1 macro test rata-rata aspek:", round(np.mean(scores), 4))
all_true = test_labels[ASPECTS].to_numpy().ravel()
all_pred = submission[ASPECTS].to_numpy().ravel()
print("F1 macro semua aspek digabung (cara IndoNLU):",
      round(f1_score(all_true, all_pred, average="macro"), 4))

# %% [markdown]
# ## 4. (Opsional) Kata kunci aspek: aspek tidak disebut → kelas netral
# Aturan sederhana yang sering membantu: kalau tidak satu pun kata kunci aspek muncul, tebakan
# dipaksa `NEUTRAL`. Nilai dulu di out-of-fold (sel 2); pakai hanya kalau F1 naik.

# %%
import numpy as np
from sklearn.metrics import f1_score

NEUTRAL = "neut"   # UBAH
KEYWORDS = {"ac": ["ac", "dingin", "panas", "gerah"], "wifi": ["wifi", "wi-fi", "internet",
                                                               "sinyal"],
            "tv": ["tv", "televisi", "channel"]}   # UBAH: aspek lain dibiarkan
for aspect, words in KEYWORDS.items():
    lowered = train[TEXT].str.lower()
    mentioned = np.zeros(len(train), dtype=bool)
    for word in words:
        mentioned |= lowered.str.contains(word, regex=False).to_numpy()
    gated = np.where(mentioned, oof[aspect], NEUTRAL)
    before = f1_score(train[aspect], oof[aspect], average="macro")
    after = f1_score(train[aspect], gated, average="macro")
    print(f"{aspect:5s} F1 sebelum {before:.4f} → sesudah aturan {after:.4f}")
