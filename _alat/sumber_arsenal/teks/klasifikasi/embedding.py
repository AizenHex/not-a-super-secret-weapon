# %% [markdown]
# # Klasifikasi teks: vektor kalimat pretrained + model sklearn (+ gabung dengan TF-IDF)
#
# **Butuh:** `train` (kolom teks + label), `test` (kolom teks + id).
# **Menghasilkan:** `submission.csv`, perbandingan skor TF-IDF vs embedding vs gabungan.
#
# Kapan: data sedikit (puluhan–ratusan per kelas), tidak sempat fine-tune, atau ingin model
# kedua untuk digabung dengan TF-IDF. TF-IDF menangkap **kata kunci**, embedding menangkap
# **makna** (sinonim, parafrase) → rata-rata probabilitas keduanya sering mengalahkan
# masing-masing.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `encoder` (MiniLM) | **ya** |
# | 2 | vektor SEMUA train + test | **ya** |
# | 3 | validasi: LogisticRegression di atas vektor | **ya** |
# | 4 | gabung probabilitas TF-IDF + embedding, cari bobot terbaik | disarankan |
# | 5 | latih di SEMUA train → `submission.csv` | **ya** |
#
# Vektor IndoBERT (rata-rata token): `jembatan/teks_ke_vektor.ipynb` sel 4.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + `encoder`

# %%
import torch
from sentence_transformers import SentenceTransformer

PRETRAINED_FOLDER = "pretrained"   # UBAH
TEXT = "text"                      # UBAH
TARGET = "label"                   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)
encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)  # UBAH: minilm_en
print(encoder.encode(["contoh kalimat"]).shape)

# %% [markdown]
# ## 2. Vektor SEMUA train + test → `X`, `X_test`, `y`

# %%
import numpy as np

X = encoder.encode(train[TEXT].tolist(), batch_size=64, normalize_embeddings=True,
                   show_progress_bar=True)
X_test = encoder.encode(test[TEXT].tolist(), batch_size=64, normalize_embeddings=True)
y = train[TARGET].to_numpy()
print(X.shape, X_test.shape)

# %% [markdown]
# ## 3. Validasi 80/20: LogisticRegression di atas vektor
# Alternatif di `#`: SVC (sering +1–2 poin), kNN cosine (data sangat sedikit).

# %%
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

vector_model = LogisticRegression(C=2, max_iter=3000, class_weight="balanced")   # UBAH: C
# from sklearn.svm import SVC
# vector_model = SVC(C=3, probability=True, class_weight="balanced")
# from sklearn.neighbors import KNeighborsClassifier
# vector_model = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
index_train, index_val = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                                          random_state=42)
vector_model.fit(X[index_train], y[index_train])
proba_vector = vector_model.predict_proba(X[index_val])
pred_vector = vector_model.classes_[proba_vector.argmax(axis=1)]
print("F1 macro embedding:", round(f1_score(y[index_val], pred_vector, average="macro"), 4))

# %% [markdown]
# ## 4. Gabung dengan TF-IDF: rata-rata probabilitas berbobot
# `WEIGHT` = bobot embedding (0 = TF-IDF saja, 1 = embedding saja). Dicari di val; pakai
# bobot yang sama untuk test. `classes_` kedua model sama (urut abjad), jadi kolom sejajar.

# %%
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline

features = FeatureUnion([
    ("kata", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("huruf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                              sublinear_tf=True))])
tfidf_model = make_pipeline(features, LogisticRegression(C=5, max_iter=3000,
                                                         class_weight="balanced"))
texts = train[TEXT].to_numpy()
tfidf_model.fit(texts[index_train], y[index_train])
proba_tfidf = tfidf_model.predict_proba(texts[index_val])
best_weight, best_f1 = 0.0, -1.0
for weight in np.arange(0, 1.01, 0.1):
    mixed = weight * proba_vector + (1 - weight) * proba_tfidf
    score = f1_score(y[index_val], vector_model.classes_[mixed.argmax(axis=1)], average="macro")
    print(f"bobot embedding {weight:.1f}: F1 {score:.4f}")
    if score > best_f1:
        best_weight, best_f1 = weight, score
print("bobot terbaik:", round(best_weight, 1), "| F1:", round(best_f1, 4))

# %% [markdown]
# ## 5. Latih kedua model di SEMUA train → `submission.csv`

# %%
import pandas as pd

vector_model.fit(X, y)
tfidf_model.fit(texts, y)
mixed = (best_weight * vector_model.predict_proba(X_test)
         + (1 - best_weight) * tfidf_model.predict_proba(test[TEXT].to_numpy()))
submission = pd.DataFrame()
submission["id"] = test["id"]                                        # UBAH
submission["label"] = vector_model.classes_[mixed.argmax(axis=1)]    # UBAH
submission.to_csv("submission.csv", index=False)
print(submission["label"].value_counts())

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
print("F1 macro test:", round(f1_score(test_labels["label"], submission["label"],
                                       average="macro"), 4))
