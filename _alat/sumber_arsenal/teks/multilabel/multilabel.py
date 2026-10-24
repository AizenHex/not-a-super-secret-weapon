# %% [markdown]
# # Teks multilabel: satu teks boleh punya BANYAK label (aspek, topik, jenis ujaran)
#
# **Butuh:** `train` (kolom teks + satu kolom 0/1 per label), `test` (teks + id).
# **Menghasilkan:** `submission.csv` (kolom 0/1 per label), ambang per label.
#
# Contoh soal: ulasan mobil menyebut aspek apa saja (bensin, mesin, harga, ...); tweet termasuk
# ujaran kebencian jenis apa saja. Cara: satu model 0/1 per label (one-vs-rest), lalu **ambang
# per label** dicari di prediksi out-of-fold (label jarang butuh ambang < 0,5).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + jumlah per label + label per teks | **ya** |
# | 2 | label dalam satu kolom teks (`"fuel,price"`) → kolom 0/1 | kalau formatnya begitu |
# | 3 | model TF-IDF + one-vs-rest → proba out-of-fold | **ya** |
# | 4 | ambang terbaik per label + skor | **ya** |
# | 5 | latih SEMUA → `submission.csv` (0/1 atau teks label) | **ya** |
#
# Fine-tune BERT multilabel: di `finetune_bert.ipynb` ganti `num_labels=len(LABELS)`,
# `problem_type="multi_label_classification"`, loss `nn.BCEWithLogitsLoss()`, target float,
# `torch.sigmoid` (bukan softmax).

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Ulasan mobil, 6 aspek (fuel, machine, others, part, price, service).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

train = pd.read_csv(f"{DEMO_FOLDER}/text/multilabel/casa_train.csv")
test = pd.read_csv(f"{DEMO_FOLDER}/text/multilabel/casa_test.csv")
test_labels = pd.read_csv(f"{DEMO_FOLDER}/text/multilabel/casa_test_labels.csv")
print(train.head(3))

# %% [markdown]
# ## 1. Konfigurasi + lihat sebaran label

# %%
import os

TEXT = "text"                                                          # UBAH
LABELS = ["fuel", "machine", "others", "part", "price", "service"]    # UBAH
os.makedirs("outputs", exist_ok=True)

train[TEXT] = train[TEXT].fillna("").astype(str)
test[TEXT] = test[TEXT].fillna("").astype(str)
print(train[LABELS].sum().sort_values())
labels_per_text = train[LABELS].sum(axis=1)
print("label per teks:\n", labels_per_text.value_counts().sort_index())

# %% [markdown]
# ## 2. (Kalau perlu) label dalam satu kolom teks → kolom 0/1
# Contoh: kolom `labels` berisi `"fuel,price"`. Lewati kalau sudah satu kolom per label.

# %%
import pandas as pd

example = pd.DataFrame({"labels": ["fuel,price", "service", "", "machine,part,price"]})
dummies = example["labels"].str.get_dummies(sep=",")    # UBAH: sep sesuai soal
print(dummies)
# train = pd.concat([train, dummies], axis=1); LABELS = list(dummies.columns)

# %% [markdown]
# ## 3. Model + proba out-of-fold → `oof`
# `cross_val_predict`: tiap baris ditebak oleh model yang TIDAK melihatnya (jujur untuk
# mencari ambang). `OneVsRestClassifier` = satu LogisticRegression per label.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline

features = FeatureUnion([
    ("kata", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("huruf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              sublinear_tf=True))])
classifier = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")   # UBAH: C
model = make_pipeline(features, OneVsRestClassifier(classifier))
X = train[TEXT].to_numpy()
Y_true = train[LABELS].to_numpy()
folds = KFold(n_splits=5, shuffle=True, random_state=42)
oof = cross_val_predict(model, X, Y_true, cv=folds, method="predict_proba")
print(oof.shape)

# %% [markdown]
# ## 4. Ambang terbaik per label → `thresholds`
# Aturan tambahan "minimal satu label": kalau semua di bawah ambang, ambil label proba
# tertinggi (aktifkan kalau setiap teks di train pasti punya ≥1 label).

# %%
import numpy as np
from sklearn.metrics import f1_score

AT_LEAST_ONE = True   # UBAH
thresholds = {}
for column, label in enumerate(LABELS):
    best, best_score = 0.5, -1.0
    for candidate in np.arange(0.05, 0.95, 0.05):
        score = f1_score(Y_true[:, column], oof[:, column] >= candidate)
        if score > best_score:
            best, best_score = candidate, score
    thresholds[label] = round(float(best), 2)
    print(f"{label:10s} ambang {best:.2f} | F1 {best_score:.4f}")
limits = np.array(list(thresholds.values()))
oof_pred = (oof >= limits).astype(int)
if AT_LEAST_ONE:
    empty = oof_pred.sum(axis=1) == 0
    oof_pred[empty, oof[empty].argmax(axis=1)] = 1
print("F1 micro:", round(f1_score(Y_true, oof_pred, average="micro"), 4),
      "| F1 macro:", round(f1_score(Y_true, oof_pred, average="macro"), 4),
      "| ambang 0.5 macro:", round(f1_score(Y_true, oof >= 0.5, average="macro"), 4))

# %% [markdown]
# ## 5. Latih di SEMUA train → `submission.csv`
# Bentuk A: satu kolom 0/1 per label (aktif). Bentuk B: satu kolom teks `"fuel,price"`.

# %%
import numpy as np
import pandas as pd

model.fit(X, Y_true)
test_proba = model.predict_proba(test[TEXT].to_numpy())
test_pred = (test_proba >= limits).astype(int)
if AT_LEAST_ONE:
    empty = test_pred.sum(axis=1) == 0
    test_pred[empty, test_proba[empty].argmax(axis=1)] = 1
submission = pd.DataFrame()
submission["id"] = test["id"]                     # UBAH
for column, label in enumerate(LABELS):
    submission[label] = test_pred[:, column]
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Bentuk B: satu kolom teks berisi nama label dipisah koma (kalau sample_submission begitu).

# %%
import pandas as pd

names = []
for row in test_pred:
    chosen = []
    for column, label in enumerate(LABELS):
        if row[column] == 1:
            chosen.append(label)
    names.append(",".join(chosen))   # UBAH: pemisah koma / spasi / titik koma
submission_text = pd.DataFrame()
submission_text["id"] = test["id"]
submission_text["labels"] = names
print(submission_text.head(3))
# submission_text.to_csv("submission.csv", index=False)   # aktifkan kalau bentuk B

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
from sklearn.metrics import f1_score

print("urutan id sama:", (test_labels["id"].to_numpy() == submission["id"].to_numpy()).all())
print("F1 macro test:", round(f1_score(test_labels[LABELS], submission[LABELS],
                                       average="macro"), 4))
