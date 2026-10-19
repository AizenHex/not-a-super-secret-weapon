# %% [markdown]
# # Teks — Klasifikasi Multilabel (satu teks bisa punya beberapa label)
# **Kapan dipakai:** tiap teks boleh punya lebih dari satu label sekaligus, misalnya ulasan
# yang membahas harga DAN layanan. Target berupa beberapa kolom 0/1.
#
# **Langkah:** TF-IDF → satu classifier per label (OneVsRest) → F1 macro, Hamming loss.
#
# **Data demo:** CASA (ulasan mobil, 6 label: fuel, machine, others, part, price, service).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/multilabel"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/casa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/casa_test.csv"

TEXT_COL = "text"
TARGETS = ["fuel", "machine", "others", "part", "price", "service"]
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
# ## EDA: seberapa sering tiap label muncul

# %%
print(train[TARGETS].mean().round(3))
label_counts = train[TARGETS].sum()
label_counts.plot.bar(title="Jumlah contoh per label")
plt.show()

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGETS], test_size=0.2, random_state=SEED
)

# %% [markdown]
# ## Model
# `OneVsRestClassifier` = satu model terpisah untuk tiap label.
# `ClassifierChain` = model label berikutnya ikut melihat tebakan label sebelumnya.
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
# vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
#                              sublinear_tf=True)   # tahan salah ketik

model = OneVsRestClassifier(LogisticRegression(C=5, max_iter=2000))
# model = OneVsRestClassifier(LinearSVC(C=0.5))
# model = OneVsRestClassifier(ComplementNB(alpha=0.3))
# model = MultiOutputClassifier(LogisticRegression(C=5, max_iter=2000))
# model = ClassifierChain(LogisticRegression(C=5, max_iter=2000), random_state=SEED)
# model = OneVsRestClassifier(RidgeClassifier(alpha=1.0))
# model = OneVsRestClassifier(SGDClassifier(loss="modified_huber", random_state=SEED))

pipe = make_pipeline(vectorizer, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# - F1 macro: rata-rata F1 semua label.
# - Hamming loss: persen sel label yang salah (makin kecil makin baik).
# - Akurasi subset: persen teks yang SEMUA labelnya benar (paling ketat).

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print("Hamming loss:", round(hamming_loss(y_val, val_predictions), 4))
print("Akurasi subset:", round(accuracy_score(y_val, val_predictions), 4))
print(classification_report(y_val, val_predictions, target_names=TARGETS, zero_division=0))

# %% [markdown]
# ## Latih ulang di semua data + submission
# Satu kolom submission per label.

# %%
pipe.fit(train[TEXT_COL], train[TARGETS])
predictions = pipe.predict(test[TEXT_COL])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["fuel"] = predictions[:, 0]
submission["machine"] = predictions[:, 1]
submission["others"] = predictions[:, 2]
submission["part"] = predictions[:, 3]
submission["price"] = predictions[:, 4]
submission["service"] = predictions[:, 5]
submission.to_csv("submission.csv", index=False)
submission.head()
