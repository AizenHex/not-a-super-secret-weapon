# %% [markdown]
# # Teks — Label Berisik (cari label yang salah dengan cleanlab)
# **Kapan dipakai:** label train dibuat manusia/crowd dan curiga sebagian salah (skor validasi
# mentok, contoh "salah prediksi" justru terlihat benar).
#
# **Langkah:**
# 1. Probabilitas out-of-fold (OOF).
# 2. cleanlab menandai baris yang labelnya tidak cocok dengan tebakan model yang yakin.
# 3. Lihat contohnya → buang (atau perbaiki) → latih ulang.
#
# **Data demo:** SmSA. Label train SENGAJA dirusak (40% teks negatif/netral ditulis "positive",
# seperti anotator malas) supaya deteksinya bisa diukur. Saat lomba, hapus sel "Rusak label".

# %%
import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/smsa_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "text"
TARGET = "label"
BROKEN_SHARE = 0.40       # hanya untuk demo
WRONG_CLASS = "positive"  # demo: label rusak selalu ditulis kelas ini
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Rusak label (HANYA DEMO — hapus sel ini saat lomba)
# `is_broken` = penanda baris yang dirusak (kunci untuk mengukur deteksi).

# %%
random_generator = np.random.default_rng(SEED)
true_labels = train[TARGET].copy()
is_broken = (true_labels != WRONG_CLASS) & (random_generator.random(len(train)) < BROKEN_SHARE)
train.loc[is_broken, TARGET] = WRONG_CLASS
print("label dirusak:", is_broken.sum())

# %% [markdown]
# ## Probabilitas out-of-fold
# Tiap baris ditebak oleh model yang TIDAK melihat baris itu saat latih (syarat cleanlab).
# cleanlab butuh label berupa nomor 0, 1, 2 sesuai urutan `classes`.

# %%
classes = sorted(train[TARGET].unique())
y = train[TARGET].map(classes.index).to_numpy()

pipe = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000),
)
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
oof_proba = cross_val_predict(pipe, train[TEXT_COL], y, cv=folds, method="predict_proba")
print(oof_proba.shape)

# %% [markdown]
# ## Cari label bermasalah
# Hasilnya nomor baris, diurutkan dari yang paling mencurigakan.
# Pilih SATU cara mengurutkan: hapus `#` di baris yang ingin dipakai.

# %%
issue_rows = find_label_issues(labels=y, pred_probs=oof_proba,
                               return_indices_ranked_by="self_confidence", n_jobs=1)
# issue_rows = find_label_issues(labels=y, pred_probs=oof_proba,
#                                return_indices_ranked_by="normalized_margin", n_jobs=1)
print("dicurigai salah label:", len(issue_rows))

examples = train.loc[issue_rows[:10], [TEXT_COL, TARGET]].copy()
examples["tebakan_model"] = ""
for row in issue_rows[:10]:
    examples.loc[row, "tebakan_model"] = classes[oof_proba[row].argmax()]
pd.set_option("display.max_colwidth", 120)
examples

# %% [markdown]
# ## Seberapa tepat deteksinya? (cek demo — hapus saat lomba)

# %%
is_flagged = np.zeros(len(train), dtype=bool)
is_flagged[issue_rows] = True
print("precision deteksi:", round(precision_score(is_broken, is_flagged), 3))
print("recall deteksi   :", round(recall_score(is_broken, is_flagged), 3))

# %% [markdown]
# ## Bersihkan + latih ulang
# Pilih SATU: buang baris bermasalah (aktif), atau ganti labelnya dengan tebakan model
# (hapus `#` di 3 baris bawah, beri `#` pada baris `drop`).

# %%
clean_train = train.drop(index=issue_rows)
# clean_train = train.copy()
# for row in issue_rows:
#     clean_train.loc[row, TARGET] = classes[oof_proba[row].argmax()]
print("baris latih:", len(train), "->", len(clean_train))

pipe.fit(clean_train[TEXT_COL], clean_train[TARGET])
predictions = pipe.predict(test[TEXT_COL])

# %% [markdown]
# ## Submission

# %%
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Pembanding: model yang sama tapi dilatih di label berisik.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
noisy_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                           LogisticRegression(max_iter=2000))
noisy_pipe.fit(train[TEXT_COL], train[TARGET])
noisy_predictions = noisy_pipe.predict(test[TEXT_COL])
print("F1 macro dilatih di label berisik:",
      round(f1_score(test_key, noisy_predictions, average="macro"), 4))
print("F1 macro setelah dibersihkan     :",
      round(f1_score(test_key, predictions, average="macro"), 4))
