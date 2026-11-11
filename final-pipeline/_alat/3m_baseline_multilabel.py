# %% [markdown]
# # 3m. Baseline MULTILABEL: TF-IDF + satu model per label, k-fold + OOF + ambang per label
# **Kapan dipakai:** satu teks boleh punya beberapa label sekaligus; target = beberapa kolom 0/1
# (`LABEL_COLS` di `konfig.py`). Pengganti `3_baseline` untuk soal multilabel.
#
# **Alur:** baca `train_bersih.csv` (hasil `2_praproses`) → k-fold → probabilitas OOF tiap label →
# ambang per label dari OOF → simpan OOF → submission kolom 0/1 → validator (`MASALAH: 0`).
#
# Lipatan sama dengan `4m_finetune_multilabel` (`KFold` acak seed 42), jadi OOF keduanya bisa
# digabung di `5m_ensemble_multilabel`.

# %%
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
N_FOLDS = 5
SEED = 42                        # jangan diubah: lipatan harus sama dengan 4m_finetune_multilabel
MODEL_NAME = "ml_tfidf_logreg"   # GANTI tiap variasi (nama file OOF, jangan menimpa model lain)
MIN_ONE_LABEL = False            # True: tiap teks minimal 1 label (label berpeluang tertinggi)
SUBMISSION_NAME = "submission.csv"

# ---- final-pipeline: nilai soal dibaca dari konfig.py (ubah DI SANA, bukan di sini) ----
exec(open("konfig.py", encoding="utf-8").read())
if LABEL_COLS is None:
    raise ValueError("Isi LABEL_COLS di sel Konfigurasi (daftar kolom label 0/1) untuk soal multilabel")
TRAIN_CSV = f"{HASIL}/praproses/train_bersih.csv"
TEST_CSV = f"{HASIL}/praproses/test_bersih.csv"
OUTPUT_FOLDER = f"{HASIL}/baseline_multilabel"
OOF_FOLDER = f"{HASIL}/oof_multilabel"
SAMPLE_CSV = f"{OUTPUT_FOLDER}/sample_submission_demo.csv"
if SAMPLE_PANITIA is not None:
    SAMPLE_CSV = SAMPLE_PANITIA
if TFIDF_BERAT:
    heavy_cols = []
    for column in TEXT_COLS:
        heavy_cols.append(column + "_tfidf")
    TEXT_COLS = heavy_cols

# %% [markdown]
# ## Metrik
# Bawaan F1 makro antarlabel (rata-rata F1 tiap kolom label). Samakan dengan `4m` dan `5m`.

# %%
METRIC_NAME = "f1_macro"         # GANTI saat lomba, ikut baris return di bawah


def score(truth, predicted):
    return f1_score(truth, predicted, average="macro", zero_division=0)
    # return f1_score(truth, predicted, average="micro", zero_division=0)
    # return f1_score(truth, predicted, average="samples", zero_division=0)


# %% [markdown]
# ## Baca data

# %%
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OOF_FOLDER, exist_ok=True)
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)


def join_text(table):
    joined = table[TEXT_COLS[0]].fillna("").astype(str)
    for column in TEXT_COLS[1:]:
        joined = joined + " " + table[column].fillna("").astype(str)
    return joined


train_text = join_text(train)
test_text = join_text(test)
label_matrix = train[LABEL_COLS].to_numpy().astype("float32")
print("train:", train.shape, "| test:", test.shape, "| label:", LABEL_COLS)
print("nilai label selain 0/1:", int((~np.isin(label_matrix, [0, 1])).sum()), "(harus 0)")
print("positif per label:", dict(zip(LABEL_COLS, label_matrix.sum(axis=0).astype(int))))
print("label per teks:", pd.Series(label_matrix.sum(axis=1)).value_counts().sort_index().to_dict())

# %% [markdown]
# ## Sample submission
# Bila `SAMPLE_PANITIA` belum diisi, dibuat sample demo (id + satu kolom per label) supaya notebook
# tetap jalan. Format panitia baru benar-benar dicek sesudah `SAMPLE_PANITIA` diisi.

# %%
demo_sample = pd.DataFrame()
demo_sample[ID_COL] = test[ID_COL]
for column in LABEL_COLS:
    demo_sample[column] = 0
if SAMPLE_CSV.startswith(OUTPUT_FOLDER):
    demo_sample.to_csv(SAMPLE_CSV, index=False)
    print("sample demo ditulis ke", SAMPLE_CSV)
sample = pd.read_csv(SAMPLE_CSV)
print("sample_submission:", sample.shape, sample.columns.tolist())
print("id test = id sample:", set(test[ID_COL]) == set(sample[ID_COL]))

# %% [markdown]
# ## Model
# Satu model terpisah per label (`OneVsRestClassifier`). Ganti model = pindahkan `#`.

# %%
FEATURES = make_union(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
)
# FEATURES = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)    # kata saja, cepat

CLASSIFIER = OneVsRestClassifier(LogisticRegression(C=5, class_weight="balanced", max_iter=2000))
# CLASSIFIER = OneVsRestClassifier(CalibratedClassifierCV(
#     LinearSVC(C=0.5, class_weight="balanced"), cv=3))     # SVM + proba, ganti MODEL_NAME
PIPE = make_pipeline(FEATURES, CLASSIFIER)

# %% [markdown]
# ## Latih k-fold
# OOF = probabilitas tiap label untuk baris yang tidak ikut dilatih. Prediksi test memakai model
# yang dilatih ulang di seluruh train.

# %%
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(train_text))
oof = np.zeros(label_matrix.shape, dtype="float32")
began = time.time()
for number, (fit_index, val_index) in enumerate(folds):
    model = clone(PIPE).fit(train_text.iloc[fit_index], label_matrix[fit_index])
    oof[val_index] = model.predict_proba(train_text.iloc[val_index])
    fold_value = score(label_matrix[val_index], oof[val_index] >= 0.5)
    print(f"fold {number}: {METRIC_NAME}@0.5 {fold_value:.4f}")
final_model = clone(PIPE).fit(train_text, label_matrix)
test_proba = final_model.predict_proba(test_text).astype("float32")
total_seconds = round(time.time() - began, 1)
print(f"total {total_seconds:.0f} detik")

# %% [markdown]
# ## Ambang per label dari OOF
# Tiap label dicari ambang 0.05..0.95 dengan F1 OOF tertinggi (0.5 jarang optimal untuk label
# langka). Label dengan positif sangat sedikit: ambang bisa melompat, periksa tabelnya.

# %%
grid = np.arange(0.05, 0.96, 0.05)
thresholds = np.full(len(LABEL_COLS), 0.5)
label_rows = []
for column in range(len(LABEL_COLS)):
    best_f1 = -1.0
    for cut in grid:
        value = f1_score(label_matrix[:, column], oof[:, column] >= cut, zero_division=0)
        if value > best_f1:
            best_f1, thresholds[column] = value, cut
    label_rows.append({"label": LABEL_COLS[column], "positif": int(label_matrix[:, column].sum()),
                       "ambang": round(thresholds[column], 2), "F1 OOF": round(best_f1, 4)})


def to_labels(proba):
    labels = (proba >= thresholds).astype(int)
    if MIN_ONE_LABEL:
        empty = labels.sum(axis=1) == 0
        labels[empty, proba[empty].argmax(axis=1)] = 1
    return labels


oof_score = score(label_matrix, to_labels(oof))
plain_score = score(label_matrix, oof >= 0.5)
print(f"{METRIC_NAME} OOF @0.5 {plain_score:.4f} | ambang per label {oof_score:.4f}")
pd.DataFrame(label_rows)

# %% [markdown]
# ## Simpan OOF dan catat percobaan

# %%
np.save(f"{OOF_FOLDER}/oof_{MODEL_NAME}.npy", oof)
np.save(f"{OOF_FOLDER}/test_{MODEL_NAME}.npy", test_proba)
row = {"waktu": time.strftime("%Y-%m-%d %H:%M:%S"), "notebook": "3m_baseline_multilabel",
       "model": MODEL_NAME, "metrik": METRIC_NAME, "skor_cv": round(oof_score, 4),
       "detik": total_seconds, "catatan": f"TF-IDF OneVsRest | KFold {N_FOLDS} | @0.5 {plain_score:.4f}"}
log_path = f"{OOF_FOLDER}/percobaan.csv"
pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)
print(pd.read_csv(log_path).tail(3).to_string())

# %% [markdown]
# ## Submission
# Satu kolom 0/1 per label; urutan kolom dan id mengikuti sample.

# %%
test_labels = pd.DataFrame(to_labels(test_proba), columns=LABEL_COLS)
test_labels.index = test[ID_COL].to_numpy()

submission = pd.DataFrame()
submission[ID_COL] = sample[ID_COL]
for column in sample.columns:
    if column != ID_COL:
        submission[column] = sample[ID_COL].map(test_labels[column]).to_numpy()
submission.to_csv(f"{OUTPUT_FOLDER}/{SUBMISSION_NAME}", index=False)
print("rata-rata label per teks: test", round(submission[LABEL_COLS].sum(axis=1).mean(), 2),
      "| train", round(label_matrix.sum(axis=1).mean(), 2))

# %% [markdown]
# ## Cek submission terhadap sample
# Harus `MASALAH: 0`.

# %%
written = pd.read_csv(f"{OUTPUT_FOLDER}/{SUBMISSION_NAME}")
problems = []
if written.columns.tolist() != sample.columns.tolist():
    problems.append("kolom / urutan kolom beda")
if written[ID_COL].tolist() != sample[ID_COL].tolist():
    problems.append("id / urutan id beda dari sample")
for column in LABEL_COLS:
    if written[column].isna().any():
        problems.append(f"{column}: ada nilai kosong")
    if not set(written[column].unique()) <= {0, 1}:
        problems.append(f"{column}: nilai selain 0/1")
for problem in problems:
    print("-", problem)
if SAMPLE_CSV.startswith(OUTPUT_FOLDER):
    print("PERINGATAN: SAMPLE_CSV masih file demo, format panitia belum dicek")
print("MASALAH:", len(problems))
