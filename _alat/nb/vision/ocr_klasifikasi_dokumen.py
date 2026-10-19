# %% [markdown]
# # Vision — OCR → Klasifikasi Dokumen (EasyOCR + TF-IDF)
# **Kapan dipakai:** gambar dokumen (struk, surat, kwitansi) harus dikelompokkan menurut jenisnya
# dan isi tulisannya membedakan jenis.
#
# **Langkah:** gambar dibaca jadi teks (EasyOCR) → teks diklasifikasi seperti notebook teks
# biasa (TF-IDF + model linear).
#
# **Data demo:** 200 gambar dokumen SINTETIS, 4 jenis (`struk`, `surat_resmi`, `undangan`,
# `kwitansi`). Alternatif tanpa OCR (tampilan dokumen saja): `vision/klasifikasi_embedding`.

# %%
import easyocr
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/vision/ocr"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/dokumen_train.csv"
TEST_CSV = f"{DATA_FOLDER}/dokumen_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/dokumen_test_labels.csv"   # hanya ada di data latihan
EASYOCR_FOLDER = "Z:/GEMASTIK-DATABASE/pretrained/easyocr"

LANGUAGES = ["en", "id"]
PATH_COL = "path"     # path gambar, relatif terhadap DATA_FOLDER
TARGET = "jenis"
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## OCR semua gambar → kolom `teks`

# %%
reader = easyocr.Reader(LANGUAGES, gpu=torch.cuda.is_available(),
                        model_storage_directory=EASYOCR_FOLDER,
                        user_network_directory=EASYOCR_FOLDER,
                        download_enabled=False, verbose=False)


def read_text(path):
    pieces = reader.readtext(f"{DATA_FOLDER}/{path}", detail=0)   # daftar potongan teks
    return " ".join(pieces)


print(read_text(train.loc[0, PATH_COL])[:200])

train["teks"] = train[PATH_COL].apply(read_text)
test["teks"] = test[PATH_COL].apply(read_text)
train.to_csv("train_ocr.csv", index=False)
train[["teks", TARGET]].head()

# %% [markdown]
# ## EDA: panjang teks per jenis

# %%
train["panjang"] = train["teks"].str.split().str.len()
train.boxplot(column="panjang", by=TARGET, figsize=(7, 4))
plt.show()

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train["teks"], train[TARGET], test_size=0.25, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# `char_wb` (potongan huruf) lebih tahan salah baca OCR daripada kata utuh.
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)

model = LogisticRegression(C=10, max_iter=2000)
# model = LinearSVC(C=1.0)
# model = MultinomialNB(alpha=0.1)

pipe = make_pipeline(vectorizer, model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=30)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(train["teks"], train[TARGET])
predictions = pipe.predict(test["teks"])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["jenis"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["jenis"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
