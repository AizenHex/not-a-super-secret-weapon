# %% [markdown]
# # Teks — Analisis Kesalahan Prediksi + Penanda Otomatis dengan Laya
# **Kapan dipakai:** SETELAH model jadi, untuk mencari pola kesalahan (bahan perbaikan dan
# presentasi).
#
# **Langkah:**
# 1. Prediksi OOF 5-fold → semua teks train punya prediksi "jujur".
# 2. Kesalahan dikelompokkan per kelas tertukar, panjang teks, kata negasi/kontras.
# 3. Tanda dari Laya (model keputusan zero-shot): sarkas? campur pujian-keluhan? topik apa?
#
# Tanda Laya tidak selalu benar (lihat README-nya), jadi dipakai untuk MENCARI POLA, bukan
# sebagai feature pasti. Selalu baca beberapa contoh sebelum menyimpulkan.
#
# **Data demo:** SmSA (sentimen). Model: TF-IDF + LogisticRegression.

# %%
import laya
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi
# `QUESTIONS`: pertanyaan untuk Laya, ditulis bebas. Tipe `choice` dengan kunci netral (A/B)
# lebih stabil daripada `noul` (ya/tidak) menurut README Laya.

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/smsa_train.csv"
LAYA_FOLDER = "../../pretrained/laya"

TEXT_COL = "text"
TARGET = "label"
N_FLAGGED = 300      # jumlah teks salah dan benar yang diberi tanda Laya (±0.2 detik per teks)
SEED = 42
QUESTIONS = {
    "sarkas": {"type": "choice", "instructions": "Apakah teks ini sindiran atau sarkasme?",
               "criteria": {"A": "ya, menyindir", "B": "tidak, makna apa adanya"}},
    "campur": {"type": "choice", "instructions": "Apakah teks memuji dan mengeluh sekaligus?",
               "criteria": {"A": "ya, ada pujian dan keluhan", "B": "tidak, satu nada saja"}},
    "topik": {"type": "choice", "instructions": "Teks ini membahas apa?",
              "criteria": {"makanan": "restoran, makanan, minuman", "politik": "politik, tokoh",
                           "layanan": "hotel, toko, aplikasi, pelayanan", "lain": "lainnya"}},
}

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(DATA_CSV)
data[TEXT_COL] = data[TEXT_COL].fillna("")
print(data.shape)
print(data[TARGET].value_counts())
data.head()

# %% [markdown]
# ## Model + prediksi OOF 5-fold
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = LogisticRegression(C=3.0, max_iter=2000)
# model = LinearSVC(C=0.5)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
pipe = make_pipeline(vectorizer, model)
folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
data["prediksi"] = cross_val_predict(pipe, data[TEXT_COL], data[TARGET], cv=folds)
data["salah"] = data["prediksi"] != data[TARGET]
print("persen salah:", round(data["salah"].mean(), 4))

# %% [markdown]
# ## 1. Kelas mana yang tertukar?

# %%
print(classification_report(data[TARGET], data["prediksi"], digits=3))
ConfusionMatrixDisplay.from_predictions(data[TARGET], data["prediksi"], normalize="true",
                                        values_format=".2f")
plt.show()

# %% [markdown]
# ## 2. Ciri sederhana: panjang teks, kata negasi, kata kontras
# Persen salah per kelompok. Kelompok yang persen salahnya jauh lebih tinggi = titik lemah model.

# %%
data["jumlah_kata"] = data[TEXT_COL].str.split().str.len()
data["ada_negasi"] = data[TEXT_COL].str.contains(r"\b(?:tidak|bukan|gak|nggak|tak|belum)\b")
data["ada_kontras"] = data[TEXT_COL].str.contains(r"\b(?:tapi|tetapi|namun|walaupun|meski)\b")

length_group = pd.qcut(data["jumlah_kata"], 4)
error_by_length = data.groupby(length_group, observed=True)["salah"].mean()
print(error_by_length.round(3), "\n")
for col in ["ada_negasi", "ada_kontras"]:
    error_by_flag = data.groupby(col)["salah"].agg(["mean", "size"])
    print(error_by_flag.round(3), "\n")

# %% [markdown]
# ## 3. Tanda dari Laya untuk sampel teks salah dan benar

# %%
agent = laya.load(LAYA_FOLDER, subfolder="multilingual")
wrong_sample = data[data["salah"]].sample(N_FLAGGED, random_state=SEED)
right_sample = data[~data["salah"]].sample(N_FLAGGED, random_state=SEED)
sample = pd.concat([wrong_sample, right_sample])

for name in QUESTIONS:
    sample[name] = ""
for row_index, text in zip(sample.index, sample[TEXT_COL]):
    answers = agent.predict(text, QUESTIONS)["answers"]
    for name in QUESTIONS:
        sample.loc[row_index, name] = answers[name]["choice"]
sample[list(QUESTIONS)].describe()

# %% [markdown]
# ## 4. Tanda mana yang lebih sering salah?
# Persen salah di sampel = 50% bila tanda tidak berhubungan dengan kesalahan.

# %%
for name in QUESTIONS:
    error_by_answer = sample.groupby(name)["salah"].agg(["mean", "size"])
    print(error_by_answer.round(3), "\n")

# %% [markdown]
# ## 5. Baca contoh (wajib sebelum menyimpulkan)
# Contoh: teks yang salah DAN ditandai "campur" oleh Laya.

# %%
pd.set_option("display.max_colwidth", 150)
mixed_wrong = sample[sample["salah"] & (sample["campur"] == "A")]
mixed_wrong[[TEXT_COL, TARGET, "prediksi"]].head(8)

# %% [markdown]
# ## Simpan tabel kesalahan (bahan laporan / cek manual)
# Langkah lanjut umum: tambah n-gram negasi ("tidak_enak"), model kontekstual (IndoBERT) untuk
# teks campur/sarkas, periksa ulang label teks yang salah dengan nada jelas.

# %%
error_table = pd.DataFrame()
error_table["id"] = data["id"]
error_table["label"] = data[TARGET]
error_table["prediksi"] = data["prediksi"]
error_table["salah"] = data["salah"]
error_table["jumlah_kata"] = data["jumlah_kata"]
error_table.to_csv("analisis_kesalahan.csv", index=False)
sample.drop(columns=[TEXT_COL]).to_csv("tanda_laya.csv", index=False)
error_table[error_table["salah"]].head()
