# %% [markdown]
# # Teks — Klasifikasi dengan Embedding Kalimat (pretrained, dibekukan)
# **Kapan dipakai:** TF-IDF kurang menangkap makna (sinonim, kalimat beda kata tapi maksud
# sama), tetapi tidak ada waktu/GPU untuk fine-tuning.
#
# **Langkah:** model pretrained mengubah tiap teks menjadi vektor angka (sekali saja) →
# classifier sklearn biasa → F1 macro.
#
# **Data demo:** SmSA (sentimen Bahasa Indonesia). Model embedding: MiniLM multibahasa lokal.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"

EMBEDDING_MODEL = "../../pretrained/minilm"
# EMBEDDING_MODEL = "../../pretrained/indobert"        # khusus Indonesia
# EMBEDDING_MODEL = "../../pretrained/distilbert_id"   # lebih ringan
# EMBEDDING_MODEL = "../../pretrained/minilm_en"       # teks bahasa Inggris

TEXT_COL = "text"
TARGET = "label"
BATCH_SIZE = 64
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## EDA: jumlah per kelas dan panjang teks

# %%
print(train[TARGET].value_counts())
word_counts = train[TEXT_COL].str.split().str.len()
word_counts.hist(bins=50)
plt.title("Jumlah kata per teks")
plt.show()

# %% [markdown]
# ## Embedding
# Teks diubah jadi vektor sekali di awal. Model embedding tidak dilatih ulang.

# %%
encoder = SentenceTransformer(EMBEDDING_MODEL)
train_vectors = encoder.encode(list(train[TEXT_COL]), batch_size=BATCH_SIZE,
                               show_progress_bar=True)
test_vectors = encoder.encode(list(test[TEXT_COL]), batch_size=BATCH_SIZE,
                              show_progress_bar=True)
print("train:", train_vectors.shape, "| test:", test_vectors.shape)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train_vectors, train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = LogisticRegression(C=1.0, max_iter=2000)                 # cepat, mudah dijelaskan
# model = LinearSVC(C=0.5)
# model = KNeighborsClassifier(n_neighbors=15, metric="cosine")
# model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=SEED)
# model = SVC(C=10, random_state=SEED)
# model = RidgeClassifier(alpha=1.0)
# model = LGBMClassifier(n_estimators=300, verbose=-1, random_state=SEED)

pipe = make_pipeline(StandardScaler(), model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
# soal 2 kelas: print("F1:", f1_score(y_val, val_predictions, pos_label="isi_kelas_positif"))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(train_vectors, train[TARGET])
predictions = pipe.predict(test_vectors)

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
