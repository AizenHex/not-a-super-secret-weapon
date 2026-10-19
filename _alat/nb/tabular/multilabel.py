# %% [markdown]
# # Tabular — Multilabel
# **Kapan dipakai:** satu baris boleh punya BANYAK label sekaligus (bukan pilih satu).
# Contoh: satu film bisa "aksi" dan "komedi"; satu protein bisa punya beberapa fungsi.
# Bentuk data: satu kolom 0/1 per label.
#
# **Data demo:** Yeast (14 label `Class1`..`Class14`).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular_extra/multilabel"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/yeast_train.csv"
TEST_CSV = f"{DATA_FOLDER}/yeast_test.csv"

LABELS = ["Class1", "Class2", "Class3", "Class4", "Class5", "Class6", "Class7",
          "Class8", "Class9", "Class10", "Class11", "Class12", "Class13", "Class14"]
ID_COL = "id"
SEED = 42

# %% [markdown]
# ## Baca data
# `FEATURES` = semua kolom test kecuali kolom id.

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
FEATURES = test.columns.drop(ID_COL).tolist()
print("train:", train.shape, "| test:", test.shape, "|", len(FEATURES), "feature")
missing_per_column = train.isna().sum()
print("sel kosong:", missing_per_column.sum())
train.head()

# %% [markdown]
# ## EDA
# Label yang sangat jarang positif biasanya paling sulit ditebak.

# %%
train[LABELS].mean().plot.bar()
plt.title("Proporsi positif per label")
plt.show()

labels_per_row = train[LABELS].sum(axis=1)
print("rata-rata jumlah label per baris:", round(labels_per_row.mean(), 2))

# %% [markdown]
# ## Split train / validasi
# `y` di sini berupa tabel: satu kolom per label.

# %%
X = train[FEATURES]
y = train[LABELS]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# %% [markdown]
# ## Model
# - `MultiOutputClassifier(...)` = satu model terpisah untuk tiap label.
# - `ClassifierChain(...)` = tebakan label sebelumnya ikut dipakai sebagai feature.
# - RandomForest / ExtraTrees / KNN bisa langsung multilabel tanpa pembungkus.
#
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=2000))
# model = MultiOutputClassifier(LGBMClassifier(n_estimators=300, verbose=-1))
# model = ClassifierChain(LogisticRegression(C=1.0, max_iter=2000), random_state=SEED)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = KNeighborsClassifier(n_neighbors=15)
# model = MultiOutputClassifier(XGBClassifier(n_estimators=300, random_state=SEED))
# model = ClassifierChain(LGBMClassifier(n_estimators=300, verbose=-1), random_state=SEED)
# model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=SEED)
# model = MultiOutputClassifier(SVC(C=1.0, random_state=SEED))

pipe = make_pipeline(StandardScaler(), model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi
# - F1 micro: semua sel label dihitung bersama (label besar lebih berpengaruh).
# - F1 macro: rata-rata F1 per label (semua label sama penting).
# - Hamming loss: proporsi sel label yang salah (makin kecil makin bagus).

# %%
val_predictions = pipe.predict(X_val)
print("F1 micro    :", round(f1_score(y_val, val_predictions, average="micro"), 4))
print("F1 macro    :", round(f1_score(y_val, val_predictions, average="macro"), 4))
print("Hamming loss:", round(hamming_loss(y_val, val_predictions), 4))

f1_per_label = f1_score(y_val, val_predictions, average=None, zero_division=0)
pd.Series(f1_per_label, index=LABELS).round(3)

# %% [markdown]
# ## Latih ulang di semua data + submission
# `predictions[:, 0]` = kolom tebakan label pertama, `[:, 1]` = label kedua, dst.

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[FEATURES])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["Class1"] = predictions[:, 0]
submission["Class2"] = predictions[:, 1]
submission["Class3"] = predictions[:, 2]
submission["Class4"] = predictions[:, 3]
submission["Class5"] = predictions[:, 4]
submission["Class6"] = predictions[:, 5]
submission["Class7"] = predictions[:, 6]
submission["Class8"] = predictions[:, 7]
submission["Class9"] = predictions[:, 8]
submission["Class10"] = predictions[:, 9]
submission["Class11"] = predictions[:, 10]
submission["Class12"] = predictions[:, 11]
submission["Class13"] = predictions[:, 12]
submission["Class14"] = predictions[:, 13]
submission.to_csv("submission.csv", index=False)
submission.head()
