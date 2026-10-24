# %% [markdown]
# # Tabular — Klasifikasi Multikelas
# **Kapan dipakai:** seperti klasifikasi biasa, tetapi labelnya lebih dari dua kelas
# dan semua feature berupa angka.
#
# **Data demo:** Covertype (target `Cover_Type`, 7 jenis tutupan hutan).
# Semua kolom selain `id` dipakai sebagai feature.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra/multiclass"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/covertype_train.csv"
TEST_CSV = f"{DATA_FOLDER}/covertype_test.csv"

TARGET = "Cover_Type"
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
# Jumlah baris per kelas: kelas kecil biasanya paling sulit ditebak.

# %%
train[TARGET].value_counts().plot.bar()
plt.title("Jumlah per kelas")
plt.show()
train.groupby(TARGET)["Elevation"].mean()

# %% [markdown]
# ## Split train / validasi

# %%
X = train[FEATURES]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print("train:", X_train.shape, "| validasi:", X_val.shape)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=SEED, verbose=-1)
# model = RandomForestClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = ExtraTreesClassifier(n_estimators=500, random_state=SEED, n_jobs=-1)
# model = HistGradientBoostingClassifier(random_state=SEED)
# model = LogisticRegression(C=1.0, max_iter=3000)
# model = KNeighborsClassifier(n_neighbors=5)
# model = SVC(C=1.0, random_state=SEED)
# model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=SEED)
# model = GaussianNB()
# model = LinearDiscriminantAnalysis()
# model = DecisionTreeClassifier(random_state=SEED)
# model = AdaBoostClassifier(n_estimators=300, random_state=SEED)

pipe = make_pipeline(StandardScaler(), model)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))
ConfusionMatrixDisplay.from_predictions(y_val, val_predictions, xticks_rotation=45)
plt.show()

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[FEATURES])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["Cover_Type"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
