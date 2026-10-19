# %% [markdown]
# # Tabular — Ensembling (Klasifikasi)
# **Kapan dipakai:** sudah punya beberapa model yang lumayan, ingin menggabungkannya agar
# tebakan lebih stabil.
#
# **Dua cara menggabungkan:**
# - **Voting:** rata-rata probabilitas (soft) atau suara terbanyak (hard) dari semua model.
# - **Stacking:** model "meta" belajar dari tebakan model-model dasar (dibuat dengan CV
#   agar tidak bocor).
#
# **Data demo:** Titanic (target `survived`).

# %%
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_COLS = ["sex", "embarked", "who", "deck"]
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## Split train / validasi

# %%
X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# %% [markdown]
# ## Model
# `base_models` = daftar model dasar, tiap model diberi nama pendek.
# Pilih SATU `ensemble`: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

base_models = [
    ("logreg", LogisticRegression(max_iter=2000)),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=SEED)),
    ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.03, verbose=-1)),
    ("cat", CatBoostClassifier(iterations=300, random_seed=SEED, verbose=0,
                               allow_writing_files=False)),
]

ensemble = StackingClassifier(base_models, final_estimator=LogisticRegression(), cv=5)
# ensemble = VotingClassifier(base_models, voting="soft")      # rata-rata probabilitas
# ensemble = VotingClassifier(base_models, voting="hard")      # suara terbanyak
# ensemble = StackingClassifier(base_models, final_estimator=LGBMClassifier(verbose=-1), cv=5)
# ensemble = StackingClassifier(base_models, final_estimator=LogisticRegression(), cv=5,
#                               passthrough=True)              # meta juga melihat feature asli
# ensemble = VotingClassifier(base_models, voting="soft", weights=[1, 1, 2, 2])

pipe = make_pipeline(preprocess, ensemble)
pipe.fit(X_train, y_train)

# %% [markdown]
# ## Evaluasi di validasi

# %%
val_predictions = pipe.predict(X_val)
print("F1 macro:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(X, y)
predictions = pipe.predict(test[NUM_COLS + CAT_COLS])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["survived"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
