# %% [markdown]
# # Tabular — Tuning Hyperparameter (Klasifikasi, Optuna)
# **Kapan dipakai:** model sudah dipilih, dan kita ingin mencari setelan (hyperparameter)
# yang lebih baik, misalnya jumlah pohon dan learning rate.
#
# **Cara kerja Optuna:** coba `N_TRIALS` kombinasi setelan. Tiap kombinasi dinilai dengan
# F1 macro cross validation. Kombinasi berikutnya dipilih dekat kombinasi yang sudah bagus.
#
# **Data demo:** Titanic (target `survived`).

# %%
import joblib
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)   # sembunyikan log tiap percobaan

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/tabular/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/titanic_train.csv"
TEST_CSV = f"{DATA_FOLDER}/titanic_test.csv"

TARGET = "survived"
NUM_COLS = ["pclass", "age", "sibsp", "parch", "fare"]
CAT_COLS = ["sex", "embarked", "who", "deck"]
N_TRIALS = 30          # jumlah kombinasi yang dicoba (makin banyak makin lama)
SEED = 42

# %% [markdown]
# ## Baca data, split, siapkan kolom

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)

X = train[NUM_COLS + CAT_COLS]
y = train[TARGET]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

num_steps = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
cat_steps = make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore"))
preprocess = ColumnTransformer([
    ("num", num_steps, NUM_COLS),
    ("cat", cat_steps, CAT_COLS),
])

# %% [markdown]
# ## Model + ruang pencarian
# `make_model(trial)` membuat satu model dengan setelan yang dipilih Optuna.
# `trial.suggest_int("nama", bawah, atas)` = Optuna memilih angka bulat di rentang itu.
# Ganti model: pindahkan `#` sehingga hanya satu `return` yang aktif.

# %%
def make_model(trial):
    # return LogisticRegression(C=trial.suggest_float("C", 1e-3, 100, log=True), max_iter=3000)
    # return RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 800),
    #     max_depth=trial.suggest_int("max_depth", 3, 20), random_state=SEED)
    # return XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 100, 800),
    #     max_depth=trial.suggest_int("max_depth", 2, 8),
    #     learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True))
    # return CatBoostClassifier(iterations=trial.suggest_int("iterations", 200, 1000),
    #     depth=trial.suggest_int("depth", 4, 10), verbose=0, allow_writing_files=False)
    return LGBMClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 800),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 4, 64),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        random_state=SEED, verbose=-1,
    )


def objective(trial):
    pipe = make_pipeline(preprocess, make_model(trial))
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro")
    return scores.mean()


sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS)
print("CV F1 terbaik:", round(study.best_value, 4))
print(study.best_params)

# %% [markdown]
# ## Evaluasi setelan terbaik di validasi

# %%
pipe = make_pipeline(preprocess, make_model(study.best_trial))
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("F1 macro validasi:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Alternatif tanpa Optuna: GridSearchCV
# Mencoba SEMUA kombinasi di daftar `grid`. Cocok kalau kombinasinya sedikit.
# Untuk memakai: hapus `#` di semua baris sel ini. Nama parameter = `namamodel__parameter`.

# %%
# grid = {
#     "lgbmclassifier__n_estimators": [200, 400],
#     "lgbmclassifier__learning_rate": [0.03, 0.1],
#     "lgbmclassifier__num_leaves": [8, 16, 32],
# }
# pipe_grid = make_pipeline(preprocess, LGBMClassifier(random_state=SEED, verbose=-1))
# search = GridSearchCV(pipe_grid, grid, cv=5, scoring="f1_macro")
# search.fit(X_train, y_train)
# print("CV F1 terbaik:", search.best_score_, search.best_params_)
# pipe = search.best_estimator_

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

# %% [markdown]
# ## Simpan model (opsional)
# Supaya bisa dipakai ulang tanpa melatih lagi: `pipe = joblib.load("model.joblib")`.

# %%
joblib.dump(pipe, "model.joblib")
