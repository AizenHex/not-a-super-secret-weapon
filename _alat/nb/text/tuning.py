# %% [markdown]
# # Teks — Tuning Hyperparameter (TF-IDF + model linear, Optuna)
# **Kapan dipakai:** baseline TF-IDF sudah jalan dan ingin skor naik tanpa ganti metode.
#
# **Yang dicari:** pengaturan TF-IDF (n-gram kata/huruf, min_df, max_features) dan kekuatan
# model (C). Tiap kombinasi dinilai dengan 5-fold CV di data latih; data validasi hanya dipakai
# di akhir.
#
# **Data demo:** SmSA (sentimen 3 kelas).

# %%
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"

TEXT_COL = "text"
TARGET = "label"
N_TRIALS = 30
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGET], test_size=0.2, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Baseline (pengaturan bawaan) sebagai pembanding

# %%
baseline = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000))
baseline_scores = cross_val_score(baseline, X_train, y_train, cv=5, scoring="f1_macro")
print("CV F1 baseline:", round(baseline_scores.mean(), 4))

# %% [markdown]
# ## Ruang pencarian + Optuna
# `analyzer="char_wb"` = potongan huruf (tahan salah ketik/singkatan); `"word"` = kata utuh.
# Ganti model: beri `#` pada `return` yang aktif, hapus `#` pada `return` lain.

# %%
def make_pipe(trial):
    analyzer = trial.suggest_categorical("analyzer", ["word", "char_wb"])
    if analyzer == "word":
        ngram_max = trial.suggest_int("ngram_max_kata", 1, 3)
    else:
        ngram_max = trial.suggest_int("ngram_max_huruf", 3, 6)
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(1, ngram_max),
        min_df=trial.suggest_int("min_df", 1, 5),
        max_features=trial.suggest_categorical("max_features", [20000, 50000, 100000]),
        sublinear_tf=trial.suggest_categorical("sublinear_tf", [True, False]),
    )
    strength = trial.suggest_float("C", 0.01, 100, log=True)
    return make_pipeline(vectorizer, LogisticRegression(C=strength, max_iter=3000))
    # return make_pipeline(vectorizer, LinearSVC(C=strength))


def objective(trial):
    scores = cross_val_score(make_pipe(trial), X_train, y_train, cv=5, scoring="f1_macro")
    return scores.mean()


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
print("CV F1 terbaik:", round(study.best_value, 4), "| baseline:", round(baseline_scores.mean(), 4))
print(study.best_params)

# %% [markdown]
# ## 10 percobaan terbaik (lihat pola: analyzer apa yang menang?)

# %%
history = study.trials_dataframe()
history.sort_values("value", ascending=False).head(10)

# %% [markdown]
# ## Evaluasi di validasi (pengaturan terbaik)

# %%
pipe = make_pipe(study.best_trial)
pipe.fit(X_train, y_train)
val_predictions = pipe.predict(X_val)
print("F1 macro validasi:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
pipe.fit(train[TEXT_COL], train[TARGET])
predictions = pipe.predict(test[TEXT_COL])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()
