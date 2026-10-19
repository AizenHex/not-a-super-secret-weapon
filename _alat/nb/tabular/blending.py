# %% [markdown]
# # Tabular — Blending OOF dengan Hill Climbing
# **Kapan dipakai:** sudah melatih beberapa model, masing-masing menyimpan probabilitas OOF
# (out-of-fold) dan probabilitas test. Kita mencari campuran model terbaik, seperti tim
# juara GEMASTIK 2024.
#
# **Syarat:** jalankan `tabular/survei_kotor` beberapa kali dengan model berbeda (pindahkan `#`
# di sel Model). Tiap kali jalan, notebook itu menyimpan `oof_<Model>.npy` dan
# `test_<Model>.npy` di folder notebook. File OOF dari notebook lain juga bisa, asal urutan
# baris dan urutan kelasnya sama.
#
# **Hill climbing (Caruana):** mulai dari model terbaik, lalu tiap langkah tambahkan satu model
# (boleh model yang sama lagi) yang paling menaikkan F1 macro OOF. Berhenti kalau tidak naik.

# %%
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/survei_kotor"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_train.csv"
TEST_CSV = f"{DATA_FOLDER}/ukt_SINTETIS_test.csv"
OOF_FOLDER = "."          # folder berisi oof_*.npy dan test_*.npy ("." = folder notebook ini)

TARGET = "kelas_ukt"
MODEL_NAMES = ["LGBMClassifier", "CatBoostClassifier", "RandomForestClassifier"]
MAX_STEPS = 30            # paling banyak 30 kali menambah model

# %% [markdown]
# ## Baca data dan probabilitas tiap model
# `oof[name]` = tabel probabilitas: satu baris per baris train, satu kolom per kelas.
# Tebakan kelas = kolom dengan probabilitas terbesar (`argmax`).

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
y = train[TARGET]
CLASSES = np.sort(y.unique())

oof = {}
test_proba = {}
for name in MODEL_NAMES:
    oof[name] = np.load(f"{OOF_FOLDER}/oof_{name}.npy")
    test_proba[name] = np.load(f"{OOF_FOLDER}/test_{name}.npy")
    guess = CLASSES[oof[name].argmax(axis=1)]
    print(name, "F1 macro OOF:", round(f1_score(y, guess, average="macro"), 4))

# %% [markdown]
# ## Hill climbing
# Hasil: berapa bagian tiap model di dalam campuran (bobot, jumlahnya 1).

# %%
def hill_climbing(oof, y_true):
    chosen = []
    total = 0
    best_score = 0
    for step in range(MAX_STEPS):
        step_best_name = None
        step_best_score = best_score
        for name in oof:
            average = (total + oof[name]) / (len(chosen) + 1)
            score = f1_score(y_true, CLASSES[average.argmax(axis=1)], average="macro")
            if score > step_best_score:
                step_best_name = name
                step_best_score = score
        if step_best_name is None:
            break
        chosen.append(step_best_name)
        total = total + oof[step_best_name]
        best_score = step_best_score
        print("langkah", step, ": tambah", step_best_name, "->", round(best_score, 4))
    return pd.Series(chosen).value_counts(normalize=True)


model_weights = hill_climbing(oof, y)
print(model_weights)

# %% [markdown]
# ## Campurkan probabilitas sesuai bobot

# %%
blend_oof = 0
blend_test = 0
for name in model_weights.index:
    blend_oof = blend_oof + model_weights[name] * oof[name]
    blend_test = blend_test + model_weights[name] * test_proba[name]

guess = CLASSES[blend_oof.argmax(axis=1)]
print("F1 macro OOF blend:", round(f1_score(y, guess, average="macro"), 4))

# %% [markdown]
# ## Pengali probabilitas per kelas (sama dengan `survei_kotor`)
# Kelas kecil sering kalah oleh kelas besar. Probabilitas tiap kelas dikali angka 0.5–3.0;
# pengali dicari satu kelas per satu kelas, dipilih yang F1 macro OOF-nya tertinggi.

# %%
def find_class_weights(proba, y_true):
    weights = np.ones(proba.shape[1])
    for round_number in range(2):
        for class_index in range(len(weights)):
            best_score = 0
            for value in np.arange(0.5, 3.01, 0.1):
                trial = weights.copy()
                trial[class_index] = value
                guess = CLASSES[(proba * trial).argmax(axis=1)]
                score = f1_score(y_true, guess, average="macro")
                if score > best_score:
                    best_score = score
                    weights[class_index] = value
    return weights


class_weights = find_class_weights(blend_oof, y)
guess = CLASSES[(blend_oof * class_weights).argmax(axis=1)]
print("pengali per kelas:", class_weights.round(2))
print("F1 macro OOF blend + pengali:", round(f1_score(y, guess, average="macro"), 4))

# %% [markdown]
# ## Submission

# %%
predictions = CLASSES[(blend_test * class_weights).argmax(axis=1)]

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["kelas_ukt"] = predictions
submission.to_csv("submission.csv", index=False)
submission["kelas_ukt"].value_counts()
