# %% [markdown]
# # Sinyal — Klasifikasi Sensor dengan Jendela Geser (statistik + FFT)
# **Kapan dipakai:** rekaman sensor panjang (akselerometer HP, getaran mesin, EEG) yang harus
# diberi label (aktivitas, kondisi mesin).
#
# **Idenya:** potong rekaman jadi **jendela** pendek yang saling tumpang tindih. Tiap jendela
# jadi satu baris feature (statistik + FFT) → LightGBM → **voting** label per segmen.
# Validasi harus **per subjek** (GroupKFold): jendela dari orang yang sama sangat mirip, jadi
# split acak membuat skor palsu tinggi.
#
# **Langkah:**
# 1. Potong rekaman jadi jendela.
# 2. Feature per jendela (statistik, frekuensi dominan, energi per pita).
# 3. Split per subjek, model, voting per segmen, submission.
#
# **Data demo:** WISDM v1.1 (akselerometer HP, 36 orang, 20 Hz, 6 aktivitas).

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/sinyal/wisdm"   # GANTI saat lomba
DATA_FOLDER = f"{DATA_FOLDER}/WISDM_ar_v1.1"
RAW_TXT = f"{DATA_FOLDER}/WISDM_ar_v1.1_raw.txt"

COLUMN_NAMES = ["user", "aktivitas", "waktu", "x", "y", "z"]
SUBJECT_COL = "user"          # orang/mesin pemilik rekaman (untuk split per grup)
LABEL_COL = "aktivitas"
AXIS_COLS = ["x", "y", "z"]
SAMPLING_HZ = 20              # frekuensi sampling (Hz)
WINDOW_SIZE = 100             # panjang jendela (sampel) = 5 detik
STEP = 50                     # geser tiap 50 sampel → tumpang tindih 50%
WINDOWS_PER_SEGMENT = 10      # 10 jendela berurutan = 1 segmen yang diberi satu label
BANDS_HZ = [(0, 1), (1, 3), (3, 6), (6, 10)]     # pita frekuensi untuk energi FFT
SEMICOLON = chr(59)           # tanda titik koma di akhir tiap baris WISDM
SEED = 42

# %% [markdown]
# ## Baca data

# %%
data = pd.read_csv(RAW_TXT, header=None, names=COLUMN_NAMES, on_bad_lines="skip")
data["z"] = data["z"].astype(str).str.replace(SEMICOLON, "", regex=False)
data["z"] = pd.to_numeric(data["z"], errors="coerce")
data = data.dropna()
print(data.shape, data[SUBJECT_COL].nunique(), "subjek")
data.head()

# %% [markdown]
# ## EDA: jumlah sampel per aktivitas dan contoh sinyal

# %%
print(data[LABEL_COL].value_counts())
fig, axes = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
for ax, activity in zip(axes, ["Walking", "Jogging", "Sitting"]):
    example = data[data[LABEL_COL] == activity]
    example = example.iloc[:200]
    for axis_name in AXIS_COLS:
        ax.plot(example[axis_name].to_numpy())
    ax.set_ylabel(activity)
plt.show()

# %% [markdown]
# ## Langkah 1 — potong rekaman jadi jendela
# Dipotong per subjek + aktivitas, supaya satu jendela tidak melintasi dua aktivitas.

# %%
windows = []       # tiap isi: array (WINDOW_SIZE, 3)
info_rows = []     # tiap isi: subjek, label, id segmen
for (subject, label), recording in data.groupby([SUBJECT_COL, LABEL_COL]):
    values = recording[AXIS_COLS].to_numpy()
    number = 0
    for start in range(0, len(values) - WINDOW_SIZE + 1, STEP):
        windows.append(values[start:start + WINDOW_SIZE])
        segment = f"{subject}_{label}_{number // WINDOWS_PER_SEGMENT}"
        info_rows.append([subject, label, segment])
        number = number + 1

info = pd.DataFrame(info_rows, columns=["subject", "label", "segment"])
print("jumlah jendela:", len(info), "| jumlah segmen:", info["segment"].nunique())

# %% [markdown]
# ## Langkah 2 — feature per jendela: statistik + FFT
# Tiap kanal (x, y, z, dan magnitudo gabungan) diringkas jadi angka statistik, frekuensi dominan,
# dan energi per pita frekuensi.

# %%
frequencies = np.fft.rfftfreq(WINDOW_SIZE, d=1 / SAMPLING_HZ)


def channel_features(name, values, features):
    features[name + "_mean"] = values.mean()
    features[name + "_std"] = values.std()
    features[name + "_min"] = values.min()
    features[name + "_max"] = values.max()
    features[name + "_energy"] = (values ** 2).mean()
    spectrum = np.abs(np.fft.rfft(values - values.mean()))
    features[name + "_dominant_hz"] = frequencies[spectrum.argmax()]
    for low, high in BANDS_HZ:
        in_band = (frequencies >= low) & (frequencies < high)
        features[f"{name}_band_{low}_{high}hz"] = (spectrum[in_band] ** 2).sum()


def window_features(window):
    features = {}
    channel_features("x", window[:, 0], features)
    channel_features("y", window[:, 1], features)
    channel_features("z", window[:, 2], features)
    channel_features("mag", np.sqrt((window ** 2).sum(axis=1)), features)
    features["corr_xy"] = np.corrcoef(window[:, 0], window[:, 1])[0, 1]
    features["corr_xz"] = np.corrcoef(window[:, 0], window[:, 2])[0, 1]
    return features


# coba pada jendela pertama
print(window_features(windows[0]))

# %%
feature_rows = []
for window in windows:
    feature_rows.append(window_features(window))
X = pd.DataFrame(feature_rows)
X = X.fillna(0)
y = info["label"]
print("feature:", X.shape)

# %% [markdown]
# ## Alternatif feature: MiniRocket (hapus `#`, menggantikan X di atas)

# %%
# rocket = MiniRocketMultivariate(num_kernels=1000, random_state=SEED)
# windows_3d = np.stack(windows).transpose(0, 2, 1)   # bentuk (jendela, sumbu, waktu)
# X = pd.DataFrame(rocket.fit_transform(windows_3d))

# %% [markdown]
# ## Langkah 3a — split per subjek
# Sebagian subjek jadi test (tidak pernah dilihat saat latih).

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_rows, test_rows = next(splitter.split(X, y, groups=info["subject"]))
X_train = X.iloc[train_rows]
y_train = y.iloc[train_rows]
X_test = X.iloc[test_rows]
train_subjects = info["subject"].iloc[train_rows]
test_subjects = info["subject"].iloc[test_rows]
print("subjek latih:", train_subjects.nunique(), "| subjek test:", test_subjects.nunique())

# %% [markdown]
# ## Langkah 3b — validasi: split acak (bocor) vs GroupKFold per subjek (jujur)

# %%
model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, random_state=SEED,
                       verbose=-1, n_jobs=4)

random_folds = StratifiedKFold(3, shuffle=True, random_state=SEED)
random_predictions = cross_val_predict(model, X_train, y_train, cv=random_folds)
group_predictions = cross_val_predict(model, X_train, y_train, cv=GroupKFold(3),
                                      groups=train_subjects)
random_f1 = f1_score(y_train, random_predictions, average="macro")
group_f1 = f1_score(y_train, group_predictions, average="macro")
print("F1 macro split acak (palsu tinggi):", round(random_f1, 4))
print("F1 macro GroupKFold subjek (jujur):", round(group_f1, 4))

# %% [markdown]
# ## Langkah 3c — prediksi per jendela, lalu voting per segmen + submission
# Label segmen = label yang paling sering ditebak di jendela-jendela segmen itu.

# %%
model.fit(X_train, y_train)
results = info.iloc[test_rows].copy()
results["prediction"] = model.predict(X_test)

votes = {}
for segment, group in results.groupby("segment"):
    votes[segment] = group["prediction"].value_counts().idxmax()
results["vote"] = results["segment"].map(votes)

submission = pd.DataFrame()
submission["segment"] = list(votes.keys())
submission["aktivitas"] = list(votes.values())
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
window_f1 = f1_score(results["label"], results["prediction"], average="macro")
vote_f1 = f1_score(results["label"], results["vote"], average="macro")
print("F1 macro per jendela      :", round(window_f1, 4))
print("F1 macro setelah voting   :", round(vote_f1, 4))
print(classification_report(results["label"], results["vote"]))
ConfusionMatrixDisplay.from_predictions(results["label"], results["vote"], xticks_rotation=45)
plt.show()
