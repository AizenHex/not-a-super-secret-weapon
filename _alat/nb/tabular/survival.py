# %% [markdown]
# # Tabular — Analisis Survival
# **Kapan dipakai:** kita ingin menebak KAPAN sebuah kejadian (event) terjadi, misalnya
# pelanggan berhenti, mesin rusak, pasien kambuh. Sebagian data belum mengalami event saat
# pengamatan selesai (disebut "censored"), jadi regresi biasa tidak cocok.
#
# **Dua kolom wajib:** durasi (berapa lama diamati) dan event (1 = terjadi, 0 = belum).
#
# **Metrik:** C-index (1.0 = urutan risiko sempurna, 0.5 = acak).
# Submission = skor risiko (makin besar = makin cepat event).
#
# **Data demo:** Rossi (`week` = minggu sejak bebas penjara, `arrest` = ditangkap lagi 1/0).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, LogNormalAFTFitter, WeibullAFTFitter
from lifelines import LogLogisticAFTFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/tabular_extra/survival"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/rossi_train.csv"
TEST_CSV = f"{DATA_FOLDER}/rossi_test.csv"

DURATION = "week"      # kolom lama pengamatan
EVENT = "arrest"       # kolom event (1 = terjadi, 0 = belum)
FEATURES = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
GROUP_COL = "fin"      # kolom untuk membandingkan kurva Kaplan-Meier
SEED = 42

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| test:", test.shape)
missing_per_column = train.isna().sum()
print("sel kosong:", missing_per_column.sum())
train.head()

# %% [markdown]
# ## EDA — kurva Kaplan-Meier
# Sumbu y = peluang belum mengalami event sampai waktu x. Kurva yang turun lebih cepat =
# kelompok yang lebih cepat mengalami event.

# %%
print("proporsi event:", round(train[EVENT].mean(), 3))

kmf = KaplanMeierFitter()
for value, group in train.groupby(GROUP_COL):
    kmf.fit(group[DURATION], group[EVENT], label=f"{GROUP_COL}={value}")
    kmf.plot_survival_function()
plt.title(f"Kaplan-Meier per {GROUP_COL}")
plt.show()

# %% [markdown]
# ## Split train / validasi

# %%
train_part, val_part = train_test_split(train, test_size=0.25, random_state=SEED)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.
# - Cox: menebak risiko relatif.
# - AFT (Weibull, LogNormal, LogLogistic): menebak waktu sampai event.

# %%
model = CoxPHFitter(penalizer=0.1)
# model = WeibullAFTFitter(penalizer=0.1)
# model = LogNormalAFTFitter(penalizer=0.1)
# model = LogLogisticAFTFitter(penalizer=0.1)
# model = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
# model = CoxPHFitter(penalizer=0.01)

model.fit(train_part[FEATURES + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
model.print_summary(decimals=3)

# %% [markdown]
# ## Evaluasi di validasi
# Cox memberi risiko (`predict_partial_hazard`). AFT memberi waktu (`predict_median`), jadi
# risikonya = minus waktu. Kalau memakai model AFT, pindahkan `#` ke baris pertama.
# `concordance_index` meminta "makin besar = makin lama hidup", jadi risikonya diberi minus.

# %%
val_risk = model.predict_partial_hazard(val_part[FEATURES])
# val_risk = -model.predict_median(val_part[FEATURES])
c_index = concordance_index(val_part[DURATION], -val_risk, val_part[EVENT])
print("C-index:", round(c_index, 4))

# %% [markdown]
# ## Latih ulang di semua data + submission

# %%
model.fit(train[FEATURES + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
predictions = model.predict_partial_hazard(test[FEATURES])
# predictions = -model.predict_median(test[FEATURES])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["risk"] = predictions.values
submission.to_csv("submission.csv", index=False)
submission.head()
