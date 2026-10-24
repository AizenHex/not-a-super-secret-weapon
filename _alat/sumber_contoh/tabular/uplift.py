# %% [markdown]
# # Tabular — Uplift / Efek Perlakuan (siapa yang perlu diberi promo?)
# **Kapan dipakai:** ada perlakuan (promo, email, bantuan) yang diberikan secara acak ke
# sebagian orang. Pertanyaannya bukan "siapa yang akan membeli", tetapi "siapa yang
# **berubah** karena diberi perlakuan".
#
# **Cara (T-learner):**
# 1. Satu model belajar dari grup yang diberi perlakuan.
# 2. Satu model lagi belajar dari grup kontrol (tidak diberi).
# 3. Uplift tiap orang = peluang jika diberi − peluang jika tidak diberi.
#
# **Data demo:** Hillstrom email (perlakuan = dapat email, hasil = `visit`).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/tabular_extra/uplift"   # GANTI saat lomba
DATA_CSV = f"{DATA_FOLDER}/hillstrom.csv"

OUTCOME = "visit"                  # hasil yang ingin dinaikkan (1/0)
GROUP_COL = "segment"              # kolom yang mencatat perlakuan
CONTROL_VALUE = "No E-Mail"        # nilai di GROUP_COL untuk grup kontrol
NUM_COLS = ["recency", "history", "mens", "womens", "newbie"]
CAT_COLS = ["zip_code", "channel"]
EDA_COLS = ["channel", "zip_code", "newbie"]
TOP_SHARE = 0.3                    # beri promo ke 30% orang dengan uplift tertinggi
SEED = 42

# %% [markdown]
# ## Baca data
# Kolom `treatment`: 1 = diberi perlakuan, 0 = kontrol.

# %%
data = pd.read_csv(DATA_CSV)
data["treatment"] = (data[GROUP_COL] != CONTROL_VALUE).astype(int)
FEATURES = NUM_COLS + CAT_COLS
print(data.shape)
print(data.groupby("treatment")[OUTCOME].agg(["mean", "count"]))
data.head()

# %% [markdown]
# ## EDA: efek rata-rata per kelompok
# `uplift` = rata-rata hasil grup perlakuan − rata-rata hasil grup kontrol.

# %%
for col in EDA_COLS:
    mean_table = data.groupby([col, "treatment"])[OUTCOME].mean()
    mean_table = mean_table.unstack()
    mean_table["uplift"] = mean_table[1] - mean_table[0]
    print(mean_table.round(4), "\n")

# %% [markdown]
# ## Split train / validasi

# %%
train_part, val_part = train_test_split(data, test_size=0.3, stratify=data["treatment"],
                                        random_state=SEED)

# %% [markdown]
# ## Model
# Pilih SATU model: hapus `#` di baris yang ingin dipakai, beri `#` pada baris lain.

# %%
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])
model = LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                       random_state=SEED, verbose=-1)
# model = LogisticRegression(max_iter=2000)
# model = RandomForestClassifier(n_estimators=300, min_samples_leaf=200, n_jobs=-1)

# %% [markdown]
# ## T-learner
# `clone` membuat salinan model yang belum dilatih, supaya dua model tidak saling menimpa.

# %%
def train_t_learner(table):
    treated = table[table["treatment"] == 1]
    control = table[table["treatment"] == 0]
    model_treated = make_pipeline(clone(preprocess), clone(model))
    model_control = make_pipeline(clone(preprocess), clone(model))
    model_treated.fit(treated[FEATURES], treated[OUTCOME])
    model_control.fit(control[FEATURES], control[OUTCOME])
    return model_treated, model_control


def predict_uplift(model_treated, model_control, table):
    proba_treated = model_treated.predict_proba(table[FEATURES])[:, 1]
    proba_control = model_control.predict_proba(table[FEATURES])[:, 1]
    return proba_treated - proba_control


model_treated, model_control = train_t_learner(train_part)
val_part = val_part.copy()
val_part["uplift"] = predict_uplift(model_treated, model_control, val_part)
val_part["uplift"].describe()

# %% [markdown]
# ## Evaluasi: uplift nyata per desil
# Orang validasi diurutkan menurut skor uplift, lalu dibagi 10 kelompok (desil 1 = skor
# tertinggi). Model bagus → desil atas punya selisih (perlakuan − kontrol) paling besar.
# Garis merah = efek rata-rata kalau semua orang diberi perlakuan.

# %%
rank = val_part["uplift"].rank(method="first", ascending=False)
val_part["decile"] = pd.qcut(rank, 10, labels=False) + 1

decile_table = val_part.groupby(["decile", "treatment"])[OUTCOME].mean()
decile_table = decile_table.unstack()
decile_table["real_uplift"] = decile_table[1] - decile_table[0]
print(decile_table.round(4))

overall = val_part.groupby("treatment")[OUTCOME].mean()
decile_table["real_uplift"].plot.bar()
plt.axhline(overall[1] - overall[0], color="red", linestyle="--", label="rata-rata semua")
plt.legend()
plt.show()

# %% [markdown]
# Uplift nyata di 30% orang dengan skor tertinggi (desil 1–3).

# %%
top = val_part[val_part["decile"] <= 3]
top_mean = top.groupby("treatment")[OUTCOME].mean()
print("uplift nyata 30% teratas:", round(top_mean[1] - top_mean[0], 4))

# %% [markdown]
# ## Latih ulang di semua data + daftar target
# Hasil: skor uplift per orang. `target_promo` = 1 untuk `TOP_SHARE` orang dengan skor tertinggi.

# %%
model_treated, model_control = train_t_learner(data)
predictions = predict_uplift(model_treated, model_control, data)
cutoff = pd.Series(predictions).quantile(1 - TOP_SHARE)

submission = pd.DataFrame()
submission["id"] = data["id"]
submission["uplift"] = predictions
submission["target_promo"] = (predictions >= cutoff).astype(int)
submission.to_csv("submission.csv", index=False)
submission.head()
