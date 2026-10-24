# %% [markdown]
# # Analisis survival (waktu-sampai-kejadian): siapa yang lebih cepat mengalami kejadian?
#
# **Butuh:** tabel dengan kolom DURASI (lama diamati) + KEJADIAN (1 = terjadi, 0 = belum
# terjadi saat pengamatan berhenti = **tersensor**) + fitur. Contoh: churn pelanggan,
# mesin rusak, pasien meninggal, peminjam gagal bayar.
# **Menghasilkan:** kurva bertahan (Kaplan–Meier), hazard ratio tiap fitur (Cox), skor
# risiko per baris, peluang bertahan sampai waktu T, `submission.csv`.
#
# Kenapa tidak regresi biasa pada durasi? Baris tersensor TIDAK punya durasi sebenarnya
# (hanya tahu "lebih dari X"). Membuangnya atau menganggap X = durasi → bias.
#
# | cara | ide | acuan |
# |---|---|---|
# | Kaplan–Meier | kurva peluang bertahan, tanpa fitur | Kaplan & Meier 1958 |
# | log-rank test | apakah dua kurva berbeda nyata? | Mantel 1966 |
# | Cox PH | risiko = dasar × exp(Σ koef·fitur); exp(koef) = hazard ratio | Cox 1972 |
# | XGBoost `survival:cox` | Cox tapi dengan pohon (interaksi, non-linear) | dok. XGBoost |
# | GBDT + target KM | (durasi, kejadian) → 1 angka risiko → regresi | lomba Kaggle CIBMTR 2025 |
# | metrik C-index | porsi pasangan yang urutan risikonya benar; 0,5 = acak | Harrell 1982 |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi kolom + cek sensor | **ya** |
# | 2 | Kaplan–Meier ditulis sendiri + `lifelines` | **ya** |
# | 3 | KM per kelompok + log-rank test | untuk laporan |
# | 4 | `c_index` ditulis sendiri | **ya** |
# | 5 | Cox PH: hazard ratio | **ya** |
# | 6 | cek asumsi proportional hazards | untuk laporan |
# | 7 | CV: Cox vs XGBoost-cox vs LightGBM target KM | disarankan |
# | 8 | model akhir → risiko + peluang bertahan sampai `HORIZON` → submission | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# 432 mantan narapidana diamati 52 minggu: `week` = minggu ditangkap lagi (atau 52 kalau
# tidak), `arrest` = 1 kalau ditangkap. `fin` = dapat bantuan keuangan.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

FOLDER = f"{DEMO_FOLDER}/tabular_extra/survival"
train = pd.read_csv(f"{FOLDER}/rossi_train.csv")
test = pd.read_csv(f"{FOLDER}/rossi_test.csv")
test_labels = pd.read_csv(f"{FOLDER}/rossi_test_labels.csv")
print(train.shape, test.shape)

# %% [markdown]
# ## 1. Konfigurasi + cek sensor

# %%
ID = "id"             # UBAH
DURATION = "week"     # UBAH: lama pengamatan
EVENT = "arrest"      # UBAH: 1 = kejadian terjadi, 0 = tersensor
FEATURES = []
for column in train.columns:
    if column not in [ID, DURATION, EVENT]:
        FEATURES.append(column)
HORIZON = 52          # UBAH: waktu T untuk "peluang bertahan sampai T"
print("porsi kejadian:", round(train[EVENT].mean(), 3))
print("durasi baris tersensor:", train.loc[train[EVENT] == 0, DURATION].describe().to_dict())

# %% [markdown]
# Di data contoh semua yang tersensor diamati penuh 52 minggu (sensor "administratif").
# Data lomba biasanya lebih berantakan (pelanggan baru, pasien pindah) → durasi sensor
# beragam, dan saat itulah cara survival jauh lebih benar daripada klasifikasi biasa.

# %% [markdown]
# ## 2. Kaplan–Meier: S(t) = Π (1 − kejadian_t / berisiko_t)
# `berisiko_t` = jumlah baris yang durasinya ≥ t (belum kejadian & belum tersensor).

# %%
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter


def kaplan_meier(durations, events):
    durations, events = np.asarray(durations), np.asarray(events)
    rows, survival = [], 1.0
    for time in np.unique(durations):
        at_risk = (durations >= time).sum()
        happened = ((durations == time) & (events == 1)).sum()
        survival *= 1 - happened / at_risk
        rows.append({"time": time, "at_risk": at_risk, "events": happened,
                     "survival": survival})
    return pd.DataFrame(rows)


curve = kaplan_meier(train[DURATION], train[EVENT])
kmf = KaplanMeierFitter().fit(train[DURATION], train[EVENT])
print(curve.tail(3).to_string(index=False))
print("lifelines S(52):", round(float(kmf.survival_function_at_times(52).iloc[0]), 4),
      "| median bertahan:", kmf.median_survival_time_)

# %% [markdown]
# ## 3. KM per kelompok + log-rank test (p < 0,05 = kurva berbeda nyata)

# %%
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

GROUP = "fin"   # UBAH: kolom kategori untuk dibandingkan
fig, ax = plt.subplots(figsize=(7, 4))
for value in sorted(train[GROUP].unique()):
    part = train[train[GROUP] == value]
    KaplanMeierFitter().fit(part[DURATION], part[EVENT], label=f"{GROUP}={value}").plot(ax=ax)
ax.set_xlabel(DURATION)
ax.set_ylabel("peluang belum kejadian")
plt.tight_layout()
plt.show()
first = train[train[GROUP] == 0]
second = train[train[GROUP] == 1]
result = logrank_test(first[DURATION], second[DURATION], first[EVENT], second[EVENT])
print("log-rank p =", round(result.p_value, 4))

# %% [markdown]
# ## 4. `c_index(durations, events, risk)`: dari pasangan yang bisa dibandingkan (yang
# lebih dulu mengalami kejadian), berapa persen risikonya diprediksi lebih tinggi?
# Pasangan bisa dibandingkan: A kejadian, dan B bertahan lebih lama (atau sama lama tapi
# tersensor). Pasangan dengan risiko sama dihitung setengah.
# Catatan: `lifelines.utils.concordance_index` minta "waktu prediksi" → beri `-risk`.

# %%
import numpy as np
from lifelines.utils import concordance_index


def c_index(durations, events, risk):
    durations, events = np.asarray(durations), np.asarray(events)
    risk = np.asarray(risk, dtype=float)
    later = durations[:, None] < durations[None, :]
    same_time_censored = (durations[:, None] == durations[None, :]) & (events[None, :] == 0)
    comparable = (later | same_time_censored) & (events[:, None] == 1)
    right = (risk[:, None] > risk[None, :]) & comparable
    tied = (risk[:, None] == risk[None, :]) & comparable
    return (right.sum() + 0.5 * tied.sum()) / comparable.sum()


risk_prio = train["prio"].to_numpy()   # coba: makin banyak catatan kriminal = makin berisiko
print("c-index sendiri :", round(c_index(train[DURATION], train[EVENT], risk_prio), 4))
print("c-index lifelines:", round(concordance_index(train[DURATION], -risk_prio,
                                                    train[EVENT]), 4))

# %% [markdown]
# ## 5. Cox proportional hazards: `exp(coef)` = hazard ratio
# HR 1,10 untuk `prio` = tiap tambah 1 catatan kriminal, laju ditangkap naik 10%.
# HR < 1 = melindungi. `penalizer` kecil (0,01–0,1) membantu kalau fitur banyak/berkorelasi.

# %%
from lifelines import CoxPHFitter

cox = CoxPHFitter(penalizer=0.01)
cox.fit(train[FEATURES + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
summary = cox.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
print(summary.round(3).to_string())
print("c-index train:", round(cox.concordance_index_, 4))

# %% [markdown]
# ## 6. Cek asumsi proportional hazards (p < 0,05 = efek fitur berubah seiring waktu)
# Kalau dilanggar: pisah model per kelompok (`strata=[kolom]`), atau pakai model pohon.
# Data contoh: `age` dan `wexp` melanggar (p < 0,05) → alasan mencoba model pohon di sel 7.

# %%
from lifelines.statistics import proportional_hazard_test

check = proportional_hazard_test(cox, train[FEATURES + [DURATION, EVENT]],
                                 time_transform="rank")
print(check.summary[["test_statistic", "p"]].round(4).to_string())

# %% [markdown]
# ## 7. CV 5-fold: tiap cara mengembalikan RISIKO (besar = lebih cepat kejadian)
# - `fit_cox`: Cox PH lifelines.
# - `fit_xgb_cox`: XGBoost `survival:cox`; label = durasi, NEGATIF untuk baris tersensor.
# - `fit_lgbm_km`: target = S_KM(durasi) (kejadian cepat → S tinggi → risiko tinggi);
#   baris tersensor dikurangi `CENSOR_SHIFT` karena kejadiannya (kalau ada) lebih lambat.

# %%
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import StratifiedKFold

folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train, train[EVENT]))


def cross_validate(fit_predict):
    oof = np.zeros(len(train))
    for index_fit, index_val in folds:
        oof[index_val] = fit_predict(train.iloc[index_fit], train.iloc[index_val])
    return c_index(train[DURATION], train[EVENT], oof)


def fit_cox(part, other):
    model = CoxPHFitter(penalizer=0.01)
    model.fit(part[FEATURES + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
    return model.predict_partial_hazard(other[FEATURES]).to_numpy()


cv_scores = {"cox": cross_validate(fit_cox)}
print(cv_scores)

# %%
import numpy as np
import xgboost as xgb


def fit_xgb_cox(part, other):
    label = np.where(part[EVENT] == 1, part[DURATION], -part[DURATION])
    model = xgb.XGBRegressor(objective="survival:cox", n_estimators=200, learning_rate=0.03,
                             max_depth=3, subsample=0.8, colsample_bytree=0.8)
    model.fit(part[FEATURES], label)
    return model.predict(other[FEATURES])


cv_scores["xgb_cox"] = cross_validate(fit_xgb_cox)
print(cv_scores)

# %%
import lightgbm as lgb
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

CENSOR_SHIFT = 0.15   # UBAH: 0,1–0,3


def km_target(part):
    kmf = KaplanMeierFitter().fit(part[DURATION], part[EVENT])
    target = kmf.survival_function_at_times(part[DURATION]).to_numpy()
    return np.where(part[EVENT] == 1, target, target - CENSOR_SHIFT)


def fit_lgbm_km(part, other):
    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.03, num_leaves=7,
                              min_child_samples=20, verbose=-1)
    model.fit(part[FEATURES], km_target(part))
    return model.predict(other[FEATURES])


cv_scores["lgbm_km"] = cross_validate(fit_lgbm_km)
board = pd.Series(cv_scores).round(4)
print(board.sort_values(ascending=False))

# %% [markdown]
# Gabungan: rata-rata PERINGKAT risiko beberapa model (skala risiko tiap model berbeda).

# %%
import pandas as pd


def fit_rank_mean(part, other):
    ranks = pd.DataFrame({"cox": fit_cox(part, other), "xgb": fit_xgb_cox(part, other),
                          "lgbm": fit_lgbm_km(part, other)})
    ranks = ranks.rank(pct=True)
    return ranks.mean(axis=1).to_numpy()


cv_scores["rata_peringkat"] = cross_validate(fit_rank_mean)
board = pd.Series(cv_scores).round(4)
print(board.sort_values(ascending=False))

# %% [markdown]
# ## 8. Model akhir di seluruh train → skor risiko test + peluang bertahan sampai `HORIZON`
# Peluang bertahan hanya dari Cox (model yang memberi kurva lengkap per baris).

# %%
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

FINAL = fit_rank_mean   # UBAH: pilih dari papan skor di atas
final_cox = CoxPHFitter(penalizer=0.01)
final_cox.fit(train[FEATURES + [DURATION, EVENT]], duration_col=DURATION, event_col=EVENT)
survival = final_cox.predict_survival_function(test[FEATURES], times=[HORIZON])
submission = pd.DataFrame()
submission[ID] = test[ID]
submission["risk"] = FINAL(train, test)                                       # UBAH
submission[f"survival_{HORIZON}"] = survival.iloc[0].to_numpy().round(4)
submission.to_csv("submission.csv", index=False)
print(submission.head())

# %% [markdown]
# Kurva bertahan 3 orang (berisiko rendah, tengah, tinggi) → bahan laporan.

# %%
import matplotlib.pyplot as plt
import numpy as np

order = np.argsort(submission["risk"].to_numpy())
chosen = [order[0], order[len(order) // 2], order[-1]]
curves = final_cox.predict_survival_function(test.iloc[chosen][FEATURES])
fig, ax = plt.subplots(figsize=(7, 4))
for column, name in zip(curves.columns, ["rendah", "tengah", "tinggi"]):
    ax.step(curves.index, curves[column], where="post", label=f"risiko {name}")
ax.set_xlabel(DURATION)
ax.set_ylabel("peluang belum kejadian")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Cek ke kunci jawaban (hanya demo).

# %%
print("urutan id sama:", (test_labels[ID].to_numpy() == submission[ID].to_numpy()).all())
true_duration = test_labels[DURATION].to_numpy()
true_event = test_labels[EVENT].to_numpy()
for name, fit_predict in {"cox": fit_cox, "xgb_cox": fit_xgb_cox, "lgbm_km": fit_lgbm_km,
                          "rata_peringkat": fit_rank_mean}.items():
    risk = fit_predict(train, test)
    print(f"{name:15s} c-index test: {c_index(true_duration, true_event, risk):.4f}")

# %% [markdown]
# Hasil data contoh: CV → rata peringkat 0,632 | LightGBM target KM 0,628 | Cox 0,616 |
# XGBoost-cox 0,611. Test (108 baris) → Cox 0,611 | rata peringkat 0,609 | LightGBM 0,608 |
# XGBoost-cox 0,593. Urutan CV dan test tidak sama: selisih < 0,02 pada data sekecil ini
# masih kebetulan. Data kecil → Cox sudah cukup dan paling mudah dijelaskan (hazard ratio);
# data besar dengan interaksi → model pohon / rata peringkat biasanya unggul.
