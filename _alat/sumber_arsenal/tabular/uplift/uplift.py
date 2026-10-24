# %% [markdown]
# # Uplift / efek perlakuan: SIAPA yang berubah perilakunya karena diberi perlakuan?
#
# **Butuh:** data eksperimen (A/B test): kolom PERLAKUAN (1 = diberi promo/email/obat,
# 0 = kontrol), kolom HASIL (beli, kunjung, sembuh), fitur. Paling sah kalau perlakuan
# dibagi ACAK. **Menghasilkan:** efek rata-rata (ATE) + selang kepercayaan, uplift per
# kelompok, skor uplift per orang, kurva Qini, daftar orang yang layak diberi perlakuan.
#
# Model "siapa yang akan beli" ≠ model uplift. Empat jenis orang (Radcliffe 2007):
# **persuadable** (beli HANYA kalau diberi promo) ← target; **sure thing** (beli juga tanpa
# promo) → promo terbuang; **lost cause**; **sleeping dog** (malah batal kalau diganggu).
#
# | cara | ide | acuan |
# |---|---|---|
# | selisih rata-rata (ATE) | rata hasil perlakuan − rata hasil kontrol | dasar A/B test |
# | S-learner | 1 model, perlakuan jadi fitur; uplift = f(x, 1) − f(x, 0) | Künzel dkk. 2019 |
# | T-learner | 2 model terpisah (perlakuan, kontrol); uplift = selisihnya | Künzel dkk. 2019 |
# | X-learner | T-learner + tebak efek per orang; grup tak seimbang | Künzel dkk. 2019 |
# | transformasi kelas | Z = 1 kalau (diberi & beli) / (kontrol & tidak) | Jaśkowski 2012 |
# | kurva Qini / uplift@k | urutkan skor, hitung tambahan hasil di k% teratas | Radcliffe 2007 |
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi: kolom perlakuan, hasil, fitur | **ya** |
# | 2 | cek acak: fitur seimbang antar grup? | **ya** |
# | 3 | ATE + selang kepercayaan 95% | **ya** |
# | 4 | uplift per kelompok (groupby) | untuk laporan |
# | 5 | bagi train/test | **ya** |
# | 6 | S-, T-, X-learner, transformasi kelas | pilih 1–4 |
# | 7 | `qini_curve` + papan skor + plot | **ya** |
# | 8 | keputusan: beri perlakuan ke k% teratas → hasil tambahan | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Hillstrom (MineThatData 2008): 64.000 pelanggan dibagi acak: email produk pria, email
# produk wanita, tanpa email. Hasil 2 minggu: `visit`, `conversion`, `spend`.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

table = pd.read_csv(f"{DEMO_FOLDER}/tabular_extra/uplift/hillstrom.csv")
print(table.shape)
rates = table.groupby("segment")[["visit", "conversion", "spend"]].mean()
print(rates.round(4))

# %% [markdown]
# ## 1. Konfigurasi. Kolom perlakuan lebih dari 2 nilai? Pilih satu pasangan perlakuan-kontrol.

# %%
import pandas as pd

GROUP_COLUMN = "segment"          # UBAH
TREATED_VALUE = "Womens E-Mail"   # UBAH
CONTROL_VALUE = "No E-Mail"       # UBAH
OUTCOME = "visit"                 # UBAH: hasil 0/1 (bisa juga angka, mis. spend)
NUMERIC = ["recency", "history", "mens", "womens", "newbie"]   # UBAH
CATEGORICAL = ["zip_code", "channel"]                          # UBAH
data = table[table[GROUP_COLUMN].isin([TREATED_VALUE, CONTROL_VALUE])].copy()
data = data.reset_index(drop=True)
data["treatment"] = (data[GROUP_COLUMN] == TREATED_VALUE).astype(int)
for column in CATEGORICAL:
    data[column] = data[column].astype("category")
FEATURES = NUMERIC + CATEGORICAL
print(len(data), "baris | porsi diberi perlakuan:", round(data["treatment"].mean(), 3))

# %% [markdown]
# ## 2. Cek pembagian acak: rata-rata fitur per grup + AUC menebak grup dari fitur
# AUC ≈ 0,5 = pembagian acak (bagus). AUC jauh di atas 0,5 = grup berbeda dari awal →
# selisih hasil bisa karena perbedaan orangnya, bukan perlakuannya (lihat catatan bawah).

# %%
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

balance = data.groupby("treatment")[NUMERIC].mean()
print(balance.round(3).T)
guess = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, verbose=-1), data[FEATURES],
                          data["treatment"], cv=5, method="predict_proba")[:, 1]
print("AUC menebak grup:", round(roc_auc_score(data["treatment"], guess), 4))

# %% [markdown]
# Data BUKAN eksperimen acak (mis. promo diberikan ke pelanggan pilihan)? Uplift langsung
# akan bias. Langkah minimum: hitung propensity p(x) = P(perlakuan | fitur) seperti di atas,
# lalu beri bobot 1/p untuk grup perlakuan dan 1/(1−p) untuk kontrol (IPW, Rosenbaum &
# Rubin 1983), dan laporkan bahwa kesimpulannya bergantung pada asumsi.

# %% [markdown]
# ## 3. Efek rata-rata (ATE) = rata hasil perlakuan − rata hasil kontrol, ± 1,96 × SE

# %%
import numpy as np

treated = data.loc[data["treatment"] == 1, OUTCOME]
control = data.loc[data["treatment"] == 0, OUTCOME]
ate = treated.mean() - control.mean()
standard_error = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
print(f"perlakuan {treated.mean():.4f} | kontrol {control.mean():.4f}")
print(f"ATE = {ate:.4f}  (95%: {ate - 1.96 * standard_error:.4f} s/d "
      f"{ate + 1.96 * standard_error:.4f}) | naik relatif {ate / control.mean():.1%}")

# %% [markdown]
# ## 4. Uplift per kelompok: kelompok mana yang paling terpengaruh?

# %%
import pandas as pd


def uplift_by(column):
    rates = data.pivot_table(index=column, columns="treatment", values=OUTCOME,
                             aggfunc="mean", observed=True)
    rates["n"] = data.groupby(column, observed=True).size()
    rates["uplift"] = rates[1] - rates[0]
    return rates.round(4).sort_values("uplift", ascending=False)


for column in ["zip_code", "channel", "newbie", "womens"]:   # UBAH
    print(uplift_by(column), "\n")

# %% [markdown]
# ## 5. Bagi train/test (stratify perlakuan × hasil supaya semua kombinasi ada di dua sisi)

# %%
from sklearn.model_selection import train_test_split

strata = data["treatment"].astype(str) + "_" + data[OUTCOME].astype(str)
train, test = train_test_split(data, test_size=0.3, random_state=42, stratify=strata)
train, test = train.reset_index(drop=True), test.reset_index(drop=True)
print(len(train), len(test))
uplifts = {}

# %% [markdown]
# ## 6a. S-learner dan T-learner

# %%
import lightgbm as lgb


def make_classifier():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=100, verbose=-1)


s_model = make_classifier().fit(train[FEATURES + ["treatment"]], train[OUTCOME])
as_treated = test[FEATURES].assign(treatment=1)
as_control = test[FEATURES].assign(treatment=0)
uplifts["s_learner"] = (s_model.predict_proba(as_treated)[:, 1]
                        - s_model.predict_proba(as_control)[:, 1])
treated_part = train[train["treatment"] == 1]
control_part = train[train["treatment"] == 0]
model_treated = make_classifier().fit(treated_part[FEATURES], treated_part[OUTCOME])
model_control = make_classifier().fit(control_part[FEATURES], control_part[OUTCOME])
uplifts["t_learner"] = (model_treated.predict_proba(test[FEATURES])[:, 1]
                        - model_control.predict_proba(test[FEATURES])[:, 1])
print("rata uplift prediksi S:", round(uplifts["s_learner"].mean(), 4),
      "| T:", round(uplifts["t_learner"].mean(), 4))

# %% [markdown]
# ## 6b. X-learner: tebak efek tiap orang dari model grup lawan, lalu regresi efek itu
# Efek orang perlakuan = hasilnya − tebakan model kontrol; efek orang kontrol = tebakan model
# perlakuan − hasilnya. Digabung dengan bobot propensity (di eksperimen acak = porsi grup).

# %%
import lightgbm as lgb


def make_regressor():
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=100, verbose=-1)


effect_treated = (treated_part[OUTCOME]
                  - model_control.predict_proba(treated_part[FEATURES])[:, 1])
effect_control = (model_treated.predict_proba(control_part[FEATURES])[:, 1]
                  - control_part[OUTCOME])
tau_treated = make_regressor().fit(treated_part[FEATURES], effect_treated)
tau_control = make_regressor().fit(control_part[FEATURES], effect_control)
propensity = train["treatment"].mean()
uplifts["x_learner"] = (propensity * tau_control.predict(test[FEATURES])
                        + (1 - propensity) * tau_treated.predict(test[FEATURES]))
print("rata uplift prediksi X:", round(uplifts["x_learner"].mean(), 4))

# %% [markdown]
# ## 6c. Transformasi kelas: Z = 1 kalau (perlakuan & hasil 1) atau (kontrol & hasil 0)
# Uplift = 2·P(Z=1) − 1. Syarat: porsi perlakuan ≈ 50%. Kalau tidak 50%, pakai versi
# regresi: target = hasil × (t − p) / (p(1 − p)) (transformed outcome, Athey & Imbens 2016).

# %%
import numpy as np

flip = train["treatment"] == train[OUTCOME]
z_model = make_classifier().fit(train[FEATURES], flip.astype(int))
uplifts["kelas_z"] = 2 * z_model.predict_proba(test[FEATURES])[:, 1] - 1
weight = (train["treatment"] - propensity) / (propensity * (1 - propensity))
outcome_model = make_regressor().fit(train[FEATURES], train[OUTCOME] * weight)
uplifts["transformed_outcome"] = outcome_model.predict(test[FEATURES])
print("korelasi skor antar cara:")
print(np.corrcoef(np.array(list(uplifts.values()))).round(2))

# %% [markdown]
# ## 7. `qini_curve`: urutkan dari uplift prediksi terbesar; di tiap titik k
# Qini(k) = hasil_perlakuan(k) − hasil_kontrol(k) × n_perlakuan(k) / n_kontrol(k)
# = tambahan hasil (mis. kunjungan) kalau k orang teratas diberi perlakuan.
# Skor = luas di atas garis acak (makin besar makin baik; 0 = sama dengan acak).

# %%
import numpy as np


def qini_curve(uplift, treatment, outcome):
    order = np.argsort(-np.asarray(uplift))
    treatment = np.asarray(treatment)[order]
    outcome = np.asarray(outcome, dtype=float)[order]
    count_treated = np.cumsum(treatment)
    count_control = np.maximum(np.cumsum(1 - treatment), 1)
    gain = (np.cumsum(outcome * treatment)
            - np.cumsum(outcome * (1 - treatment)) * count_treated / count_control)
    share = np.arange(1, len(order) + 1) / len(order)
    return share, gain


def uplift_at(uplift, treatment, outcome, top=0.3):
    chosen = np.argsort(-np.asarray(uplift))[:int(top * len(uplift))]
    treatment, outcome = np.asarray(treatment)[chosen], np.asarray(outcome)[chosen]
    return outcome[treatment == 1].mean() - outcome[treatment == 0].mean()


share, gain = qini_curve(uplifts["t_learner"], test["treatment"], test[OUTCOME])
print("tambahan hasil kalau semua diberi:", round(gain[-1], 1))

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

uplifts["acak"] = np.random.default_rng(0).random(len(test))
rows = []
fig, ax = plt.subplots(figsize=(7, 4))
for name, score in uplifts.items():
    share, gain = qini_curve(score, test["treatment"], test[OUTCOME])
    area = gain.mean() - gain[-1] * share.mean()
    rows.append({"cara": name, "qini_area": round(area, 2),
                 "uplift@30%": round(uplift_at(score, test["treatment"], test[OUTCOME]), 4)})
    ax.plot(share, gain, label=name)
ax.plot([0, 1], [0, gain[-1]], "k--", label="garis acak")
ax.set_xlabel("porsi orang diberi perlakuan (urut skor)")
ax.set_ylabel(f"tambahan {OUTCOME}")
ax.legend()
plt.tight_layout()
plt.show()
board = pd.DataFrame(rows).sort_values("qini_area", ascending=False)
print(board.to_string(index=False))

# %% [markdown]
# ## 8. Keputusan: beri perlakuan ke `TOP` teratas saja → berapa tambahan hasil yang didapat?
# Biaya per perlakuan dan nilai per hasil? Isi `COST` dan `VALUE` → untung bersih.

# %%
import numpy as np
import pandas as pd

CHOSEN = board["cara"].iloc[0]   # UBAH
COST, VALUE = 0.02, 1.0          # UBAH: biaya 1 perlakuan, nilai 1 hasil
share, gain = qini_curve(uplifts[CHOSEN], test["treatment"], test[OUTCOME])
rows = []
for top in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    index = int(top * len(share)) - 1
    people = top * len(test)
    rows.append({"porsi": top, "tambahan_hasil": round(gain[index], 1),
                 "untung_bersih": round(gain[index] * VALUE - people * COST, 1)})
print(CHOSEN)
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# Hasil data contoh (12.808 orang test, hasil = kunjungan): Qini S-learner 56,8 > X 38,4 >
# T 31,1 > transformed outcome 23,3 > kelas Z 12,1 > acak −2,8. Uplift@30%: 8,5 poin vs
# rata-rata 4,5 poin. Efek di data ini sederhana (pembeli produk wanita jauh lebih
# terpengaruh, lihat sel 4) → S-learner cukup. 50% teratas sudah memberi ±86% tambahan
# kunjungan; dengan biaya 0,02 per email, untung bersih tertinggi di sekitar 50%.
# Cara mana yang menang BERBEDA per data → selalu bandingkan dengan Qini di data test.

# %% [markdown]
# Simpan skor uplift tiap orang di test (urut dari yang paling layak diberi perlakuan).

# %%
import os

import pandas as pd

scores = pd.DataFrame()
scores["id"] = test["id"]                    # UBAH
scores["uplift"] = uplifts[CHOSEN]
scores = scores.sort_values("uplift", ascending=False)
os.makedirs("outputs", exist_ok=True)
scores.to_csv("outputs/uplift.csv", index=False)
print(scores.head())
