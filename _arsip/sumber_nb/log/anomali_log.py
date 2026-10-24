# %% [markdown]
# # Log-sekuens — Anomali pada Event Log (DFG + varian + durasi → IsolationForest)
# **Kapan dipakai:** event log proses (tiket, pesanan, transaksi, klik) dan diminta mencari kasus
# yang "tidak biasa": urutan aktivitas janggal, langkah hilang/berulang, atau durasi terlalu lama.
#
# **Idenya:** tiga sudut pandang, semuanya dengan pandas biasa (tanpa pm4py):
# - **DFG** (directly-follows graph): hitung pasangan aktivitas berurutan A → B. Pasangan langka
#   = transisi mencurigakan.
# - **Varian**: urutan aktivitas lengkap per kasus dan seberapa sering urutan itu muncul.
# - **Durasi**: lama kasus dari event pertama sampai terakhir.
#
# **Langkah:**
# 1. Hitung DFG dari semua kasus.
# 2. Feature per kasus (skor DFG, frekuensi varian, durasi).
# 3. IsolationForest → skor anomali → submission.
#
# **Data demo:** log helpdesk (4580 tiket). Anomali **disisipkan sendiri** pada 5% kasus (tukar
# urutan, hapus langkah, ulang langkah, durasi 10× lebih lambat) supaya skornya bisa diukur.

# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/log"   # GANTI saat lomba
LOG_CSV = f"{DATA_FOLDER}/helpdesk.csv"

CASE_COL = "case_id"
ACTIVITY_COL = "aktivitas"
TIME_COL = "waktu"
ANOMALY_SHARE = 0.05    # perkiraan porsi anomali (di demo juga porsi kasus yang dirusak)
SEED = 42

# %% [markdown]
# ## Baca data

# %%
log = pd.read_csv(LOG_CSV, parse_dates=[TIME_COL])
log = log.sort_values([CASE_COL, TIME_COL])
print(log.shape, "|", log[CASE_COL].nunique(), "kasus,", log[ACTIVITY_COL].nunique(), "aktivitas")
log.head()

# %% [markdown]
# ## Sisipkan anomali (HANYA DEMO — hapus sel ini saat lomba)
# Tiap kasus terpilih dirusak satu cara. `corrupted` mencatat kasus yang benar-benar berubah.

# %%
rng = np.random.default_rng(SEED)


def corrupt_case(activities, times):
    kind = rng.choice(["swap", "drop", "repeat", "slow"])
    position = int(rng.integers(1, len(activities) - 1))
    if kind == "swap":
        other = activities[position + 1]
        activities[position + 1] = activities[position]
        activities[position] = other
    if kind == "drop":
        del activities[position]
        del times[position]
    if kind == "repeat":
        activities.insert(position, activities[position])
        times.insert(position, times[position])
    if kind == "slow":
        start = times[0]
        for i in range(len(times)):
            times[i] = start + (times[i] - start) * 10


# coba pada urutan contoh
example = ["A", "B", "C", "D"]
corrupt_case(example, [0, 1, 2, 3])
print(example)

# %%
all_cases = log[CASE_COL].unique()
chosen = set(rng.choice(all_cases, int(ANOMALY_SHARE * len(all_cases)), replace=False))
corrupted = set()
new_rows = []
for case, group in log.groupby(CASE_COL):
    activities = list(group[ACTIVITY_COL])
    times = list(group[TIME_COL])
    if case in chosen and len(activities) >= 3:
        corrupt_case(activities, times)
        corrupted.add(case)
    for activity, time in zip(activities, times):
        new_rows.append([case, activity, time])
log = pd.DataFrame(new_rows, columns=[CASE_COL, ACTIVITY_COL, TIME_COL])
print("kasus dirusak:", len(corrupted))

# %% [markdown]
# ## Langkah 1 — DFG: hitung pasangan aktivitas berurutan
# Tiap kasus diberi penanda `[START]` dan `[END]`.
# `probability` = P(aktivitas berikut | aktivitas sebelumnya).

# %%
pair_counts = {}
from_counts = {}
sequences = {}
for case, group in log.groupby(CASE_COL):
    sequence = ["[START]"] + list(group[ACTIVITY_COL]) + ["[END]"]
    sequences[case] = sequence
    for before, after in zip(sequence[:-1], sequence[1:]):
        pair_counts[(before, after)] = pair_counts.get((before, after), 0) + 1
        from_counts[before] = from_counts.get(before, 0) + 1

dfg_rows = []
for (before, after), count in pair_counts.items():
    dfg_rows.append([before, after, count])
dfg = pd.DataFrame(dfg_rows, columns=["from", "to", "count"])
dfg["probability"] = dfg["count"] / dfg["from"].map(from_counts)
dfg = dfg.sort_values("count", ascending=False)
dfg.head(10)

# %% [markdown]
# ## Langkah 2 — feature per kasus (DFG, varian, durasi)
# Skor DFG = -log peluang transisi. Transisi langka → skor besar.

# %%
variant = {}
dfg_max = {}
dfg_mean = {}
for case, sequence in sequences.items():
    variant[case] = " > ".join(sequence)
    transition_scores = []
    for before, after in zip(sequence[:-1], sequence[1:]):
        probability = pair_counts[(before, after)] / from_counts[before]
        transition_scores.append(-np.log(probability))
    dfg_max[case] = max(transition_scores)
    dfg_mean[case] = np.mean(transition_scores)

variant = pd.Series(variant)
variant_counts = variant.value_counts()
per_case = log.groupby(CASE_COL)
duration = per_case[TIME_COL].max() - per_case[TIME_COL].min()

features = pd.DataFrame()
features["dfg_max"] = pd.Series(dfg_max)
features["dfg_mean"] = pd.Series(dfg_mean)
features["variant_frequency"] = np.log(variant.map(variant_counts))
features["n_events"] = per_case.size()
features["log_duration_days"] = np.log1p(duration.dt.total_seconds() / 86400)
features["n_unique_activities"] = per_case[ACTIVITY_COL].nunique()
print("varian unik:", len(variant_counts))
features.describe()

# %% [markdown]
# ## Langkah 3 — IsolationForest → skor anomali
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
X = StandardScaler().fit_transform(features)
model = IsolationForest(n_estimators=300, contamination=ANOMALY_SHARE, random_state=SEED)
# from pyod.models.ecod import ECOD
# model = ECOD(contamination=ANOMALY_SHARE)     # tanpa parameter, cepat
model.fit(X)
scores = -model.score_samples(X)        # makin besar = makin aneh

top = pd.DataFrame()
top["score"] = scores
top["variant"] = variant[features.index].to_numpy()
top = top.sort_values("score", ascending=False)
pd.set_option("display.max_colwidth", 90)
top.head(8)

# %% [markdown]
# ## Submission

# %%
submission = pd.DataFrame()
submission["case_id"] = features.index
submission["anomaly_score"] = scores
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban (HAPUS saat lomba)

# %%
truth = features.index.isin(corrupted).astype(int)
print("ROC-AUC:", round(roc_auc_score(truth, scores), 3))
print("PR-AUC :", round(average_precision_score(truth, scores), 3))
for name in features.columns:
    print("  ROC-AUC satu feature", name, ":", round(roc_auc_score(truth, features[name]), 3))
