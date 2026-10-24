# %% [markdown]
# # Banyak tabel → satu tabel fitur: cek kunci, agregasi sebelum tanggal kejadian, gabung aman
#
# **Butuh:** tabel utama (satu baris per hal yang ditebak, mis. pinjaman) + tabel pendukung
# (transaksi, akun, wilayah, ...). **Menghasilkan:** `features` (satu baris per baris tabel
# utama), skor CV per kelompok tabel, `submission.csv`.
#
# Tiga aturan yang paling sering dilanggar:
# 1. **Cek kunci dulu**: unik di sisi "satu"? `merge(..., validate="many_to_one")` memastikan.
# 2. **Jumlah baris tabel utama TIDAK boleh berubah** setelah merge (cek `len` sebelum/sesudah).
# 3. **Waktu**: hanya pakai baris tabel pendukung yang terjadi SEBELUM tanggal kejadian
#    (transaksi setelah pinjaman diberikan = bocoran masa depan).
#
# Agregasi umum (tanpa waktu) ada di `umum/tabel_olah` bagian 8; notebook ini versi lengkap
# untuk data relasional (gaya lomba PKDD'99 Berka / Home Credit).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | `check_key`: kunci unik? berapa baris per kunci? | **ya** |
# | 2 | tabel utama train+test → `loans` | **ya** |
# | 3 | transaksi SEBELUM tanggal pinjaman → agregasi angka | **ya** |
# | 4 | agregasi per jenis (pivot) + jendela waktu terakhir | disarankan |
# | 5 | tabel statis: order, akun, wilayah, klien, kartu | **ya** |
# | 6 | gabung semua dengan `validate` + cek baris | **ya** |
# | 7 | skor CV per kelompok tabel → submission | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Bank Ceko (Berka): pinjaman `bad` = 1 kalau macet. Transaksi ± 1 juta baris.

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

FOLDER = f"{DEMO_FOLDER}/multitabel/berka"   # UBAH
loan_train = pd.read_csv(f"{FOLDER}/loan_train.csv", parse_dates=["date"])
loan_test = pd.read_csv(f"{FOLDER}/loan_test.csv", parse_dates=["date"])
test_labels = pd.read_csv(f"{FOLDER}/loan_test_labels.csv")
trans = pd.read_csv(f"{FOLDER}/trans.csv", parse_dates=["date"], low_memory=False)
account = pd.read_csv(f"{FOLDER}/account.csv", parse_dates=["date"])
district = pd.read_csv(f"{FOLDER}/district.csv")
disp = pd.read_csv(f"{FOLDER}/disp.csv")
client = pd.read_csv(f"{FOLDER}/client.csv", parse_dates=["birth_date"])
card = pd.read_csv(f"{FOLDER}/card.csv", parse_dates=["issued"])
order = pd.read_csv(f"{FOLDER}/order.csv")
print(loan_train.shape, loan_test.shape, trans.shape, "| porsi macet:",
      round(loan_train["bad"].mean(), 3))

# %% [markdown]
# ## 1. `check_key(table, key)`: kunci unik? berapa baris per kunci?
# Relasi Berka: loan → account (banyak-ke-satu); trans/order → account (banyak per akun);
# account → district; disp menghubungkan client ↔ account; card → disp.

# %%
import pandas as pd


def check_key(table, key, name):
    per_key = table.groupby(key).size()
    return {"tabel": name, "kunci": key, "baris": len(table), "kunci_unik": len(per_key),
            "unik?": bool(per_key.max() == 1), "maks_baris_per_kunci": int(per_key.max())}


checks = [check_key(loan_train, "account_id", "loan_train"), check_key(account, "account_id",
                                                                       "account"),
          check_key(trans, "account_id", "trans"), check_key(order, "account_id", "order"),
          check_key(disp, "account_id", "disp"), check_key(district, "district_id", "district"),
          check_key(card, "disp_id", "card")]
print(pd.DataFrame(checks).to_string(index=False))
print("disp per jenis:", disp["type"].value_counts().to_dict())

# %% [markdown]
# ## 2. Tabel utama train + test → `loans` (fitur dibuat sekali untuk keduanya)

# %%
import pandas as pd

KEY = "account_id"     # UBAH: kunci ke tabel pendukung
EVENT_DATE = "date"    # UBAH: tanggal kejadian (pinjaman)
TARGET = "bad"         # UBAH

loan_train["is_test"] = 0
loan_test["is_test"] = 1
loans = pd.concat([loan_train, loan_test], ignore_index=True)
loans = loans.rename(columns={EVENT_DATE: "loan_date"})
loans["loan_year"] = loans["loan_date"].dt.year
loans["amount_per_month"] = loans["amount"] / loans["duration"]
print(loans.shape, "| akun unik:", loans[KEY].nunique())

# %% [markdown]
# ## 3. Transaksi SEBELUM tanggal pinjaman → agregasi angka per pinjaman
# Langkah: tempelkan tanggal pinjaman ke tiap transaksi akun itu → buang transaksi
# `date >= loan_date` → groupby `loan_id`.

# %%
import pandas as pd

history = trans.merge(loans[["loan_id", KEY, "loan_date"]], on=KEY, how="inner")
before = len(history)
history = history[history["date"] < history["loan_date"]].copy()
print("transaksi akun peminjam:", before, "→ sebelum pinjaman:", len(history))
history["days_before"] = (history["loan_date"] - history["date"]).dt.days
history["signed_amount"] = history["amount"]
is_out = history["type"].isin(["VYDAJ", "VYBER"])    # UBAH: jenis uang keluar
history.loc[is_out, "signed_amount"] = -history.loc[is_out, "amount"]
by_loan = history.groupby("loan_id")
trans_features = by_loan.agg(trans_count=("trans_id", "count"),
                             amount_mean=("amount", "mean"), amount_max=("amount", "max"),
                             balance_mean=("balance", "mean"), balance_min=("balance", "min"),
                             balance_std=("balance", "std"), net_flow=("signed_amount", "sum"),
                             history_days=("days_before", "max"))
negative = history["balance"] < 0
trans_features["negative_balance_count"] = negative.groupby(history["loan_id"]).sum()
trans_features["out_share"] = is_out.groupby(history["loan_id"]).mean()
print(trans_features.shape)
print(trans_features.head(3).round(1))

# %% [markdown]
# ## 4. Agregasi per jenis transaksi (pivot) + jendela waktu terakhir (90 hari)
# Pivot: satu kolom per nilai kategori (mis. total per `k_symbol`). Jendela terakhir
# menangkap kondisi keuangan menjelang pinjaman; rasio jendela/seluruh = tren.

# %%
import pandas as pd

CATEGORY = "k_symbol"   # UBAH: kolom jenis transaksi
WINDOW_DAYS = 90        # UBAH

history[CATEGORY] = history[CATEGORY].fillna("kosong").str.strip()
history.loc[history[CATEGORY] == "", CATEGORY] = "kosong"
per_type = history.pivot_table(index="loan_id", columns=CATEGORY, values="amount",
                               aggfunc="sum", fill_value=0)
per_type.columns = "sum_" + per_type.columns.str.lower()
recent = history[history["days_before"] <= WINDOW_DAYS]
recent_features = recent.groupby("loan_id").agg(recent_count=("trans_id", "count"),
                                                 recent_balance_min=("balance", "min"),
                                                 recent_net_flow=("signed_amount", "sum"))
trans_features = trans_features.join(per_type).join(recent_features)
trans_features["recent_share"] = trans_features["recent_count"] / trans_features["trans_count"]
print(trans_features.shape, "|", list(per_type.columns)[:6])

# %% [markdown]
# ## 5. Tabel statis (tanpa tanggal kejadian): order, akun, wilayah
# Satu baris per akun/wilayah → nanti digabung ke `loans`.

# %%
import pandas as pd

order_features = order.pivot_table(index=KEY, columns="k_symbol", values="amount",
                                   aggfunc="sum", fill_value=0)
order_names = order_features.columns.str.strip()
order_features.columns = "order_" + order_names.str.lower()
order_features["order_count"] = order.groupby(KEY).size()
DISTRICT_NAMES = {"A4": "district_population", "A11": "district_salary",          # UBAH
                  "A12": "unemployment_95", "A13": "unemployment_96",
                  "A14": "entrepreneurs_per_1000"}
district_features = district.rename(columns=DISTRICT_NAMES)
district_features = district_features[["district_id"] + list(DISTRICT_NAMES.values())].copy()
district_features["unemployment_95"] = pd.to_numeric(district_features["unemployment_95"],
                                                     errors="coerce")   # ada "?" di data
account_features = account.rename(columns={"date": "account_date"})
print(order_features.shape, district_features.shape, account_features.shape)

# %% [markdown]
# Pemilik akun (disp `OWNER`) → umur & jenis kelamin; jumlah pemakai akun; jenis kartu.

# %%
import pandas as pd

owners = disp[disp["type"] == "OWNER"].merge(client, on="client_id", how="left",
                                             validate="one_to_one")
owners = owners.rename(columns={"district_id": "client_district_id"})
owners = owners[[KEY, "disp_id", "gender", "birth_date", "client_district_id"]]
users_per_account = disp.groupby(KEY).size()
users_per_account.name = "users_per_account"
owner_card = owners.merge(card[["disp_id", "type", "issued"]], on="disp_id", how="left",
                          validate="one_to_one")
owner_card = owner_card.rename(columns={"type": "card_type"})
owner_card["card_type"] = owner_card["card_type"].fillna("tanpa_kartu")
print(owner_card.shape, "| akun unik:", owner_card[KEY].nunique())
print(owner_card["card_type"].value_counts().to_dict())

# %% [markdown]
# ## 6. Gabung semua ke `loans` dengan `validate` + cek jumlah baris tidak berubah

# %%
import pandas as pd

rows_before = len(loans)
features = loans.merge(trans_features, left_on="loan_id", right_index=True, how="left",
                       validate="one_to_one")
features = features.merge(account_features, on=KEY, how="left", validate="many_to_one")
features = features.merge(district_features, on="district_id", how="left",
                          validate="many_to_one")
features = features.merge(order_features, left_on=KEY, right_index=True, how="left",
                          validate="many_to_one")
features = features.merge(owner_card, on=KEY, how="left", validate="many_to_one")
features = features.merge(users_per_account, left_on=KEY, right_index=True, how="left",
                          validate="many_to_one")
print("baris sebelum", rows_before, "sesudah", len(features), "→ aman:",
      len(features) == rows_before)

# %% [markdown]
# Fitur turunan dari tanggal. Kartu yang terbit SETELAH pinjaman → dianggap belum punya
# kartu (aturan waktu lagi).

# %%
late_card = features["issued"] >= features["loan_date"]
features.loc[late_card, "card_type"] = "tanpa_kartu"
features["owner_age"] = (features["loan_date"] - features["birth_date"]).dt.days / 365.25
features["account_age_days"] = (features["loan_date"] - features["account_date"]).dt.days
features["payment_to_net_flow"] = features["payments"] / (features["net_flow"].abs() + 1)
print(features.shape, "| kartu terbit setelah pinjaman:", int(late_card.sum()))

# %% [markdown]
# ## 7. Skor CV per kelompok tabel
# Data kecil (± 500 pinjaman) → CV 5 fold diulang 3 seed supaya perbandingan tidak kebetulan.

# %%
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

for column in ["frequency", "gender", "card_type"]:
    features[column] = features[column].astype("category")
train = features[features["is_test"] == 0].reset_index(drop=True)
test = features[features["is_test"] == 1].reset_index(drop=True)
GROUPS = {"pinjaman": ["amount", "duration", "payments", "amount_per_month", "loan_year"],
          "+transaksi": list(trans_features.columns),
          "+akun/wilayah/klien": ["frequency", "account_age_days", "district_salary",
                                  "unemployment_96", "owner_age", "gender", "card_type",
                                  "users_per_account", "payment_to_net_flow"],
          "+order": list(order_features.columns)}   # UBAH
MODEL = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                           min_child_samples=10, verbose=-1)
columns = []
for name, group in GROUPS.items():
    columns = columns + group
    aucs = []
    for seed in [0, 1, 2]:
        splitter = StratifiedKFold(5, shuffle=True, random_state=seed)
        aucs.append(cross_val_score(MODEL, train[columns], train[TARGET], cv=splitter,
                                    scoring="roc_auc").mean())
    print(f"{name:22s} {len(columns):3d} fitur | AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# %% [markdown]
# Lonjakan besar dari tabel TANPA tanggal (di sini `order`, terutama `order_sipo`) = curigai.
# Tabel statis biasanya potret di AKHIR periode data (setelah pinjaman) → bisa berisi akibat
# dari macet (mis. pembayaran rutin dihentikan). Di lomba: boleh dipakai bila tabel yang sama
# tersedia untuk test, tetapi sebutkan di laporan bahwa fitur ini tidak ada saat keputusan.

# %% [markdown]
# Latih di semua train → proba test → `submission.csv`. Cek ke kunci (hanya demo).

# %%
import pandas as pd
from sklearn.metrics import roc_auc_score

MODEL.fit(train[columns], train[TARGET])
submission = pd.DataFrame()
submission["loan_id"] = test["loan_id"]                                   # UBAH
submission["bad"] = MODEL.predict_proba(test[columns])[:, 1]              # UBAH
submission.to_csv("submission.csv", index=False)
answer = submission["loan_id"].map(dict(zip(test_labels["loan_id"], test_labels["bad"])))
print("AUC test:", round(roc_auc_score(answer, submission["bad"]), 4))
