# %% [markdown]
# # Mengolah tabel: pilih, urut, kelompok, pivot, gabung, fitur dari tabel lain, fitur waktu
#
# **Butuh:** tabel pandas yang sudah bersih (`tabel_bersih.ipynb`).
# **Menghasilkan:** tabel baru per bagian (nama variabel ditulis di judul bagian).
#
# ## Cara menempel ke notebook-mu
# Tiap bagian berdiri sendiri. Salin bagian yang perlu, ganti nama tabel/kolom di baris `# UBAH`.
# Data contoh: `customers` (satu baris per pelanggan) dan `transactions` (banyak baris per
# pelanggan, ada tanggal).
#
# | ingin | bagian |
# |---|---|
# | ambil baris/kolom tertentu | 1 |
# | urut, peringkat, N teratas per kelompok | 2 |
# | ringkasan per kelompok (jumlah, rata-rata, ...) | 3 |
# | tiap baris diberi nilai kelompoknya (rata-rata kota, % dari total) | 4 |
# | tabel silang / pivot (baris × kolom) | 5 |
# | lebar ↔ panjang (kolom bulan → baris) | 6 |
# | gabung dua tabel lewat kunci | 7 |
# | **fitur dari tabel lain** (banyak transaksi → 1 baris per pelanggan) | 8 |
# | fitur waktu: lag, rolling, selisih hari, kumulatif | 9 |
# | angka → kelompok (umur → rentang) | 10 |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)

# %%
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
customers = pd.DataFrame({"customer_id": [1, 2, 3, 4],
                          "city": ["jakarta", "bandung", "jakarta", "surabaya"],
                          "age": [23, 35, 47, 31]})
dates = pd.date_range("2026-01-01", periods=12, freq="7D")
transactions = pd.DataFrame({"customer_id": rng.choice([1, 2, 3], 12),
                             "date": dates,
                             "category": rng.choice(["makanan", "elektronik", "baju"], 12),
                             "amount": rng.integers(10, 200, 12) * 1000})
print(customers)
print(transactions.head())

# %% [markdown]
# ## 1. Pilih baris dan kolom
# Syarat banyak: `&` (dan), `|` (atau), `~` (bukan), tiap syarat dalam kurung.

# %%
selected = customers[(customers["age"] > 25) & (customers["city"] == "jakarta")]
print(selected)
selected = customers[customers["city"].isin(["jakarta", "bandung"])]
print(selected[["customer_id", "city"]])
big = transactions[~(transactions["category"] == "baju") & (transactions["amount"] >= 100_000)]
print(len(big), "transaksi besar bukan baju")
numeric_only = transactions.select_dtypes("number")   # semua kolom angka
print(numeric_only.columns.tolist())

# %% [markdown]
# ## 2. Urut, peringkat, N teratas per kelompok
# `rank(ascending=False)`: 1 = terbesar. N teratas per kelompok: urutkan, lalu `groupby().head(N)`.

# %%
TOP_N = 2   # UBAH

ordered = transactions.sort_values(["customer_id", "amount"], ascending=[True, False])
top_per_customer = ordered.groupby("customer_id").head(TOP_N)
print(top_per_customer)
transactions["rank_in_customer"] = transactions.groupby("customer_id")["amount"].rank(
    ascending=False)
print(transactions.head())

# %% [markdown]
# ## 3. Ringkasan per kelompok → `summary`
# `as_index=False` supaya hasil tetap tabel biasa. Nama kolom hasil = `nama_baru=(kolom, fungsi)`.
# Fungsi: "count", "sum", "mean", "median", "min", "max", "std", "nunique", "first", "last".

# %%
summary = transactions.groupby("customer_id", as_index=False).agg(
    n_transactions=("amount", "count"),
    total_amount=("amount", "sum"),
    mean_amount=("amount", "mean"),
    n_categories=("category", "nunique"),
)
print(summary)
two_keys = transactions.groupby(["customer_id", "category"], as_index=False)["amount"].sum()
print(two_keys)

# %% [markdown]
# ## 4. Tiap baris diberi nilai kelompoknya (`transform`)
# Hasil `transform` sepanjang tabel asli, jadi bisa langsung jadi kolom baru.

# %%
group_mean = transactions.groupby("customer_id")["amount"].transform("mean")
transactions["amount_vs_mean"] = transactions["amount"] / group_mean
group_total = transactions.groupby("customer_id")["amount"].transform("sum")
transactions["share_of_customer"] = transactions["amount"] / group_total
print(transactions[["customer_id", "amount", "amount_vs_mean", "share_of_customer"]])

# %% [markdown]
# ## 5. Tabel silang / pivot → `pivot`
# `crosstab`: hitung jumlah kemunculan. `pivot_table`: nilai + fungsi agregat. Kosong → 0.

# %%
import pandas as pd

counts = pd.crosstab(transactions["customer_id"], transactions["category"])
print(counts)
pivot = transactions.pivot_table(index="customer_id", columns="category", values="amount",
                                 aggfunc="sum", fill_value=0)
new_names = []
for name in pivot.columns:
    new_names.append("amount_" + name)   # nama kolom rata
pivot.columns = new_names
pivot = pivot.reset_index()
print(pivot)

# %% [markdown]
# ## 6. Lebar ↔ panjang
# `melt`: banyak kolom (jan, feb, mar) → dua kolom (month, value). `pivot`: kebalikannya.

# %%
import pandas as pd

wide = pd.DataFrame({"store": ["A", "B"], "jan": [10, 20], "feb": [11, 21], "mar": [12, 22]})
long = wide.melt(id_vars=["store"], var_name="month", value_name="sales")
print(long)
back = long.pivot(index="store", columns="month", values="sales")
back = back.reset_index()
print(back)

# %% [markdown]
# ## 7. Gabung dua tabel lewat kunci → `merged`
# `how="left"`: semua baris kiri tetap, pasangan tidak ada → NaN. Cek jumlah baris: kalau naik,
# kunci di tabel kanan dobel (satu baris kiri jadi beberapa).
# Hanya butuh satu kolom dari tabel lain? Lebih sederhana pakai `map` (baris kedua).

# %%
merged = transactions.merge(customers, on="customer_id", how="left")
print("baris:", len(transactions), "→", len(merged))
print(merged.head())
city_of = dict(zip(customers["customer_id"], customers["city"]))
transactions["city"] = transactions["customer_id"].map(city_of)
print(transactions[["customer_id", "city"]].head(3))

# %% [markdown]
# ## 8. Fitur dari tabel lain: banyak baris per pelanggan → 1 baris per pelanggan → `features`
# Pola soal multi-tabel: tabel utama (1 baris per id) + tabel riwayat (banyak baris per id).
# Ringkas riwayat per id (bagian 3 + 5), lalu `merge` ke tabel utama. Id tanpa riwayat → isi 0.

# %%
ID = "customer_id"   # UBAH

history = transactions.groupby(ID, as_index=False).agg(
    n_transactions=("amount", "count"),
    total_amount=("amount", "sum"),
    max_amount=("amount", "max"),
    last_date=("date", "max"),
)
history["days_since_last"] = (pd.Timestamp("2026-04-01") - history["last_date"]).dt.days
history = history.drop(columns=["last_date"])
per_category = transactions.pivot_table(index=ID, columns="category", values="amount",
                                        aggfunc="sum", fill_value=0)
new_names = []
for name in per_category.columns:
    new_names.append("total_" + name)
per_category.columns = new_names
features = customers.merge(history, on=ID, how="left")
features = features.merge(per_category.reset_index(), on=ID, how="left")
features["days_since_last"] = features["days_since_last"].fillna(999)   # tidak pernah belanja
features = features.fillna(0)   # pelanggan tanpa transaksi: jumlah/total = 0

print(features.to_string())

# %% [markdown]
# ## 9. Fitur waktu per kelompok: lag, rolling, selisih, kumulatif
# **Wajib urut waktu dulu** di dalam tiap kelompok. `shift(1)` = nilai transaksi sebelumnya
# (tidak bocor ke masa depan). `rolling(3)` = rata-rata 3 transaksi terakhir **termasuk** baris ini;
# untuk prediksi masa depan pakai `shift(1)` dulu.

# %%
ordered = transactions.sort_values([ID, "date"]).copy()
group = ordered.groupby(ID)["amount"]
ordered["prev_amount"] = group.shift(1)
windows = ordered.groupby(ID)["prev_amount"].rolling(3, min_periods=1)   # UBAH: 3 = panjang jendela
ordered["rolling_mean_3"] = windows.mean().droplevel(0)

ordered["cumulative_amount"] = group.cumsum()
ordered["days_since_prev"] = ordered.groupby(ID)["date"].diff().dt.days
ordered["order_number"] = ordered.groupby(ID).cumcount() + 1
ordered["month"] = ordered["date"].dt.month
ordered["weekday"] = ordered["date"].dt.dayofweek   # 0 = Senin
print(ordered[[ID, "date", "amount", "prev_amount", "rolling_mean_3", "days_since_prev",
               "order_number"]].to_string())

# %% [markdown]
# ## 10. Angka → kelompok
# `cut`: batas ditentukan sendiri. `qcut`: dibagi supaya tiap kelompok berisi jumlah baris sama.

# %%
import pandas as pd

AGE_BINS = [0, 25, 35, 50, 200]                        # UBAH
AGE_LABELS = ["<25", "25-34", "35-49", "50+"]          # UBAH

customers["age_group"] = pd.cut(customers["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
transactions["amount_quartile"] = pd.qcut(transactions["amount"], q=4, labels=False)
print(customers)
print(transactions["amount_quartile"].value_counts().sort_index())
