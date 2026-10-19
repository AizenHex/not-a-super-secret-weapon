# %% [markdown]
# # Tabular — Optimasi Keputusan (Linear Programming, Integer/MIP, Simulasi Monte Carlo)
# **Kapan dipakai:** soal meminta **keputusan terbaik** di bawah batasan, bukan tebakan.
# Contoh: berapa unit tiap produk diproduksi (bahan terbatas), gudang mana yang dibuka
# (anggaran terbatas), seberapa berisiko rencana itu kalau permintaan tidak pasti.
#
# - LP: jumlah boleh pecahan → `scipy.optimize.linprog`.
# - MIP: jumlah harus bulat / keputusan ya-tidak → `scipy.optimize.milp`.
# - Monte Carlo: ulangi ribuan skenario acak → sebaran keuntungan dan peluang rugi.
#
# Semua angka soal ada di tabel tiap bagian. Ganti dengan data soal. Sering dipakai setelah
# model ramalan: hasil ramalan permintaan → masuk ke optimasi.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

# %% [markdown]
# ## Konfigurasi

# %%
N_SIMULATIONS = 10000
SEED = 42

# %% [markdown]
# ## 1. Linear Programming: rencana produksi
# Tiga produk, dua bahan baku terbatas. Tujuan: keuntungan sebesar mungkin.

# %%
products = pd.DataFrame()
products["nama"] = ["roti", "kue", "donat"]
products["untung_per_unit"] = [3.0, 5.0, 4.0]
products["tepung_per_unit"] = [2.0, 3.0, 1.5]       # kg
products["gula_per_unit"] = [0.5, 2.0, 1.0]         # kg
products["permintaan_maks"] = [400, 150, 300]       # unit (misalnya dari model ramalan)
FLOUR_STOCK = 1000
SUGAR_STOCK = 400
products

# %% [markdown]
# `linprog` selalu **meminimumkan**, jadi keuntungan diberi tanda minus.
# - `A_ub @ x <= b_ub`: pemakaian bahan tidak boleh melebihi stok.
# - `bounds`: tiap produk dibuat antara 0 dan permintaan maksimum.

# %%
cost = -products["untung_per_unit"].to_numpy()
usage = np.array([products["tepung_per_unit"], products["gula_per_unit"]])
stock = np.array([FLOUR_STOCK, SUGAR_STOCK])
limits = []
for max_demand in products["permintaan_maks"]:
    limits.append((0, max_demand))

lp_result = linprog(cost, A_ub=usage, b_ub=stock, bounds=limits, method="highs")
print(lp_result.message)
products["produksi_lp"] = lp_result.x.round(2)
lp_profit = -lp_result.fun
print("keuntungan maksimum:", round(lp_profit, 2))
print("sisa tepung, gula:", (stock - usage @ lp_result.x).round(2))
products[["nama", "produksi_lp"]]

# %% [markdown]
# ## 2. Integer / MIP: gudang mana yang dibuka?
# Tiap gudang punya biaya buka dan manfaat (misalnya jumlah pelanggan terlayani).
# Keputusan 0/1. Batasan: total biaya ≤ anggaran, dan minimal `MIN_WAREHOUSES` gudang dibuka.

# %%
warehouses = pd.DataFrame()
warehouses["nama"] = ["Medan", "Jakarta", "Surabaya", "Makassar", "Balikpapan", "Denpasar"]
warehouses["biaya_buka"] = [40, 90, 70, 50, 45, 35]
warehouses["manfaat"] = [55, 140, 100, 60, 50, 45]
BUDGET = 200
MIN_WAREHOUSES = 3

# %% [markdown]
# `milp` juga meminimumkan, jadi manfaat diberi minus. `integrality=1` = harus bilangan bulat;
# ditambah `Bounds(0, 1)` → nilainya hanya 0 (tidak dibuka) atau 1 (dibuka).

# %%
n_warehouses = len(warehouses)
objective = -warehouses["manfaat"].to_numpy()
budget_rule = LinearConstraint([warehouses["biaya_buka"].to_numpy()], 0, BUDGET)
count_rule = LinearConstraint([np.ones(n_warehouses)], MIN_WAREHOUSES, n_warehouses)
mip_result = milp(
    objective,
    constraints=[budget_rule, count_rule],
    integrality=np.ones(n_warehouses),
    bounds=Bounds(0, 1),
)
print(mip_result.message)
warehouses["dibuka"] = mip_result.x.round().astype(int)
total_cost = (warehouses["biaya_buka"] * warehouses["dibuka"]).sum()
print("total manfaat:", -mip_result.fun, "| total biaya:", total_cost)
warehouses

# %% [markdown]
# ## 3. Simulasi Monte Carlo: seberapa berisiko rencana produksi LP?
# Permintaan sebenarnya tidak pasti: dianggap normal dengan rata-rata = permintaan_maks dan
# simpangan `DEMAND_UNCERTAINTY` × rata-rata. Barang tak terjual dibuang, rugi `LEFTOVER_LOSS`
# per unit.

# %%
DEMAND_UNCERTAINTY = 0.25
LEFTOVER_LOSS = 1.5
random_generator = np.random.default_rng(SEED)

simulated_profit = []
for simulation in range(N_SIMULATIONS):
    total = 0
    for i in range(len(products)):
        mean_demand = products["permintaan_maks"][i]
        demand = random_generator.normal(mean_demand, DEMAND_UNCERTAINTY * mean_demand)
        demand = max(0, demand)
        made = products["produksi_lp"][i]
        sold = min(made, demand)
        leftover = made - sold
        total = total + sold * products["untung_per_unit"][i] - leftover * LEFTOVER_LOSS
    simulated_profit.append(total)
simulated_profit = np.array(simulated_profit)

# %% [markdown]
# Persentil 5% = keuntungan pada skenario buruk (hanya 5% skenario yang lebih buruk).

# %%
print("rata-rata keuntungan:", round(simulated_profit.mean(), 1))
print("persentil 5% (skenario buruk):", round(np.percentile(simulated_profit, 5), 1))
print("peluang untung < 80% target LP:", (simulated_profit < 0.8 * lp_profit).mean())

plt.hist(simulated_profit, bins=50)
plt.axvline(lp_profit, color="red", label="rencana LP (permintaan pasti)")
plt.xlabel("keuntungan")
plt.legend()
plt.title("Sebaran keuntungan dari 10.000 skenario permintaan")
plt.savefig("monte_carlo.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Simpan keputusan

# %%
products.to_csv("rencana_produksi.csv", index=False)
warehouses.to_csv("gudang_dibuka.csv", index=False)
