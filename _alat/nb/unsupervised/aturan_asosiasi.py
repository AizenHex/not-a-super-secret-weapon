# %% [markdown]
# # Unsupervised — Aturan Asosiasi (market basket)
# **Kapan dipakai:** data berupa keranjang belanja / daftar item per transaksi, dan diminta
# "item apa yang sering dibeli bersama" atau "kalau beli A, biasanya beli B".
#
# **Idenya:** cari kumpulan item yang sering muncul bersama (itemset), lalu ubah jadi aturan
# A → B. Ukuran aturan:
# - **support** = porsi transaksi yang berisi A dan B.
# - **confidence** = dari transaksi berisi A, berapa porsi yang juga berisi B.
# - **lift** = confidence dibagi porsi B biasa. Lift > 1 = A benar-benar menaikkan peluang B.
#
# **Langkah:**
# 1. Baca transaksi, ubah jadi tabel 0/1 (baris = transaksi, kolom = item).
# 2. Cari itemset yang sering muncul, lalu aturan.
# 3. Urutkan aturan berdasarkan lift, simpan.
#
# **Data demo:** Groceries (tanpa header, satu transaksi per baris, item dipisah koma).

# %%
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, hmine
from mlxtend.preprocessing import TransactionEncoder

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/unsupervised"   # GANTI saat lomba
TRANSACTIONS_CSV = f"{DATA_FOLDER}/groceries.csv"

ITEM_SEPARATOR = ","
MIN_SUPPORT = 0.01       # itemset harus ada di minimal 1% transaksi
MIN_CONFIDENCE = 0.2     # confidence minimal aturan

# %% [markdown]
# ## Baca data
# Tiap baris file dipecah per koma, spasi di tepi item dibuang, item kosong dilewati.

# %%
transactions = []
with open(TRANSACTIONS_CSV, encoding="utf-8") as file:
    for line in file:
        basket = []
        for item in line.split(ITEM_SEPARATOR):
            item = item.strip()
            if item != "":
                basket.append(item)
        transactions.append(basket)

total_items = 0
for basket in transactions:
    total_items += len(basket)
print("jumlah transaksi:", len(transactions))
print("rata-rata item per transaksi:", round(total_items / len(transactions), 2))
print(transactions[:3])

# %% [markdown]
# ## Langkah 1 — tabel 0/1
# True = item ada di transaksi itu.

# %%
encoder = TransactionEncoder()
encoder.fit(transactions)
onehot = pd.DataFrame(encoder.transform(transactions), columns=encoder.columns_)
print(onehot.shape)

item_counts = onehot.sum()
item_counts = item_counts.sort_values(ascending=False)
item_counts.head(15).plot.barh()
plt.title("15 item paling sering (jumlah transaksi)")
plt.gca().invert_yaxis()
plt.show()

# %% [markdown]
# ## Langkah 2 — itemset dan aturan
# Pilih SATU algoritma: hapus `#` di depan baris yang ingin dipakai. Hasilnya sama,
# bedanya hanya kecepatan (fpgrowth biasanya paling cepat untuk data besar).

# %%
itemsets = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
# itemsets = fpgrowth(onehot, min_support=MIN_SUPPORT, use_colnames=True)
# itemsets = hmine(onehot, min_support=MIN_SUPPORT, use_colnames=True)

rules = association_rules(itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print("itemset:", len(itemsets), "| aturan:", len(rules))

# %% [markdown]
# ## Langkah 3 — urutkan berdasarkan lift

# %%
rules = rules.sort_values("lift", ascending=False)
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

rules.plot.scatter(x="support", y="confidence", c="lift", cmap="viridis", s=20)
plt.show()

# %% [markdown]
# ## Simpan aturan
# Kumpulan item (frozenset) diubah jadi teks "item1, item2" supaya mudah dibaca di CSV.

# %%
antecedent_texts = []
consequent_texts = []
for antecedent, consequent in zip(rules["antecedents"], rules["consequents"]):
    antecedent_texts.append(", ".join(sorted(antecedent)))
    consequent_texts.append(", ".join(sorted(consequent)))

submission = pd.DataFrame()
submission["antecedents"] = antecedent_texts
submission["consequents"] = consequent_texts
submission["support"] = rules["support"].to_numpy()
submission["confidence"] = rules["confidence"].to_numpy()
submission["lift"] = rules["lift"].to_numpy()
submission.to_csv("submission.csv", index=False)
submission.head()
