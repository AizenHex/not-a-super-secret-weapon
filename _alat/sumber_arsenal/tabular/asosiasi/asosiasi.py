# %% [markdown]
# # Aturan asosiasi (market basket): "yang membeli A juga membeli B"
#
# **Butuh:** daftar transaksi (tiap transaksi = kumpulan item), atau tabel panjang
# (id_transaksi, item). **Menghasilkan:** tabel aturan `A → B` dengan support, confidence,
# lift; rekomendasi item untuk keranjang; `outputs/aturan.csv`.
#
# | ukuran | arti | rumus |
# |---|---|---|
# | support(A) | porsi transaksi yang memuat A | n(A) / N |
# | confidence(A→B) | dari yang membeli A, berapa persen juga membeli B | n(A,B) / n(A) |
# | lift(A→B) | berapa kali lebih sering dari kebetulan; > 1 = saling menarik | conf / sup(B) |
# | leverage | sup(A,B) − sup(A)·sup(B); 0 = tidak berhubungan | |
# | conviction | (1 − sup(B)) / (1 − conf); besar = aturan jarang salah | |
#
# Apriori (Agrawal & Srikant 1994): kumpulan item hanya bisa sering kalau semua bagiannya
# sering → kandidat besar dibangun dari kandidat kecil yang lolos `MIN_SUPPORT`.
# Aturan dengan confidence tinggi belum tentu berarti: "→ whole milk" confidence-nya tinggi
# karena susu memang dibeli di 26% transaksi. Selalu lihat **lift** juga.
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | transaksi → tabel 0/1 (baris transaksi × kolom item) | **ya** |
# | 2 | support satu item | **ya** |
# | 3 | pasangan item: support, confidence, lift (numpy, ditulis sendiri) | **ya** |
# | 4 | apriori + aturan lengkap dengan `mlxtend` (≥ 2 item di kiri) | disarankan |
# | 5 | `recommend(basket)`: item yang disarankan untuk satu keranjang | opsional |
# | 6 | uji rekomendasi: sembunyikan 1 item, tebak kembali (hit rate) | opsional |
# | 7 | simpan | **ya** |

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# 9835 transaksi toko kelontong; satu baris = item dipisah koma (jumlah item berbeda-beda).

# %%
DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

baskets = []
with open(f"{DEMO_FOLDER}/unsupervised/groceries.csv", encoding="utf-8") as file:
    for line in file:
        items = []
        for item in line.strip().split(","):
            if item:
                items.append(item.strip())
        baskets.append(sorted(set(items)))
print(len(baskets), "transaksi | contoh:", baskets[:2])
# Tabel panjang (id, item)? → baskets = table.groupby("id")["item"].apply(list).tolist()

# %% [markdown]
# ## 1. Transaksi → tabel 0/1 `onehot` (baris = transaksi, kolom = item)

# %%
import numpy as np
import pandas as pd

all_items = []
for basket in baskets:
    all_items.extend(basket)
all_items = sorted(set(all_items))
position = dict(zip(all_items, range(len(all_items))))
matrix = np.zeros((len(baskets), len(all_items)), dtype=bool)
for row, basket in enumerate(baskets):
    for item in basket:
        matrix[row, position[item]] = True
onehot = pd.DataFrame(matrix, columns=all_items)
print(onehot.shape, "| rata-rata item per transaksi:", round(matrix.sum(axis=1).mean(), 2))

# %% [markdown]
# ## 2. Support satu item: item paling sering dibeli

# %%
item_support = onehot.mean().sort_values(ascending=False)
print(item_support.head(10).round(3))
print("item yang muncul < 1% transaksi:", int((item_support < 0.01).sum()), "dari",
      len(item_support))

# %% [markdown]
# ## 3. Pasangan item (A → B) dengan numpy: hitung bersama lewat perkalian matriks
# `together[a, b]` = jumlah transaksi yang memuat a DAN b. Cepat untuk ratusan item.

# %%
import numpy as np
import pandas as pd

MIN_COUNT = 50   # UBAH: minimal jumlah transaksi yang memuat pasangan itu
counts = matrix.astype(np.int32)
together = counts.T @ counts
single = np.diag(together)
total = len(baskets)
rows = []
for first, second in np.argwhere(together >= MIN_COUNT):
    if first != second:
        confidence = together[first, second] / single[first]
        lift = confidence / (single[second] / total)
        rows.append({"A": all_items[first], "B": all_items[second],
                     "support": together[first, second] / total,
                     "confidence": confidence, "lift": lift})
pairs = pd.DataFrame(rows).round(3)
pairs = pairs.sort_values("lift", ascending=False)
print(len(pairs), "aturan pasangan")
print(pairs.head(10).to_string(index=False))

# %% [markdown]
# ## 4. Apriori `mlxtend`: kumpulan item sering (bisa 3+ item) → aturan lengkap
# `MIN_SUPPORT` terlalu kecil = sangat lambat & aturan meledak; mulai dari 0,01 lalu turunkan.

# %%
from mlxtend.frequent_patterns import apriori, association_rules

MIN_SUPPORT = 0.01      # UBAH
MIN_CONFIDENCE = 0.3    # UBAH
itemsets = apriori(onehot, min_support=MIN_SUPPORT, use_colnames=True)
itemsets["size"] = itemsets["itemsets"].apply(len)
size_counts = itemsets["size"].value_counts().sort_index()
print(size_counts.to_dict(), "(ukuran: jumlah)")
rules = association_rules(itemsets, num_itemsets=len(onehot), metric="confidence",
                          min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > 1].sort_values("lift", ascending=False).round(3)
columns = ["antecedents", "consequents", "support", "confidence", "lift", "conviction"]
print(len(rules), "aturan")
print(rules[columns].head(10).to_string(index=False))
# Data besar & apriori lambat? Ganti apriori → fpgrowth (hasil sama, lebih cepat).

# %% [markdown]
# Cara baca: `{citrus fruit, root vegetables} → {other vegetables}` lift 3 = pembeli dua
# item kiri 3× lebih mungkin membeli sayuran lain dibanding pembeli acak.

# %% [markdown]
# ## 5. `recommend(basket)`: aturan yang sisi kirinya ada di keranjang → item sisi kanan

# %%
import pandas as pd


def recommend(basket, rules, top=5, sort_by="lift"):
    basket = set(basket)
    scores = {}
    for left, right, value in zip(rules["antecedents"], rules["consequents"],
                                  rules[sort_by]):
        if left <= basket:
            for item in right:
                if item not in basket:
                    scores[item] = max(scores.get(item, 0), value)
    ranked = pd.Series(scores, dtype=float).sort_values(ascending=False)
    return ranked.head(top)


print(recommend(["whole milk", "yogurt"], rules))

# %% [markdown]
# ## 6. Uji rekomendasi: aturan dari 80% transaksi, lalu di 20% sisanya sembunyikan 1 item
# Hit rate@5 = seberapa sering item yang disembunyikan ada di 5 rekomendasi teratas.
# Pembanding: selalu menyarankan 5 item terlaris.

# %%
import numpy as np

order = np.random.default_rng(42).permutation(len(baskets))
cut = int(0.8 * len(baskets))
fit_itemsets = apriori(onehot.iloc[order[:cut]], min_support=MIN_SUPPORT, use_colnames=True)
fit_rules = association_rules(fit_itemsets, num_itemsets=cut, metric="confidence",
                              min_threshold=0.1)
popular = onehot.iloc[order[:cut]].mean().sort_values(ascending=False).index


def hit_rate(sort_by, top=5):
    rng = np.random.default_rng(0)
    hits, tried = 0, 0
    for row in order[cut:]:
        if len(baskets[row]) >= 2:
            hidden = baskets[row][rng.integers(len(baskets[row]))]
            kept = set(baskets[row]) - {hidden}
            guess = []
            if sort_by != "laris saja":
                guess = list(recommend(kept, fit_rules, top, sort_by).index)
            for item in popular:
                if len(guess) < top and item not in kept and item not in guess:
                    guess.append(item)
            hits += hidden in guess
            tried += 1
    return hits / tried


for sort_by in ["laris saja", "lift", "confidence"]:
    print(f"hit rate@5 {sort_by:11s}: {hit_rate(sort_by):.3f}")

# %% [markdown]
# Hasil data contoh: laris 0,274 | urut lift 0,244 | urut confidence 0,285. Urut **lift**
# menyarankan item langka (menarik untuk laporan, tapi jarang tepat); untuk MENEBAK item,
# urut **confidence** + isi sisa dengan item terlaris. Bedanya kecil → data belanja ini
# memang didominasi item populer. Laporkan dua-duanya: lift untuk insight, confidence untuk
# rekomendasi.

# %% [markdown]
# ## 7. Simpan aturan (frozenset → teks "a, b")

# %%
import os

saved = rules.copy()
for column in ["antecedents", "consequents"]:
    saved[column] = saved[column].apply(sorted).str.join(", ")
os.makedirs("outputs", exist_ok=True)
saved[columns].to_csv("outputs/aturan.csv", index=False)
pairs.to_csv("outputs/aturan_pasangan.csv", index=False)
print("tersimpan:", len(saved), "aturan |", len(pairs), "pasangan")
