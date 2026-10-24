# %% [markdown]
# # Unsupervised — Pola Sekuensial ("setelah A, orang biasanya B")
# **Kapan dipakai:** data berisi urutan kejadian per orang (film yang ditonton, halaman yang
# dibuka, produk yang dibeli dari waktu ke waktu) dan diminta pola urutan.
#
# **Idenya:** beda dengan aturan asosiasi biasa, di sini **urutan waktu** dipakai. Pola A → B
# dihitung bila B muncul paling jauh `WINDOW` langkah sesudah A pada urutan satu orang.
# - **support** = porsi orang yang punya pola itu.
# - **confidence** = dari orang yang pernah A, berapa porsi yang lalu B.
# - **lift** = confidence dibagi porsi B biasa. Lift > 1 = B lebih sering muncul sesudah A.
#
# **Langkah:**
# 1. Urutkan kejadian per orang berdasarkan waktu.
# 2. Hitung pola di 70% orang, cek ulang confidence-nya di 30% sisanya.
# 3. Hitung ulang dengan semua orang, simpan pola.
#
# **Data demo:** MovieLens 100k (urut `timestamp` per user). Banyak rating diberi pada detik
# yang sama, jadi urutan dalam satu sesi tidak pasti.

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/recsys/ml-100k"   # GANTI saat lomba
EVENTS_CSV = f"{DATA_FOLDER}/ratings.csv"
ITEMS_CSV = f"{DATA_FOLDER}/movies.csv"

USER_COL = "user"
ITEM_COL = "item"
TIME_COL = "timestamp"
TITLE_COL = "title"
WINDOW = 10          # B dihitung bila muncul paling jauh 10 langkah sesudah A
MIN_USERS = 20       # pola harus dimiliki minimal 20 orang
SEED = 42

# %% [markdown]
# ## Baca data

# %%
events = pd.read_csv(EVENTS_CSV)
items = pd.read_csv(ITEMS_CSV)
titles = pd.Series(items[TITLE_COL].to_numpy(), index=items[ITEM_COL])
events = events.sort_values([USER_COL, TIME_COL, ITEM_COL], kind="stable")
print(events.shape)
print("user:", events[USER_COL].nunique(), "| item:", events[ITEM_COL].nunique())
events.head()

# %% [markdown]
# ## Pilih data urutan (satu baris aktif)

# %%
sequence = events
# sequence = events[events["rating"] >= 4]       # hanya film yang disukai

# %% [markdown]
# ## Langkah 1 — hitung pola A → B
# Tiap orang dihitung sekali per pola, walau polanya muncul berkali-kali di urutannya.

# %%
def pattern_table(table):
    pair_users = {}
    item_users = {}
    for _, user_items in table.groupby(USER_COL)[ITEM_COL]:
        user_items = user_items.tolist()
        user_pairs = set()
        for i in range(len(user_items)):
            for later in user_items[i + 1:i + 1 + WINDOW]:
                user_pairs.add((user_items[i], later))
        for pair in user_pairs:
            pair_users[pair] = pair_users.get(pair, 0) + 1
        for item in set(user_items):
            item_users[item] = item_users.get(item, 0) + 1
    patterns = pd.DataFrame(list(pair_users.keys()), columns=["A", "B"])
    patterns["n_users"] = list(pair_users.values())
    patterns["support"] = patterns["n_users"] / table[USER_COL].nunique()
    patterns["confidence"] = patterns["n_users"] / patterns["A"].map(item_users)
    b_share = patterns["B"].map(item_users) / table[USER_COL].nunique()
    patterns["lift"] = patterns["confidence"] / b_share
    return patterns


# coba pada 3 orang pertama
first_users = sequence[USER_COL].unique()[:3]
print(pattern_table(sequence[sequence[USER_COL].isin(first_users)]).head())

# %% [markdown]
# ## Langkah 2 — cari pola di 70% orang, cek di 30% sisanya
# Pola yang bagus tetap punya confidence tinggi di orang lain (bukan kebetulan).

# %%
rng = np.random.default_rng(SEED)
all_users = sequence[USER_COL].unique()
train_users = rng.choice(all_users, int(len(all_users) * 0.7), replace=False)
is_train = sequence[USER_COL].isin(train_users)

patterns = pattern_table(sequence[is_train])
patterns = patterns[(patterns["n_users"] >= MIN_USERS) & (patterns["lift"] > 1)]
patterns = patterns.sort_values("confidence", ascending=False)
print("jumlah pola:", len(patterns))

test_patterns = pattern_table(sequence[~is_train])
test_confidence = {}
for item_a, item_b, value in zip(test_patterns["A"], test_patterns["B"],
                                 test_patterns["confidence"]):
    test_confidence[(item_a, item_b)] = value

top = patterns.head(20).copy()
top_test = []
for item_a, item_b in zip(top["A"], top["B"]):
    top_test.append(test_confidence.get((item_a, item_b), 0))
top["confidence_test"] = top_test
top["title_A"] = top["A"].map(titles)
top["title_B"] = top["B"].map(titles)
print("rata confidence train:", round(top["confidence"].mean(), 3))
print("rata confidence uji  :", round(top["confidence_test"].mean(), 3))
top[["title_A", "title_B", "n_users", "confidence", "confidence_test", "lift"]].round(3)

# %% [markdown]
# ## Langkah 3 — hitung ulang dengan semua orang, simpan

# %%
patterns = pattern_table(sequence)
patterns = patterns[(patterns["n_users"] >= MIN_USERS) & (patterns["lift"] > 1)]
patterns = patterns.sort_values("confidence", ascending=False)

submission = pd.DataFrame()
submission["A"] = patterns["A"].map(titles)
submission["B"] = patterns["B"].map(titles)
submission["support"] = patterns["support"]
submission["confidence"] = patterns["confidence"]
submission["lift"] = patterns["lift"]
submission.to_csv("submission.csv", index=False)
submission.head(10)
