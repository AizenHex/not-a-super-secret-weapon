# %% [markdown]
# # Multi-tabel — Entity Resolution (mencocokkan record antar dua tabel)
# **Kapan dipakai:** dua tabel berisi hal yang sama (toko, perusahaan, produk, artikel) tetapi
# ditulis beda (huruf besar/kecil, singkatan, salah ketik) dan tidak punya kunci bersama.
#
# **Idenya:**
# - **Blocking:** hanya bandingkan pasangan yang kunci kasarnya sama (tahun + 4 huruf awal
#   judul), supaya tidak membandingkan jutaan pasangan.
# - Tiap pasangan kandidat diberi skor kemiripan teks (RapidFuzz, 0–100).
# - Model belajar dari pasangan berlabel: cocok / tidak.
#
# **Langkah:**
# 1. Bersihkan teks, buat kunci blocking, daftar pasangan kandidat.
# 2. Skor kemiripan tiap pasangan.
# 3. Latih model pasangan, pilih ambang, simpan pasangan yang cocok.
#
# **Data demo:** DBLP-ACM (benchmark Leipzig): 2.616 artikel DBLP vs 2.294 artikel ACM,
# 2.224 pasangan benar.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/multitabel/dblp_acm"   # GANTI saat lomba
TABLE_A_CSV = f"{DATA_FOLDER}/DBLP2.csv"
TABLE_B_CSV = f"{DATA_FOLDER}/ACM.csv"
MATCHES_CSV = f"{DATA_FOLDER}/DBLP-ACM_perfectMapping.csv"   # pasangan benar (berlabel)
ENCODING = "latin-1"

ID_COL = "id"
MATCH_A_COL = "idDBLP"        # kolom id tabel A di file pasangan benar
MATCH_B_COL = "idACM"
NAME_COL = "title"            # kolom teks utama yang dibandingkan
EXTRA_COL = "authors"         # kolom teks kedua
BLOCK_COL = "year"            # kolom yang harus SAMA agar sepasang record dibandingkan
BLOCK_PREFIX = 4              # huruf awal nama yang juga harus sama (kecilkan bila recall rendah)
THRESHOLD = 0.5               # naikkan = precision lebih tinggi, turunkan = recall lebih tinggi
SEED = 42

# %% [markdown]
# ## Baca data

# %%
table_a = pd.read_csv(TABLE_A_CSV, encoding=ENCODING)
table_b = pd.read_csv(TABLE_B_CSV, encoding=ENCODING)
matches = pd.read_csv(MATCHES_CSV)
print(table_a.shape, table_b.shape, matches.shape)
table_b.head()

# %% [markdown]
# ## Langkah 1a — bersihkan teks + kunci blocking (sama untuk kedua tabel)

# %%
for table in [table_a, table_b]:
    name = table[NAME_COL].str.lower()
    name = name.str.replace("[^a-z0-9 ]", " ", regex=True)    # buang tanda baca
    name = name.str.replace(" +", " ", regex=True)            # rapikan spasi
    name = name.str.strip()
    table["clean_name"] = name
    extra = table[EXTRA_COL].fillna("")
    table["clean_extra"] = extra.str.lower()
    table["block"] = table[BLOCK_COL].astype(str) + "_" + name.str[:BLOCK_PREFIX]
print(table_a[["clean_name", "block"]].head(3))

# %% [markdown]
# ## Langkah 1b — pasangan kandidat (kunci blok sama)
# `rows_by_block` = untuk tiap kunci blok, daftar nomor baris tabel B yang punya kunci itu.

# %%
rows_by_block = {}
for row_b, block in enumerate(table_b["block"]):
    if block not in rows_by_block:
        rows_by_block[block] = []
    rows_by_block[block].append(row_b)

candidates = []
for row_a, block in enumerate(table_a["block"]):
    for row_b in rows_by_block.get(block, []):
        candidates.append((row_a, row_b))
print("kandidat:", len(candidates), "dari", len(table_a) * len(table_b), "pasangan penuh")

# %% [markdown]
# ## Langkah 2 — skor kemiripan tiap pasangan
# - token_set: kata yang sama, urutan diabaikan. ratio: kemiripan huruf per huruf.
# - jaro_winkler: tahan salah ketik, menghargai awalan yang sama.

# %%
true_pairs = set(zip(matches[MATCH_A_COL], matches[MATCH_B_COL]))
rows = []
for row_a, row_b in candidates:
    name_a = table_a["clean_name"][row_a]
    name_b = table_b["clean_name"][row_b]
    id_a = table_a[ID_COL][row_a]
    id_b = table_b[ID_COL][row_b]
    rows.append([id_a, id_b,
                 fuzz.token_set_ratio(name_a, name_b),
                 fuzz.ratio(name_a, name_b),
                 JaroWinkler.similarity(name_a, name_b) * 100,
                 fuzz.token_set_ratio(table_a["clean_extra"][row_a], table_b["clean_extra"][row_b]),
                 abs(len(name_a) - len(name_b)),
                 int((id_a, id_b) in true_pairs)])

FEATURE_COLS = ["token_set_name", "ratio_name", "jaro_winkler_name", "token_set_extra",
                "length_gap"]
pairs = pd.DataFrame(rows, columns=["id_a", "id_b"] + FEATURE_COLS + ["is_match"])
print(pairs["is_match"].value_counts())
print("recall blocking:", round(pairs["is_match"].sum() / len(matches), 3))
pairs.groupby("is_match")[FEATURE_COLS].mean()

# %%
not_match = pairs[pairs["is_match"] == 0]
is_match = pairs[pairs["is_match"] == 1]
plt.hist(not_match["token_set_name"], bins=30, alpha=0.6, label="tidak cocok")
plt.hist(is_match["token_set_name"], bins=30, alpha=0.6, label="cocok")
plt.xlabel("token_set_ratio nama")
plt.legend()
plt.show()

# %% [markdown]
# ## Langkah 3a — split per record tabel A
# Satu record A tidak boleh muncul di latih dan uji sekaligus.

# %%
splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED)
train_rows, test_rows = next(splitter.split(pairs, groups=pairs["id_a"]))
train = pairs.iloc[train_rows]
test = pairs.iloc[test_rows]
print("latih:", len(train), "| uji:", len(test))

# %% [markdown]
# ## Langkah 3b — model pasangan
# Pilih SATU model: hapus `#` di depan baris yang ingin dipakai, beri `#` pada baris lain.

# %%
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
# model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=8,
#                        random_state=SEED, verbose=-1)

model.fit(train[FEATURE_COLS], train["is_match"])
probabilities = model.predict_proba(test[FEATURE_COLS])
test_scores = probabilities[:, 1]

# %% [markdown]
# ## Langkah 3c — lihat efek ambang, tulis pilihan di `THRESHOLD`

# %%
for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    guesses = (test_scores >= threshold).astype(int)
    print(threshold,
          "| precision:", round(precision_score(test["is_match"], guesses), 4),
          "| recall:", round(recall_score(test["is_match"], guesses), 4),
          "| F1:", round(f1_score(test["is_match"], guesses), 4))

# %% [markdown]
# ## Simpan hasil pencocokan pasangan uji
# Saat lomba: susun `pairs` dari tabel test dengan langkah yang sama, latih model di semua
# pasangan berlabel, lalu prediksi dengan `THRESHOLD`.

# %%
predictions = (test_scores >= THRESHOLD).astype(int)

submission = pd.DataFrame()
submission["id_a"] = test["id_a"]
submission["id_b"] = test["id_b"]
submission["is_match"] = predictions
submission.to_csv("submission.csv", index=False)
print("pasangan diprediksi cocok:", submission["is_match"].sum())
submission.head()
