# %% [markdown]
# # Teks — Label Lemah (weak supervision)
# **Kapan dipakai:** data teks banyak tetapi **tanpa label** (atau label sangat sedikit), dan
# kamu tahu "aturan kasar" tiap kelas (kata kunci, pola regex, kamus).
#
# **Langkah:**
# 1. Beberapa **fungsi pelabelan** memberi suara kelas, atau abstain bila tidak cocok.
# 2. Nilai tiap fungsi (coverage, konflik, akurasi di sedikit teks berlabel).
# 3. Suara digabung (voting) → label lemah.
# 4. Model TF-IDF dilatih di label lemah → menebak teks yang tidak kena aturan.
#
# **Data demo:** SmSA (sentimen). Label asli hanya dipakai untuk mengukur, seolah-olah tidak ada.
# Ganti isi `LABELING_RULES` sesuai soal.

# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/sentiment"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/smsa_train.csv"
TEST_CSV = f"{DATA_FOLDER}/smsa_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/smsa_test_labels.csv"   # hanya ada di data latihan

TEXT_COL = "text"
TARGET = "label"     # hanya untuk mengukur, saat lomba kolom ini tidak ada
N_DEV = 200          # jumlah teks yang kamu labeli tangan untuk menilai tiap fungsi
SEED = 42

# Fungsi pelabelan: nama -> (kelas yang dipilih, pola regex). Tambah/hapus baris sesuka hati.
# Pola dicari di teks huruf kecil. Teks yang tidak cocok = abstain (tidak memberi suara).
LABELING_RULES = {
    "kata_positif": ("positive", r"\b(?:enak|mantap|puas|ramah|nyaman|recommended|suka|lezat)\b"),
    "kata_bagus": ("positive", r"\b(?:bagus|keren|terbaik|sempurna|cocok)\b"),
    "tertawa": ("positive", r"__laugh__|haha|wkwk"),
    "kata_negatif": ("negative", r"\b(?:kecewa|buruk|jelek|parah|kesal|menyesal|payah)\b"),
    "tidak_x": ("negative", r"\btidak (?:enak|bagus|puas|ramah|becus|sesuai)\b"),
    "keluhan": ("negative", r"\b(?:lambat|lama sekali|kotor|mahal|bau|korupsi)\b"),
    "berita": ("neutral", r"\b(?:periksa|kpk|rapat|resmi|menurut|mengatakan|akan)\b"),
}

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train[TEXT_COL] = train[TEXT_COL].fillna("")
test[TEXT_COL] = test[TEXT_COL].fillna("")
print("train:", train.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## 1. Jalankan semua fungsi pelabelan
# Hasilnya tabel `votes`: satu kolom per fungsi, isi = kelas atau kosong (abstain).

# %%
lower_text = train[TEXT_COL].str.lower()
votes = pd.DataFrame(index=train.index)
for name in LABELING_RULES:
    label, pattern = LABELING_RULES[name]
    matched = lower_text.str.contains(pattern, regex=True)
    votes[name] = None
    votes.loc[matched, name] = label
votes.head(10)

# %% [markdown]
# ## 2. Nilai tiap fungsi
# - **coverage**: persen teks yang kena fungsi ini.
# - **overlap**: kena fungsi ini dan minimal satu fungsi lain.
# - **konflik**: kena fungsi ini dan fungsi lain memberi kelas berbeda.
# - **akurasi_dev**: persen benar pada N_DEV teks yang dilabeli tangan.
#
# Fungsi dengan akurasi_dev rendah (di demo: `berita`) atau konflik tinggi sebaiknya diperbaiki
# polanya atau dihapus dari `LABELING_RULES`, lalu jalankan ulang.

# %%
dev = train.sample(N_DEV, random_state=SEED)   # GANTI: saat lomba, ini teks yang kamu labeli
votes_per_row = votes.notna().sum(axis=1)
report = pd.DataFrame(index=list(LABELING_RULES))
for name in LABELING_RULES:
    hit = votes[name].notna()
    conflicts = 0
    for i in votes.index[hit]:
        for other_name in LABELING_RULES:
            other_label = votes.loc[i, other_name]
            if other_label is not None and other_label != votes.loc[i, name]:
                conflicts += 1
                break
    report.loc[name, "coverage"] = hit.mean()
    report.loc[name, "overlap"] = (hit & (votes_per_row > 1)).mean()
    report.loc[name, "konflik"] = conflicts / len(votes)
    dev_hit = votes.loc[dev.index, name].notna()
    dev_correct = votes.loc[dev.index, name][dev_hit] == dev[TARGET][dev_hit]
    report.loc[name, "akurasi_dev"] = dev_correct.mean()
print(report.round(3))

# %% [markdown]
# ## 3. Gabungkan suara: majority vote
# Kelas dengan suara terbanyak menang. Seri atau tanpa suara = tidak berlabel (tidak dipakai
# latih).

# %%
def majority_vote(row_votes):
    counts = {}
    for name in LABELING_RULES:
        label = row_votes[name]
        if label is None:
            continue
        weight = 1.0
        # weight = report.loc[name, "akurasi_dev"]   # alternatif: voting berbobot akurasi
        counts[label] = counts.get(label, 0) + weight
    if len(counts) == 0:
        return None
    highest = max(counts.values())
    winners = []
    for label in counts:
        if counts[label] == highest:
            winners.append(label)
    if len(winners) > 1:
        return None
    return winners[0]


print(majority_vote(votes.iloc[0]))

weak_labels = []
for i in votes.index:
    weak_labels.append(majority_vote(votes.loc[i]))
train["label_lemah"] = weak_labels

# %% [markdown]
# ## Hasil label lemah

# %%
labeled = train[train["label_lemah"].notna()]
print("teks berlabel lemah:", len(labeled), "dari", len(train))
print(labeled["label_lemah"].value_counts())
print("akurasi label lemah (cek latihan):", (labeled["label_lemah"] == labeled[TARGET]).mean())

# %% [markdown]
# ## 4. Model: latih di label lemah
# Model belajar dari kata lain di teks yang kena aturan, sehingga bisa menebak teks yang tidak
# kena aturan sama sekali. Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
model = LogisticRegression(max_iter=2000, class_weight="balanced")
# model = LinearSVC(C=0.5, class_weight="balanced")
pipe = make_pipeline(vectorizer, model)
pipe.fit(labeled[TEXT_COL], labeled["label_lemah"])

# %% [markdown]
# ## 5. Submission

# %%
predictions = pipe.predict(test[TEXT_COL])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan (HAPUS saat lomba)
# Pembanding: model yang sama dilatih di label asli (batas atas).

# %%
test_key = pd.read_csv(TEST_KEY_CSV)[TARGET]
print("F1 macro label lemah:", round(f1_score(test_key, predictions, average="macro"), 4))
print(classification_report(test_key, predictions))

true_pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=2000, class_weight="balanced"))
true_pipe.fit(train[TEXT_COL], train[TARGET])
true_predictions = true_pipe.predict(test[TEXT_COL])
print("F1 macro label asli (batas atas):",
      round(f1_score(test_key, true_predictions, average="macro"), 4))
