# %% [markdown]
# # Teks — Augmentasi Data Teks (data latih sedikit)
# **Kapan dipakai:** data berlabel hanya ratusan baris. Augmentasi = membuat salinan teks yang
# sedikit diubah agar model melihat variasi lebih banyak:
# - **hapus** kata acak, **tukar** posisi 2 kata acak;
# - **slang**: ganti kata baku dengan versi alay (kamus alay dibalik: baku → slang).
#
# **Aturan penting:** augmentasi HANYA pada data latih; validasi dan test tetap teks asli.
#
# **Data demo:** EmoT (emosi tweet), sengaja hanya dipakai 400 baris (sel Baca data).

# %%
import random
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/classification"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/emot_train.csv"
TEST_CSV = f"{DATA_FOLDER}/emot_test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/emot_test_labels.csv"   # hanya ada di data latihan
SLANG_CSV = "Z:/GEMASTIK-DATABASE/data_latihan/preprocessing/kamus_alay_salsabila.csv"

TEXT_COL = "text"
TARGET = "label"
N_COPIES = 2          # jumlah teks augmentasi per teks asli
CHANGE_CHANCE = 0.15  # peluang tiap kata diubah
SEED = 42
random.seed(SEED)

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
test = pd.read_csv(TEST_CSV)
train = train.sample(400, random_state=SEED)   # HAPUS baris ini saat lomba (simulasi data sedikit)
print("train:", train.shape, "| test:", test.shape)
print(train[TARGET].value_counts())
train.head()

# %% [markdown]
# ## Bersihkan teks

# %%
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\[username\]|http\S+|@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


print(clean_text("[USERNAME] Seneng bgt hari ini!!! http://x.id"))

train[TEXT_COL] = train[TEXT_COL].fillna("").apply(clean_text)
test[TEXT_COL] = test[TEXT_COL].fillna("").apply(clean_text)

# %% [markdown]
# ## Split train / validasi

# %%
X_train, X_val, y_train, y_val = train_test_split(
    train[TEXT_COL], train[TARGET], test_size=0.25, stratify=train[TARGET], random_state=SEED
)

# %% [markdown]
# ## Kamus terbalik: kata baku → daftar versi slang

# %%
slang_table = pd.read_csv(SLANG_CSV)
standard_to_slang = {}
for slang, standard in zip(slang_table["slang"], slang_table["baku"]):
    if " " in str(standard):
        continue                       # lewati padanan lebih dari satu kata
    if standard not in standard_to_slang:
        standard_to_slang[standard] = []
    standard_to_slang[standard].append(slang)
print(len(standard_to_slang), "kata baku punya versi slang")

# %% [markdown]
# ## Fungsi augmentasi satu teks
# 1. Hapus kata acak. 2. Tukar posisi 2 kata acak. 3. Ganti kata baku dengan slang.

# %%
def augment(text):
    words = text.split()
    if len(words) < 3:
        return text
    kept = []
    for word in words:
        if random.random() > CHANGE_CHANCE:
            kept.append(word)
    if len(kept) >= 2:
        first, second = random.sample(range(len(kept)), 2)
        kept[first], kept[second] = kept[second], kept[first]
    result = []
    for word in kept:
        if word in standard_to_slang and random.random() < CHANGE_CHANCE * 2:
            word = random.choice(standard_to_slang[word])
        result.append(word)
    return " ".join(result)


print(X_train.iloc[0])
print(augment(X_train.iloc[0]))

# %% [markdown]
# ## Tambahkan salinan augmentasi ke data latih
# Tiap teks asli mendapat `N_COPIES` salinan dengan label yang sama.

# %%
def add_augmented(texts, labels):
    new_texts = []
    new_labels = []
    for text, label in zip(texts, labels):
        for copy_number in range(N_COPIES):
            new_texts.append(augment(text))
            new_labels.append(label)
    all_texts = pd.concat([texts, pd.Series(new_texts)], ignore_index=True)
    all_labels = pd.concat([labels, pd.Series(new_labels)], ignore_index=True)
    return all_texts, all_labels


X_augmented, y_augmented = add_augmented(X_train, y_train)
print("data latih:", len(X_train), "→ setelah augmentasi:", len(X_augmented))

# %% [markdown]
# ## Model
# Pilih SATU vectorizer dan SATU model: hapus `#` di baris yang ingin dipakai.

# %%
vectorizer = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2))
# vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)

model = LogisticRegression(C=5, max_iter=3000)
# model = LinearSVC(C=0.5)
# model = ComplementNB(alpha=0.3)

pipe = make_pipeline(vectorizer, model)

# %% [markdown]
# ## Evaluasi: tanpa vs dengan augmentasi (validasi = teks asli)

# %%
pipe.fit(X_train, y_train)
plain_predictions = pipe.predict(X_val)
print("F1 macro tanpa augmentasi :", round(f1_score(y_val, plain_predictions, average="macro"), 4))

pipe.fit(X_augmented, y_augmented)
val_predictions = pipe.predict(X_val)
print("F1 macro dengan augmentasi:", round(f1_score(y_val, val_predictions, average="macro"), 4))
print(classification_report(y_val, val_predictions))

# %% [markdown]
# ## Latih ulang di semua data (+ augmentasi) + submission
# Augmentasi tidak membantu di validasi? Latih dengan `train[TEXT_COL], train[TARGET]` saja.

# %%
X_all, y_all = add_augmented(train[TEXT_COL], train[TARGET])
pipe.fit(X_all, y_all)
predictions = pipe.predict(test[TEXT_COL])

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["label"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: file kunci hanya ada di data latihan.

# %%
test_key = pd.read_csv(TEST_KEY_CSV)["label"]
print("F1 macro test (kunci latihan):", round(f1_score(test_key, predictions, average="macro"), 4))
