# %% [markdown]
# # Teks — POS Tagging / Pelabelan Urutan per Token (CRF)
# **Kapan dipakai:** tiap kata diberi kelas kata (kata benda, kerja, sifat, ...). Pola yang sama
# dipakai untuk chunking, tanda baca, atau label per token lain.
#
# **Bedanya dengan NER:** hampir semua token punya tag penting, jadi metrik = akurasi token +
# F1 macro per tag.
#
# **Data demo:** POSP (IndoNLU, 26 tag).

# %%
import pandas as pd
import sklearn_crfsuite
from sklearn.metrics import accuracy_score, classification_report, f1_score

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/pos"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/posp_train.csv"
VALID_CSV = f"{DATA_FOLDER}/posp_valid.csv"
TEST_CSV = f"{DATA_FOLDER}/posp_test.csv"

SENTENCE_COL = "kalimat_id"
TOKEN_COL = "token"
TAG_COL = "tag"

# %% [markdown]
# ## Baca data
# Satu baris = satu token. `keep_default_na=False` supaya token seperti "NA" tidak jadi kosong.

# %%
train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
valid = pd.read_csv(VALID_CSV, keep_default_na=False)
test = pd.read_csv(TEST_CSV, keep_default_na=False)
print("train:", train.shape, "| valid:", valid.shape, "| test:", test.shape)
print("test terurut per kalimat:", test[SENTENCE_COL].is_monotonic_increasing)
train.head(10)

# %% [markdown]
# ## EDA: jumlah tiap tag dan panjang kalimat

# %%
print(train[TAG_COL].value_counts())
sentence_lengths = train.groupby(SENTENCE_COL).size()
print(sentence_lengths.describe())

# %% [markdown]
# ## Kelompokkan token per kalimat
# Sudah ada file valid. Tidak ada? Pecah `kalimat_id` train dengan `GroupShuffleSplit`.

# %%
def to_sentences(table, col):
    sentences = []
    for sentence_id, group in table.groupby(SENTENCE_COL):
        sentences.append(list(group[col]))
    return sentences


train_tokens = to_sentences(train, TOKEN_COL)
train_tags = to_sentences(train, TAG_COL)
valid_tokens = to_sentences(valid, TOKEN_COL)
valid_tags = to_sentences(valid, TAG_COL)
test_tokens = to_sentences(test, TOKEN_COL)
print(len(train_tokens), "kalimat train,", len(valid_tokens), "valid,", len(test_tokens), "test")

# %% [markdown]
# ## Feature per token
# Tambah/hapus baris di fungsi ini untuk mencoba feature lain.

# %%
def token_features(sentence, i):
    word = sentence[i]
    features = {"bias": 1.0, "kata": word.lower(), "huruf_besar": word.istitle()}
    features["angka"] = word.isdigit()
    features["tanda_baca"] = not word.isalnum()
    features["akhiran2"] = word[-2:].lower()
    features["akhiran3"] = word[-3:].lower()
    features["awalan2"] = word[:2].lower()
    features["awalan3"] = word[:3].lower()
    if i > 0:
        features["kata_sebelum"] = sentence[i - 1].lower()
    else:
        features["awal_kalimat"] = True
    if i < len(sentence) - 1:
        features["kata_sesudah"] = sentence[i + 1].lower()
    else:
        features["akhir_kalimat"] = True
    return features


print(token_features(train_tokens[0], 0))

# %% [markdown]
# ## Ubah semua kalimat menjadi daftar feature

# %%
def all_features(sentences):
    result = []
    for sentence in sentences:
        sentence_features = []
        for i in range(len(sentence)):
            sentence_features.append(token_features(sentence, i))
        result.append(sentence_features)
    return result


X_train = all_features(train_tokens)
X_val = all_features(valid_tokens)
X_test = all_features(test_tokens)

# %% [markdown]
# ## Model (CRF)
# `c1`, `c2` = kekuatan regularisasi. Pilih SATU model: hapus `#` di baris yang ingin dipakai.

# %%
model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100,
                             all_possible_transitions=True)
# model = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=0.01, max_iterations=200,
#                              all_possible_transitions=True)
# model = sklearn_crfsuite.CRF(algorithm="l2sgd", c2=0.1, max_iterations=200)
# model = sklearn_crfsuite.CRF(algorithm="ap", max_iterations=50)   # averaged perceptron

model.fit(X_train, train_tags)

# %% [markdown]
# ## Evaluasi di valid (per token)
# Tebakan per kalimat disambung menjadi satu daftar per token.

# %%
val_predictions = []
for sentence_tags in model.predict(X_val):
    val_predictions.extend(sentence_tags)
val_true = []
for sentence_tags in valid_tags:
    val_true.extend(sentence_tags)

print("akurasi token:", round(accuracy_score(val_true, val_predictions), 4))
print("F1 macro per tag:", round(f1_score(val_true, val_predictions, average="macro"), 4))
print(classification_report(val_true, val_predictions, digits=3, zero_division=0))

# %% [markdown]
# ## Latih ulang (train + valid) + submission
# Urutan token sama dengan file test karena kalimat sudah terurut.

# %%
model.fit(X_train + X_val, train_tags + valid_tags)
predictions = []
for sentence_tags in model.predict(X_test):
    predictions.extend(sentence_tags)

submission = pd.DataFrame()
submission["kalimat_id"] = test["kalimat_id"]
submission["token"] = test["token"]
submission["tag"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: kolom `tag` di file test hanya ada di data latihan.

# %%
print("akurasi token test:", round(accuracy_score(test[TAG_COL], predictions), 4))
print("F1 macro test:", round(f1_score(test[TAG_COL], predictions, average="macro"), 4))
