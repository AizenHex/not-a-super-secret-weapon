# %% [markdown]
# # Teks — Named Entity Recognition (NER) dengan CRF
# **Kapan dipakai:** tiap kata (token) dalam kalimat diberi tag entitas, misalnya nama orang,
# tempat, organisasi, dengan format BIO (`B-PER` = awal nama orang, `I-PER` = lanjutan,
# `O` = bukan entitas).
#
# **Langkah:** kelompokkan token per kalimat → feature tiap token (kata itu, awalan/akhiran,
# kata sebelum/sesudah) → CRF → evaluasi per token.
#
# **Data demo:** NERP (kalimat Bahasa Indonesia, tag BIO). Test berlabel dipakai sebagai
# evaluasi akhir (jangan dipakai untuk tuning).

# %%
import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../data_latihan/text/ner"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/nerp_train.csv"
TEST_CSV = f"{DATA_FOLDER}/nerp_test.csv"

SENTENCE_COL = "sentence_id"
TOKEN_COL = "token"
TAG_COL = "tag"

# %% [markdown]
# ## Baca data
# Satu baris = satu token. `keep_default_na=False` supaya token seperti "NA" tidak jadi kosong.

# %%
train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
test = pd.read_csv(TEST_CSV, keep_default_na=False)
print("train:", train.shape, "| test:", test.shape)
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
# CRF butuh daftar kalimat; tiap kalimat = daftar token (dan daftar tag).

# %%
def to_sentences(table, col):
    sentences = []
    for sentence_id, group in table.groupby(SENTENCE_COL):
        sentences.append(list(group[col]))
    return sentences


train_tokens = to_sentences(train, TOKEN_COL)
train_tags = to_sentences(train, TAG_COL)
test_tokens = to_sentences(test, TOKEN_COL)
test_tags = to_sentences(test, TAG_COL)
print(len(train_tokens), "kalimat train,", len(test_tokens), "kalimat test")
print(train_tokens[0])
print(train_tags[0])

# %% [markdown]
# ## Feature per token
# Tambah/hapus baris di fungsi ini untuk mencoba feature lain.

# %%
def token_features(sentence, i):
    word = sentence[i]
    features = {"bias": 1.0, "kata": word.lower(), "huruf_besar": word.istitle()}
    features["angka"] = word.isdigit()
    features["akhiran3"] = word[-3:].lower()
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


def sentence_features(sentence):
    features = []
    for i in range(len(sentence)):
        features.append(token_features(sentence, i))
    return features


print(token_features(train_tokens[0], 0))

# %% [markdown]
# ## Ubah semua kalimat menjadi daftar feature

# %%
X_train = []
for sentence in train_tokens:
    X_train.append(sentence_features(sentence))
X_test = []
for sentence in test_tokens:
    X_test.append(sentence_features(sentence))

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
# model = sklearn_crfsuite.CRF(algorithm="pa", max_iterations=50)   # passive aggressive

model.fit(X_train, train_tags)

# %% [markdown]
# ## Evaluasi di test
# Skor per token, tanpa tag `O` (tag `O` sangat banyak dan membuat skor terlihat bagus palsu).

# %%
test_predictions = model.predict(X_test)
entity_tags = sorted(set(train[TAG_COL]) - {"O"})
weighted_f1 = metrics.flat_f1_score(test_tags, test_predictions, average="weighted",
                                    labels=entity_tags)
print("F1 weighted:", round(weighted_f1, 4))
print(metrics.flat_classification_report(test_tags, test_predictions, labels=entity_tags,
                                         digits=3))

# %% [markdown]
# ## Submission
# Model sudah dilatih di seluruh train. Tebakan per kalimat disambung menjadi satu daftar per
# token; urutannya sama dengan file test karena kalimat sudah terurut.

# %%
predicted_tags = []
for sentence_tags in test_predictions:
    predicted_tags.extend(sentence_tags)

submission = pd.DataFrame()
submission["sentence_id"] = test["sentence_id"]
submission["token"] = test["token"]
submission["tag"] = predicted_tags
submission.to_csv("submission.csv", index=False)
submission.head()
