# %% [markdown]
# # Teks — Tanya Jawab Ekstraktif (jawaban = potongan dari konteks)
# **Kapan dipakai:** diberi pertanyaan + paragraf, cari potongan teks yang menjawab.
#
# **Cara:** model QA pretrained langsung dipakai tanpa latih (mDeBERTa SQuAD2 multibahasa,
# `pretrained/qa_mdeberta`). Model memberi skor "awal jawaban" dan "akhir jawaban" untuk tiap
# token konteks; jawaban = pasangan (awal, akhir) dengan skor tertinggi.
#
# **Metrik gaya SQuAD:** Exact Match (jawaban persis) dan F1 token (irisan kata).
#
# **Data demo:** FacQA (berita Indonesia). Di data latihan, file test masih punya kolom
# `jawaban` (untuk sel kunci).

# %%
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "Z:/GEMASTIK-DATABASE/data_latihan/text/qa"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/facqa_train.csv"
VALID_CSV = f"{DATA_FOLDER}/facqa_valid.csv"
TEST_CSV = f"{DATA_FOLDER}/facqa_test.csv"
QA_MODEL = "Z:/GEMASTIK-DATABASE/pretrained/qa_mdeberta"

QUESTION_COL = "pertanyaan"
CONTEXT_COL = "konteks"
ANSWER_COL = "jawaban"
MAX_ANSWER_TOKENS = 15     # naikkan (mis. 30) bila jawaban sering terpotong
MAX_LENGTH = 384           # panjang maksimal pertanyaan + konteks (token)
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
train = pd.read_csv(TRAIN_CSV)
valid = pd.read_csv(VALID_CSV)
test = pd.read_csv(TEST_CSV)
print("train:", train.shape, "| valid:", valid.shape, "| test:", test.shape)
train.head()

# %% [markdown]
# ## EDA: panjang konteks, panjang jawaban, kata tanya

# %%
context_words = train[CONTEXT_COL].str.split().str.len()
print("kata per konteks:")
print(context_words.describe().round(1))

answer_words = train[ANSWER_COL].str.split().str.len()
answer_word_counts = answer_words.value_counts().sort_index()
answer_word_counts.head(10).plot.bar()
plt.title("Jumlah kata per jawaban")
plt.show()

question_word = train[QUESTION_COL].str.split().str[0]
question_word = question_word.str.lower()
print(question_word.value_counts().head(8))

# %% [markdown]
# ## Metrik: Exact Match dan F1 token
# Teks dinormalisasi dulu: huruf kecil, tanda baca dibuang, dipecah per kata.

# %%
def normalize(text):
    text = re.sub(r"[^\w\s]", " ", str(text).lower())
    return text.split()


def token_f1(prediction, truth):
    common = Counter(normalize(prediction)) & Counter(normalize(truth))
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(normalize(prediction))
    recall = n_same / len(normalize(truth))
    return 2 * precision * recall / (precision + recall)


print(token_f1("Presiden Joko Widodo", "Joko Widodo"))

# %% [markdown]
# ## Model QA

# %%
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL)
model = model.to(DEVICE)
model.eval()

# %% [markdown]
# ## Fungsi menjawab satu pertanyaan
# 1. Tokenisasi pertanyaan + konteks. `offsets` = posisi huruf tiap token di konteks.
# 2. Model memberi skor awal dan akhir untuk tiap token.
# 3. Cari pasangan (awal, akhir) di dalam konteks dengan skor total tertinggi, panjang maksimal
#    `MAX_ANSWER_TOKENS`.

# %%
def answer(question, context):
    encoded = tokenizer(question, context, truncation="only_second", max_length=MAX_LENGTH,
                        return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping")[0].numpy()
    in_context = np.array(encoded.sequence_ids(0)) == 1
    with torch.no_grad():
        output = model(**encoded.to(DEVICE))
    start_scores = output.start_logits[0].cpu().numpy()
    end_scores = output.end_logits[0].cpu().numpy()
    best_score = -np.inf
    best_start = 0
    best_end = 0
    for start in np.where(in_context)[0]:
        for end in range(start, min(start + MAX_ANSWER_TOKENS, len(end_scores))):
            score = start_scores[start] + end_scores[end]
            if in_context[end] and score > best_score:
                best_score = score
                best_start = start
                best_end = end
    return context[offsets[best_start][0]:offsets[best_end][1]].strip()


# coba pada contoh validasi pertama
print("pertanyaan:", valid.loc[0, QUESTION_COL])
print("jawaban model:", answer(valid.loc[0, QUESTION_COL], valid.loc[0, CONTEXT_COL]))
print("jawaban asli :", valid.loc[0, ANSWER_COL])

# %% [markdown]
# ## Evaluasi di data validasi

# %%
val_predictions = []
for question, context in zip(valid[QUESTION_COL], valid[CONTEXT_COL]):
    val_predictions.append(answer(question, context))

exact_matches = []
f1_scores = []
for prediction, truth in zip(val_predictions, valid[ANSWER_COL]):
    exact_matches.append(normalize(prediction) == normalize(truth))
    f1_scores.append(token_f1(prediction, truth))
print("Exact Match:", round(np.mean(exact_matches), 4), "| F1 token:", round(np.mean(f1_scores), 4))

examples = valid[[QUESTION_COL, ANSWER_COL]].copy()
examples["prediksi"] = val_predictions
pd.set_option("display.max_colwidth", 80)
examples.head(10)

# %% [markdown]
# ## Prediksi test + submission (model tidak dilatih, langsung dipakai)

# %%
predictions = []
for question, context in zip(test[QUESTION_COL], test[CONTEXT_COL]):
    predictions.append(answer(question, context))

submission = pd.DataFrame()
submission["id"] = test["id"]
submission["jawaban"] = predictions
submission.to_csv("submission.csv", index=False)
submission.head()

# %% [markdown]
# ## Cek dengan kunci jawaban latihan
# **HAPUS sel ini saat lomba**: kolom `jawaban` di file test hanya ada di data latihan.

# %%
test_f1 = []
for prediction, truth in zip(predictions, test[ANSWER_COL]):
    test_f1.append(token_f1(prediction, truth))
print("F1 token test (kunci latihan):", round(np.mean(test_f1), 4))
