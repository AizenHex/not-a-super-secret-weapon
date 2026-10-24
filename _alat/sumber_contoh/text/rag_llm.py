# %% [markdown]
# # Teks — RAG: Cari Dokumen + Jawab dengan LLM Kecil
# **Kapan dipakai:** ada kumpulan dokumen (berita, FAQ, peraturan) dan pertanyaan TANPA
# paragraf jawabannya.
#
# **Langkah:**
# 1. Retrieval: cari dokumen paling relevan untuk tiap pertanyaan.
# 2. Generation: LLM membaca dokumen itu dan menjawab singkat.
#
# Keduanya diukur terpisah supaya tahu mana yang lemah.
#
# **Data demo:** FacQA. Semua `konteks` digabung jadi satu kumpulan dokumen, pertanyaan dicari
# ulang. Model: TF-IDF / MiniLM untuk mencari, Qwen2.5-0.5B-Instruct (`pretrained/qwen_05b`)
# untuk menjawab.

# %%
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/qa"   # GANTI saat lomba
TRAIN_CSV = f"{DATA_FOLDER}/facqa_train.csv"
VALID_CSV = f"{DATA_FOLDER}/facqa_valid.csv"
TEST_CSV = f"{DATA_FOLDER}/facqa_test.csv"
LLM_FOLDER = "../../../pretrained/qwen_05b"
EMBEDDING_MODEL = "../../../pretrained/minilm"

QUESTION_COL = "pertanyaan"
CONTEXT_COL = "konteks"
ANSWER_COL = "jawaban"
TOP_K = 2               # jumlah dokumen yang diberikan ke LLM
MAX_ANSWER_TOKENS = 24
BATCH_SIZE = 16
INSTRUCTION = ("Jawab pertanyaan hanya berdasarkan dokumen. Jawab sangat singkat (nama, angka, "
               "atau frasa pendek), salin kata dari dokumen, tanpa kalimat penjelasan.")
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data + kumpulan dokumen

# %%
train = pd.read_csv(TRAIN_CSV)
valid = pd.read_csv(VALID_CSV)
test = pd.read_csv(TEST_CSV)
documents = pd.concat([train[CONTEXT_COL], valid[CONTEXT_COL], test[CONTEXT_COL]])
documents = list(documents.drop_duplicates())
print("dokumen:", len(documents), "| pertanyaan valid:", len(valid), "| test:", len(test))
valid.head()

# %% [markdown]
# ## Retrieval
# `search` mengembalikan nomor TOP_K dokumen teratas untuk tiap pertanyaan.
# Pilih SATU blok: TF-IDF (aktif) atau embedding (hapus `#` di blok embedding, beri `#` pada
# blok TF-IDF).

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
document_vectors = vectorizer.fit_transform(documents)


def search(questions):
    scores = (vectorizer.transform(questions) @ document_vectors.T).toarray()
    return np.argsort(-scores, axis=1)[:, :TOP_K]


# encoder = SentenceTransformer(EMBEDDING_MODEL)
# document_vectors = encoder.encode(documents, batch_size=64, normalize_embeddings=True)
#
#
# def search(questions):
#     question_vectors = encoder.encode(questions, batch_size=64, normalize_embeddings=True)
#     return np.argsort(-(question_vectors @ document_vectors.T), axis=1)[:, :TOP_K]


print(search(["Siapa presiden pertama Indonesia?"]))

# %% [markdown]
# ## Evaluasi retrieval: apakah dokumen yang benar ada di TOP_K?

# %%
val_found = search(list(valid[QUESTION_COL]))
found_first = []
found_top_k = []
for found_rows, true_context in zip(val_found, valid[CONTEXT_COL]):
    found_texts = []
    for row in found_rows:
        found_texts.append(documents[row])
    found_first.append(found_texts[0] == true_context)
    found_top_k.append(true_context in found_texts)
found_top_k = np.array(found_top_k)
print("Recall@1:", round(np.mean(found_first), 4))
print(f"Recall@{TOP_K}:", round(found_top_k.mean(), 4))

# %% [markdown]
# ## Muat LLM

# %%
tokenizer = AutoTokenizer.from_pretrained(LLM_FOLDER, padding_side="left")
llm = AutoModelForCausalLM.from_pretrained(LLM_FOLDER, dtype="auto")
llm = llm.to(DEVICE)
llm.eval()

# %% [markdown]
# ## Generation: LLM menjawab berdasarkan dokumen yang ditemukan
# `make_prompt`: gabungkan instruksi + dokumen + pertanyaan menjadi satu prompt chat.
# `generate`: kirim beberapa prompt sekaligus, ambil teks jawaban saja.

# %%
def make_prompt(question, found_rows):
    found_texts = []
    for row in found_rows:
        found_texts.append(documents[row])
    content = INSTRUCTION + "\n\nDokumen:\n" + "\n\n".join(found_texts)
    content = content + "\n\nPertanyaan: " + question
    chat = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def generate(prompts):
    answers = []
    for start in range(0, len(prompts), BATCH_SIZE):
        encoded = tokenizer(prompts[start:start + BATCH_SIZE], return_tensors="pt",
                            padding=True).to(DEVICE)
        with torch.no_grad():
            output = llm.generate(**encoded, max_new_tokens=MAX_ANSWER_TOKENS, do_sample=False)
        new_tokens = output[:, encoded["input_ids"].shape[1]:]   # buang bagian prompt
        for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True):
            answers.append(text.strip())
    return answers


print(generate([make_prompt(valid.loc[0, QUESTION_COL], val_found[0])]))
print("jawaban asli:", valid.loc[0, ANSWER_COL])

# %% [markdown]
# ## Evaluasi jawaban: F1 token gaya SQuAD

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


val_prompts = []
for question, found_rows in zip(valid[QUESTION_COL], val_found):
    val_prompts.append(make_prompt(question, found_rows))
val_predictions = generate(val_prompts)

val_f1 = []
for prediction, truth in zip(val_predictions, valid[ANSWER_COL]):
    val_f1.append(token_f1(prediction, truth))
val_f1 = np.array(val_f1)
print("F1 token semua:", round(val_f1.mean(), 4),
      "| saat dokumen benar ditemukan:", round(val_f1[found_top_k].mean(), 4))

# %% [markdown]
# ## Lihat contoh jawaban

# %%
examples = valid[[QUESTION_COL, ANSWER_COL]].copy()
examples["prediksi"] = val_predictions
pd.set_option("display.max_colwidth", 80)
examples.head(10)

# %% [markdown]
# ## Prediksi test + submission

# %%
test_prompts = []
for question, found_rows in zip(test[QUESTION_COL], search(list(test[QUESTION_COL]))):
    test_prompts.append(make_prompt(question, found_rows))
predictions = generate(test_prompts)

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
