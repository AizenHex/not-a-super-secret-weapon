# %% [markdown]
# # Teks — Generasi (ringkasan / parafrase dengan LLM kecil lokal + ROUGE, chrF, BLEU)
# **Kapan dipakai:** soal meminta teks buatan model (ringkasan berita, parafrase kalimat,
# jawaban bebas) yang dinilai terhadap teks acuan memakai ROUGE / BLEU / chrF.
#
# **Cara kerja:** Qwen2.5-0.5B-Instruct (`pretrained/qwen_05b`, offline) diberi instruksi + 1
# contoh → teks dibuat tanpa sampling acak (hasil sama tiap dijalankan) → dinilai dengan
# `rouge_score` (ROUGE-1/2/L) dan `sacrebleu` (chrF, BLEU).
#
# **Pembanding wajib: Lead-3** (3 kalimat pertama artikel), karena model kecil sering kalah dari
# baseline ini.
#
# **Data demo:** 50 artikel berita IndoSum (Apache-2.0) + ringkasan manusia sebagai acuan.
# Ringkasan IndoSum sangat ekstraktif (mirip kalimat awal artikel), makanya Lead-3 kuat.

# %%
import re

import pandas as pd
import torch
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% [markdown]
# ## Konfigurasi

# %%
DATA_FOLDER = "../../../data_latihan/text/generasi"   # GANTI saat lomba
EXAMPLE_CSV = f"{DATA_FOLDER}/contoh.csv"
TEST_CSV = f"{DATA_FOLDER}/test.csv"
TEST_KEY_CSV = f"{DATA_FOLDER}/test_labels.csv"   # hanya ada di data latihan
LLM_FOLDER = "../../../pretrained/qwen_05b"

TEXT_COL = "artikel"
TARGET = "ringkasan"
MAX_ARTICLE_WORDS = 350   # artikel dipotong supaya muat di memori GPU
MAX_NEW_TOKENS = 110      # panjang ringkasan maksimal (acuan sekitar 60 kata = 100 token)
BATCH_SIZE = 5
INSTRUCTION = "Ringkas artikel berikut menjadi 3 kalimat berbahasa Indonesia."
# INSTRUCTION = "Tulis ulang teks berikut dengan kata-kata berbeda tetapi artinya sama."
DEVICE = "cpu"
if torch.cuda.is_available():
    DEVICE = "cuda"

# %% [markdown]
# ## Baca data

# %%
example = pd.read_csv(EXAMPLE_CSV)
test = pd.read_csv(TEST_CSV)
print(test.shape)
article_words = test[TEXT_COL].str.split().str.len()
reference_words = example[TARGET].str.split().str.len()
print("rata-rata kata artikel:", round(article_words.mean()))
print("rata-rata kata acuan  :", round(reference_words.mean()))

# %% [markdown]
# ## Pembanding: Lead-3
# Tiga kalimat pertama artikel (kalimat dipisah di tanda titik).

# %%
lead3 = []
for article in test[TEXT_COL]:
    sentences = re.split(r"(?<=[.!?])\s+", article)
    lead3.append(" ".join(sentences[:3]))
print(lead3[0][:300])

# %% [markdown]
# ## Muat LLM + 1 contoh percakapan (few-shot)
# Contoh dari `contoh.csv` dimasukkan sebagai percakapan sebelumnya.

# %%
tokenizer = AutoTokenizer.from_pretrained(LLM_FOLDER, padding_side="left")
llm = AutoModelForCausalLM.from_pretrained(LLM_FOLDER, dtype="auto")
llm = llm.to(DEVICE)
llm.eval()

example_article = " ".join(example.loc[0, TEXT_COL].split()[:MAX_ARTICLE_WORDS])
few_shot_messages = [
    {"role": "user", "content": INSTRUCTION + "\n\n" + example_article},
    {"role": "assistant", "content": example.loc[0, TARGET]},
]

# %% [markdown]
# ## Buat ringkasan
# Tiap batch: susun prompt chat → generate tanpa sampling → ambil teks barunya saja.

# %%
summaries = []
for start in range(0, len(test), BATCH_SIZE):
    prompts = []
    for article in test[TEXT_COL][start:start + BATCH_SIZE]:
        short_article = " ".join(article.split()[:MAX_ARTICLE_WORDS])
        messages = few_shot_messages + [{"role": "user",
                                         "content": INSTRUCTION + "\n\n" + short_article}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True))
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        output = llm.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                              repetition_penalty=1.1)
    new_tokens = output[:, encoded["input_ids"].shape[1]:]   # buang bagian prompt
    summaries.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
print(summaries[0])

# %% [markdown]
# ## Evaluasi: ROUGE, chrF, BLEU (memakai kunci, hanya untuk latihan)
# - ROUGE-1/2 = kecocokan kata / pasangan kata, ROUGE-L = urutan kata terpanjang yang sama.
# - chrF = kecocokan huruf (cocok untuk bahasa Indonesia yang banyak imbuhan).
#
# Semua dalam persen (0–100).

# %%
references = list(pd.read_csv(TEST_KEY_CSV)[TARGET])
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])


def score_outputs(outputs):
    totals = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    for output, reference in zip(outputs, references):
        scores = scorer.score(reference, output)     # urutan: (acuan, hasil)
        for name in totals:
            totals[name] = totals[name] + scores[name].fmeasure
    row = {}
    for name in totals:
        row[name] = 100 * totals[name] / len(references)
    row["chrF"] = CHRF().corpus_score(outputs, [references]).score
    row["BLEU"] = BLEU().corpus_score(outputs, [references]).score
    return row


result = pd.DataFrame()
result["LLM Qwen 0.5B"] = pd.Series(score_outputs(summaries))
result["Lead-3"] = pd.Series(score_outputs(lead3))
result.T.round(1)

# %% [markdown]
# ## Panjang hasil dan contoh berdampingan
# Hasil yang jauh lebih panjang/pendek dari acuan biasanya merusak ROUGE.

# %%
llm_words = pd.Series(summaries).str.split()
llm_words = llm_words.str.len()
key_words = pd.Series(references).str.split()
key_words = key_words.str.len()
print("rata-rata kata: LLM", round(llm_words.mean()), "| acuan", round(key_words.mean()))
print("ACUAN :", references[1][:300])
print("LLM   :", summaries[1][:300])
print("LEAD-3:", lead3[1][:300])

# %% [markdown]
# ## Submission
# Lead-3 lebih baik di validasi? Ganti `summaries` dengan `lead3`.

# %%
submission = pd.DataFrame()
submission["id"] = test["id"]
submission["ringkasan"] = summaries
submission.to_csv("submission.csv", index=False)
submission.head()
