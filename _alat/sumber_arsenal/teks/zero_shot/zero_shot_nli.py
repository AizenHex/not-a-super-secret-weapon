# %% [markdown]
# # Klasifikasi teks TANPA data berlabel (zero-shot) + self-training
#
# **Butuh:** `texts` (tabel dengan kolom teks), daftar nama kelas. Sangat dianjurkan: 30–50
# contoh yang kamu labeli sendiri untuk memilih kalimat template.
# **Menghasilkan:** `texts["pred"]`, `texts["confidence"]`, `submission.csv`.
#
# Cara (Yin dkk. 2019): model NLI menilai apakah teks (premis) **menyiratkan** kalimat
# "Teks ini tentang {kelas}." (hipotesis). Kelas dengan proba entailment tertinggi = tebakan.
# Hasil sangat bergantung pada kata kelas dan template → selalu bandingkan beberapa (sel 2).
# Lalu self-training (Gera dkk. 2022): tebakan paling yakin dipakai sebagai label untuk melatih
# TF-IDF, yang lalu menebak semua teks (sering naik beberapa poin).
#
# ## Cara menempel ke notebook-mu
#
# | sel | isi | wajib? |
# |---|---|---|
# | 1 | konfigurasi + `zero_shot_scores` (mDeBERTa NLI) | **ya** |
# | 2 | pilih template + kata kelas pada contoh berlabel sendiri | disarankan |
# | 3 | terapkan ke SEMUA teks | **ya** |
# | 4 | alternatif cepat: kemiripan embedding ke deskripsi kelas | opsional |
# | 5 | self-training: tebakan yakin → latih TF-IDF → tebak ulang | disarankan |
# | 6 | `submission.csv` | **ya** |
#
# Label sendiri dengan cepat: urutkan `confidence` terendah, cek di Excel, perbaiki.

# %% [markdown]
# ## 0. Data contoh (lewati saat lomba)
# Tweet emosi; label asli dipura-purakan tidak ada (hanya untuk menilai di akhir).

# %%
import pandas as pd

DEMO_FOLDER = "data_latihan"   # folder data_latihan (di sebelah repo)

texts = pd.read_csv(f"{DEMO_FOLDER}/text/classification/emot_train.csv")
texts = texts.sample(600, random_state=42).reset_index(drop=True)
print(texts["label"].value_counts())

# %% [markdown]
# ## 1. Konfigurasi + `zero_shot_scores(text_list, label_words, template)`
# **Menghasilkan:** array (jumlah teks, jumlah kelas), tiap baris berjumlah 1.
# `LABEL_WORDS`: nama kelas di soal → kata yang dimasukkan ke template (bahasa teksnya).

# %%
import numpy as np
import torch
from transformers import pipeline

PRETRAINED_FOLDER = "pretrained"   # UBAH
TEXT = "text"                      # UBAH
LABEL_WORDS = {"anger": "marah", "fear": "takut", "happy": "senang", "love": "cinta",
               "sadness": "sedih"}   # UBAH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
classifier = pipeline("zero-shot-classification", model=f"{PRETRAINED_FOLDER}/nli_mdeberta",
                      device=DEVICE)


def zero_shot_scores(text_list, label_words, template):
    words = list(label_words.values())
    results = classifier(text_list, candidate_labels=words, hypothesis_template=template,
                         batch_size=16)
    scores = np.zeros((len(text_list), len(words)))
    for row, result in enumerate(results):
        for word, score in zip(result["labels"], result["scores"]):
            scores[row, words.index(word)] = score
    return scores


print(zero_shot_scores(["Aku kesal sekali, antrean nggak maju-maju!"], LABEL_WORDS,
                       "Penulis teks ini merasa {}.").round(3))

# %% [markdown]
# ## 2. Pilih template + kata kelas pada contoh berlabel sendiri → `TEMPLATE`
# `labeled` = 30–50 teks yang kamu labeli sendiri (di demo: 60 baris pertama).
# Coba juga mengganti kata kelas (`"senang"` → `"bahagia"`), lalu pilih yang F1-nya tertinggi.

# %%
import numpy as np
from sklearn.metrics import f1_score

TEMPLATES = ["Teks ini tentang {}.", "Penulis teks ini merasa {}.", "Perasaan: {}.",
             "Orang ini sedang {}."]   # UBAH
labeled = texts.head(60)              # UBAH: contoh berlabel buatanmu
class_names = np.array(list(LABEL_WORDS.keys()))
best_template, best_f1 = TEMPLATES[0], -1.0
for template in TEMPLATES:
    scores = zero_shot_scores(labeled[TEXT].tolist(), LABEL_WORDS, template)
    guess = class_names[scores.argmax(axis=1)]
    score = f1_score(labeled["label"], guess, average="macro")
    print(f"{template:30s} F1 macro: {score:.4f}")
    if score > best_f1:
        best_template, best_f1 = template, score
print("terpilih:", best_template)

# %% [markdown]
# ## 3. Terapkan ke SEMUA teks → `texts["pred"]`, `texts["confidence"]`
# Multilabel (satu teks boleh beberapa kelas): tambahkan `multi_label=True` di
# `classifier(...)`, lalu tiap kelas dengan skor ≥ 0,5 dianggap ada.

# %%
import numpy as np

TEMPLATE = best_template   # UBAH: atau tulis langsung, mis. "Orang ini sedang {}."
all_scores = zero_shot_scores(texts[TEXT].tolist(), LABEL_WORDS, TEMPLATE)
texts["pred"] = class_names[all_scores.argmax(axis=1)]
texts["confidence"] = all_scores.max(axis=1)
print(texts["pred"].value_counts())
texts.sort_values("confidence").to_csv("zero_shot_cek.csv", index=False)  # paling ragu di atas

# %% [markdown]
# ## 4. Alternatif cepat: kemiripan embedding ke deskripsi kelas
# Tiap kelas diberi 1–3 kalimat deskripsi; teks masuk ke kelas dengan cosine tertinggi.
# 10× lebih cepat dari NLI, biasanya lebih lemah; bagus untuk ribuan teks / CPU.

# %%
import numpy as np
from sentence_transformers import SentenceTransformer

DESCRIPTIONS = {"anger": "aku marah, kesal, benci", "fear": "aku takut, cemas, ngeri",
                "happy": "aku senang, bahagia, bersyukur", "love": "aku cinta, sayang kamu",
                "sadness": "aku sedih, kecewa, rindu"}   # UBAH
encoder = SentenceTransformer(f"{PRETRAINED_FOLDER}/minilm", device=DEVICE)
class_vectors = encoder.encode(list(DESCRIPTIONS.values()), normalize_embeddings=True)
text_vectors = encoder.encode(texts[TEXT].tolist(), normalize_embeddings=True)
similarity = text_vectors @ class_vectors.T
texts["pred_embedding"] = np.array(list(DESCRIPTIONS.keys()))[similarity.argmax(axis=1)]
print(texts["pred_embedding"].value_counts())

# %% [markdown]
# ## 5. Self-training → `texts["pred_self"]`
# Ambil `CONFIDENT_SHARE` tebakan paling yakin PER KELAS (supaya kelas kecil tetap ada),
# latih LogisticRegression di atas vektor MiniLM (sel 4), lalu tebak semua teks.
# Tidak selalu naik: bandingkan dulu di contoh berlabel sendiri sebelum dipakai.

# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression

CONFIDENT_SHARE = 0.4   # UBAH: 0.3–0.5
confident = []
for label, group in texts.groupby("pred"):
    count = max(int(len(group) * CONFIDENT_SHARE), 5)
    confident.append(group.nlargest(count, "confidence"))
confident = pd.concat(confident)
student = LogisticRegression(C=1, max_iter=3000, class_weight="balanced")   # UBAH: C
student.fit(text_vectors[confident.index], confident["pred"])
texts["pred_self"] = student.predict(text_vectors)
print(len(confident), "teks yakin dipakai melatih")

# %% [markdown]
# Bandingkan ke label asli (hanya demo; saat lomba bandingkan di contoh berlabel sendiri).

# %%
from sklearn.metrics import f1_score

for column in ["pred", "pred_embedding", "pred_self"]:
    print(f"{column:15s} F1 macro: {f1_score(texts['label'], texts[column], average='macro'):.4f}")

# %% [markdown]
# ## 6. `submission.csv`

# %%
import pandas as pd

submission = pd.DataFrame()
submission["id"] = texts["id"]           # UBAH
submission["label"] = texts["pred"]      # UBAH: kolom tebakan terbaik
submission.to_csv("submission.csv", index=False)
print(submission.head())
